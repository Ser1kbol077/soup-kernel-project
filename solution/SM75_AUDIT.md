# SM75 compatibility audit and guarded native backward

## Confirmed defect and limits

The team reports that both solution runs of
`3351e4b4cab8fcd4abbded676cdcea973256d3e5` failed on Kaggle T4, while baseline
training completed and the 79 protected Soup regression tests passed. No
`train.log` traceback or per-run exit codes have been supplied. The cause of
those failures is therefore still unknown.

The audit did find a concrete dispatch defect: `mlp_input_grad` admitted SM75
assuming that ordinary `tl.dot` would use FP16 Tensor Cores. In official
[Triton 3.6.0 AccelerateMatmul.cpp](https://github.com/triton-lang/triton/blob/v3.6.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L344-L350),
the compiler explicitly declines MMA acceleration below SM80 and uses FMA.
The local SM75 PTX for both MLP orientations contains `fma.rn.f32`, with no
`mma.sync` instructions. Local assembly also reports register spills. This
can cause substantial extra work and local-memory traffic, but does not by
itself prove a Kaggle timeout, OOM, or other particular failure.

## Minimal correction

- `mlp_dx._supported` requires SM80+ before selecting the custom GEMM.
- `mlp_backward._eligible_context` also requires SM80+, so T4 invokes the
  original `MatMul4Bit.backward` before our decode or matrix-product path.
- Direct helper calls on SM75 retain native `torch.matmul` semantics.
- Vocabulary, causal loss, RMSNorm, compact SwiGLU and patch order are unchanged.
  No task, LoRA, optimizer, allocator or measurement settings are changed.
- No new kernels, tuning, artificial work or exception-suppression fallback is
  added. The T4 MLP GEMM is no longer claimed toward original-kernel share.

This fixes a demonstrated compatibility assumption. Successful training and
all ten official checks, including the required >=5% share, still require T4
measurement. Other new kernels remain unverified in a complete training run.

## Other contracts checked against the pinned sources

- [bitsandbytes 0.50.2 MatMul4Bit](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/autograd/_functions.py#L273-L352):
  canonical packed layout, saved context, native decode/cast, and five backward
  return values match our wrapper. No transpose error was identified.
- [Transformers 5.16.1 loss function property](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/modeling_utils.py#L4287-L4303)
  supports the instance setter. Causal shifting and token normalization match
  its loss utility; mapping identity remains unchanged.
- [Qwen2 RMSNorm](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py#L215-L229)
  matches the fused frozen-FP32 operation order and first-order derivative.
- [Linear4bit](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/nn/modules.py#L564-L590)
  casts to compute dtype and returns the input dtype. Under the existing guard,
  the compact cache is lossless and its activation gradients remain FP32.
- `__init__.py` applies the example MLP patch before its compact wrapper;
  the other patches affect separate operations. No ordering defect was found.

## Local verification boundaries

Fifteen existing kernel signatures were compiled offline to SM75 cubin using
isolated `triton-windows==3.6.0.post24`, Python 3.14 and CUDA 12.8 ptxas. They
include both MLP orientations, RMSNorm forward/backward, FP16/FP32 vocabulary
paths and the mixed FP16-cache/FP32-gradient SwiGLU backward. Compilation
completed; **zero GPU kernels were executed**.

The Windows package is not Kaggle's official Linux package. Its JIT, language
core/semantic and code-generator Python files match official v3.6.0 by SHA256;
the NVIDIA backend has Windows-specific differences. This is supporting
compiler evidence, not T4 correctness or performance validation.

The compile-only probe and its logs are in ignored `runs/triton36_compile_*`.
Dispatch regression tests live in `solution/test_mlp_dispatch.py`, and the
existing real NF4 integration tests check the selected path by architecture.
CPU tests using FakeTensor metadata do not allocate CUDA tensors.

Before the correction, the new dispatch suite reproduced **7 failures / 9
passes**: SM75 entered the forbidden custom launch and the NF4 guard accepted
the unsupported architecture. After the correction, all **16 dispatch tests
passed on CPU**. With the existing MLP test file included, the local result was
**17 passed, 16 skipped** (CUDA or unavailable bitsandbytes). These are not T4
results. Existing numerical tolerances were not relaxed; the real NF4 test now
requires exact native gradients on SM75.

Python syntax for all 16 solution Python files, `git diff --check`, and the
unmodified `source_kernels()` inspection completed. It finds seven original
definitions and two excluded examples, no forbidden imports, parse errors or
example lookalikes. Definition count does not establish runtime kernel share.
The protected directories match `3351e4b` unchanged.

## Next Kaggle run

Pin the new full commit SHA in `kaggle_t4.ipynb`. Preserve the failed run logs.
Run solution tests and the protected Soup suite, then the unmodified harness
with a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 CUDA_VISIBLE_DEVICES=0 python -m pytest tests -q
python harness/run_task.py --repeats 2 --profile --out runs_sm75_fix
python harness/check_rules.py --result runs_sm75_fix/result.json
```

Collect both `train.log` files, per-run `exit_code`, `wall_s` and peak memory,
`result.json`, `checks.json`, raw solution `metrics.json` if present, actual
package versions and `TESTED_COMMIT`. A traceback or the explicit harness
`TIMEOUT after ...` line is still needed to identify the previous failure.
