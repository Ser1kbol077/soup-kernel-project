# Check 4 candidate: real MLP input-gradient GEMM

## Measured starting point and target

The team confirmed that commit `6b175fb51c0ee1d272d9b3098d1b3df275bbc270`
achieved 9/10 checks on Kaggle NVIDIA T4: only check 4 failed at **2.4%**.
Reported speedup was 1.084x, peak memory 14345 -> 12385 MiB and evaluation loss
2.04967 -> 2.04965. All 79 Soup regression tests passed. Exact per-kernel CUDA
event times and timed_loop_s were not supplied.

The parent candidate `b7d39a8fc4faa735fcb4a2559dc578de1e9666f8` adds RMSNorm and
memory optimizations but has not been measured on T4. Neither those changes nor
this candidate inherit the older commit's PASS status. The goal is **at least
6% measured original-kernel share**, giving margin above the mandatory 5%.

## Why this operation

RMSNorm is already present in the parent. Adding another implementation of it
would not expand coverage. Rank-16 q/v LoRA arithmetic is comparatively small;
changing its dropout/RNG/adapter path would add risk without evidence of enough
useful work. Large-vocabulary operations are already covered.

For frozen NF4 MLP linears, native bitsandbytes 0.50.2 backward still computes a
large dense input gradient:

```python
grad_input = grad_output @ dequantize_4bit(packed_weight, state).to(grad_output.dtype)
```

We replace only that matrix product. For Qwen2.5-1.5B's 28 layers, three MLP
projections and a full 4 x 512 token batch, the work is:

```text
2 * 2048 * 28 * 3 * 1536 * 8960 = 4.73520144384e12 FLOPs per step
```

This is a workload calculation, not a timing measurement or proof of 6%.
Dynamic padding and CPU/other GPU work affect the measured denominator.

## Implementation

- `mlp_dx.py`: original `@triton.jit` **mlp_input_grad_kernel**, implementing
  FP16 GEMM with FP32 accumulation and FP16 output. Fixed 64 x 128 x 32 tiles,
  grouped row scheduling, four warps, two pipeline stages. Tails are masked.
  Runtime M avoids recompiling for each padded sequence length; N/K specialize
  to the two MLP matrix shapes. No split-K, atomics, workspace, autotuning,
  repeated calculations or artificial delay is added.
- `mlp_backward.py`: idempotent wrapper for the pinned MatMul4Bit.backward.
  Requires the actual frozen NF4 MLP shapes, CUDA FP16 contiguous grad_output,
  canonical packed weights, no requested weight/bias gradient and at least 128
  rows. The native dequantization and FP16 cast run exactly as before.
  Unsupported cases and higher-order autograd call the original backward.
- `__init__.py`: installs the new hook through solution.apply().
- `test_mlp_dx.py`: numerical and actual NF4-layer integration tests.

The forward and its outputs are unchanged. LoRA forward/backward, dropout,
adapter parameters, label/loss/metric handling and every task setting are
unchanged. The output gradient buffer and dequantized weight are the same sizes
as native; no additional saved activations or dense workspace is introduced.
Global matmul precision flags are not changed. Example SwiGLU remains unchanged
and excluded from original-kernel share.

GEMM accumulation order can differ from cuBLAS, so numerical equivalence is
tested with documented FP16 tolerances and an FP64 reference as well as native
FP16 matmul. Previous test tolerances were not changed. Adapter quality, exact
resource use and speed still require the complete T4 training/evaluation run;
a generic Triton GEMM can be slower than cuBLAS on some shapes.

## Checks actually executed here

- Python syntax, notebook code-cell syntax and git diff --check.
- Unmodified source_kernels(): seven original kernels, two excluded example
  kernels, no banned imports, parse errors or example lookalikes.
- Grouped tile indexing covers 44 M/N combinations exactly once, including
  partial groups. This checks scalar indexing, not GPU execution.
- Six native CPU fallback cases compare first and second autograd derivatives.
- Forty dispatch checks use real CPU PyTorch with FakeTensor metadata: eight
  supported cases and 32 native fallbacks. They allocate no CUDA tensors and
  validate no GPU arithmetic.

The isolated CPU probes use the actual helper definitions extracted with AST
because the local Windows environment lacks Triton. They are not full pytest.
**No new GPU PASS, speed/memory result or >=5/6% share is claimed.**

## Required Kaggle validation

Set COMMIT in kaggle_t4.ipynb to the new full SHA and run on idle T4 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 CUDA_VISIBLE_DEVICES=0 python -m pytest tests -q
python harness/run_task.py --repeats 2
python harness/check_rules.py
```

Check that mlp_input_grad_kernel appears in triton_launches / own_kernel_ms in
both solution runs. The authoritative share is the minimum over measured runs
of eligible original-kernel CUDA event time divided by timed_loop_s, steps 3+.
The desired value is >=6%; all ten official checks must also pass. If it is
absent or performance regresses, use the unmodified --profile mode and actual
per-kernel times to revise dispatch/tile choices; do not add useless work.

Primary sources: [Qwen configuration](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json),
[bnb 0.50.2 backward](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/autograd/_functions.py),
[PEFT 0.20 LoRA](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/bnb.py),
[Triton dot](https://triton-lang.org/main/python-api/generated/triton.language.dot.html).
