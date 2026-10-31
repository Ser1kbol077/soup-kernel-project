# Separate candidate: strided token accuracy without a dense logits copy

## Starting point and target

The current Kaggle run tests `298a64c` on `codex/triton-vocab-t4`. Its outcome
has not been supplied. This experiment is isolated on
`codex/t4-strided-accuracy`, based on that commit. It preserves the previous
candidate for comparison and introduces one additional training operation.

The last successful user-reported run was `6b175fb`: 80.0 s, 12385 MiB,
1.084x, 9/10 checks, 2.4% original-kernel share. Those measurements do not
validate this branch or `298a64c`.

The new requested target is <=45 s, including compilation, and lower memory.
For reference, reducing 80 to 45 s requires eliminating at least 43.75% of
the full time even with unlimited acceleration of that portion. At 2x local
acceleration the required affected fraction is 87.5%, before accounting for
any new compilation cost. No measured bottleneck fractions are available yet.
This accuracy change alone must not be presented as sufficient for 45 s.

## Concrete allocation found in the pinned training path

[Accelerate 1.14.0](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/operations.py#L817-L856)
converts returned FP16 logits to full FP32. That output contract is retained.
[TRL 0.29.1](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/sft_trainer.py#L1177-L1211)
then copies `outputs.logits[..., :-1, :]` with `.contiguous()` solely to call
`argmax` for token accuracy. For batch size greater than one the time slice
has gaps between batches and therefore usually requires a new allocation.

For the maximum fixed task shape, that extra tensor is:

```text
4 * 511 * 151936 * 4 bytes / 2**20 = 1184.681640625 MiB
```

The new path reads the original strides. At this shape its two statistics
arrays and int64 predictions total about 0.608 MiB of tensor storage. Shapes,
buffer lifetimes and the allocator determine the actual external memory peak;
the removed allocation is not a measured peak reduction. Cold JIT cost can
offset time saved by eliminating the copy, so complete training must be timed.

## Implementation and preservation of behavior

- `argmax_ops.py` contains two original `@triton.jit` kernels. The first
  reduces 4096 vocabulary values per tile to a maximum and first index; the
  second combines tiles. Negative partial indices encode the first NaN.
  Ties, infinities, NaNs and signed zero match PyTorch's first-index semantics.
  Positive batch/time/vocabulary strides are supported without a dense clone.
- Time length and batch stride remain runtime scalars to avoid a compilation
  for each dynamically padded sequence length. There is no autotuning.
- `accuracy_patch.py` recognizes the exact two accuracy expressions in the
  pinned TRL method and replaces them with a guarded view and argmax. It keeps
  the original method's live globals, `__class__` closure, `super()`, signature,
  defaults and all other statements. Unknown source structure, versions or
  closures leave the original method installed. CUDA FP32, three dimensions,
  vocabulary 151936 and unit vocabulary stride guard the production hook.
- `solution.apply()` installs this patch after the existing ones. Results are
  consumed by the existing comparison with shifted labels, ignored-token mask,
  distributed gathering and accuracy logging. This is actual metric work in
  training, with no artificial operations or delay added for kernel share.
- Full FP32 model outputs, loss, entropy, token counts, rewind, LoRA, dropout,
  optimizer, task settings, and allocator settings are preserved. Accuracy
  stays in its original `torch.no_grad()` block; the loss graph is unchanged.
  The example SwiGLU stays excluded from original-kernel accounting.

The AST integration avoids copying the full third-party training method or
changing global Tensor operations. The strict source test in the solution
suite must confirm that the actual installed pinned method is recognized.

## Local evidence, not GPU validation

Executed locally in isolated CPU dependencies:

- Targeted pytest across the new argmax/integration tests and existing MLP
  dispatch tests: **42 passed, 59 skipped**. CUDA cases, the unavailable bnb
  integration and actual installed-TRL acceptance were skipped locally.
  This is not the complete solution GPU suite or the 79 protected regressions.
- CPU fallback and loss-graph tests; rewrite tests compare exact loss, first
  gradients, FP32 outputs, metrics, closure/global aliases and guard behavior.
- The exact unmodified method extracted from downloaded TRL 0.29.1 was also
  exercised in 32 CPU scenarios: train/eval, explicit shifting, virtual tokens,
  output modes and attention-mask/position-id paths. An argmax proxy was used
  for these integration cases, not a CUDA kernel.
- The actual argmax kernels ran through Triton's CPU interpreter in 62 cases,
  matching `torch.argmax` exactly with unchanged source storage. These cover
  FP16/FP32, vocabularies 1/65/4096/4103/151936, strided layouts, ties and NaNs.
- Nineteen signatures compiled to SM75 cubin with isolated
  `triton-windows==3.6.0.post24`, Python 3.14 and CUDA 12.8 ptxas. No spills
  occurred. The production tile used 116 registers/thread and 16 B shared;
  this is compiler evidence, not an occupancy or latency measurement.

**Zero GPU kernels were executed locally.** CUDA correctness, real hook
activation, all 79 protected regressions, training success, quality, memory,
speed and the >=5% share still require Kaggle T4 validation. The Windows fork
and local CPU torch are not the pinned Kaggle execution environment.

## Research findings for subsequent work

An independently written inline-PTX proof can emit Turing's documented
`mma.sync.m16n8k8` despite Triton 3.6's ordinary-dot FMA fallback. Its tiny
32x16 global-load tile has only about 5.33 FLOP per operand byte and lacks
reuse/pipelining. It is not integrated or claimed faster than cuBLAS.
[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-1688)
defines the fragment mapping used in the proof.

Native bitsandbytes 0.50.2 already has shape-dependent fused NF4 forward;
the large Qwen MLP cases use dequantization plus linear. One dense MLP weight
is only 26.25 MiB, so eliminating its temporary decode is not by itself a
multi-GiB peak saving. Removing intermediate MLP casts while retaining native
Tensor Core products is another candidate, but branch-gradient rounding must
stay separate. Neither direction is added here without a measured profile.
[Native dispatch](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/backends/cuda/ops.py),
[native backward](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/autograd/_functions.py#L333-L352).

## Next measurement

Finish and archive the currently running `298a64c` measurement first. Do not
run another GPU job during it. Then pin this candidate's full commit SHA and
run from a separate clean checkout/output directory:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 CUDA_VISIBLE_DEVICES=0 python -m pytest tests -q
python harness/run_task.py --repeats 2 --profile --out runs_accuracy
python harness/check_rules.py --result runs_accuracy/result.json
```

Inspect `vocab_argmax_tile_kernel` and `vocab_argmax_finish_kernel` in both
solution runs' launches and timings. Compute original share with the unchanged
checker, not with profiler GPU percentages. Compare all ten checks and complete
wall time, including compilation, against the measured parent.

Optional, only after other GPU processes have exited:

```bash
CUDA_VISIBLE_DEVICES=0 python -m solution.benchmark_argmax
```

This prints actual synthetic operation latency, added allocated memory and
first-call wall time. It is not a training result or official score. Collect
`result.json`, `checks.json`, raw per-run metrics/logs and profiler JSON before
choosing a further optimization.
