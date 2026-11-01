# Speed candidate: resident decoder and earlier cold compilation

**Historical experiment (`274f3cb`), not the selected production policy.**
After clarification that memory must also improve below 3671MiB, decoder
offload was restored. The current next step is
[fused RoPE](ROTARY_CANDIDATE.md). No T4 result exists for this resident variant.
The discussion below records its original rationale, not a confirmed gain.

The target is **at most 40 seconds for the complete measured training command**,
while satisfying all ten unmodified challenge checks. The latest user-supplied
T4 summary is baseline 86.6s / 14345MiB / eval 2.04963 versus solution
83.1s / 3671MiB / eval 2.04965. Its SHA, raw per-run metrics and check results
have not yet been supplied. It is not a measurement of this candidate.

83.1 -> 40 seconds requires another 2.078x speedup, or removing 51.9% of the
current complete time. Source inspection cannot establish that this target is
attainable: `wall_s` includes imports, model/data loading, cold compilation,
training and adapter saving. `train_loop_s`, `timed_loop_s`, the raw per-step
times and profiler table are needed to identify the remaining cost.

## Implementation

For an accepted streamed-head forward with batch 1..4 and sequence 1..512,
without gradient checkpointing, decoder activations now remain on the GPU.
A scoped ContextVar makes each patched decoder layer call its preceding
forward immediately, before offload eligibility checks, parameter-storage
inspection, saved-tensor hooks, pinned-host allocation, streams or copy events.
The context ends before head computation and is restored after exceptions and
nested calls. Other calls retain the previous offload implementation.

This removes real transfer/setup work. At 4x512, the two saved FP16 MLP caches
alone total `2 * 2048 * 8960 * 2 * 28 = 2055208960` bytes, about 1.914GiB per
step, previously transferred in both directions. Attention and other saved
activations add further transfers. The previous policy also waited for late
D2H batches during the decoder forward and at its end.

The old policy caps offloaded payload at 256MiB per layer. Across 28 layers,
the retained payload could add up to 7168MiB; adding this to the reported
3671MiB suggests roughly 10839MiB as an initial memory estimate. **This is not
a peak-memory guarantee**: allocator fragmentation, workspace and lifetimes
can change the measured peak. The new peak must fit the GPU and the original
rule limiting memory to 110% of baseline.

Cold compilation now starts before the lazy `transformers.Trainer` import,
allowing it to overlap dependency loading as well as model loading. Warmup
descriptions also cover the existing FP32, FP16 and mixed-cache SwiGLU pointer
signatures, using the runtime's unchanged default launch/fusion options.
Warmup compiles only; it neither executes kernels nor advances Torch RNG.
The worker still finishes before training and at process exit. Compilation is
inside the measured process and uses the harness's fresh cache.

## Preserved contracts

- No model parameters, NF4 format, compute dtype, GEMM dimensions, head chunk
  size, tensor arithmetic, LoRA, dropout/RNG order or adapter format changes.
- SM75 MLP packing remains disabled after the reported gradient mismatch.
- Loss, entropy, accuracy, token counts, rewind and all Trainer/task settings
  use the same path as the preceding candidate.
- No changes to `harness/`, `task/` or `tests/`; no warm cache outside timing,
  extra GPU work, disabled checks or artificial kernel-share work.
- NF4 decoding and other useful original kernels still run normally. Shipped
  SwiGLU warmups do not become original kernels. The minimum measured share
  across both solution runs must still be at least 5%.

## Validation

Local CPU dependencies: Python 3.14.3, PyTorch 2.15.0+cpu, Triton 3.6.0,
pytest 9.2.0. `python -m pytest solution -x -q --tb=short`:
**176 passed, 221 skipped**. CUDA and unavailable training-package cases were
skipped; this does not establish T4 correctness or performance.

New CPU regressions verify exact original decoder output, input/parameter
gradients and dropout RNG; no offload setup in the resident scope; nested and
exception restoration; task-shape/checkpointing bounds; decoder-only residency;
and compilation startup order, pointer signatures and original launch options.
The T4 residency tests require exact outputs/gradients/RNG for FP16 and FP32.
They have not been executed here.

The protected Soup suite remains unavailable locally because the training
dependencies are absent. Python/notebook syntax, source classification and
`git diff --check` are checked separately before commit.

## Required T4 measurement

Pin the new commit SHA in `kaggle_t4.ipynb` and run its unchanged acceptance
flow: solution tests, the 79 protected Soup tests, base/solution/solution/base,
all ten checks and scoring. Its profile run is separate from scored runs.

```bash
python -m pytest solution -x -vv --tb=short
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
python harness/run_task.py --repeats 2 --profile
python harness/check_rules.py
python harness/score.py runs/result.json
```

Compare `wall_s`, `train_loop_s`, `timed_loop_s`, `apply_s`, GPU peaks and
original-kernel share for both solution repeats. Preserve `result.json`,
`checks.json`, raw `solution_*/metrics.json`, `train.log` and the separate
`solution_profile/profile.json` with the tested SHA. If time is still above
40s, use this split and profile to choose the next useful change; do not infer
GPU speed from CPU tests or relax the task to reach the time target.
