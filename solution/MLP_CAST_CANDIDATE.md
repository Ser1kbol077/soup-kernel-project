# Candidate: eliminate redundant casts in the frozen NF4 MLP

This experiment is isolated on `codex/t4-mlp-cast-elision`, based on
`6908b391a3a32ccd2a07e6e1f767479baf1c4ae9`. The parent accuracy candidate is
currently being measured. Neither branch is declared validated on T4.

The latest user-reported completed run took 79.0 s versus 79.9 s baseline,
9243 versus 14345 MiB, and passed 9/10 rules plus 79 Soup regressions. Its
original-kernel share was 2.5%, below 5%. These numbers are not this candidate's
results. Reaching 60 s from 79 s requires eliminating 24.05% of full wall time;
there is no measured profile establishing that this change can do that.

## Operation and numerical contract

In [bitsandbytes 0.50.2 Linear4bit](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/nn/modules.py),
an FP32 input is cast to the configured FP16 compute dtype and the FP16 result
is converted back to FP32. The next projection converts that result to FP16
again. The previous compact-cache path still materialized these FP32 outputs
and gradients for every gate/up/down projection.

`mlp_casts.py` wraps the complete frozen MLP in a first-order autograd node.
It casts the input to FP16 once, invokes the original gate/up projections,
runs the **unchanged supplied** SwiGLU forward with FP16 storage, invokes the
original down projection and returns FP32. SwiGLU arithmetic remains FP32.
The only saved activations are the two FP16 projection outputs. Packed frozen
weights and their quantization states are retained; dense weights are not
cached or kept between steps.

Backward preserves every original FP16 rounding boundary:

```text
down_grad_half = FP16(incoming_FP32_grad)
activation_grad_half = native_down_input_GEMM(down_grad_half)
gate_grad_half, up_grad_half = unchanged_example_backward(activation_grad_half)
gate_x_half = native_gate_input_GEMM(gate_grad_half)
up_x_half = native_up_input_GEMM(up_grad_half)
input_grad_FP32 = FP32(gate_x_half) + FP32(up_x_half)
```

The final add uses a fresh FP32 buffer and promotes the second operand before
addition. Sharing the forward input conversion does **not** introduce a shared
FP16 gradient accumulation node. Each input GEMM uses the original
[bnb decode/cast/matmul](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/autograd/_functions.py)
sequence. There is no new GEMM, precision setting, weight cache, recomputation,
autotuning, optimizer change or task change.

Dispatch requires bnb 0.50.2, exact Qwen2MLP and exact frozen bias-free
Linear4bit projections, canonical uint8 NF4 weights, FP16 quantization/compute
dtype, original task dimensions 1536/8960, contiguous FP32 CUDA input requiring
grad, at least 128 token rows and SM75. BF16 autocast, hooks, overridden
forwards, LoRA-wrapped MLP projections, other devices/layouts, inference and
unknown versions use the preceding implementation. The patch is installed
after the existing example/compact wrappers through `solution.apply()`.

This change adds no original kernel. The supplied SwiGLU kernels remain
excluded even when specialized for FP16 storage. Existing vocabulary,
RMSNorm and parent-branch argmax kernels retain their accounting. Faster casts
alone do not establish a 5% share: at unchanged original-kernel milliseconds,
the timed loop would have to halve to increase 2.5% to 5%.

## Expected scope, not measured benefit

For 28 layers and a maximum 4x512-token batch, source-level counting estimates
about 31.36 GB of avoided cast/pointwise traffic per step relative to the
previous compact path. Shorter padded sequences reduce this amount. Native
GEMMs, model loading, compilation, saving, logging and other work remain.
This traffic estimate is not a latency or total training speedup prediction.

The old compact path already saved gate/up activations in FP16. That saving
cannot be counted again. Additional peak reduction concerns temporary buffers
of the current layer; allocator overlap and real batch shapes must be measured.
The parent's removed accuracy clone is a separate, still unmeasured change.

## Local evidence

- Full locally available `solution/` pytest: **93 passed, 179 skipped**.
  The new file contributes 22 CPU passes and 14 CUDA skips. No GPU integration
  test, protected Soup regression, training or evaluation ran locally.
- The actual production autograd class matched the upstream bnb autograd graph
  plus compact SwiGLU in **60 CPU-interpreter forward/input-gradient cases**,
  including strided upstream gradients and scale 65536. The supplied kernels
  ran in the interpreter, but native CUDA GEMM/NF4 decode were replaced by a
  fixed dense CPU backend. This tests cast boundaries, not GPU NF4 correctness.
- Four old/new SwiGLU dtype signatures compiled offline for SM75 using isolated
  triton-windows 3.6.0.post24 and CUDA 12.8 ptxas. The FP32 arithmetic graphs
  agree after register renaming; new forward/backward use 37/55 registers,
  with zero stack/spills/shared memory. This is not a Kaggle Linux run.
- The unchanged harness source classifier found nine original definitions,
  two excluded supplied examples, no lookalikes, banned imports or parse
  errors. Static classification does not establish runtime launch/share.
- Python/notebook syntax and `git diff --check` succeed. `harness/`, `task/`,
  `tests/` and the supplied `solution/swiglu.py` are unchanged from the parent.

**GPU launches performed locally: zero.** T4 precision, actual dispatch,
training success, all 79 Soup regressions, quality, speed, memory and kernel
share remain unverified for this candidate.

## Kaggle validation

Finish and archive the currently running parent first. Do not run simultaneous
GPU workloads. Use a clean checkout of this candidate's **full SHA** and the
existing task preparation. The notebook now selects this candidate branch and
enables the harness's optional extra profiler run; pin its `COMMIT` before
Save & Run All. The profiler run is separate from the two measured repeats.
After checking rules, the notebook saves a `*-logs.zip` in `/kaggle/working`,
including the tested SHA and run JSON/logs, even if a rule fails. It preserves
the checker's failing exit status and does not archive adapter weights.

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 CUDA_VISIBLE_DEVICES=0 python -m pytest tests -q
python harness/run_task.py --repeats 2 --profile
python harness/check_rules.py
```

Keep `runs/result.json`, `runs/checks.json`, both solution `train.log` and raw
`metrics.json` files, plus `runs/solution_profile/profile.json`. Compare full
wall and timed-loop time, peaks, loss curves, quality and all ten rules.
CUDA tests compare the complete MLP forward and input gradients against native
bnb plus the supplied example, rather than an isolated pointwise surrogate.
Do not loosen test tolerances after a failure without numerical evidence.

No training or GPU correctness claim is made until these checks run on T4.
