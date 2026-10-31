# Second optimization stage: implementation and validation boundary

This stage targets large activation buffers and repeated pointwise/reduction
work while keeping the fixed Soup 0.77.0 task. The model architecture, weights,
data and batch order, LoRA configuration and dropout, optimizer, scheduler,
training steps, and Trainer flags are unchanged. The protected `harness/`,
`task/`, and `tests/` directories are not edited.

**This version has not been executed on NVIDIA T4.** Its GPU correctness,
training speed, peak memory, full regression result, and all harness rule
checks remain unverified. In particular, the original-kernel share of at least
5% must be measured again; more original kernels do not prove that condition.

## Earlier user-reported measurements

The user supplied these results from an earlier Kaggle run. Its exact commit
SHA was not supplied with the measurements, so these numbers must not be
attributed to the new stage or treated as its acceptance result.

| Run | Training wall time | GPU memory peak |
|---|---:|---:|
| Baseline 1 | 91.21 s | 14,345 MiB |
| Solution 1 | 79.43 s | 12,385 MiB |
| Solution 2 | 80.63 s | 12,385 MiB |
| Baseline 2 | 82.33 s | 14,345 MiB |

The reported summary was **1.084x speedup**. Reported held-out evaluation loss
was **2.04967 for baseline** and **2.04965 for solution**. These observations
establish a useful previous comparison; they do not validate the changes below.

## FP32 cross-entropy without a dense FP32 logits copy

`causal_loss.py` installs a guarded loss function on newly constructed matching
Qwen2 model instances. It uses the supported `loss_function` setter and leaves
the original `ForCausalLMLoss` symbol and `LOSS_MAPPING` entries intact. This
matters because Transformers 5.16.1 Trainer recognizes causal label shifting
through those mapping identities when counting `num_items_in_batch`.

The wrapper preserves the original causal shift, terminal ignored label,
explicit `shift_labels`, target device transfer, and mean or sum reduction with
the original batch-token denominator. For supported CUDA inputs it invokes
`cross_entropy_fp32` directly on FP16 or FP32 logits. Triton promotes each
loaded tile to FP32 and returns FP32 loss. Backward stores the input gradient
in the logits' original dtype, matching the cast boundary of native
`cross_entropy(logits.float(), ...)`. Unsupported cases call the original
causal loss. The older `cross_entropy` API retains its FP16 native fallback.

The custom autograd node saves the original logits and small row statistics,
avoiding a vocabulary-sized FP32 cast retained for backward. It also avoids
forming a dense FP32 input-gradient buffer before casting that gradient to
FP16. Full model logits and their input gradients still exist, and other
wrappers or metrics can still allocate FP32 tensors.

For a full `4 x 512 x 151936` tensor, **one FP32 logits matrix is 1.159 GiB**:

```text
4 * 512 * 151936 * 4 bytes / 2**30 = 1.1591796875 GiB
```

This is a tensor-size calculation, not a measured peak-memory reduction.
Buffer lifetimes, dynamic sequence lengths, allocator behavior, and other
training work determine the actual external memory peak.

## Skip ignored rows in vocabulary cross-entropy

The partition, finalization, and backward kernels branch on the row target.
Prompt and padding rows with `ignore_index` avoid vocabulary loads and
exponentials. Backward still writes the required zero input gradient.
All-ignored mean loss preserves the native NaN loss and exactly zero logits
gradients, without multiplying an ignored row by a NaN upstream gradient.
Entropy still processes every requested row because it is a separate metric.

## Original fused RMSNorm

`rmsnorm.py` adds original forward and backward Triton kernels. Each program
normalizes one token row; backward recomputes normalized values using the
saved input and one inverse RMS per row. The production hook accepts the
1536-wide CUDA FP32 path with frozen FP32 norm weights. Low-precision inputs,
trainable norm weights, unsupported layouts, and unsupported shapes retain
the native path, including native weight gradients.

The guard preserves Qwen2's observable normalization cast before multiplication
by its weight. This optimization replaces an operation, without changing norm
parameters, residual connections, or training precision settings. The custom
path supports first-order autograd, as does the custom vocabulary CE path.

## Lossless half cache for the existing SwiGLU example

`compact_swiglu.py` reuses the two shipped example SwiGLU kernels. They remain
excluded from original-kernel accounting. The change compacts only saved
activations whose FP16 origin is established by the pinned frozen NF4 layers.

In bitsandbytes 0.50.2, a `Linear4bit` call with FP16 compute first converts its
input to that compute dtype and returns its result in the original input
dtype. Its FP32 output therefore contains FP16-representable values. For the
guarded gate/up projections, saving those values as FP16 loses no information.
The pointwise output and both input gradients remain FP32, and the original
projection calls and separate branch cast boundaries are preserved.

The dispatch guard requires exact `Linear4bit` types, fixed FP16 compute,
frozen quantized weights, no bias, and no overriding or observing forward
hooks. Other cases use the existing MLP implementation. LoRA wrappers and
arbitrary FP32 activations are not assumed to have a lossless half cache.

For 28 layers with a full batch and sequence length, the theoretical saved
gate/up cache reduction is **1.914 GiB**:

```text
4 * 512 * 8960 * 28 layers * 2 caches * (4 - 2) bytes / 2**30
    = 1.9140625 GiB
```

This is a storage calculation assuming all eligible caches are retained at the
stated shape. It is not a measured peak reduction and must not be added to the
logits figure as a prediction of the final peak.

## Kernel accounting

There are six original kernels eligible for runtime accounting, if they launch
inside measured training steps:

| File | Original kernels |
|---|---|
| `vocab_ops.py` | `vocab_partition_kernel`, `vocab_ce_finalize_kernel`, `vocab_ce_backward_kernel`, `vocab_entropy_finalize_kernel` |
| `rmsnorm.py` | `rmsnorm_row_forward_kernel`, `rmsnorm_row_backward_kernel` |

The two `example_swiglu_*` kernels in `swiglu.py`, including their use by the
compact cache, are excluded. The harness's denominator is synchronized timed
training-step wall time, not total profiler GPU-kernel time. The minimum share
across solution runs must satisfy the unmodified harness rule.

## Correctness tests and remaining checks

New and retained tests live only under `solution/`:

| Test file | Coverage |
|---|---|
| `test_vocab_ops.py` | CE values and gradients, ignored labels, reductions, strided rows, repeated backward, entropy offsets, FP16 entropy precision, native autograd, aliases and rewind integration |
| `test_causal_loss.py` | FP16 logits with FP32 loss versus the native cast, saved-tensor dtype, causal alignment, explicit shifting, denominators, all-ignored gradients and constructor/mapping identity |
| `test_rmsnorm.py` | Forward/backward, epsilon and magnitudes, arbitrary weights, strided upstream gradients, input/weight preservation, low-precision casts, trainable-weight fallback and patch integration |
| `test_compact_swiglu.py` | Exact equality against the example for half-origin inputs, FP32 gradients, cache contents/size, unchanged inputs, provenance guard rejections, patch idempotence and real Linear4bit integration |

Only **15 isolated CPU checks of the dispatch/provenance guards** have been
executed for this stage. They are not full pytest execution and do not compile
or execute CUDA kernels. The full `solution/` suite, protected Soup regression
suite, measured training, evaluation and rule checker still require the pinned
dependencies on an idle Kaggle NVIDIA T4.

Run from a clean, pinned commit using the provided notebook. Its validation
must complete the solution tests and protected tests before the benchmark,
then run both baseline and solution repeats and `harness/check_rules.py`.
Record the commit SHA, runtime torch/Triton versions, evaluation, peak memory,
and kernel share. CUDA skips provide no evidence of kernel correctness.

## Pinned source contracts reviewed

- [Transformers 5.16.1 causal loss](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/loss/loss_utils.py): FP32 loss, shifted targets, reduction and denominator.
- [Transformers 5.16.1 Trainer](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/trainer.py): mapping-based causal token counting.
- [Transformers 5.16.1 model base](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/modeling_utils.py): instance loss getter/setter.
- [Transformers 5.16.1 Qwen2](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py): loss call, RMSNorm cast order and MLP structure.
- [bitsandbytes 0.50.2 Linear4bit](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/nn/modules.py): compute dtype and output cast.
- [TRL 0.29.1 SFTTrainer](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/sft_trainer.py): metric calls and label alignment.
