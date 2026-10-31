# Architecture and rule audit

This records the first-stage audit. The next candidate adds two original RMSNorm
kernels, direct FP32 loss accumulation from FP16 logits, ignored-row skipping and
lossless compact activation storage; see [OPTIMIZATION_NOTES.md](OPTIMIZATION_NOTES.md).
Its additional T4 correctness/performance measurements are still pending.

## Baseline and execution path

Before implementation, all 24 tracked files at local baseline `05cb258` were
compared with the [official challenge template](https://github.com/MakazhanAlpamys/soup-kernel-challenge)
using Git blob SHAs. Every file matched. The protected `harness/`, `tests/` and
`task/` directories remain unchanged.

The required dependency versions are `soup-cli[train]==0.77.0`,
`transformers==5.16.1`, `peft==0.20.0`, `trl==0.29.1`,
`bitsandbytes==0.50.2` and `accelerate==1.14.0`, with PyTorch, Triton and pytest.
The template leaves PyTorch and Triton to the Kaggle runtime; their actual
versions are recorded in `runs/result.json` rather than claimed as fixed here.

`harness/run_task.py` starts baseline and solution training in separate fresh
processes. The solution process calls `solution.apply()` before Soup builds the
model and trainer. The fixed task trains Qwen2.5-1.5B-Instruct with NF4 double
quantization, FP16 compute, LoRA rank 16 and alpha 32, batch size 4, maximum
sequence length 512 and one epoch over 256 rows. Soup's automatic LoRA selection
uses the query and value projections with dropout 0.05. The optimizer, schedule,
data order, training settings and standard PEFT adapter format are preserved.
The baseline uses SDPA attention, `adamw_torch`, a cosine schedule and no gradient
checkpointing. K-bit preparation leaves norms and the tied embedding/head in
FP32; model compute otherwise follows the baseline's autocast path.

Soup uses its `RewindSFTTrainer` subclass of TRL's `SFTTrainer`. In addition to
training loss, TRL calculates entropy and token accuracy, and Soup records each
row's loss and token count in `rewind.jsonl`. These outputs must be preserved.

## First optimization: vocabulary reductions

The first change targets repeated reductions over the 151,936-token vocabulary.
It retains full model logits and replaces the cross-entropy and entropy
reductions with original Triton code in `vocab_ops.py`.

| Original kernel | Work used by training |
|---|---|
| `vocab_partition_kernel` | Per-row, 4,096-column tile maxima and shifted exponential sums; optional entropy moments |
| `vocab_ce_finalize_kernel` | Merge tile statistics and calculate each token's cross-entropy |
| `vocab_ce_backward_kernel` | Recompute probabilities and calculate the logits gradient |
| `vocab_entropy_finalize_kernel` | Merge tile moments and calculate categorical entropy |

Tile statistics use FP32. Cross-entropy backward retains logits and two scalars
per row, then recomputes probabilities instead of retaining a vocabulary-sized
log-softmax result. Entropy uses shifted moments to avoid cancellation from a
large common logit offset. Vocabulary tails are masked. The cross-entropy path
supports contiguous vocabulary columns with a noncontiguous row stride.

The integration in `patches.py` installs idempotent wrappers with CUDA, FP32 and
vocabulary-size guards:

- `transformers.loss.loss_utils.fixed_cross_entropy`: preserves the original
  mean or sum reduction and `num_items_in_batch` denominator. The original
  `ForCausalLMLoss`, label shifting and `LOSS_MAPPING` identities remain intact;
  Transformers Trainer relies on that identity for its token bookkeeping.
- `trl.trainer.utils.entropy_from_logits` and the alias imported by
  `trl.trainer.sft_trainer`: both must be patched because TRL imports the utility
  by value. The fused path serves metrics without autograd; differentiable
  entropy retains the original path.
- `soup_cli.trainer.rewind_hf.row_losses`: preserves row iteration, shifted
  labels, ignored tokens, row averaging and the return format.

Unsupported inputs retain native operations. All-ignored mean cross-entropy
retains PyTorch's NaN loss and zero logits gradients. The shipped SwiGLU example
is retained as a separate optimization; its kernels do not count as original.

## Acceptance checks

`harness/check_rules.py` is authoritative. Static source recognition alone does
not establish that any runtime check passes.

| Check | Required condition |
|---|---|
| 1. Finished runs | Every requested baseline and solution run exits with code 0 |
| 2. Libraries | No banned imports or runtime libraries; no packages changed during measurement; `torchao` has only the harness's stated runtime exception |
| 3. Original kernels | Original `@triton.jit` functions defined under `solution/` launch in every solution run; no unparseable Python files |
| 4. Kernel share | Original kernels together account for at least 5% of timed training steps in every solution run |
| 5. Same task and outputs | Steps, adapter settings/tensor names/shapes/dtypes, trainer settings, metric names, token counts and rewind row identities agree; mean relative gaps for entropy, accuracy and rewind row losses are at most 3% |
| 6. Loss curve | First-step relative loss gap at most 1%; mean relative loss gap over all steps at most 3% |
| 7. Quality | Held-out evaluation loss at most 0.5% above baseline |
| 8. Time and memory | Solution median wall time and maximum GPU-memory peak each at most 110% of baseline; summary agrees with run measurements |
| 9. Clean measurement | T4; only GPU 0 visible and one device; unchanged `expandable_segments:True` allocator; starting memory summed across GPUs at most 1,000 MiB; external peak at least 97% of PyTorch's reserved peak |
| 10. Soup regressions | Exactly 79 tests from the protected `tests/` directory pass with `SOUP_APPLY_SOLUTION=1` |

For each solution run, check 4 calculates:

```text
share = sum(original_kernel_cuda_event_ms) / (1,000 * timed_loop_s)
accepted_share = minimum(share across all solution runs)
```

Both numerator and denominator cover steps 3 onward. The denominator is the sum
of synchronized step wall times, not a profiler's total GPU-kernel time. Only
original kernel names launched from `solution/` in every solution run count.
The example SwiGLU and kernels generated by `torch.compile` are excluded. A
small margin above 5% may not survive different hidden rows; the harness flags
results below 6% for attention.

Scored training time covers the complete command, including startup, Triton
compilation and adapter saving. Each run starts with fresh compiler caches.
Scored memory is the external `nvidia-smi` peak sampled every 100 ms and summed
across GPUs. Evaluation loads the saved adapter without importing the solution.

## Validation status and limitations

No GPU execution, T4 correctness result, measured speedup, memory reduction or
5% kernel-share result is available for this implementation. The tile size and
four-warps launch configuration are initial choices, not measured optima.
`test_vocab_ops.py` includes forward/backward comparisons, ignored-label cases,
strided rows, repeated backward, entropy offsets, fallback behavior and hook
integration. These tests still require execution with the target dependencies
and GPU; skipped tests do not demonstrate CUDA correctness.

The optimized cross-entropy supports hard class labels and first-order
autograd. Higher-order derivatives are deliberately unsupported by its custom
autograd function. The installed hooks accelerate FP32 CUDA logits for this
model's vocabulary; other dtypes, devices and unsupported layouts use native
operations. Full logits, the language-model head and its gradient still exist,
so this change does not remove all vocabulary-sized memory costs.

Before treating the implementation as accepted, run kernel tests and then the
unmodified harness on an idle Kaggle T4 with `--repeats 2`, followed by all ten
rule checks. The 5% condition is unresolved until those measurements exist.
