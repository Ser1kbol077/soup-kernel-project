# FP16 entropy fallback precision fix

The team reported 51 passed and 2 failed tests on Kaggle T4 for commit
`6e6781bb7628aa11810c14f94d1a41454645cdf8`. Both failures were FP16
`test_entropy_large_logit_offsets` cases at offsets -1000 and +1000. Training and
benchmarks did not start. This report does not establish results for this fix.

## Cause and comparison

The test compares against `_reference_entropy(logits.float())`. The old fallback
instead kept `log_softmax`, `exp`, their product and the returned sum in FP16.
Even when log-softmax internally uses an FP32 accumulator, its output is rounded
to FP16. With equal or coarsely quantized logits, these rounding errors can be
correlated across thousands of classes. Stable log-softmax prevents overflow;
it does not make the subsequent FP16 entropy calculation match FP32.

The team's exact failing tensors/torch version were not supplied. The following
values were measured separately on **CPU, PyTorch 2.15.0+cpu**, not on T4:

| Input (3 rows, 4103 classes) | Old FP16 entropy | FP32 reference | Fixed entropy | Maximum absolute error, old → fixed |
|---|---|---|---|---|
| Uniform FP16 logits at either -1000 or +1000 | 8.3125 | 8.3194770813 | 8.3203125 | 0.0069770813 → 0.0008354187 |
| CPU random seed 31, first row, either offset | 7.84375 | 7.8179640770 | 7.81640625 | 0.0257859230 → 0.0015578270 |

CPU and CUDA RNG sequences differ, so the random row is not a reconstruction of
the team's exact failing CUDA tensor. The deterministic uniform case needs no RNG
and isolates the intermediate-rounding defect. The mathematical uniform entropy
is `log(4103) = 8.3194736924`; the FP32 reference also has ordinary rounding error.

## Change and compatibility

Only the direct FP16 entropy fallback without a tracked gradient graph changes:
`log_softmax(..., dtype=torch.float32)`, FP32 exp/product/sum, then cast the reduced
entropy to the input dtype. Grad-enabled inputs requiring gradients keep the
exact original native operations. Input data is not modified.

Production hooks in `patches.py` still accept only CUDA FP32 for custom metrics;
FP16 calls through TRL still use the original TRL function. No hook, label shift,
loss denominator, CE forward/backward kernel, model or training setting changed.
The four Triton kernel bodies are unchanged. No protected files changed.

Existing tests and their tolerances remain unchanged. Twelve new parameter cases
cover deterministic uniform logits, both offsets, CPU/CUDA, contiguous/strided
inputs, output dtype, no mutation and exact native forward/backward equivalence.

## Validation limits

- Python syntax, unmodified source classifier and `git diff --check` were checked.
- A CPU-only diagnostic executed the actual native helper/CPU dispatch functions
  extracted from `vocab_ops.py` using AST, since this Windows setup has no Triton.
  It checked 12 dtype/offset/layout combinations for metric precision, input
  preservation and exact native forward/backward agreement. It is not a full
  pytest run and does not compile or execute Triton.
- Full pytest, the 79 Soup regressions, GPU correctness and benchmarks for this
  fix still require execution in Kaggle. No T4 PASS is claimed.

Repeat on Kaggle with `COMMIT` set to the new full SHA:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 CUDA_VISIBLE_DEVICES=0 python -m pytest tests -q
python harness/run_task.py --repeats 2
python harness/check_rules.py
```

Primary sources:
[PyTorch log_softmax dtype contract](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.log_softmax.html),
[CUDA log-softmax output cast](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/SoftMax.cu#L35).
