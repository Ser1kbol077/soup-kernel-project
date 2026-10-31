# Submission draft — Soup Kernel Challenge

Status: `3351e4b` failed both T4 solution runs. An SM75 dispatch correction is
pending training validation; see solution/SM75_AUDIT.md. The traceback is still
needed to establish the original failure's cause.

## Links

- Repository: https://github.com/Ser1kbol077/soup-kernel-project
- Final tested commit: pending T4 validation; use the full SHA from the pull request.
- Public Kaggle saved notebook version: not published yet; add the same link to README.md.
- Demo video (up to 3 minutes): pending.
- Presentation: pending.
- Team name, members and captain contact: to be filled by the team.

## Own kernels

| Kernel | File | Operation | Measured training share |
|---|---|---|---|
| vocab_partition_kernel | solution/vocab_ops.py | Tiled partition statistics | Pending T4 |
| vocab_ce_finalize_kernel | solution/vocab_ops.py | Token CE | Pending T4 |
| vocab_ce_backward_kernel | solution/vocab_ops.py | CE backward | Pending T4 |
| vocab_entropy_finalize_kernel | solution/vocab_ops.py | Categorical entropy | Pending T4 |
| rmsnorm_row_forward_kernel | solution/rmsnorm.py | Frozen FP32 RMSNorm forward | Pending T4 |
| rmsnorm_row_backward_kernel | solution/rmsnorm.py | Frozen FP32 RMSNorm input gradient | Pending T4 |
| mlp_input_grad_kernel | solution/mlp_dx.py | Frozen NF4 MLP input-gradient matrix product, SM80+ only | Disabled on T4; native backward |

Written from scratch using standard mathematics. Integration contracts and primary
source links are in README.md. The shipped example SwiGLU remains enabled and is
excluded from our share.

The next candidate also uses FP32 CE accumulation directly from half logits,
skips ignored-row vocabulary work and stores proven half-origin MLP caches
losslessly in FP16. See solution/OPTIMIZATION_NOTES.md. The team's reported
previous 1.084x / 12385 MiB result is not a measurement of this new candidate.

## Results on Kaggle T4

| Metric | Base Soup | Ours | Ratio/delta |
|---|---|---|---|
| soup train wall seconds | Pending | Pending | Pending |
| Peak GPU MiB | Pending | Pending | Pending |
| Public eval loss | Pending | Pending | Pending |
| Minimum own-kernel training share | — | Pending | Required ≥5% |

- [ ] T4 forward/backward and hook tests passed.
- [ ] All 10 unmodified harness checks passed on two repeats.
- [ ] Public notebook saved with full immutable tested COMMIT SHA.
- [ ] Notebook link added to README and this file.
- [ ] Team, final commit, video and presentation links filled.

No GPU correctness or performance result is claimed before these runs.

Check-4 follow-up: `6b175fb` measured 2.4% and 9/10 PASS. `3351e4b` produced no
successful solution run. The corrected T4 dispatch preserves the other patches;
its training, quality and original-kernel share require new measurements.
