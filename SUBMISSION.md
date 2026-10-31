# Submission draft — Soup Kernel Challenge

Status: first implementation; Kaggle T4 correctness and benchmark pending.

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

Written from scratch using standard mathematics. Integration contracts and primary
source links are in README.md. The shipped example SwiGLU remains enabled and is
excluded from our share.

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
