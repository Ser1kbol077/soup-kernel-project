# Admission-first candidate: fused FP32 Qwen RoPE

The joint targets are full `soup train` time <=40s, including model loading and
cold JIT; GPU peak <=3671 MiB (lower is better); useful original-kernel share >=6%; all ten
official checks; and quality within the baseline thresholds. Admission comes
first. No target is claimed from source inspection or skipped GPU tests.

## Evidence and saved candidates

The user reports baseline 86.6s / 14345 MiB / eval 2.04963 versus solution
83.1s / 3671 MiB / eval 2.04965, with 9/10 checks and all 79 Soup tests passing.
Check 4 failed at 4.7%, below 5%. The exact SHA is unconfirmed; the user
reports that Kaggle reset the session and the raw files are no longer available.
This is a user-reported measurement, not a measurement of the code below.

`9acceccbe918baf7deb28c60c366f919b2ffd97d` preserves the full-shape SM75 MLP
fallback and the economical offload implementation. It is not assigned the
reported benchmark's PASS status without its tested SHA.

`274f3cb2d482cf6eaa031c56fccb22581a3b88c7` preserves the resident-decoder speed
experiment. It is unmeasured and may increase peak memory. Its automatic
residency scope and earlier-start JIT experiment are removed from production
for a controlled comparison with the preceding economical pipeline. The
offload implementation itself remains unchanged.

## Selected optimization

The pinned [Qwen2 source](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py)
applies `q*cos + rotate_half(q)*sin`, and likewise for k. The new original
Triton forward writes both rotated halves directly, without allocating rotated
copies and the two separate product tensors. Its backward evaluates the
transpose rotation directly, without retaining Q/K values. Each kernel is
launched once for Q and once for K; no extra mathematical work is added.

The kernel uses separate FP32 products and addition with FP fusion disabled.
It does not assume equal sine/cosine halves: the backward reads the other
half's sine where required by the Jacobian transpose. Projection GEMMs,
attention masks, SDPA selection, LoRA and dropout are unchanged.

Dispatch is limited to SM75, exact FP32 tensors, 12 Q / 2 KV heads of width 128,
sequence length >1, and standard dense BHTD or BTHD-backed input layouts.
Cos/sin must be nontrainable contiguous tensors and may broadcast over batch.
Forward output strides match the original dense input layout; backward uses
the native contiguous BHTD gradient layout. A one-token
call stays native because its singleton strides can differ. Other dtypes,
layouts, trainable trigonometric inputs and package versions retain the
installed original function. CUDA supports first-order input gradients;
native fallbacks retain ordinary autograd.

Runtime dimensions and strides avoid compiling separate variants for Q/K,
sequence lengths, layouts and broadcast batches. Two compile-only warmups
cover the forward/backward signatures inside the measured process. Neither
warmup nor a static kernel definition proves runtime share.

## Amdahl and limits

83.1 -> 40s needs another 2.078x speedup: at least 51.9% of complete wall time must
be removed. The reported 4.7% share concerns timed training steps, not full
startup-to-save wall time. Even making only that existing portion infinitely
fast would accelerate the timed loop by at most `1/(1-0.047)=1.049x` under the
usual non-overlap assumption. It cannot by itself explain a 2.078x full gain.

Keeping original-kernel milliseconds constant would require cutting the
timed-loop denominator by 21.7% to reach 6% share (`4.7/6=0.7833`). Instead,
this candidate transfers necessary existing RoPE work from native operations
to fused original kernels while reducing temporary allocations and launches.
Its actual numerator, denominator and external GPU peak must be measured.
It is not legitimate to retain useless decode, slow a kernel, or add delay to
meet the percentage.

RoPE alone is not expected to account for 43.1s of savings. The next decision
needs `wall_s`, `train_loop_s`, `timed_loop_s`, `apply_s`, per-step timings,
the measured cold-compilation log and the actual GPU kernel table. These
separate startup, matmul, decode, head and transfer costs. No complete profile
or new T4 measurement is available locally.

## Validation and Kaggle procedure

CPU tests cover independent FP64 rotation/Jacobian formulas with unequal trig
halves, repeated and strided VJPs, native trainable-trig/higher-order fallback,
strict dispatch metadata, package guards and installer idempotence. A separate
CPU Triton-interpreter probe executes the actual kernels on eight small
layout/broadcast/tail cases and matches native FP32 outputs/VJPs exactly.
A full frozen Q/K/V projection -> RoPE -> SDPA interpreter check also matches
outputs, Q/K/V and hidden VJPs, all relevant strides, RNG and unchanged inputs.
This is **zero GPU execution** and does not validate T4 performance or CUDA.

T4 regressions require exact native pointwise outputs/VJPs and output layouts,
RNG/input preservation, tail masking, unused outputs and the pinned Qwen/SDPA
integration. An independent FP64 reference also bounds actual arithmetic.
Existing test tolerances are not loosened. Local full solution tests:
**200 passed, 235 skipped** (Python 3.14.3, PyTorch 2.15.0+cpu, Triton 3.6.0).
The protected suite cannot start locally because `bitsandbytes` is absent;
the historical 79 PASS do not apply automatically to this candidate.
Both kernels compile offline to SM75 cubin, without FMA or register spills;
the warmup and eight runtime-shape/layout JIT keys match. This uses the local
Windows compiler package, not the Kaggle Linux runtime, and runs no GPU code.

Pin the new full SHA in `kaggle_t4.ipynb`, then run:

```bash
python -m pytest solution -x -vv --tb=short
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
python harness/run_task.py --repeats 2 --profile
python harness/check_rules.py
python harness/score.py runs/result.json
```

Preserve the tested SHA, `result.json`, `checks.json`, both solution logs and
raw metrics, plus the separate `solution_profile/profile.json`. Verify both
rotary kernels actually execute, all ten checks pass and the minimum original
share across repeats is >=5% (target >=6%). Compare full time and external peak
against 83.1s / 3671 MiB, not just one metric. If any numerical regression fails,
do not benchmark past it or relax its tolerance without isolating the cause.

No files under `harness/`, `task/` or `tests/`, task data/order, batch size,
optimizer, model/LoRA parameters, dropout, metric outputs or rewind format
are changed. No forbidden library or built-in Soup acceleration mode is used.
