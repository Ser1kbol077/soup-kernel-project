# Architecture comparison and the next controlled change

Analysis date: 1 November 2026. The priority is admission with all ten checks,
then a measured improvement in both complete time and GPU peak. Targets remain
40 seconds, no more than 3671 MiB where achievable, and at least 6% useful
original-kernel share to give margin above the mandatory 5%.

## Evidence boundaries and provenance

| Evidence | What is known | What is not established |
|---|---|---|
| Our last measured run, reported by the user | 86.6s / 14345 MiB baseline; 83.1s / 3671 MiB solution; eval 2.04963 vs 2.04965; 79 Soup tests and 9/10 checks pass; own share 4.7% | Tested SHA and repeat count are unconfirmed; raw result/checks/profile were lost when Kaggle reset |
| Other team's notebook, reported by the user | 90.8s / 14345 MiB baseline; 51.4s / 599 MiB solution; reported ratios 1.768x and 23.948x; nearly equal eval; 14.5% own share; 10/10 checks for one pair | Raw artifacts and the exact implementation were not independently verified here; no second pair or hidden-row result |
| Our current source | Implementation and mathematical tensor sizes can be inspected | Later commits do not inherit the old run's timing, memory, quality or PASS status |
| Local `runs/` | CPU/interpreter and offline-compiler diagnostics exist | No current run's GPU `profile.json`, complete `result.json`, raw step timings or `train.log` were found |

The external reference supplied by the user is
[BAITC-Hacks/hack-8c5e0501-saikyo, commit 450312b8082fef5811dd3d030b951dedaf243391](https://github.com/BAITC-Hacks/hack-8c5e0501-saikyo/commit/450312b8082fef5811dd3d030b951dedaf243391).
Fetching that commit did not succeed in this analysis. No foreign kernel
implementation was used to author the new code. The external kernel names
below are user-provided clues, not proof of algorithms or data placement.

Sources for the analysis are our repository, the
[official challenge rules](https://github.com/MakazhanAlpamys/soup-kernel-challenge#rules),
the [Qwen configuration](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json),
and [PEFT 0.20 preparation](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/other.py).
The proposed future online softmax approach is attributed separately below.
No source-code similarity is used to infer the foreign architecture.

The checker accepts a result with the requested repeat count >=1. Thus a
single base/solution pair can print 10/10 without establishing the required
base/solution/solution/base jury protocol. Our comparison must use two pairs,
median complete wall time, maximum external peak and minimum own share.
Different baseline sessions also prevent interpreting 51.4 vs 83.1 as an
isolated implementation comparison.

## What the kernel names suggest, and what our source actually does

| Operation | User-provided foreign names | Our inspected implementation | Inference limit / safe next direction |
|---|---|---|---|
| Residual and normalization | `add_rmsnorm_fwd_kernel`, `add_rmsnorm_bwd_kernel` | `rmsnorm.py` fuses frozen FP32 RMSNorm; decoder residual adds are separate | Names suggest residual/norm fusion. They do not prove layout, rounding, saved-state policy or backward formula. Our own fusion is feasible, but replacing decoder forward is a wider integration change |
| Embedding | `embedding_gather_kernel` | Standard GPU embedding, with the shared frozen FP32 table resident | A gather kernel alone does not imply CPU storage. A future exact CPU row gather/staging path could keep the table off GPU; input order, tying and dtype must remain unchanged |
| Vocabulary head | `head_chunk_kernel`, `head_finish_kernel` | `streamed_head.py`: token chunks of 256, full FP16 head cast per invocation, bounded logits, CE/entropy/argmax, precomputed FP32 hidden VJP | Names do not distinguish token chunks, vocabulary slices, one/two passes, or weight streaming. Our head already streams token rows but still holds the whole head matrix |
| NF4 | `nf4_dequant_kernel` | `nf4_decode.py` fuses nested-scale decode, offset and NF4 conversion; all packed decoder weights remain on GPU | Decode fusion does not imply low residency. Layer windows would require exact packed weights/quant state and stream-safe buffer lifetimes |
| Padding and dropout | `pack_rows_kernel`, `packed_dropout_kernel` | Head gathers observed/supervised rows. Decoder attention and LoRA dropout keep full shapes; SM75 MLP packing is disabled after a measured gradient mismatch | Names suggest packing/dropout work, but do not establish RNG compatibility or which rows are packed. Packing trainable LoRA linears changes reduction shapes even if dropout RNG is preserved |
| RoPE | `rope_scatter_kernel`, `rope_gather_bwd_kernel` | Original FP32 RoPE kernels preserve forward layout and native backward layout; full T4 validation remains pending | Names may indicate packing-aware indexing or just layout operations. Our implementation does not assume their meaning |
| Slices and accumulation | `slice_accumulate_kernel`, `slice_update_kernel` | Head and MLP use several native casts, copies and additions; decoder offload has per-layer leases | Names do not identify the tensor being accumulated. Independent fusion of existing branch casts/addition is straightforward and is the only new optimization in this commit |

All architectural interpretations in the final column are hypotheses or our
own proposed designs. None is presented as a description of the foreign code.

## Why the present residency cannot reach 599 MiB

The fixed dimensions are vocabulary 151936, width 1536, MLP width 8960,
28 layers, 12 query heads and 2 KV heads of dimension 128. PEFT's standard
k-bit preparation promotes nonquantized FP16/BF16 parameters to FP32. In this
pipeline the frozen embedding and head share one FP32 table; count it once.

| Tensor allocation at maximum task shape | Calculation | MiB |
|---|---|---:|
| Shared FP32 embedding/head | 151936 * 1536 * 4 | 890.250 |
| Full temporary FP16 head matrix | 151936 * 1536 * 2 | 445.125 |
| All packed NF4 decoder projections | 28 * (2 * 1536^2 + 2 * 1536 * 256 + 3 * 1536 * 8960) / 2 | 624.750 |
| First nested quantized scale arrays | packed elements * 2 / 64 | 19.523 |
| FP32 q/v LoRA parameters | 2179072 * 4 | 8.313 |
| LoRA params + gradients + two Adam states | 4 * parameter bytes | 33.250 |
| One FP16 token-logits chunk | 256 * 151936 * 2 | 74.188 |
| Full FP32 precomputed head VJP | 2048 * 1536 * 4 | 12.000 |

For clarity, the decoder NF4 count per layer is
`2 * 1536^2 + 2 * 1536 * 256 + 3 * 1536 * 8960` elements
(q/o, k/v, gate/up/down). It excludes biases and small second-level states.
These are sizes, not measured peaks; lifetimes and workspaces also matter.

The persistent FP32 table plus packed weights and first scales are already
about 1534.5 MiB. Packed NF4 bytes alone exceed 599 MiB. Therefore neither
activation offload nor a faster gather/norm kernel can produce that peak with
our current residency. This is a mathematical constraint, not an accusation
about the reported foreign result.

Sub-1-GiB execution would require a different loading/residency design:

1. Keep the original tied FP32 table on CPU; gather only needed embedding rows
   into GPU staging, reconstructing the exact original row order. The head
   reads the same frozen table through bounded vocabulary slices.
2. Keep most immutable packed NF4 weights/scales on CPU, staging a bounded
   layer window for forward and again for backward. Preserve original GPU
   quantization bytes/states rather than silently switching quantizers.
3. Load/quantize/stage incrementally from the beginning. Offloading after a
   complete GPU model load does not undo the loading peak already measured.
4. Bound GPU activation, decoded-weight, logits and VJP workspaces together
   with CUDA/cuBLAS overhead. Publish pinned-host capacity and process RAM too.

The final peak is the maximum over loading, forward, head, backward and saving,
not their sum or a steady-state tensor census. This design is feasible to
investigate, not a promise of 599 MiB or faster execution: it adds real CPU
work, H2D/D2H transfers, synchronization and loading complexity.

## Padding and dropout are correctness constraints

Our head shifts labels inside each batch row before gathering; metric-mask
positions and supervised query positions are retained. Attention still sees
the same original batch and positions. The SM75 MLP remains full-shape because
packing 256 to 161 rows failed the supplied T4 gradient tolerance.

Dropping padding before q/v LoRA dropout changes the number/order of random
draws. Matching probability or seed is insufficient. A packed implementation
would need the original full-shape random stream, the same q-then-v invocation
order and correct scatter/gather Jacobians. It must not cross sequence or
batch boundaries or consume additional Torch RNG. It also changes LoRA dW
reduction dimensions, so an unchanged random mask still does not prove
numerical equivalence. No decoder/LoRA packing is implemented here.

A lower-risk future storage optimization is losslessly packing the already
generated bool dropout masks: preserve RNG and mask values, transfer fewer
bytes, reconstruct the exact mask for backward. This is distinct from
generating a new packed dropout mask. PyTorch's
[CUDA native dropout](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/Dropout.cu)
and [autograd derivatives](https://github.com/pytorch/pytorch/blob/v2.11.0/tools/autograd/derivatives.yaml)
describe the saved bool mask. At maximum shape, two q/v masks per layer are
168 MiB raw or 21 MiB bit-packed; an actual saved-tensor census must establish
their occurrence, lifetime and transfer contribution in our Kaggle run.

## Head slicing, NF4 and buffer reuse

The current head casts the full matrix once per forward, not once per token
chunk. Vocabulary slices of 4096 rows would need a 12 MiB FP16 matrix instead
of 445.125 MiB; 8192 rows need 24 MiB. This can reduce a real workspace, but
does not remove the resident 890.25 MiB original table.

First retain existing full-chunk logits and global statistics while evaluating
weight slices. A fully streamed two-pass head recomputes logits after global
normalization: three head GEMMs instead of two can add about 50% head FLOPs.
At maximum shape that is about 0.956 TFLOP extra per step, not a speedup claim.
One-pass weighted VJP accumulation with online normalization avoids that extra
pass but moves normalization relative to FP16 probability casts and changes
accumulation order. It needs independent FP64 and actual T4 checks. The online
normalizer idea is attributed to
[Milakov and Gimelshein, 2018](https://arxiv.org/abs/1805.02867); no kernel code
from that work or another team is used here.

Low-risk future head changes include writing statistics/VJP directly to final
chunk views and reusing a local logits allocation. They preserve GEMM shapes.
A global persistent head workspace could instead increase decoder peak; use
explicit lifetimes rather than assuming every cache helps.

Our NF4 decoder returns fresh output buffers because callers/autograd may
retain them. Blindly returning a reused public decoder buffer is unsafe.
Caching every decoded decoder weight would add about 2499 MiB and remove
steady-state decode work, conflicting with both memory and possibly share.
A future bounded stage/decode window needs events, ownership and version/state
invalidation, not a global scratch tensor overwritten between projections.

Current activation offload has real costs: per-save Python hooks, parameter
alias checks, pinned slab allocation/consolidation, individual copies and
bounded-queue waits. Removing all waits can grow the live GPU queue if PCIe
falls behind. Repeated whole-layer prefetch after one cache miss may restore
unneeded payloads. Both require a measured tensor/transfer census before
choosing the next change; neither is modified in this commit.

## Rules and measurement

The official rules explicitly allow recomputation/offloading in our own code,
with its time counted. This is different from enabling Soup's built-in
activation-offloading/checkpointing/packing switches. Those flags, optimizer,
schedule, LoRA, dropout, batch order, outputs and adapter remain unchanged.
See local `harness/common.py`, `_train.py` and `check_rules.py` for the exact
measurement and comparisons; none is edited.

CPU placement must not involve another GPU, external/background GPU work,
warm caches prepared outside timing, changed allocator settings, substituted
adapters or measurement-dependent behavior. The external poller covers the
whole child process and sums all GPUs. Loading peaks count. Check 9 compares
the external peak to PyTorch reserved memory; it is not permission to exploit
the 100 ms sampler. Genuine host RAM and transfer costs must be reported.

## Amdahl and the missing profile

We cannot rank actual current bottlenecks without the lost profile. The other
team's timings and kernel names are not our profile. Our offline compiler
reports explicitly execute zero GPU kernels.

From 83.1 to 40 seconds requires removing 43.1 seconds, or 51.9% of complete
wall time. Let C be the measured wall time outside the training loop and L
the loop time. Required loop acceleration is `L / (40-C)`; C must be <40.
For illustration only: assumed C=20/25/30 seconds requires loop speedups
3.16x/3.87x/5.31x respectively. These are scenarios, not measured startup costs.

The reported 4.7% is a timed-loop kernel share, not a full-wall fraction.
Similarly, the reported 14.5% alone does not rank kernel efficiency: share
depends on useful operation coverage and the measured loop denominator.
Making only that portion infinitely fast gives at most 1.049x loop speedup
under the simple non-overlap assumption. With own milliseconds unchanged,
reaching 6% share requires cutting the denominator by 21.7%. Useful fusion
can instead cover native work, but the resulting percentage must be measured.

## The only new optimization implemented

`mlp_branch_sum.py` replaces the gate-dX FP16-to-FP32 materialization followed
by in-place addition of up-dX. The two cuBLAS/bitsandbytes GEMMs, their shapes
and their independent FP16 output rounding stay unchanged. One original
kernel loads both half results, promotes each to FP32, adds once and writes
the FP32 input gradient. The implementation comes from our existing graph
and the elementary sum, not a foreign kernel.

At the maximum 2048x1536 input, the previous cast plus add transfers roughly
16N bytes; the fused sum transfers 8N bytes, saving 24 MiB per eligible layer,
up to 672 MiB per step over 28 layers. It removes one launch per eligible
MLP. Keeping gate-dX half until the up GEMM completes reduces that live buffer
from 12 to 6 MiB; the final sum needs both half inputs plus its FP32 output.
Global peak reduction and speed cannot be inferred from this local lifetime.

The selected tile is 256 elements / four warps. Offline SM75 compilation of
the same formula showed no shared-memory layout conversion or barrier for
this tile; the 1024-element tile used 4 KiB shared and one barrier. This is a
compiler-resource choice, not a measured throughput optimum. No arithmetic
or extra work was added to influence kernel recognition or measured share.

This is selected first for verifiable arithmetic and coverage on an existing
training path. It will not by itself remove the 890/445 MiB head residency,
establish 6% share or explain 40 seconds. No sliced head, CPU-weight loader,
padding/dropout change or offload policy change is bundled with it.

## Priority and acceptance plan

1. Run the entire solution CUDA suite on the target T4, then all 79 protected
   Soup tests. Fix failures before benchmarking; historical PASS does not carry
   to a new SHA. Preserve the previous `907c0f7` candidate for comparison.
2. Measure baseline/solution/solution/baseline with fresh compiler caches,
   exact tested SHA and the unmodified harness. Require all ten checks, then
   assess minimum own share with a 6% target. No artificial work if it is low.
3. Preserve `result.json`, `checks.json`, every `metrics.json` and `train.log`,
   plus the separate GPU `profile.json`. Add an out-of-band diagnostic tensor
   census, CUDA memory trace, phase timing/JIT log, transfer bytes and host RSS;
   keep these diagnostic runs separate from scored runs.
4. Use that evidence to choose one next local allocation/transfer fusion:
   head destination reuse, exact mask storage or targeted prefetch. Measure
   each independently; retain admitted time/memory tradeoffs.
5. Prototype vocabulary-sliced head separately with exact global metrics and
   a FP64 reference. Only then evaluate CPU-backed tied table and staged
   embedding/head. Verify plain-transformers adapter evaluation unchanged.
6. For sub-1-GiB experiments, implement host-first exact NF4 loading and a
   bounded forward/backward weight window as a separate architecture. Keep
   baseline loader and all admitted candidates available. If no candidate
   reaches both targets, choose among measured 10/10 variants, not the lowest
   unverified number.

Local CPU checks for this commit: **232 passed, 252 skipped**. Real T4 branch
sum/NF4 tests require exact native values, including independent branch
rounding, finite overflow boundaries, RNG and unchanged inputs/parameters.
The protected Soup suite could not start locally because `bitsandbytes` is
absent; the historical 79 PASS are not a validation of this new commit.
CPU interpreter and SM75 compile diagnostics are supporting evidence only.
No complete T4 benchmark, 10/10 claim, new peak or speedup is asserted.
