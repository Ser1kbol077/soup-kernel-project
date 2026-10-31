"""Original strided vocabulary argmax, without a dense contiguous logits copy.

Each tile keeps a maximum and its first vocabulary index. A second reduction
selects the first winner across tiles. Negative partial indices encode the
first NaN, matching torch.argmax's first-NaN and first-index tie semantics.
Only small per-token statistics and the final int64 predictions are allocated.
"""

import torch
import triton
import triton.language as tl

ARGMAX_TILE = 4096
INDEX_SENTINEL = 2147483647


@triton.jit(do_not_specialize=["TIMES", "STRIDE_BATCH"])
def vocab_argmax_tile_kernel(Logits, PartialMax, PartialIndex,
                             TIMES, VOCAB: tl.constexpr,
                             STRIDE_BATCH, STRIDE_TIME: tl.constexpr,
                             STRIDE_VOCAB: tl.constexpr, PARTS: tl.constexpr,
                             BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    columns = tile * BLOCK + tl.arange(0, BLOCK)
    valid = columns < VOCAB
    start = (row // TIMES) * STRIDE_BATCH + (row % TIMES) * STRIDE_TIME
    values = tl.load(Logits + start + columns * STRIDE_VOCAB, valid,
                     other=-float("inf")).to(tl.float32)
    is_nan = values != values
    maximum = tl.max(tl.where(is_nan, -float("inf"), values), 0)
    first_nan = tl.min(tl.where(valid & is_nan, columns, 2147483647), 0)
    first_max = tl.min(tl.where(valid & (values == maximum), columns, 2147483647), 0)
    index = tl.where(first_nan < 2147483647, -first_nan - 1, first_max)
    slot = row * PARTS + tile
    tl.store(PartialMax + slot, maximum)
    tl.store(PartialIndex + slot, index)


@triton.jit
def vocab_argmax_finish_kernel(PartialMax, PartialIndex, Predictions,
                               PARTS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    valid = part < PARTS
    maxima = tl.load(PartialMax + row * PARTS + part, valid, other=-float("inf"))
    indices = tl.load(PartialIndex + row * PARTS + part, valid, other=2147483647)
    first_nan = tl.min(tl.where(valid & (indices < 0), -indices - 1, 2147483647), 0)
    maximum = tl.max(maxima, 0)
    first_max = tl.min(tl.where(valid & (indices >= 0) & (maxima == maximum),
                                indices, 2147483647), 0)
    prediction = tl.where(first_nan < 2147483647, first_nan, first_max)
    tl.store(Predictions + row, prediction.to(tl.int64))


def _supported(logits):
    return (
        isinstance(logits, torch.Tensor)
        and logits.is_cuda
        and logits.dtype in (torch.float16, torch.float32)
        and logits.ndim == 3
        and 0 < logits.numel() <= INDEX_SENTINEL
        and all(stride > 0 for stride in logits.stride())
        and sum((size - 1) * stride for size, stride in zip(logits.shape, logits.stride()))
        <= INDEX_SENTINEL
    )


def argmax_last_dim(logits):
    """Return exactly the same indices as logits.argmax(dim=-1).

    The CUDA FP16/FP32 three-dimensional path accepts positive strides, notably
    logits[:, :-1, :] with gaps between batches. Argmax has no gradient; logits
    are read only and their training graph is retained by its original owners.
    Unsupported devices, layouts, dtypes and empty shapes use PyTorch.
    """
    if not _supported(logits):
        return logits.argmax(dim=-1)

    batches, times, vocab = logits.shape
    rows = batches * times
    parts = triton.cdiv(vocab, ARGMAX_TILE)
    maxima = torch.empty((rows, parts), device=logits.device, dtype=torch.float32)
    indices = torch.empty((rows, parts), device=logits.device, dtype=torch.int32)
    predictions = torch.empty((batches, times), device=logits.device, dtype=torch.int64)
    with torch.cuda.device(logits.device):
        vocab_argmax_tile_kernel[(rows, parts)](
            logits, maxima, indices, times, vocab, *logits.stride(), parts, ARGMAX_TILE,
            num_warps=4,
        )
        vocab_argmax_finish_kernel[(rows,)](
            maxima, indices, predictions, parts, triton.next_power_of_2(parts),
            num_warps=4,
        )
    return predictions
