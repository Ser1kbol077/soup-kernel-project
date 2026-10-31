"""Original split-vocabulary reductions for the challenge's large, dense logits.

No external kernel code is used. Each tile produces a shifted partition function;
a second reduction combines tiles. Backward recomputes probabilities from logits
instead of retaining a vocabulary-sized log_softmax tensor. Entropy uses the same
partition statistics with an additional shifted first moment.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

VOCAB_TILE = 4096


@triton.jit
def vocab_partition_kernel(X, Max, Sum, Moment, V: tl.constexpr,
                           ROW_STRIDE: tl.constexpr, PARTS: tl.constexpr,
                           ENTROPY: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + row * ROW_STRIDE + col, col < V, other=-float("inf")).to(tl.float32)
    maximum = tl.max(value, 0)
    shifted = value - maximum
    probability = tl.exp(shifted)
    index = row * PARTS + part
    tl.store(Max + index, maximum)
    tl.store(Sum + index, tl.sum(probability, 0))
    if ENTROPY:
        # Avoid 0 * -inf in the masked tail. Shift before the moment reduction
        # to retain precision when every logit has a large common offset.
        moment = probability * tl.where(col < V, shifted, 0.0)
        tl.store(Moment + index, tl.sum(moment, 0))


@triton.jit
def vocab_ce_finalize_kernel(X, Target, Max, Sum, RowMax, LogSum, Loss,
                             V: tl.constexpr, ROW_STRIDE: tl.constexpr,
                             PARTS: tl.constexpr, IGNORE: tl.constexpr,
                             BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    maxima = tl.load(Max + row * PARTS + part, part < PARTS, other=-float("inf"))
    sums = tl.load(Sum + row * PARTS + part, part < PARTS, other=0.0)
    maximum = tl.max(maxima, 0)
    log_sum = tl.log(tl.sum(sums * tl.exp(maxima - maximum), 0))
    label = tl.load(Target + row)
    valid = (label != IGNORE) & (label >= 0) & (label < V)
    selected = tl.load(X + row * ROW_STRIDE + label, valid, other=0.0).to(tl.float32)
    tl.store(RowMax + row, maximum)
    tl.store(LogSum + row, log_sum)
    tl.store(Loss + row, tl.where(valid, (maximum - selected) + log_sum, 0.0))


@triton.jit
def vocab_ce_backward_kernel(X, Target, RowMax, LogSum, GradLoss, GradX,
                             V: tl.constexpr, ROW_STRIDE: tl.constexpr,
                             IGNORE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    label = tl.load(Target + row)
    maximum = tl.load(RowMax + row)
    log_sum = tl.load(LogSum + row)
    value = tl.load(X + row * ROW_STRIDE + col, col < V, other=-float("inf")).to(tl.float32)
    grad = tl.load(GradLoss + row).to(tl.float32)
    probability = tl.exp((value - maximum) - log_sum)
    # where, rather than multiplication by zero: all-ignored mean loss has
    # NaN upstream scaling, while PyTorch still returns exactly zero grad.
    result = tl.where(label != IGNORE, (probability - (col == label).to(tl.float32)) * grad, 0.0)
    tl.store(GradX + row * V + col, result, col < V)


@triton.jit
def vocab_entropy_finalize_kernel(Max, Sum, Moment, Out,
                                  PARTS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    maxima = tl.load(Max + row * PARTS + part, part < PARTS, other=-float("inf"))
    sums = tl.load(Sum + row * PARTS + part, part < PARTS, other=0.0)
    moments = tl.load(Moment + row * PARTS + part, part < PARTS, other=0.0)
    maximum = tl.max(maxima, 0)
    delta = tl.where(part < PARTS, maxima - maximum, 0.0)
    rescale = tl.exp(delta)
    partition = tl.sum(sums * rescale, 0)
    moment = tl.sum((moments + sums * delta) * rescale, 0)
    tl.store(Out + row, tl.log(partition) - moment / partition)


def _supported(logits):
    return (logits.is_cuda and logits.ndim == 2 and logits.size(0) > 0
            and logits.size(1) > 0 and logits.stride(1) == 1
            and logits.dtype == torch.float32)


def _partition(logits, entropy=False):
    rows, vocab = logits.shape
    parts = triton.cdiv(vocab, VOCAB_TILE)
    maxima = torch.empty((rows, parts), device=logits.device, dtype=torch.float32)
    sums = torch.empty_like(maxima)
    moments = torch.empty_like(maxima) if entropy else sums
    vocab_partition_kernel[(rows, parts)](
        logits, maxima, sums, moments, vocab, logits.stride(0), parts,
        entropy, VOCAB_TILE, num_warps=4, enable_fp_fusion=False,
    )
    return maxima, sums, moments


class _TokenCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, ignore_index):
        # Match the native error for out-of-range class indices without a
        # host synchronization. The normal training labels already satisfy it.
        torch._assert_async(((targets == ignore_index)
                             | ((targets >= 0) & (targets < logits.size(1)))).all(),
                            "cross_entropy target out of bounds")
        rows, vocab = logits.shape
        maxima, sums, _ = _partition(logits)
        row_max = torch.empty(rows, device=logits.device, dtype=torch.float32)
        log_sum = torch.empty_like(row_max)
        loss = torch.empty(rows, device=logits.device, dtype=logits.dtype)
        parts = maxima.size(1)
        vocab_ce_finalize_kernel[(rows,)](
            logits, targets, maxima, sums, row_max, log_sum, loss,
            vocab, logits.stride(0), parts, ignore_index, triton.next_power_of_2(parts),
            num_warps=4, enable_fp_fusion=False,
        )
        ctx.save_for_backward(logits, targets, row_max, log_sum)
        ctx.ignore_index = ignore_index
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss):
        logits, targets, row_max, log_sum = ctx.saved_tensors
        grad_loss = grad_loss.contiguous()
        grad_logits = torch.empty(logits.shape, device=logits.device, dtype=logits.dtype)
        rows, vocab = logits.shape
        vocab_ce_backward_kernel[(rows, triton.cdiv(vocab, VOCAB_TILE))](
            logits, targets, row_max, log_sum, grad_loss, grad_logits,
            vocab, logits.stride(0), ctx.ignore_index, VOCAB_TILE,
            num_warps=4, enable_fp_fusion=False,
        )
        return grad_logits, None, None


def cross_entropy(logits, targets, reduction="mean", ignore_index=-100):
    """Hard-label CE; first-order autograd, with a native unsupported-case path.

    The hook supplies fp32 logits just as Transformers does. Keeping reductions
    in PyTorch also preserves its mean-of-zero-valid-tokens NaN semantics.
    """
    if (not _supported(logits) or targets.ndim != 1
            or targets.shape[0] != logits.shape[0] or targets.dtype != torch.long
            or targets.device != logits.device or not targets.is_contiguous()
            or reduction not in ("none", "sum", "mean")):
        return F.cross_entropy(logits, targets, reduction=reduction, ignore_index=ignore_index)
    with torch.cuda.device(logits.device):
        losses = _TokenCrossEntropy.apply(logits, targets, ignore_index)
    if reduction == "none":
        return losses
    total = losses.sum()
    return total if reduction == "sum" else total / (targets != ignore_index).sum()


def _native_entropy(logits):
    log_probs = logits.log_softmax(dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def entropy_from_logits(logits, chunk_size=128):
    """Exact categorical entropy, fused only for the no-grad metrics path.

    chunk_size is accepted for TRL API compatibility; tiling is by vocabulary.
    Autograd callers keep a native differentiable path.
    """
    if (logits.ndim < 2 or not logits.is_contiguous()
            or (torch.is_grad_enabled() and logits.requires_grad)):
        return _native_entropy(logits)
    flat = logits.view(-1, logits.size(-1))
    if not _supported(flat):
        return _native_entropy(logits)
    with torch.cuda.device(logits.device):
        maxima, sums, moments = _partition(flat, entropy=True)
        out = torch.empty(flat.size(0), device=logits.device, dtype=logits.dtype)
        parts = maxima.size(1)
        vocab_entropy_finalize_kernel[(flat.size(0),)](
            maxima, sums, moments, out, parts, triton.next_power_of_2(parts),
            num_warps=4, enable_fp_fusion=False,
        )
    return out.view(logits.shape[:-1])
