"""Frozen linear head, causal loss and metrics without retaining dense logits.

Token rows are processed in bounded chunks. Backward recomputes those chunks,
so the actual upstream loss scale is applied before the FP16 cast, as in the
native autocast linear -> FP32 cross-entropy graph. Model weights stay on GPU.
All kernels in this module are original implementations of tiled reductions.
"""

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU contract tests do not need Triton.
    triton = None


CHUNK_ROWS = 64
VOCAB_TILE = 4096


if triton is not None:
    @triton.jit
    def head_statistics_tile_kernel(X, Max, Sum, Moment, Index,
                                    V: tl.constexpr, PARTS: tl.constexpr,
                                    BLOCK: tl.constexpr):
        row, tile = tl.program_id(0), tl.program_id(1)
        col = tile * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(X + row * V + col, col < V, other=-float("inf")).to(tl.float32)
        maximum = tl.max(values, 0)
        shifted = values - maximum
        probability = tl.exp(shifted)
        index = tl.min(tl.where((col < V) & (values == maximum), col, 2147483647), 0)
        out = row * PARTS + tile
        tl.store(Max + out, maximum)
        tl.store(Sum + out, tl.sum(probability, 0))
        tl.store(Moment + out, tl.sum(probability * tl.where(col < V, shifted, 0.0), 0))
        tl.store(Index + out, index)


    @triton.jit
    def head_statistics_merge_kernel(X, Targets, Max, Sum, Moment, Index,
                                     RowMax, LogSum, Loss, Entropy, Predictions,
                                     V: tl.constexpr, PARTS: tl.constexpr,
                                     IGNORE: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        tile = tl.arange(0, BLOCK)
        maxima = tl.load(Max + row * PARTS + tile, tile < PARTS, other=-float("inf"))
        sums = tl.load(Sum + row * PARTS + tile, tile < PARTS, other=0.0)
        moments = tl.load(Moment + row * PARTS + tile, tile < PARTS, other=0.0)
        indices = tl.load(Index + row * PARTS + tile, tile < PARTS, other=2147483647)
        maximum = tl.max(maxima, 0)
        delta = tl.where(tile < PARTS, maxima - maximum, 0.0)
        rescale = tl.exp(delta)
        total = tl.sum(sums * rescale, 0)
        logsum = tl.log(total)
        moment = tl.sum((moments + sums * delta) * rescale, 0)
        prediction = tl.min(tl.where(maxima == maximum, indices, 2147483647), 0)
        target = tl.load(Targets + row)
        valid = (target != IGNORE) & (target >= 0) & (target < V)
        selected = tl.load(X + row * V + target, valid, other=0.0).to(tl.float32)
        tl.store(RowMax + row, maximum)
        tl.store(LogSum + row, logsum)
        tl.store(Loss + row, tl.where(valid, maximum - selected + logsum, 0.0))
        tl.store(Entropy + row, logsum - moment / total)
        tl.store(Predictions + row, prediction.to(tl.int64))


def _statistics(logits, targets, ignore_index):
    """CE, entropy, first-index argmax and two small backward statistics."""
    if logits.is_cuda:
        rows, vocab = logits.shape
        parts = triton.cdiv(vocab, VOCAB_TILE)
        partial = torch.empty((3, rows, parts), dtype=torch.float32, device=logits.device)
        indices = torch.empty((rows, parts), dtype=torch.int32, device=logits.device)
        maximum, logsum, loss, entropy = torch.empty((4, rows), dtype=torch.float32,
                                                    device=logits.device).unbind()
        predictions = torch.empty(rows, dtype=torch.long, device=logits.device)
        head_statistics_tile_kernel[(rows, parts)](
            logits, partial[0], partial[1], partial[2], indices, vocab, parts, VOCAB_TILE,
            num_warps=4, enable_fp_fusion=False,
        )
        head_statistics_merge_kernel[(rows,)](
            logits, targets, partial[0], partial[1], partial[2], indices,
            maximum, logsum, loss, entropy, predictions, vocab, parts, ignore_index,
            triton.next_power_of_2(parts), num_warps=4, enable_fp_fusion=False,
        )
        return loss, entropy, predictions, maximum, logsum
    # Same chunked algorithm on CPU for numerical/autograd contract tests.
    source = logits if logits.dtype == torch.float64 else logits.float()
    maximum = source.max(-1).values
    shifted = source - maximum[:, None]
    logsum = shifted.exp().sum(-1).log()
    logprob = shifted - logsum[:, None]
    entropy = -(logprob.exp() * logprob).sum(-1)
    loss = F.cross_entropy(source, targets, ignore_index=ignore_index, reduction="none")
    return loss, entropy, source.argmax(-1), maximum, logsum


def _logits_gradient(logits, targets, maximum, logsum, upstream, ignore_index):
    if logits.is_cuda:
        from .vocab_ops import vocab_ce_backward_kernel

        rows, vocab = logits.shape
        gradient = torch.empty_like(logits)
        factors = upstream.expand(rows).contiguous()
        vocab_ce_backward_kernel[(rows, triton.cdiv(vocab, VOCAB_TILE))](
            logits, targets, maximum, logsum, factors, gradient,
            vocab, logits.stride(0), ignore_index, VOCAB_TILE,
            num_warps=4, enable_fp_fusion=False,
        )
        return gradient
    source = logits if logits.dtype == torch.float64 else logits.float()
    probability = ((source - maximum[:, None]) - logsum[:, None]).exp()
    live = targets != ignore_index
    safe = torch.where(live, targets, 0)
    probability.scatter_add_(1, safe[:, None], -live.to(source.dtype)[:, None])
    probability.mul_(upstream)
    probability.masked_fill_(~live[:, None], 0.0)
    return probability.to(logits.dtype)


class _StreamedLinearHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, denominator, compute_dtype, chunk_rows, ignore_index):
        shape, width = hidden.shape, hidden.shape[-1]
        # Explicit operands make forward/backward independent of the ambient
        # autocast state, which normally ends before loss.backward().
        rows = hidden.reshape(-1, width).to(compute_dtype)
        matrix = weight.to(compute_dtype)
        n = rows.shape[0]
        accumulation = torch.float64 if compute_dtype == torch.float64 else torch.float32
        losses = torch.empty(n, dtype=accumulation, device=hidden.device)
        entropies = torch.empty_like(losses)
        predictions = torch.empty(n, dtype=torch.long, device=hidden.device)
        maxima, logsums = torch.empty_like(losses), torch.empty_like(losses)
        with torch.autocast(hidden.device.type, enabled=False):
            for start in range(0, n, chunk_rows):
                stop = min(start + chunk_rows, n)
                logits = torch.mm(rows[start:stop], matrix.t())
                values = _statistics(logits, targets[start:stop], ignore_index)
                for destination, value in zip((losses, entropies, predictions, maxima, logsums), values):
                    destination[start:stop].copy_(value)
                del logits, values
        loss = losses.sum() / denominator
        ctx.save_for_backward(rows, matrix, targets, maxima, logsums, denominator)
        ctx.shape, ctx.input_dtype = shape, hidden.dtype
        ctx.chunk_rows, ctx.ignore_index = chunk_rows, ignore_index
        metrics = (losses.view(shape[:-1]), entropies.view(shape[:-1]), predictions.view(shape[:-1]))
        ctx.mark_non_differentiable(*metrics)
        return loss, *metrics

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss, _grad_tokens, _grad_entropy, _grad_predictions):
        if not ctx.needs_input_grad[0]:
            return (None,) * 7
        rows, matrix, targets, maxima, logsums, denominator = ctx.saved_tensors
        result = torch.empty(rows.shape, dtype=ctx.input_dtype, device=rows.device)
        factor = (grad_loss / denominator).reshape(())
        with torch.autocast(rows.device.type, enabled=False):
            for start in range(0, rows.shape[0], ctx.chunk_rows):
                stop = min(start + ctx.chunk_rows, rows.shape[0])
                logits = torch.mm(rows[start:stop], matrix.t())
                gradient = _logits_gradient(logits, targets[start:stop], maxima[start:stop],
                                             logsums[start:stop], factor, ctx.ignore_index)
                # Rounds at the native linear's FP16 boundary before conversion
                # back to the original FP32 residual stream.
                result[start:stop].copy_(torch.mm(gradient, matrix))
                del logits, gradient
        return result.view(ctx.shape), None, None, None, None, None, None


def linear_head_loss(hidden, weight, labels, num_items_in_batch=None,
                     attention_mask=None, *, shift_labels=None, ignore_index=-100,
                     chunk_rows=CHUNK_ROWS):
    """Return causal loss and detached source-aligned token statistics.

    ``token_loss[..., t]`` predicts ``labels[..., t+1]``; the terminal label is
    ignored unless explicit ``shift_labels`` are supplied. Entropy/predictions
    include every source position; the caller applies the original metric mask.
    Only hidden-state gradients are supported: this path requires a frozen head.
    CPU executes the same chunk/recompute algorithm for contract testing.
    """
    if hidden.ndim != 3 or weight.ndim != 2 or hidden.shape[-1] != weight.shape[1]:
        raise ValueError("expected hidden[B,T,H] and weight[V,H]")
    if weight.requires_grad:
        raise ValueError("streamed head requires frozen weight")
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    if hidden.numel() == 0 or weight.shape[0] == 0:
        raise ValueError("streamed head requires nonempty inputs")
    if hidden.device != weight.device or not hidden.is_floating_point() or not weight.is_floating_point():
        raise ValueError("hidden and weight must be floating tensors on the same device")
    targets = shift_labels
    if targets is None:
        if labels is None:
            raise ValueError("labels or shift_labels required")
        targets = F.pad(labels, (0, 1), value=ignore_index)[..., 1:]
    if targets.shape != hidden.shape[:-1] or targets.dtype != torch.long:
        raise ValueError("labels must be int64 and match hidden token dimensions")
    targets = targets.to(hidden.device).reshape(-1).contiguous()
    torch._assert_async(((targets == ignore_index) | ((targets >= 0) & (targets < weight.shape[0]))).all(),
                        "streamed head target out of bounds")
    dtype = hidden.dtype
    if torch.is_autocast_enabled(hidden.device.type):
        dtype = torch.get_autocast_dtype(hidden.device.type)
    if hidden.is_cuda and (triton is None or dtype not in (torch.float16, torch.float32)):
        raise ValueError("CUDA streamed head requires Triton and FP16/FP32 compute")
    if not torch.is_autocast_enabled(hidden.device.type) and hidden.dtype != weight.dtype:
        raise ValueError("without autocast hidden and weight dtypes must match")
    denominator = ((targets != ignore_index).sum() if num_items_in_batch is None
                   else torch.as_tensor(num_items_in_batch, device=hidden.device))
    if denominator.ndim != 0:
        raise ValueError("num_items_in_batch must be scalar")
    denominator = denominator.to(device=hidden.device, dtype=torch.float64 if dtype == torch.float64 else torch.float32)
    loss, tokens, entropy, predictions = _StreamedLinearHead.apply(
        hidden, weight, targets, denominator, dtype, chunk_rows, ignore_index,
    )
    return loss, {"token_loss": tokens, "entropy": entropy, "predictions": predictions}
