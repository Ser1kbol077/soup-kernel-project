"""Fuse Qwen2's FP32 query/key rotation without changing attention or LoRA.

Each program rotates both halves of several head rows. Separate FP32 products
and their final addition preserve the native arithmetic; FP fusion is disabled.
Backward needs only the original cosine/sine tensors, not query/key values.
Unsupported calls retain ordinary PyTorch or the installed Qwen implementation.
"""

from functools import wraps
from importlib import metadata

import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


BLOCK = 256
_MARKER = "_soup_rotary_patch"
_DYNAMIC = ["n_pairs", "time", "heads", "x_s0", "x_s1", "x_s2", "out_s0", "out_s1",
            "out_s2", "cos_batch_stride", "sin_batch_stride"]


if triton is not None:
    @triton.jit(do_not_specialize=_DYNAMIC)
    def rotary_forward_kernel(X, Cos, Sin, Output, n_pairs, time, heads,
                              x_s0, x_s1, x_s2, out_s0, out_s1, out_s2,
                              cos_batch_stride, sin_batch_stride,
                              BLOCK: tl.constexpr):
        pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = pair < n_pairs
        row, lane = pair // 64, pair % 64
        token = row % time
        head = (row // time) % heads
        batch = row // (time * heads)
        source = batch * x_s0 + head * x_s1 + token * x_s2 + lane
        target = batch * out_s0 + head * out_s1 + token * out_s2 + lane
        cosine = batch * cos_batch_stride + token * 128 + lane
        sine = batch * sin_batch_stride + token * 128 + lane
        low, high = tl.load(X + source, live, other=0.), tl.load(X + source + 64, live, other=0.)
        c0, c1 = tl.load(Cos + cosine, live, other=0.), tl.load(Cos + cosine + 64, live, other=0.)
        s0, s1 = tl.load(Sin + sine, live, other=0.), tl.load(Sin + sine + 64, live, other=0.)
        low_out = low * c0 + (-high) * s0
        high_out = high * c1 + low * s1
        tl.store(Output + target, low_out, live)
        tl.store(Output + target + 64, high_out, live)


    @triton.jit(do_not_specialize=_DYNAMIC)
    def rotary_backward_kernel(Grad, Cos, Sin, Output, n_pairs, time, heads,
                               x_s0, x_s1, x_s2, out_s0, out_s1, out_s2,
                               cos_batch_stride, sin_batch_stride,
                               BLOCK: tl.constexpr):
        pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = pair < n_pairs
        row, lane = pair // 64, pair % 64
        token = row % time
        head = (row // time) % heads
        batch = row // (time * heads)
        source = batch * x_s0 + head * x_s1 + token * x_s2 + lane
        target = batch * out_s0 + head * out_s1 + token * out_s2 + lane
        cosine = batch * cos_batch_stride + token * 128 + lane
        sine = batch * sin_batch_stride + token * 128 + lane
        low, high = tl.load(Grad + source, live, other=0.), tl.load(Grad + source + 64, live, other=0.)
        c0, c1 = tl.load(Cos + cosine, live, other=0.), tl.load(Cos + cosine + 64, live, other=0.)
        s0, s1 = tl.load(Sin + sine, live, other=0.), tl.load(Sin + sine + 64, live, other=0.)
        low_out = low * c0 + high * s1
        high_out = high * c1 + (-(low * s0))
        tl.store(Output + target, low_out, live)
        tl.store(Output + target + 64, high_out, live)


def _dense_layout(tensor):
    batch, heads, time, width = tensor.shape
    return (tensor.storage_offset() == 0 and tensor.stride() in (
        (heads * time * width, time * width, width, 1),
        (time * heads * width, width, heads * width, 1),
    ))


def _supported(q, k, cos, sin, unsqueeze_dim=1):
    if (triton is None or type(unsqueeze_dim) is not int or unsqueeze_dim != 1
            or any(type(value) is not torch.Tensor for value in (q, k, cos, sin))
            or not q.is_cuda or torch.version.hip is not None
            or any(value.device != q.device or value.dtype != torch.float32
                   or value.layout != torch.strided or value.is_conj() or value.is_neg()
                   for value in (q, k, cos, sin))
            or q.ndim != 4 or k.ndim != 4):
        return False
    batch, heads, time, width = q.shape
    if (batch < 1 or time <= 1 or heads != 12 or width != 128
            or tuple(k.shape) != (batch, 2, time, width)
            or not _dense_layout(q) or not _dense_layout(k)):
        return False
    for value in (cos, sin):
        if (value.requires_grad or value.ndim != 3
                or value.shape[0] not in (1, batch)
                or tuple(value.shape[1:]) != (time, width)
                or value.storage_offset() != 0
                or value.stride() != (time * width, width, 1)):
            return False
    return torch.cuda.get_device_capability(q.device) == (7, 5)


def _native_rotary(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(value):
        half = value.shape[-1] // 2
        return torch.cat((-value[..., half:], value[..., :half]), dim=-1)

    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def _launch(kernel, values, cos, sin, output):
    batch, heads, time, _ = values.shape
    n_pairs = batch * heads * time * 64
    kernel[(triton.cdiv(n_pairs, BLOCK),)](
        values, cos, sin, output, n_pairs, time, heads,
        *values.stride()[:3], *output.stride()[:3],
        0 if cos.shape[0] == 1 else cos.stride(0),
        0 if sin.shape[0] == 1 else sin.stride(0), BLOCK,
        num_warps=4, enable_fp_fusion=False,
    )


class _Rotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        ctx.set_materialize_grads(False)
        q_out, k_out = torch.empty_like(q), torch.empty_like(k)
        _launch(rotary_forward_kernel, q, cos, sin, q_out)
        _launch(rotary_forward_kernel, k, cos, sin, k_out)
        ctx.save_for_backward(cos, sin)
        ctx.shapes = (q.shape, k.shape)
        return q_out, k_out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_q, grad_k):
        cos, sin = ctx.saved_tensors
        results = []
        with torch.cuda.device(cos.device):
            for needs_grad, grad, shape in zip(
                    ctx.needs_input_grad[:2], (grad_q, grad_k), ctx.shapes):
                if not needs_grad or grad is None:
                    results.append(None)
                    continue
                # The native rotate-half backward accumulates into a canonical
                # BHTD tensor, including for token-major inputs and upstreams.
                result = torch.empty(shape, dtype=grad.dtype, device=grad.device)
                if grad.stride(-1) != 1:
                    grad = grad.contiguous()
                _launch(rotary_backward_kernel, grad, cos, sin, result)
                results.append(result)
        return results[0], results[1], None, None


def rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Rotate FP32 task-shaped Q/K; other inputs retain native autograd."""
    if not _supported(q, k, cos, sin, unsqueeze_dim):
        return _native_rotary(q, k, cos, sin, unsqueeze_dim)
    with torch.cuda.device(q.device):
        return _Rotary.apply(q, k, cos, sin)


def patch_qwen2_rotary(modeling_qwen2=None, version=None):
    """Install only the pinned Qwen function; delegate unsupported calls intact."""
    if version is None:
        try:
            version = metadata.version("transformers")
        except metadata.PackageNotFoundError:
            return False
    if version != "5.16.1":
        return False
    if modeling_qwen2 is None:
        from transformers.models.qwen2 import modeling_qwen2
    original = modeling_qwen2.apply_rotary_pos_emb
    if getattr(original, _MARKER, False):
        return False

    @wraps(original)
    def forward(q, k, cos, sin, unsqueeze_dim=1):
        if not _supported(q, k, cos, sin, unsqueeze_dim):
            return original(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
        with torch.cuda.device(q.device):
            return _Rotary.apply(q, k, cos, sin)

    setattr(forward, _MARKER, True)
    modeling_qwen2.apply_rotary_pos_emb = forward
    return True
