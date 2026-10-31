"""Original row-wise RMSNorm kernels for Qwen2's frozen FP32 norms.

The forward retains one inverse RMS per token. Backward recomputes normalized
values from the input instead of keeping another full activation. No norm
parameters, residuals, projection weights, or training settings are changed.

Semantics checked against Transformers v5.16.1 Qwen2RMSNorm:
https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py
Only the FP32 path is fused. In particular, casting normalized FP16 values before
multiplication by an FP32 weight is observable and stays on the original path.
"""

import math

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def rmsnorm_row_forward_kernel(X, Weight, Y, InvRms,
                               WIDTH: tl.constexpr, EPS: tl.constexpr,
                               BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    valid = column < WIDTH
    value = tl.load(X + row * WIDTH + column, valid, other=0.0)
    weight = tl.load(Weight + column, valid, other=0.0)
    variance = tl.sum(value * value, 0) / WIDTH
    inv_rms = tl.rsqrt(variance + EPS)
    normalized = value * inv_rms
    tl.store(Y + row * WIDTH + column, normalized * weight, valid)
    tl.store(InvRms + row, inv_rms)


@triton.jit
def rmsnorm_row_backward_kernel(GradY, X, Weight, InvRms, GradX,
                                WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    valid = column < WIDTH
    value = tl.load(X + row * WIDTH + column, valid, other=0.0)
    weight = tl.load(Weight + column, valid, other=0.0)
    upstream = tl.load(GradY + row * WIDTH + column, valid, other=0.0)
    inv_rms = tl.load(InvRms + row)
    normalized = value * inv_rms
    scaled_grad = upstream * weight
    correction = tl.sum(scaled_grad * normalized, 0) / WIDTH
    result = inv_rms * (scaled_grad - normalized * correction)
    tl.store(GradX + row * WIDTH + column, result, valid)


def _native_rms_norm(hidden_states, weight, eps):
    """Preserve the normalization cast and subsequent weight promotion."""
    values = hidden_states.float()
    inverse = torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    return weight * (values * inverse).to(hidden_states.dtype)


def _supported(hidden_states, weight, eps):
    return (
        hidden_states.is_cuda
        and hidden_states.dtype == torch.float32
        and hidden_states.ndim >= 1
        and hidden_states.numel() > 0
        and hidden_states.is_contiguous()
        and 0 < hidden_states.size(-1) <= 8192
        and weight.device == hidden_states.device
        and weight.dtype == torch.float32
        and weight.shape == (hidden_states.size(-1),)
        and weight.is_contiguous()
        and not weight.requires_grad
        and isinstance(eps, (int, float))
        and math.isfinite(eps)
        and eps > 0
    )


class _FrozenRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, eps):
        width = hidden_states.size(-1)
        rows = hidden_states.numel() // width
        out = torch.empty_like(hidden_states)
        inverse = torch.empty(rows, device=hidden_states.device, dtype=torch.float32)
        rmsnorm_row_forward_kernel[(rows,)](
            hidden_states, weight, out, inverse, width, eps,
            triton.next_power_of_2(width), num_warps=4, enable_fp_fusion=False,
        )
        ctx.save_for_backward(hidden_states, weight, inverse)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        hidden_states, weight, inverse = ctx.saved_tensors
        width = hidden_states.size(-1)
        rows = hidden_states.numel() // width
        grad_out = grad_out.contiguous()
        grad_input = torch.empty_like(hidden_states)
        with torch.cuda.device(hidden_states.device):
            rmsnorm_row_backward_kernel[(rows,)](
                grad_out, hidden_states, weight, inverse, grad_input,
                width, triton.next_power_of_2(width), num_warps=4,
                enable_fp_fusion=False,
            )
        return grad_input, None, None


def rms_norm(hidden_states, weight, eps=1e-6):
    """Qwen2 RMSNorm with first-order fused autograd for frozen FP32 weights.

    CPU, strided inputs, low precision inputs, trainable weights, and unsupported
    shapes retain ordinary PyTorch autograd, including weight gradients.
    """
    if not _supported(hidden_states, weight, eps):
        return _native_rms_norm(hidden_states, weight, eps)
    with torch.cuda.device(hidden_states.device):
        return _FrozenRMSNorm.apply(hidden_states, weight, eps)


def patch_qwen2_rmsnorm():
    """Idempotently fuse the 1536-wide frozen FP32 Qwen2 norm path."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2RMSNorm.forward
    if getattr(original, "_soup_rmsnorm_patch", False):
        return

    def forward(self, hidden_states):
        if (hidden_states.ndim < 1 or hidden_states.size(-1) != 1536
                or not _supported(hidden_states, self.weight, self.variance_epsilon)):
            return original(self, hidden_states)
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)

    forward._soup_rmsnorm_patch = True
    modeling_qwen2.Qwen2RMSNorm.forward = forward
