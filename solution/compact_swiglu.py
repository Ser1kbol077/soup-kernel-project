"""Lossless compact activation storage for the pinned frozen NF4 Qwen MLP.

This reuses the shipped example's SwiGLU kernels. Neither the reused kernels
nor this storage change is claimed as an original kernel for the 5% rule.

bitsandbytes 0.50.2 Linear4bit computes in ``compute_dtype`` and then casts its
output back to the input dtype. With FP32 inputs and FP16 compute, its FP32
outputs contain exactly representable FP16 values. Saving those values as FP16
loses no information; the SwiGLU output and its input gradients remain FP32.
"""

from functools import wraps

import torch
import torch.nn.functional as F
import triton
from torch.autograd.function import once_differentiable
from torch.nn.modules import module as module_hooks

from .swiglu import (
    BLOCK,
    example_swiglu_bwd_kernel,
    example_swiglu_fwd_kernel,
    swiglu,
)


class _HalfOriginSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        gate = gate.contiguous()
        up = up.contiguous()
        out = torch.empty_like(gate)
        n = gate.numel()
        example_swiglu_fwd_kernel[(triton.cdiv(n, BLOCK),)](
            gate, up, out, n, BLOCK=BLOCK
        )
        ctx.input_dtype = gate.dtype
        ctx.save_for_backward(gate.to(torch.float16), up.to(torch.float16))
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        gate, up = ctx.saved_tensors
        # The cache dtype must not become the gradient dtype. In the original
        # graph the FP32 gradients are rounded by each Linear4bit output cast,
        # independently, before that branch's matrix-product backward.
        grad = grad.to(ctx.input_dtype).contiguous()
        dgate = torch.empty(gate.shape, dtype=ctx.input_dtype, device=gate.device)
        dup = torch.empty(up.shape, dtype=ctx.input_dtype, device=up.device)
        n = gate.numel()
        example_swiglu_bwd_kernel[(triton.cdiv(n, BLOCK),)](
            grad, gate, up, dgate, dup, n, BLOCK=BLOCK
        )
        return dgate, dup


def half_origin_swiglu(gate, up):
    """FP32 SwiGLU with a lossless FP16 saved cache for half-origin inputs.

    Precondition for the compact CUDA path: both FP32 tensors were obtained by
    casting FP16 values to FP32. This must be established by the caller's layer
    provenance, never assumed for arbitrary FP32 activations. Unsupported
    devices and dtypes use the regular operation without compacting the cache.
    """
    if not gate.is_cuda or not up.is_cuda:
        return F.silu(gate) * up
    if (gate.dtype != torch.float32 or up.dtype != torch.float32
            or gate.device != up.device or gate.shape != up.shape
            or gate.numel() == 0):
        return swiglu(gate, up)
    with torch.cuda.device(gate.device):
        return _HalfOriginSwiGLU.apply(gate, up)


def _eligible_mlp(mlp, x, linear4bit_type):
    if (not x.is_cuda or x.dtype != torch.float32
            or getattr(getattr(mlp, "config", None), "hidden_act", None) != "silu"
            or module_hooks._global_forward_hooks
            or module_hooks._global_forward_pre_hooks):
        return False
    for name in ("gate_proj", "up_proj"):
        layer = getattr(mlp, name, None)
        # Exact type excludes LoRA wrappers and subclasses with another forward
        # implementation. This task only trains the attention q/v adapters.
        if (type(layer) is not linear4bit_type
                or layer.compute_dtype != torch.float16
                or not layer.compute_type_is_set
                or layer.weight.requires_grad
                or layer.bias is not None
                or getattr(layer.weight, "quant_state", None) is None
                or "forward" in layer.__dict__
                or layer._forward_hooks or layer._forward_pre_hooks):
            return False
    return True


def patch_compact_qwen2_mlp():
    """Wrap the installed example MLP and compact only proven FP16 results.

    Call after ``patch_qwen2_mlp``. Model parameters, compute dtype, each linear
    call, and the original autograd graph around the pointwise operation stay
    unchanged. In particular the two input casts are not merged: that would
    change where branch gradients accumulate and round.
    """
    from bitsandbytes.nn import Linear4bit
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2MLP.forward
    if getattr(original, "_soup_compact_patch", False):
        return

    @wraps(original)
    def forward(self, x):
        if not _eligible_mlp(self, x, Linear4bit):
            return original(self, x)
        gate, up = self.gate_proj(x), self.up_proj(x)
        activated = half_origin_swiglu(gate, up)
        # The custom autograd node retains the half cache. Release the original
        # FP32 buffers before the down projection allocates its workspace.
        del gate, up
        return self.down_proj(activated)

    forward._soup_compact_patch = True
    modeling_qwen2.Qwen2MLP.forward = forward
