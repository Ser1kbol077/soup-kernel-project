"""Remove redundant casts inside the pinned, frozen NF4 MLP.

Native bitsandbytes GEMMs and weight decoding are retained. The supplied
SwiGLU kernels are reused unchanged, and remain excluded from own-kernel share.
This is a first-order VJP: each branch rounds to FP16 independently before
conversion and accumulation in FP32, exactly as the original linear casts do.
"""

from functools import wraps

import torch
import triton
from torch.autograd.function import once_differentiable
from torch.nn.modules import module as module_hooks

from .swiglu import BLOCK, example_swiglu_bwd_kernel, example_swiglu_fwd_kernel

HIDDEN = 1536
INTERMEDIATE = 8960
MIN_ROWS = 128
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_SHAPES = ((INTERMEDIATE, HIDDEN), (INTERMEDIATE, HIDDEN), (HIDDEN, INTERMEDIATE))
_LOCAL_HOOKS = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")
_GLOBAL_HOOKS = tuple("_global" + name for name in _LOCAL_HOOKS)


def _observed(module):
    return ("forward" in module.__dict__ or hasattr(module, "_hf_hook")
            or hasattr(module, "_old_forward")
            or any(getattr(module, name, None) for name in _LOCAL_HOOKS))


def _eligible_mlp(mlp, x, linear4bit_type):
    if (type(x) is not torch.Tensor or not x.is_cuda
            or x.dtype != torch.float32 or x.ndim not in (2, 3)
            or not x.is_contiguous() or not x.requires_grad
            or x.shape[-1] != HIDDEN or x.numel() // HIDDEN < MIN_ROWS
            or not torch.is_grad_enabled() or not mlp.training
            or torch.version.hip is not None
            or torch.cuda.get_device_capability(x.device) != (7, 5)
            or getattr(getattr(mlp, "config", None), "hidden_act", None) != "silu"
            or _observed(mlp)
            or any(getattr(module_hooks, name, None) for name in _GLOBAL_HOOKS)):
        return False
    if (torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") != torch.float16):
        return False
    for name, shape in zip(_PROJECTIONS, _SHAPES):
        layer = getattr(mlp, name, None)
        if (type(layer) is not linear4bit_type or _observed(layer)
                or layer.compute_dtype != torch.float16
                or not layer.compute_type_is_set or layer.bias is not None):
            return False
        weight = layer.weight
        state = getattr(weight, "quant_state", None)
        if (weight.requires_grad or weight.device != x.device
                or weight.dtype != torch.uint8 or weight.ndim != 2
                or weight.shape != ((shape[0] * shape[1] + 1) // 2, 1)
                or not weight.is_contiguous()
                or tuple(getattr(state, "shape", ()) or ()) != shape
                or getattr(state, "dtype", None) != torch.float16
                or getattr(state, "quant_type", None) != "nf4"):
            return False
    return True


def _input_grad(grad, packed, state, dequantize):
    # Same canonical packed layout, decode, dtype cast and matmul as
    # bitsandbytes 0.50.2 MatMul4Bit.backward. No dense weight cache is kept.
    weight = dequantize(packed, state).to(grad.dtype)
    return torch.matmul(grad, weight)


class _CastElidedMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mlp, dequantize):
        half_x = x.to(torch.float16)
        gate = mlp.gate_proj(half_x).contiguous()
        up = mlp.up_proj(half_x).contiguous()
        del half_x
        activated = torch.empty_like(gate)
        n = gate.numel()
        example_swiglu_fwd_kernel[(triton.cdiv(n, BLOCK),)](
            gate, up, activated, n, BLOCK=BLOCK
        )
        out = mlp.down_proj(activated)
        weights = tuple(getattr(mlp, name).weight for name in _PROJECTIONS)
        ctx.states = tuple(weight.quant_state for weight in weights)
        ctx.dequantize = dequantize
        ctx.save_for_backward(gate, up, *(weight.view(-1, 1) for weight in weights))
        return out.to(torch.float32)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        gate, up, gate_weight, up_weight, down_weight = ctx.saved_tensors
        gate_state, up_state, down_state = ctx.states
        grad_half = grad.to(torch.float16)
        grad_activated = _input_grad(grad_half, down_weight, down_state, ctx.dequantize)
        del grad_half
        dgate, dup = torch.empty_like(gate), torch.empty_like(up)
        n = gate.numel()
        example_swiglu_bwd_kernel[(triton.cdiv(n, BLOCK),)](
            grad_activated, gate, up, dgate, dup, n, BLOCK=BLOCK
        )
        del grad_activated
        # Do not add these branches in FP16. In the native graph their separate
        # input-cast backward nodes convert each rounded GEMM result to FP32
        # before the autograd engine accumulates into the original FP32 x.
        dx = _input_grad(dgate, gate_weight, gate_state, ctx.dequantize).float()
        del dgate
        dx_up = _input_grad(dup, up_weight, up_state, ctx.dequantize)
        dx.add_(dx_up)  # FP32 accumulation; TensorIterator promotes the FP16 operand.
        return dx, None, None


def patch_cast_elided_qwen2_mlp():
    """Install after the compact/example wrappers; unsupported calls keep them."""
    import bitsandbytes as bnb
    from bitsandbytes import functional as bnb_functional
    from transformers.models.qwen2 import modeling_qwen2

    if bnb.__version__ != "0.50.2":
        return
    original = modeling_qwen2.Qwen2MLP.forward
    if getattr(original, "_soup_mlp_cast_patch", False):
        return
    if not getattr(original, "_soup_compact_patch", False):
        return

    @wraps(original)
    def forward(self, x):
        if type(self) is not modeling_qwen2.Qwen2MLP or not _eligible_mlp(self, x, bnb.nn.Linear4bit):
            return original(self, x)
        with torch.cuda.device(x.device):
            return _CastElidedMLP.apply(x, self, bnb_functional.dequantize_4bit)

    forward._soup_mlp_cast_patch = True
    modeling_qwen2.Qwen2MLP.forward = forward
