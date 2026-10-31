"""Replace the actual frozen NF4 MLP input-gradient GEMM, and nothing else.

Pinned contract: bitsandbytes 0.50.2 MatMul4Bit.backward computes dX = dY @ W
after its native dequantize_4bit. The forward, weight decode, LoRA parameters,
dropout, bias gradients and original cast boundaries are preserved.
T4 retains the original backward: Triton 3.6.0 lowers ordinary tl.dot to FMA
below SM80 instead of the Tensor Core instructions required by this GEMM.
"""

from functools import wraps

import torch

from .mlp_dx import mlp_input_grad

MLP_WEIGHT_SHAPES = {(8960, 1536), (1536, 8960)}
MIN_ROWS = 128  # Small GEMMs retain native dispatch rather than underfill T4.


def _eligible_context(ctx, grad_output):
    if (torch.is_grad_enabled() or torch.version.hip is not None
            or not isinstance(grad_output, torch.Tensor)
            or not grad_output.is_cuda or grad_output.dtype != torch.float16
            or grad_output.ndim not in (2, 3) or not grad_output.is_contiguous()
            or grad_output.numel() == 0 or getattr(ctx, "is_empty", True)
            or getattr(ctx, "dtype_A", None) not in (torch.float16, torch.bfloat16, torch.float32)
            or getattr(ctx, "dtype_B", None) != torch.uint8):
        return False
    if torch.cuda.get_device_capability(grad_output.device) < (8, 0):
        return False
    needs = getattr(ctx, "needs_input_grad", ())
    if len(needs) != 5 or not needs[0] or needs[1] or needs[3]:
        return False
    state = getattr(ctx, "state", None)
    shape = tuple(getattr(state, "shape", ()) or ())
    if (shape not in MLP_WEIGHT_SHAPES
            or getattr(state, "quant_type", None) != "nf4"
            or getattr(state, "dtype", None) not in (torch.float16, torch.bfloat16, torch.float32)
            or grad_output.size(-1) != shape[0]
            or grad_output.numel() // shape[0] < MIN_ROWS):
        return False
    tensors = getattr(ctx, "tensors", ())
    if len(tensors) != 2:
        return False
    packed = tensors[1]
    return (isinstance(packed, torch.Tensor) and packed.device == grad_output.device
            and packed.dtype == torch.uint8 and packed.ndim == 2
            and packed.size(1) == 1 and packed.is_contiguous()
            and packed.numel() == (shape[0] * shape[1] + 1) // 2)


def patch_mlp_input_grad():
    """Idempotent, version/shape/dtype-guarded first-order backward patch."""
    import bitsandbytes as bnb
    from bitsandbytes import functional as bnb_functional
    from bitsandbytes.autograd import _functions

    if bnb.__version__ != "0.50.2":
        return
    original = _functions.MatMul4Bit.backward
    if getattr(original, "_soup_mlp_dx_patch", False):
        return

    @wraps(original)
    def backward(ctx, grad_output):
        if not _eligible_context(ctx, grad_output):
            return original(ctx, grad_output)
        _, packed = ctx.tensors
        with torch.cuda.device(grad_output.device):
            # Exactly the original decode and cast; the only replaced operation
            # is torch.matmul(grad_output, weight). No activation is recomputed.
            weight = bnb_functional.dequantize_4bit(packed, ctx.state).to(grad_output.dtype)
            grad_input = mlp_input_grad(grad_output, weight)
        return grad_input, None, None, None, None

    backward._soup_mlp_dx_patch = True
    _functions.MatMul4Bit.backward = staticmethod(backward)
