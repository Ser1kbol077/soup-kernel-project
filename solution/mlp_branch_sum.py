"""Combine independently rounded frozen-MLP input gradients in FP32.

The gate and up input GEMMs retain their original FP16 outputs. Promoting
both operands before adding preserves the native branch accumulation, while
avoiding a separately materialized FP32 gate buffer and a second GPU launch.
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


# SM75 compile inspection: this tile avoids the shared-memory layout
# conversion/barrier generated for a 1024-element mixed FP16/FP32 tile.
BLOCK = 256


if triton is not None:
    @triton.jit(do_not_specialize=["n_elements"])
    def mlp_branch_sum_kernel(Gate, Up, Result, n_elements, BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = index < n_elements
        gate = tl.load(Gate + index, live, other=0.).to(tl.float32)
        up = tl.load(Up + index, live, other=0.).to(tl.float32)
        tl.store(Result + index, gate + up, live)


def _supported(gate_dx, up_dx):
    if (triton is None or type(gate_dx) is not torch.Tensor or type(up_dx) is not torch.Tensor
            or not gate_dx.is_cuda or not up_dx.is_cuda
            or gate_dx.device != up_dx.device
            or gate_dx.dtype != torch.float16 or up_dx.dtype != torch.float16
            or gate_dx.layout != torch.strided or up_dx.layout != torch.strided
            or gate_dx.shape != up_dx.shape or gate_dx.numel() == 0
            or not gate_dx.is_contiguous() or not up_dx.is_contiguous()
            or gate_dx.requires_grad or up_dx.requires_grad
            or gate_dx.is_conj() or up_dx.is_conj() or gate_dx.is_neg() or up_dx.is_neg()
            or torch.version.hip is not None):
        return False
    return torch.cuda.get_device_capability(gate_dx.device) == (7, 5)


def sum_input_branches(gate_dx, up_dx):
    """Return FP32(gate_dx) + FP32(up_dx), preserving native fallback autograd."""
    if not _supported(gate_dx, up_dx):
        return gate_dx.float() + up_dx.float()
    with torch.cuda.device(gate_dx.device):
        result = torch.empty(gate_dx.shape, dtype=torch.float32, device=gate_dx.device)
        mlp_branch_sum_kernel[(triton.cdiv(gate_dx.numel(), BLOCK),)](
            gate_dx, up_dx, result, gate_dx.numel(), BLOCK,
            num_warps=4, enable_fp_fusion=False,
        )
    return result
