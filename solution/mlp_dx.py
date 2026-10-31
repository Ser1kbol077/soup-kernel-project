"""Original FP16 matrix product for frozen MLP input gradients.

For a frozen linear projection, backward needs ``grad_output @ weight`` but
does not need a weight gradient. This implementation computes that same dense
product with FP32 accumulators and one FP16 output rounding. Its production
callers provide the FP16 operands already used by bitsandbytes backward; it
does not change the quantized weights or the activation/gradient cast points.

The blocked product and tile grouping below are independently implemented.
The only matrix primitive is Triton's documented ``tl.dot`` intrinsic:
https://triton-lang.org/main/python-api/generated/triton.language.dot.html
No split-K reduction, temporary tensor, autotuning, or repeated work is used.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def mlp_input_grad_kernel(Grad, Weight, Result,
                          M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                          BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    # Traverse several row tiles for each column tile before advancing the
    # columns. Neighboring programs can reuse the same weight tile in L2.
    row_tiles = tl.cdiv(M, BLOCK_M)
    column_tiles = tl.cdiv(N, BLOCK_N)
    tile = tl.program_id(0)
    group_slots = GROUP_M * column_tiles
    first_row_tile = (tile // group_slots) * GROUP_M
    rows_in_group = tl.minimum(GROUP_M, row_tiles - first_row_tile)
    position = tile % group_slots
    row_tile = first_row_tile + position % rows_in_group
    column_tile = position // rows_in_group

    rows = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = column_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, BLOCK_K)
    accumulator = tl.full((BLOCK_M, BLOCK_N), 0.0, tl.float32)

    for block in range(tl.cdiv(K, BLOCK_K)):
        inner = block * BLOCK_K + reduction
        grad = tl.load(
            Grad + rows[:, None] * K + inner[None, :],
            (rows[:, None] < M) & (inner[None, :] < K), other=0.0,
        )
        weight = tl.load(
            Weight + inner[:, None] * N + columns[None, :],
            (inner[:, None] < K) & (columns[None, :] < N), other=0.0,
        )
        accumulator = tl.dot(grad, weight, accumulator, out_dtype=tl.float32)

    tl.store(
        Result + rows[:, None] * N + columns[None, :],
        accumulator.to(tl.float16),
        (rows[:, None] < M) & (columns[None, :] < N),
    )


def _supported(grad, weight):
    if (
        not grad.is_cuda or not weight.is_cuda
        or grad.device != weight.device
        or grad.dtype != torch.float16 or weight.dtype != torch.float16
        or grad.ndim not in (2, 3) or weight.ndim != 2
        or not grad.is_contiguous() or not weight.is_contiguous()
        or grad.numel() == 0 or weight.numel() == 0
        or grad.size(-1) != weight.size(0)
        or (torch.is_grad_enabled() and (grad.requires_grad or weight.requires_grad))
    ):
        return False
    return torch.cuda.get_device_capability(grad.device) >= (7, 5)


def mlp_input_grad(grad, weight):
    """Compute ``grad @ weight`` without changing first/higher-order autograd.

    The Triton path accepts contiguous CUDA FP16 operands on SM 7.5 or newer,
    with a two- or three-dimensional gradient and a two-dimensional weight.
    Calls that require an autograd graph, unsupported layouts/devices/dtypes,
    and empty shapes keep native ``torch.matmul`` semantics.
    """
    if not _supported(grad, weight):
        return torch.matmul(grad, weight)

    k, n = weight.shape
    m = grad.numel() // k
    result = torch.empty((*grad.shape[:-1], n), device=grad.device, dtype=grad.dtype)
    # Two pipeline stages need 24 KiB of operand tiles. This conservative
    # fixed configuration fits T4 without an autotuning warm-up or workspace.
    block_m, block_n, block_k = 64, 128, 32
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    with torch.cuda.device(grad.device):
        mlp_input_grad_kernel[grid](
            grad, weight, result, m, n, k,
            block_m, block_n, block_k, 8,
            num_warps=4, num_stages=2,
        )
    return result
