"""Decode the task's double-quantized NF4 weights in one original Triton launch.

bitsandbytes 0.50.2 materializes decoded FP32 scales, adds their offset, then
decodes the packed NF4 values. This combines those operations, retaining each
FP32 rounding and the final FP16 cast. Outputs are fresh: a decoded matrix may
be saved by autograd, so it must never alias the next projection's scratch.
Unsupported inputs keep the installed bitsandbytes implementation unchanged.
"""

from functools import wraps
from math import prod

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


VALUES_PER_BLOCK = 64
BLOCKS_PER_PROGRAM = 16
_TABLES = {}

# NF4's format constants, in packed-index order. bnb's CUDA decoder uses this
# fixed map, not QuantState.code (which can be changed independently by callers).
NF4_VALUES = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
)


if triton is not None:
    @triton.jit(do_not_specialize=["n_blocks"])
    def nf4_double_decode_kernel(Packed, QuantizedScale, ScaleCode, ScaleAbsmax,
                                 Offset, NF4Code, Output, n_blocks,
                                 BLOCKS: tl.constexpr):
        blocks = tl.program_id(0) * BLOCKS + tl.arange(0, BLOCKS)
        live = blocks < n_blocks
        scale_index = tl.load(QuantizedScale + blocks, live, other=0).to(tl.int32)
        scale_code = tl.load(ScaleCode + scale_index)
        scale_absmax = tl.load(ScaleAbsmax + blocks // 256, live, other=0.0)
        # enable_fp_fusion=False keeps the two FP32 roundings of the native
        # scale decode and its separate in-place offset addition.
        scale = scale_code * scale_absmax
        scale = scale + tl.load(Offset)
        lanes = tl.arange(0, 64)
        positions = blocks[:, None] * 64 + lanes[None, :]
        packed = tl.load(Packed + positions // 2, live[:, None], other=0).to(tl.int32)
        code = tl.where((lanes[None, :] & 1) == 0, packed >> 4, packed & 15)
        value = tl.load(NF4Code + code) * scale[:, None]
        tl.store(Output + positions, value, live[:, None])


def _tensor(tensor, device, dtype, size):
    return (isinstance(tensor, torch.Tensor) and tensor.device == device
            and tensor.dtype == dtype and tensor.layout == torch.strided
            and tensor.is_contiguous() and tensor.numel() == size)


def _supported(packed, state, out):
    if (triton is None or not isinstance(packed, torch.Tensor)
            or not packed.is_cuda or torch.version.hip is not None
            or packed.dtype != torch.uint8 or packed.ndim != 2
            or (packed.shape[1] != 1 and packed.shape[0] != 1)
            or packed.numel() < 2
            or not packed.is_contiguous()
            or torch.cuda.get_device_capability(packed.device) != (7, 5)
            or state is None or getattr(state, "quant_type", None) != "nf4"
            or getattr(state, "blocksize", None) != 64
            or getattr(state, "dtype", None) != torch.float16
            or not getattr(state, "nested", False)):
        return False
    shape = getattr(state, "shape", None)
    if (not isinstance(shape, (tuple, list, torch.Size)) or len(shape) != 2
            or any(not isinstance(size, int) or size <= 0 for size in shape)):
        return False
    n = prod(shape)
    if n % 64 or packed.numel() != n // 2:
        return False
    blocks = n // 64
    second = getattr(state, "state2", None)
    if (second is None or getattr(second, "blocksize", None) != 256
            or getattr(second, "dtype", None) != torch.float32
            or getattr(second, "nested", False)
            or not _tensor(getattr(state, "absmax", None), packed.device, torch.uint8, blocks)
            or not _tensor(getattr(second, "code", None), packed.device, torch.float32, 256)
            or not _tensor(getattr(second, "absmax", None), packed.device, torch.float32,
                           (blocks + 255) // 256)
            or not _tensor(getattr(state, "offset", None), packed.device, torch.float32, 1)):
        return False
    if out is not None and (
            not _tensor(out, packed.device, torch.float16, n)
            or tuple(out.shape) != tuple(shape) or out.requires_grad):
        return False
    return True


def _nf4_table(device):
    table = _TABLES.get(device)
    if table is None:
        table = torch.tensor(NF4_VALUES, device=device, dtype=torch.float32)
        _TABLES[device] = table
    return table


def make_nf4_decoder(original):
    """Wrap the public bnb API; every unsupported call is delegated verbatim."""
    @wraps(original)
    def dequantize_4bit(A, quant_state=None, absmax=None, out=None,
                        blocksize=None, quant_type="fp4"):
        if not _supported(A, quant_state, out):
            return original(A, quant_state=quant_state, absmax=absmax, out=out,
                            blocksize=blocksize, quant_type=quant_type)
        state = quant_state
        with torch.cuda.device(A.device):
            result = (torch.empty(state.shape, dtype=torch.float16, device=A.device)
                      if out is None else out)
            blocks = prod(state.shape) // VALUES_PER_BLOCK
            nf4_double_decode_kernel[(triton.cdiv(blocks, BLOCKS_PER_PROGRAM),)](
                A, state.absmax, state.state2.code, state.state2.absmax,
                state.offset, _nf4_table(A.device), result, blocks, BLOCKS_PER_PROGRAM,
                num_warps=4, enable_fp_fusion=False,
            )
        # Preserve bnb's public API compatibility for callers that pass the
        # same packed bytes as [1, N] rather than [N, 1]. The output buffer
        # itself is always in quant_state.shape; the returned value is a view.
        return result.t() if A.shape[0] == 1 else result

    dequantize_4bit._soup_nf4_decode_patch = True
    return dequantize_4bit


def patch_nf4_decode(functional=None, version=None):
    """Install before MLP wrappers capture bnb's decoder. Returns whether active."""
    if functional is None:
        import bitsandbytes as bnb
        from bitsandbytes import functional

        version = bnb.__version__
    if version != "0.50.2":
        return False
    original = functional.dequantize_4bit
    if getattr(original, "_soup_nf4_decode_patch", False):
        return True
    functional.dequantize_4bit = make_nf4_decoder(original)
    return True
