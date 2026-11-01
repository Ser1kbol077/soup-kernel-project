"""Exact bnb values, out semantics, fresh storage and guarded decode fallback."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from solution import nf4_decode as decode


requires_t4 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5),
    reason="requires Kaggle T4",
)


def test_cpu_and_unquantized_calls_keep_original_arguments_and_object_identity():
    calls = []
    sentinel = object()

    def original(A, **kwargs):
        calls.append((A, kwargs))
        return sentinel

    wrapped = decode.make_nf4_decoder(original)
    packed = torch.zeros((32, 1), dtype=torch.uint8)
    state, absmax, out = object(), object(), torch.empty(8, 8)
    assert wrapped(packed, state, absmax, out, 128, "fp4") is sentinel
    assert calls == [(packed, {"quant_state": state, "absmax": absmax, "out": out,
                               "blocksize": 128, "quant_type": "fp4"})]


def test_installer_is_version_guarded_and_idempotent():
    original = lambda *args, **kwargs: None
    module = SimpleNamespace(dequantize_4bit=original)
    assert not decode.patch_nf4_decode(module, "0.50.1")
    assert module.dequantize_4bit is original
    assert decode.patch_nf4_decode(module, "0.50.2")
    installed = module.dequantize_4bit
    assert decode.patch_nf4_decode(module, "0.50.2")
    assert module.dequantize_4bit is installed
    assert installed.__wrapped__ is original


def _bnb():
    bnb = pytest.importorskip("bitsandbytes")
    if bnb.__version__ != "0.50.2":
        pytest.skip("requires pinned bitsandbytes 0.50.2")
    from bitsandbytes import functional as F

    reference = F.dequantize_4bit
    while getattr(reference, "_soup_nf4_decode_patch", False):
        reference = reference.__wrapped__
    return F, reference


@requires_t4
@pytest.mark.parametrize("shape", [(8, 8), (65, 64), (128, 128), (257, 64), (1536, 1536), (8960, 1536)])
@pytest.mark.parametrize("transposed", [False, True])
def test_fused_nested_nf4_matches_native_bit_exactly_and_returns_fresh_storage(shape, transposed):
    F, original = _bnb()
    torch.manual_seed(701)
    values = torch.randn(shape, dtype=torch.float16, device="cuda")
    packed, state = F.quantize_4bit(values, blocksize=64, quant_type="nf4", compress_statistics=True)
    if transposed:
        packed = packed.t()
    assert decode._supported(packed, state, None)
    expected = original(packed, state)
    wrapped = decode.make_nf4_decoder(original)
    actual = wrapped(packed, state)
    other = wrapped(packed, state)
    assert actual.data_ptr() != other.data_ptr()
    assert actual.shape == expected.shape and actual.dtype == torch.float16
    assert actual.stride() == expected.stride()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(other, expected, rtol=0, atol=0)


@requires_t4
def test_all_nibbles_nested_scales_tail_and_fixed_native_nf4_map():
    F, original = _bnb()
    values = torch.randn((65, 64), dtype=torch.float16, device="cuda")
    packed, state = F.quantize_4bit(values, quant_type="nf4", compress_statistics=True)
    packed.copy_(torch.arange(packed.numel(), device="cuda").remainder(256).to(torch.uint8).view_as(packed))
    # bnb's CUDA NF4 decoder uses the fixed format, not this state field.
    state.code = torch.zeros_like(state.code)
    expected = original(packed, state)
    actual = decode.make_nf4_decoder(original)(packed, state)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_t4
@pytest.mark.parametrize("transposed", [False, True])
def test_supported_out_buffer_is_returned_and_written_in_place(transposed):
    F, original = _bnb()
    packed, state = F.quantize_4bit(torch.randn(128, 64, dtype=torch.float16, device="cuda"),
                                   quant_type="nf4", compress_statistics=True)
    if transposed:
        packed = packed.t()
    expected_out = torch.empty(state.shape, dtype=state.dtype, device="cuda")
    expected = original(packed, state, out=expected_out)
    out = torch.empty_like(expected_out)
    assert decode._supported(packed, state, out)
    actual = decode.make_nf4_decoder(original)(packed, state, out=out)
    assert actual.data_ptr() == out.data_ptr()
    assert actual.stride() == expected.stride()
    assert (actual is out) == (not transposed)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_t4
@pytest.mark.parametrize("case", ["fp4", "plain", "bf16", "block128"])
def test_unsupported_cuda_formats_keep_native_fallback(case):
    F, original = _bnb()
    values = torch.randn(128, 128, dtype=torch.bfloat16 if case == "bf16" else torch.float16, device="cuda")
    packed, state = F.quantize_4bit(values, blocksize=128 if case == "block128" else 64,
                                   quant_type="fp4" if case == "fp4" else "nf4",
                                   compress_statistics=case != "plain")
    assert not decode._supported(packed, state, None)
    calls = []

    def native(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    actual = decode.make_nf4_decoder(native)(packed, state)
    assert calls == [True]
    torch.testing.assert_close(actual, original(packed, state), rtol=0, atol=0)
