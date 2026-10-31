"""Exact token predictions, strided views, ties/NaNs and unchanged gradients.

CUDA tests require execution on the target T4; CPU tests validate fallback.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/T4")


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("vocab", [1, 65, 4096, 4103, 151936])
@pytest.mark.parametrize("layout", ["contiguous", "causal_shift", "vocab_stride"])
def test_argmax_strides_match_torch_exactly(dtype, vocab, layout):
    from solution.argmax_ops import argmax_last_dim

    torch.manual_seed(193)
    if layout == "vocab_stride":
        base = torch.randn((2, 5, vocab * 2), device="cuda", dtype=dtype)
        logits = base[..., ::2]
    else:
        base = torch.randn((2, 5, vocab), device="cuda", dtype=dtype)
        logits = base[:, :-1, :] if layout == "causal_shift" else base
    before = base.clone()
    actual = argmax_last_dim(logits)
    expected = logits.argmax(dim=-1)
    assert actual.shape == expected.shape and actual.dtype == torch.int64
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(base, before, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("case", ["tie", "negative_inf", "positive_inf", "nan", "all_nan"])
def test_argmax_special_values_across_tiles(dtype, case):
    from solution.argmax_ops import argmax_last_dim

    vocab = 8203
    logits = torch.full((2, 3, vocab), -1.0, device="cuda", dtype=dtype)
    if case == "tie":
        logits[..., 4095] = 9.0
        logits[..., 4096] = 9.0
    elif case == "negative_inf":
        logits.fill_(-float("inf"))
    elif case == "positive_inf":
        logits[..., 17] = float("inf")
        logits[..., 8199] = float("inf")
    elif case == "nan":
        logits[..., 13] = float("nan")
        logits[..., 5001] = float("nan")
        logits[..., 0] = float("inf")
    else:
        logits.fill_(float("nan"))
    torch.testing.assert_close(argmax_last_dim(logits), logits.argmax(dim=-1), rtol=0, atol=0)


@requires_cuda
def test_argmax_shift_does_not_allocate_dense_copy():
    from solution.argmax_ops import argmax_last_dim

    # Inputs are allocated before resetting peak accounting. A clone here
    # would cost ~35 MiB; the tile statistics/predictions need well under 1 MiB.
    logits = torch.randn((3, 21, 151936), device="cuda", dtype=torch.float32)[:, :-1, :]
    torch.cuda.synchronize()
    argmax_last_dim(logits)  # compile before measuring allocations
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    predictions = argmax_last_dim(logits)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - baseline
    assert extra < logits.numel() * logits.element_size() // 4
    torch.testing.assert_close(predictions, logits.argmax(dim=-1), rtol=0, atol=0)


@requires_cuda
def test_accuracy_hook_uses_strided_storage_and_retains_loss_gradients(monkeypatch):
    from solution import accuracy_patch, argmax_ops

    logits = torch.randn((2, 5, 151936), device="cuda", dtype=torch.float32, requires_grad=True)
    before = logits.detach().clone()
    shifted = accuracy_patch._shifted_view(logits)
    assert shifted.untyped_storage().data_ptr() == logits.untyped_storage().data_ptr()
    assert shifted.stride() == logits.stride()
    assert not shifted.is_contiguous()
    calls = []
    original = argmax_ops.argmax_last_dim

    def observed(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(argmax_ops, "argmax_last_dim", observed)
    with torch.no_grad():
        actual = accuracy_patch._accuracy_argmax(shifted)
    assert len(calls) == 1 and calls[0] is shifted
    torch.testing.assert_close(actual, shifted.argmax(dim=-1), rtol=0, atol=0)
    logits.square().sum().backward()
    torch.testing.assert_close(logits.grad, 2 * before, rtol=0, atol=0)
    torch.testing.assert_close(logits, before, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(2, 5, 19), (0, 5, 19)])
def test_argmax_cpu_fallback_and_loss_graph(dtype, shape):
    from solution.argmax_ops import argmax_last_dim

    logits = torch.randn(shape, dtype=dtype, requires_grad=True)
    before = logits.detach().clone()
    actual = argmax_last_dim(logits[:, :-1, :])
    expected = logits[:, :-1, :].argmax(dim=-1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not actual.requires_grad
    logits.square().sum().backward()
    torch.testing.assert_close(logits.grad, 2 * before, rtol=0, atol=0)


def test_argmax_cpu_empty_vocabulary_retains_native_error():
    from solution.argmax_ops import argmax_last_dim

    with pytest.raises(IndexError):
        argmax_last_dim(torch.empty((2, 3, 0)))
