"""Vocabulary reductions: native PyTorch references and autograd regressions.

Run on the target T4 after installing the challenge requirements:
    python -m pytest solution/test_vocab_ops.py -q

Install the challenge's PyTorch/Triton dependencies for CPU fallback tests too.
CUDA tests are skipped explicitly when a GPU is unavailable; a skipped test is
not evidence of GPU correctness.
"""

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU (target: NVIDIA T4)"
)


def _assert_close(actual, expected, dtype, *, gradient=False):
    # FP16 exercises native fallback. FP32 exercises the custom reductions;
    # tolerances include the changed reduction tree and approximate exponentials.
    if dtype == torch.float16:
        rtol, atol = (4e-3, 2e-5) if gradient else (2e-3, 3e-3)
    else:
        rtol, atol = (5e-5, 2e-7) if gradient else (2e-5, 2e-5)
    torch.testing.assert_close(
        actual.float(), expected.float(), rtol=rtol, atol=atol, equal_nan=True
    )


def _reference_entropy(logits):
    # log_softmax is stable even when the absolute logit offset is large.
    log_probs = logits.log_softmax(dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(3, 151936), (17, 151936), (17, 4103)])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_cross_entropy_forward_and_backward(dtype, shape, reduction):
    from solution.vocab_ops import cross_entropy

    torch.manual_seed(3407)
    n_rows, vocab_size = shape
    logits = (torch.randn(shape, device="cuda", dtype=dtype) * 2).requires_grad_()
    reference_logits = logits.detach().clone().requires_grad_()
    original_logits = logits.detach().clone()
    targets = torch.randint(vocab_size, (n_rows,), device="cuda")
    targets[::4] = -100
    original_targets = targets.clone()

    actual = cross_entropy(logits, targets, reduction=reduction, ignore_index=-100)
    expected = F.cross_entropy(
        reference_logits, targets, reduction=reduction, ignore_index=-100
    )
    _assert_close(actual, expected, dtype)
    # Random upstream gradients exercise each unreduced row independently.
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream.to(expected.dtype))
    _assert_close(logits.grad, reference_logits.grad, dtype, gradient=True)
    assert torch.count_nonzero(logits.grad[targets == -100]) == 0
    torch.testing.assert_close(logits.detach(), original_logits, rtol=0, atol=0)
    torch.testing.assert_close(targets, original_targets, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_cross_entropy_all_ignored(dtype, reduction):
    from solution.vocab_ops import cross_entropy

    torch.manual_seed(19)
    logits = torch.randn((3, 4103), device="cuda", dtype=dtype, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_()
    targets = torch.full((3,), -100, device="cuda", dtype=torch.long)
    actual = cross_entropy(logits, targets, reduction=reduction)
    expected = F.cross_entropy(reference_logits, targets, reduction=reduction)
    _assert_close(actual, expected, dtype)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream.to(expected.dtype))
    torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=0, atol=0)
    assert torch.count_nonzero(logits.grad) == 0


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_cross_entropy_row_stride_and_repeated_backward(dtype):
    from solution.vocab_ops import cross_entropy

    torch.manual_seed(23)
    # The vocabulary axis is contiguous but rows are separated by unused rows.
    storage = torch.randn((34, 4103), device="cuda", dtype=dtype)
    logits = storage[::2].detach().requires_grad_()
    assert logits.stride(0) == 2 * logits.shape[1]
    reference_logits = logits.detach().clone().requires_grad_()
    original_storage = storage.clone()
    targets = torch.randint(4103, (17,), device="cuda")
    targets[::5] = -1
    actual = cross_entropy(logits, targets, reduction="none", ignore_index=-1)
    expected = F.cross_entropy(
        reference_logits, targets, reduction="none", ignore_index=-1
    )
    _assert_close(actual, expected, dtype)
    for retain_graph in (True, False):
        upstream = torch.randn_like(actual)
        actual.backward(upstream, retain_graph=retain_graph)
        expected.backward(upstream.to(expected.dtype), retain_graph=retain_graph)
        _assert_close(logits.grad, reference_logits.grad, dtype, gradient=True)
        torch.testing.assert_close(storage, original_storage, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(3, 151936), (17, 151936), (2, 3, 4103), (3, 1)])
def test_entropy_no_grad_matches_stable_pytorch(dtype, shape):
    from solution.vocab_ops import entropy_from_logits

    torch.manual_seed(29)
    logits = torch.randn(shape, device="cuda", dtype=dtype) * 2
    original_logits = logits.clone()
    with torch.no_grad():
        actual = entropy_from_logits(logits, chunk_size=128)
        expected = _reference_entropy(logits.float())
    assert actual.shape == logits.shape[:-1]
    _assert_close(actual, expected, dtype)
    torch.testing.assert_close(logits, original_logits, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("offset", [-1000.0, 1000.0])
def test_entropy_large_logit_offsets(dtype, offset):
    from solution.vocab_ops import entropy_from_logits

    torch.manual_seed(31)
    logits = torch.randn((3, 4103), device="cuda", dtype=dtype) + offset
    with torch.no_grad():
        actual = entropy_from_logits(logits)
        expected = _reference_entropy(logits.float())
    # Offset cancellation must not lose the entropy in a large absolute value.
    torch.testing.assert_close(
        actual.float(), expected, rtol=3e-5,
        atol=5e-3 if dtype == torch.float16 else 3e-4,
    )


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_entropy_preserves_autograd(dtype):
    from solution.vocab_ops import entropy_from_logits

    torch.manual_seed(37)
    logits = torch.randn((2, 3, 4103), device="cuda", dtype=dtype, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_()
    actual = entropy_from_logits(logits)
    expected = _reference_entropy(reference_logits)
    _assert_close(actual, expected, dtype)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    _assert_close(logits.grad, reference_logits.grad, dtype, gradient=True)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_cross_entropy_cpu_fallback(reduction):
    from solution.vocab_ops import cross_entropy

    torch.manual_seed(41)
    # FP64 and a noncontiguous vocabulary axis must retain native semantics.
    logits = torch.randn((5, 38), dtype=torch.float64)[:, ::2].requires_grad_()
    reference_logits = logits.detach().clone().requires_grad_()
    targets = torch.tensor([0, 18, -100, 7, 12])
    actual = cross_entropy(logits, targets, reduction=reduction)
    expected = F.cross_entropy(reference_logits, targets, reduction=reduction)
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(logits.grad, reference_logits.grad)


def test_entropy_cpu_fallback_preserves_autograd():
    from solution.vocab_ops import entropy_from_logits

    torch.manual_seed(43)
    logits = torch.randn((2, 3, 19), requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_()
    actual = entropy_from_logits(logits)
    expected = _reference_entropy(reference_logits)
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(logits.grad, reference_logits.grad)


@pytest.fixture(scope="module")
def training_hooks():
    """Exercise the real pinned dependency modules when they are installed."""
    pytest.importorskip("triton")
    loss_utils = pytest.importorskip("transformers.loss.loss_utils")
    trl_utils = pytest.importorskip("trl.trainer.utils")
    sft_trainer = pytest.importorskip("trl.trainer.sft_trainer")
    rewind_hf = pytest.importorskip("soup_cli.trainer.rewind_hf")
    import solution

    mapping_before = dict(loss_utils.LOSS_MAPPING)
    solution.apply()
    installed = (
        loss_utils.fixed_cross_entropy,
        trl_utils.entropy_from_logits,
        sft_trainer.entropy_from_logits,
        rewind_hf.row_losses,
    )
    solution.apply()
    repeated = (
        loss_utils.fixed_cross_entropy,
        trl_utils.entropy_from_logits,
        sft_trainer.entropy_from_logits,
        rewind_hf.row_losses,
    )
    assert all(first is second for first, second in zip(installed, repeated))
    assert loss_utils.LOSS_MAPPING == mapping_before
    assert all(getattr(fn, "_soup_vocab_patch", False) for fn in installed)
    return loss_utils, trl_utils, sft_trainer, rewind_hf


def test_apply_idempotent_and_hooks_preserve_cpu_values(training_hooks):
    loss_utils, trl_utils, sft_trainer, rewind_hf = training_hooks
    torch.manual_seed(47)
    logits = torch.randn((5, 19), requires_grad=True)
    targets = torch.tensor([0, 18, -100, 7, 12])
    actual = loss_utils.fixed_cross_entropy(logits, targets, num_items_in_batch=7)
    expected = loss_utils.fixed_cross_entropy.__wrapped__(
        logits, targets, num_items_in_batch=7
    )
    torch.testing.assert_close(actual, expected)
    for module in (trl_utils, sft_trainer):
        entropy = module.entropy_from_logits
        torch.testing.assert_close(
            entropy(logits, chunk_size=2), entropy.__wrapped__(logits, chunk_size=2)
        )
    sequence_logits = torch.randn((2, 5, 19))
    labels = torch.tensor([[-100] * 5, [1, -100, 0, 18, 2]])
    assert rewind_hf.row_losses(sequence_logits, labels) == (
        rewind_hf.row_losses.__wrapped__(sequence_logits, labels)
    )


@requires_cuda
@pytest.mark.parametrize("denominator", ["default", "integer", "cpu_tensor"])
def test_fixed_cross_entropy_hook_denominator_and_backward(training_hooks, denominator):
    loss_utils, _, _, _ = training_hooks
    torch.manual_seed(53)
    logits = torch.randn((17, 151936), device="cuda", requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_()
    targets = torch.randint(151936, (17,), device="cuda")
    targets[::4] = -100
    num_items = None if denominator == "default" else 23
    if denominator == "cpu_tensor":
        num_items = torch.tensor(23)
    actual = loss_utils.fixed_cross_entropy(
        logits, targets, num_items_in_batch=num_items
    )
    expected = loss_utils.fixed_cross_entropy.__wrapped__(
        reference_logits, targets, num_items_in_batch=num_items
    )
    _assert_close(actual, expected, torch.float32)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    _assert_close(logits.grad, reference_logits.grad, torch.float32, gradient=True)


@requires_cuda
def test_both_trl_entropy_aliases_and_rewind_hook(training_hooks):
    _, trl_utils, sft_trainer, rewind_hf = training_hooks
    torch.manual_seed(59)
    logits = torch.randn((2, 5, 151936), device="cuda")
    labels = torch.tensor(
        [[-100] * 5, [1, -100, 0, 151935, 2]], device="cuda"
    )
    original_logits = logits.clone()
    with torch.no_grad():
        for module in (trl_utils, sft_trainer):
            entropy = module.entropy_from_logits
            _assert_close(
                entropy(logits, chunk_size=2),
                entropy.__wrapped__(logits, chunk_size=2), torch.float32,
            )
        actual_losses, actual_tokens = rewind_hf.row_losses(logits, labels)
        expected_losses, expected_tokens = rewind_hf.row_losses.__wrapped__(logits, labels)
    assert actual_tokens == expected_tokens == [0, 3]
    assert actual_losses[0] == 0.0
    torch.testing.assert_close(
        torch.tensor(actual_losses), torch.tensor(expected_losses), rtol=2e-5, atol=2e-5
    )
    torch.testing.assert_close(logits, original_logits, rtol=0, atol=0)
