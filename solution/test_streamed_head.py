"""Loss/gradient/metric contracts and bounded saved storage for the frozen head."""

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional


def _reference(hidden, weight, labels, denominator=None, shift_labels=None):
    logits = F.linear(hidden, weight)
    source = logits if logits.dtype == torch.float64 else logits.float()
    targets = (F.pad(labels, (0, 1), value=-100)[..., 1:] if shift_labels is None else shift_labels)
    ce = F.cross_entropy(source.reshape(-1, weight.shape[0]), targets.reshape(-1),
                         ignore_index=-100, reduction="none").view_as(targets)
    divisor = (targets != -100).sum() if denominator is None else denominator
    logprob = source.log_softmax(-1)
    return ce.sum() / divisor, ce, -(logprob.exp() * logprob).sum(-1), source.argmax(-1)


@pytest.mark.parametrize("chunk_rows", [1, 5, 64])
@pytest.mark.parametrize("denominator", [None, 23, "tensor"])
@pytest.mark.parametrize("explicit_shift", [False, True])
def test_chunking_preserves_causal_loss_metrics_and_scaled_input_gradient(chunk_rows, denominator, explicit_shift):
    from solution.streamed_head import linear_head_loss

    torch.manual_seed(603)
    hidden = torch.randn(2, 7, 9, dtype=torch.float64, requires_grad=True)
    reference = hidden.detach().clone().requires_grad_()
    weight = torch.randn(37, 9, dtype=torch.float64)
    labels = torch.randint(37, (2, 7))
    labels[:, :3] = -100
    shifted = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous() if explicit_shift else None
    divisor = torch.tensor(23.0) if denominator == "tensor" else denominator
    actual, stats = linear_head_loss(hidden, weight, labels, divisor,
                                     shift_labels=shifted, chunk_rows=chunk_rows)
    expected, ce, entropy, prediction = _reference(reference, weight, labels, divisor, shifted)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(stats["token_loss"], ce, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(stats["entropy"], entropy, rtol=1e-12, atol=1e-12)
    assert torch.equal(stats["predictions"], prediction)
    assert all(not value.requires_grad for value in stats.values())
    scale = torch.tensor(123.25, dtype=torch.float64)
    actual.backward(scale)
    expected.backward(scale)
    torch.testing.assert_close(hidden.grad, reference.grad, rtol=1e-11, atol=1e-11)
    assert weight.grad is None


def test_all_ignored_labels_preserve_nan_mean_and_exact_zero_gradient():
    from solution.streamed_head import linear_head_loss

    hidden = torch.randn(2, 3, 5, requires_grad=True)
    weight = torch.randn(11, 5)
    labels = torch.full((2, 3), -100, dtype=torch.long)
    loss, stats = linear_head_loss(hidden, weight, labels, chunk_rows=2)
    assert loss.isnan()
    assert torch.count_nonzero(stats["token_loss"]) == 0
    loss.backward()
    assert torch.count_nonzero(hidden.grad) == 0


def test_saved_tensors_do_not_include_dense_logits_and_repeated_backward_works():
    from solution.streamed_head import linear_head_loss

    hidden = torch.randn(2, 7, 9, requires_grad=True)
    weight = torch.randn(37, 9)
    labels = torch.randint(37, (2, 7))
    saved = []

    def pack(tensor):
        saved.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss, _ = linear_head_loss(hidden, weight, labels, chunk_rows=3)
    assert (14, 37) not in saved and (2, 7, 37) not in saved
    assert (3, 37) not in saved  # even a chunk must be recomputed
    loss.backward(retain_graph=True)
    first = hidden.grad.clone()
    hidden.grad = None
    loss.backward()
    torch.testing.assert_close(hidden.grad, first, rtol=0, atol=0)


def test_noncontiguous_hidden_and_first_index_ties():
    from solution.streamed_head import linear_head_loss

    hidden = torch.randn(7, 2, 5).transpose(0, 1).requires_grad_()
    weight = torch.zeros(13, 5)
    labels = torch.randint(13, (2, 7))
    loss, stats = linear_head_loss(hidden, weight, labels, chunk_rows=4)
    assert not hidden.is_contiguous()
    assert torch.count_nonzero(stats["predictions"]) == 0
    torch.testing.assert_close(loss, torch.tensor(13.0).log())
    loss.backward()
    assert torch.count_nonzero(hidden.grad) == 0


def test_rejects_trainable_head_instead_of_silently_dropping_weight_gradients():
    from solution.streamed_head import linear_head_loss

    with pytest.raises(ValueError, match="frozen"):
        linear_head_loss(torch.randn(1, 2, 3), torch.randn(7, 3, requires_grad=True),
                         torch.tensor([[0, 1]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Kaggle T4")
@pytest.mark.parametrize("vocab", [4103, 151936])
@pytest.mark.parametrize("scale", [1.0, 65536.0])
def test_cuda_autocast_forward_metrics_and_grad_scaler_boundary(vocab, scale):
    pytest.importorskip("triton")
    from solution.streamed_head import linear_head_loss

    torch.manual_seed(607)
    hidden = torch.randn(2, 7, 32, device="cuda", requires_grad=True)
    reference = hidden.detach().clone().requires_grad_()
    weight = torch.randn(vocab, 32, device="cuda") * 0.025
    labels = torch.randint(vocab, (2, 7), device="cuda")
    labels[:, :3] = -100
    with torch.autocast("cuda", dtype=torch.float16):
        actual, stats = linear_head_loss(hidden, weight, labels, chunk_rows=5)
        expected, ce, entropy, predictions = _reference(reference, weight, labels)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(stats["token_loss"], ce, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(stats["entropy"], entropy, rtol=3e-5, atol=3e-5)
    assert torch.equal(stats["predictions"], predictions)
    actual.backward(torch.tensor(scale, device="cuda"))
    expected.backward(torch.tensor(scale, device="cuda"))
    torch.testing.assert_close(hidden.grad, reference.grad, rtol=5e-3, atol=2e-6 * scale)
    assert torch.count_nonzero(hidden.grad[:, :2]) == 0
