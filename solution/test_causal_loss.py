"""FP32 accumulation on half logits, causal alignment and instance-hook tests."""

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/T4")


def _native_causal(logits, labels, vocab_size, num_items_in_batch=None,
                   ignore_index=-100, shift_labels=None, **kwargs):
    if shift_labels is None:
        shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
    loss = F.cross_entropy(logits.float().view(-1, vocab_size),
                           shift_labels.view(-1).to(logits.device),
                           ignore_index=ignore_index,
                           reduction="mean" if num_items_in_batch is None else "sum")
    if num_items_in_batch is not None:
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("vocab", [4103, 151936])
def test_fp32_accumulating_ce_matches_native_cast(dtype, reduction, vocab):
    from solution.vocab_ops import cross_entropy_fp32

    torch.manual_seed(71)
    logits = torch.randn((7, vocab), device="cuda", dtype=dtype, requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    targets = torch.randint(vocab, (7,), device="cuda")
    targets[::3] = -100
    actual = cross_entropy_fp32(logits, targets, reduction=reduction)
    expected = F.cross_entropy(reference.float(), targets, reduction=reduction)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(logits.grad, reference.grad,
                               rtol=2e-3 if dtype == torch.float16 else 5e-5,
                               atol=1e-6 if dtype == torch.float16 else 2e-7)
    assert torch.count_nonzero(logits.grad[targets == -100]) == 0


@requires_cuda
@pytest.mark.parametrize("denominator", [None, 19, "cpu_tensor"])
@pytest.mark.parametrize("explicit_shift", [False, True])
def test_causal_loss_alignment_and_denominator(denominator, explicit_shift):
    from solution.causal_loss import make_causal_loss

    torch.manual_seed(73)
    vocab = 4103
    logits = torch.randn((2, 7, vocab), device="cuda", dtype=torch.float16, requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    labels = torch.randint(vocab, (2, 7), device="cuda")
    labels[:, :3] = -100
    num_items = torch.tensor(19) if denominator == "cpu_tensor" else denominator
    shift = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous() if explicit_shift else None
    loss_fn = make_causal_loss(_native_causal, expected_vocab_size=vocab)
    actual = loss_fn(logits, labels, vocab, num_items_in_batch=num_items, shift_labels=shift)
    expected = _native_causal(reference, labels, vocab, num_items_in_batch=num_items, shift_labels=shift)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(logits.grad, reference.grad, rtol=2e-3, atol=1e-6)


@requires_cuda
def test_half_ce_keeps_half_logits_in_saved_tensors():
    from solution.vocab_ops import cross_entropy_fp32

    logits = torch.randn((7, 4103), device="cuda", dtype=torch.float16, requires_grad=True)
    targets = torch.tensor([0, 1, -100, 3, 4, -100, 4102], device="cuda")
    saved = []

    def pack(tensor):
        saved.append((tensor.shape, tensor.dtype))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = cross_entropy_fp32(logits, targets)
    assert (logits.shape, torch.float16) in saved
    assert (logits.shape, torch.float32) not in saved
    loss.backward()


@requires_cuda
def test_fp32_accumulating_ce_all_ignored_has_zero_grad():
    from solution.vocab_ops import cross_entropy_fp32

    logits = torch.randn((3, 4103), device="cuda", dtype=torch.float16, requires_grad=True)
    targets = torch.full((3,), -100, device="cuda", dtype=torch.long)
    loss = cross_entropy_fp32(logits, targets)
    assert loss.isnan()
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


@requires_cuda
def test_causal_loss_against_real_transformers_with_explicit_shift():
    pytest.importorskip("transformers")
    from transformers.loss import loss_utils
    from solution.causal_loss import make_causal_loss

    # Keep the reference on native PyTorch even when apply() ran in another
    # test and wrapped fixed_cross_entropy globally.
    fixed = loss_utils.fixed_cross_entropy
    native_fixed = getattr(fixed, "__wrapped__", fixed)
    original_globals = dict(loss_utils.ForCausalLMLoss.__globals__)
    original_globals["fixed_cross_entropy"] = native_fixed
    import types
    native_loss = types.FunctionType(loss_utils.ForCausalLMLoss.__code__,
                                     original_globals,
                                     argdefs=loss_utils.ForCausalLMLoss.__defaults__)
    native_loss.__kwdefaults__ = loss_utils.ForCausalLMLoss.__kwdefaults__
    patched = make_causal_loss(native_loss, expected_vocab_size=4103)
    logits = torch.randn((2, 7, 4103), device="cuda", dtype=torch.float16, requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    shifted = torch.randint(4103, (2, 7), device="cuda")
    shifted[:, ::3] = -7
    denominator = torch.tensor(19, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        actual = patched(logits, None, 4103, shift_labels=shifted,
                         ignore_index=-7, num_items_in_batch=denominator)
        expected = native_loss(reference, None, 4103, shift_labels=shifted,
                               ignore_index=-7, num_items_in_batch=denominator)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    upstream = torch.tensor(128.0, device="cuda")
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(logits.grad, reference.grad, rtol=2e-3, atol=1e-6)


def test_causal_loss_cpu_fallback_and_constructor_identity():
    pytest.importorskip("transformers")
    from transformers import Qwen2Config
    from transformers.loss import loss_utils
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    from solution.causal_loss import make_causal_loss, patch_qwen2_causal_loss

    original_mapping = dict(loss_utils.LOSS_MAPPING)
    original_loss = loss_utils.ForCausalLMLoss
    patch_qwen2_causal_loss()
    init = Qwen2ForCausalLM.__init__
    patch_qwen2_causal_loss()
    assert Qwen2ForCausalLM.__init__ is init
    assert loss_utils.LOSS_MAPPING == original_mapping
    assert loss_utils.ForCausalLMLoss is original_loss
    config = Qwen2Config(vocab_size=151936, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    with torch.device("meta"):
        model = Qwen2ForCausalLM(config)
    assert getattr(model.loss_function, "_soup_causal_loss_patch", False)
    assert loss_utils.LOSS_MAPPING.get(model.loss_type) is original_loss
    logits = torch.randn((2, 3, 19), dtype=torch.float16, requires_grad=True)
    labels = torch.tensor([[1, 2, -100], [0, 18, 5]])
    actual = make_causal_loss(_native_causal, expected_vocab_size=19)(logits, labels, 19)
    expected = _native_causal(logits, labels, 19)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
