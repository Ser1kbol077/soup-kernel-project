"""Integration contracts for a logits-free step, without downloading a model.

The tiny CPU head below supplies independent dense reference values. T4 tests
of the actual bounded-memory head live in test_streamed_head.py.
"""

from dataclasses import dataclass
from collections import defaultdict
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from solution import streamed_head_patch as patch


@dataclass
class _Step:
    token_entropy: object
    token_loss: object
    correct: object
    labelled: object


class _Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(7, 3)
        self.calls = 0

    def forward(self, input_ids=None, **kwargs):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self.embed(input_ids), past_key_values=None)


class _Causal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Decoder()
        self.lm_head = torch.nn.Linear(3, 7, bias=False)
        self.lm_head.weight.requires_grad_(False)
        self.config = SimpleNamespace(vocab_size=7, hidden_size=3)
        self.native_calls = 0

    def forward(self, input_ids=None, labels=None, num_items_in_batch=None, **kwargs):
        self.native_calls += 1
        logits = self.lm_head(self.model(input_ids=input_ids).last_hidden_state)
        if labels is None:
            return SimpleNamespace(logits=logits, loss=None)
        targets = F.pad(labels, (0, 1), value=-100)[..., 1:]
        terms = F.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1),
                                reduction="none").view_as(labels)
        denom = (targets != -100).sum() if num_items_in_batch is None else num_items_in_batch
        return SimpleNamespace(logits=logits, loss=terms.sum() / denom)


def _dense_head(hidden, weight, labels, num_items_in_batch=None, attention_mask=None,
                shift_labels=None):
    logits = F.linear(hidden, weight)
    targets = (F.pad(labels, (0, 1), value=-100)[..., 1:]
               if shift_labels is None else shift_labels)
    token_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                                reduction="none").view_as(targets)
    denominator = (targets != -100).sum() if num_items_in_batch is None else num_items_in_batch
    log_probs = logits.log_softmax(-1)
    return token_loss.sum() / denominator, {
        "token_loss": token_loss.detach(),
        "entropy": -(log_probs.exp() * log_probs).sum(-1).detach(),
        "predictions": logits.argmax(-1),
    }


def _trainer_loss(trainer, model, inputs, return_outputs=False, num_items_in_batch=None):
    outputs = model(**inputs, num_items_in_batch=num_items_in_batch)
    # HF's token/device scaling must still own this calculation.
    loss = outputs.loss * trainer.loss_scale
    return (loss, outputs) if return_outputs else loss


def _log(trainer, mode, inputs, step):
    mask = inputs["attention_mask"]
    trainer.seen.append((mode, (step.token_entropy * mask).sum() / mask.sum(),
                         step.correct / step.labelled.clamp(min=1), int(mask.sum())))


def _original(trainer, model, inputs, return_outputs=False, num_items_in_batch=None):
    trainer.fallback_calls += 1
    return _trainer_loss(trainer, model, inputs, return_outputs, num_items_in_batch)


def _trainer():
    return SimpleNamespace(loss_scale=1.7, fallback_calls=0, seen=[])


def _batch():
    return {"input_ids": torch.tensor([[0, 1, 2, 3], [4, 5, 6, 0]]),
            "labels": torch.tensor([[-100, 1, 2, 3], [-100, 5, -100, -100]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])}


def test_active_head_skips_padding_without_changing_targets_or_hidden_gradients():
    torch.manual_seed(311)
    hidden = torch.randn(2, 5, 3, requires_grad=True)
    reference = hidden.detach().clone().requires_grad_()
    weight = torch.randn(7, 3)
    labels = torch.tensor([[-100, 2, 3, -100, -100], [-100, 4, 5, 6, -100]])
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    seen = []

    def head(*args, **kwargs):
        seen.append((args[0].shape, kwargs.get("shift_labels")))
        return _dense_head(*args, **kwargs)

    actual, stats = patch._active_head_loss(head, hidden, weight, labels, 11, mask)
    expected, expected_stats = _dense_head(reference, weight, labels, 11)
    assert seen[0][0] == (1, 7, 3)
    assert torch.equal(seen[0][1], torch.tensor([[2, 3, -100, 4, 5, 6, -100]]))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(hidden.grad, reference.grad)
    torch.testing.assert_close(stats["token_loss"], expected_stats["token_loss"])
    torch.testing.assert_close(stats["entropy"] * mask, expected_stats["entropy"] * mask)


def test_active_head_keeps_supervised_positions_even_outside_attention_mask():
    hidden = torch.randn(1, 5, 3, requires_grad=True)
    weight = torch.randn(7, 3)
    labels = torch.tensor([[-100, -100, -100, 4, -100]])
    mask = torch.tensor([[1, 1, 0, 0, 0]])
    seen = []

    def head(*args, **kwargs):
        seen.append(args[0].shape)
        return _dense_head(*args, **kwargs)

    actual, stats = patch._active_head_loss(head, hidden, weight, labels, None, mask)
    expected, original = _dense_head(hidden, weight, labels)
    assert seen == [(1, 3, 3)]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(stats["token_loss"], original["token_loss"])


@pytest.mark.parametrize("denominator", [None, 11, torch.tensor(9)])
def test_streamed_step_preserves_loss_gradient_metrics_and_rewind(monkeypatch, denominator):
    torch.manual_seed(307)
    model = _Causal()
    reference = _Causal()
    reference.load_state_dict(model.state_dict())
    batch = _batch()
    expected = reference(**batch, num_items_in_batch=denominator)
    expected_loss = expected.loss * 1.7
    expected_loss.backward()
    monkeypatch.setattr(patch, "_eligible_trainer", lambda *args: True)
    monkeypatch.setattr(patch, "_eligible_forward", lambda model, request, *args: request is not None)
    monkeypatch.setattr(_Causal, "forward", patch._make_forward(
        _Causal.forward, _dense_head, SimpleNamespace))
    compute = patch._make_compute_loss(_original, _trainer_loss, _Causal, _Step,
                                       _log, "_soup_chunked_step", lambda z: z)
    trainer = _trainer()
    loss, outputs = compute(trainer, model, dict(batch), return_outputs=True,
                            num_items_in_batch=denominator)
    assert outputs.logits is None
    assert trainer.fallback_calls == 0
    assert model.native_calls == 0 and model.model.calls == 1
    torch.testing.assert_close(loss, expected_loss)
    loss.backward()
    torch.testing.assert_close(model.model.embed.weight.grad, reference.model.embed.weight.grad)
    step = trainer._soup_chunked_step
    assert torch.count_nonzero(step.token_loss[:, -1]) == 0
    assert step.labelled == 4
    expected_entropy = -(expected.logits.log_softmax(-1).exp()
                         * expected.logits.log_softmax(-1)).sum(-1)
    mask = batch["attention_mask"]
    torch.testing.assert_close(trainer.seen[0][1], (expected_entropy * mask).sum() / mask.sum())
    shifted = F.pad(batch["labels"], (0, 1), value=-100)[..., 1:]
    expected_terms = F.cross_entropy(expected.logits.flatten(0, 1), shifted.flatten(),
                                     reduction="none").view_as(shifted)
    torch.testing.assert_close(step.token_loss, expected_terms)
    assert patch._REQUEST.get() is None


def test_unsupported_call_keeps_original_forward_and_clears_stale_step(monkeypatch):
    model = _Causal()
    trainer = _trainer()
    trainer._soup_chunked_step = object()
    monkeypatch.setattr(patch, "_eligible_trainer", lambda *args: False)
    compute = patch._make_compute_loss(_original, _trainer_loss, _Causal, _Step,
                                       _log, "_soup_chunked_step", lambda z: z)
    loss, outputs = compute(trainer, model, _batch(), return_outputs=True)
    assert trainer._soup_chunked_step is None
    assert trainer.fallback_calls == 1 and model.native_calls == 1
    assert outputs.logits.shape == (2, 4, 7)
    assert loss.requires_grad


def test_request_context_is_restored_after_model_error(monkeypatch):
    monkeypatch.setattr(patch, "_eligible_trainer", lambda *args: True)

    def broken(*args, **kwargs):
        assert patch._REQUEST.get() is not None
        raise RuntimeError("decoder failure")

    compute = patch._make_compute_loss(_original, broken, _Causal, _Step,
                                       _log, "_soup_chunked_step", lambda z: z)
    outer = patch._Request(object())
    token = patch._REQUEST.set(outer)
    try:
        with pytest.raises(RuntimeError, match="decoder failure"):
            compute(_trainer(), _Causal(), _batch())
        assert patch._REQUEST.get() is outer
    finally:
        patch._REQUEST.reset(token)


def test_cpu_and_generation_forward_do_not_request_streamed_loss():
    model = _Causal()

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported path must retain native logits")

    forward = patch._make_forward(_Causal.forward, forbidden, SimpleNamespace)
    inputs = _batch()
    output = forward(model, **inputs)
    assert output.logits.shape == (2, 4, 7)
    assert not patch._eligible_forward(model, None, inputs["labels"], None, None, 0, {})


@pytest.mark.parametrize("setting", ["smoothing", "custom_loss", "aux", "virtual", "eval", "inputs",
                                     "distributed", "trainable_head", "head_hook"])
def test_unrelated_training_features_fail_closed(monkeypatch, setting):
    # Meta tensors verify guards at the real model shapes without allocating a
    # 890 MiB matrix or requiring a GPU. Only the CUDA predicate is simulated.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    model = _Causal()
    model.config = SimpleNamespace(vocab_size=151936, hidden_size=1536)
    model.lm_head = torch.nn.Linear(1536, 151936, bias=False, device="meta")
    model.lm_head.weight.requires_grad_(False)
    trainer = SimpleNamespace(args=SimpleNamespace(loss_type="nll"),
                              accelerator=SimpleNamespace(), label_smoother=None,
                              compute_loss_func=None, num_virtual_tokens=0,
                              aux_loss_enabled=False)
    inputs = {key: tensor.to("meta") for key, tensor in _batch().items()}
    assert patch._eligible_trainer(trainer, model, inputs, _Causal)
    if setting == "smoothing": trainer.label_smoother = object()
    if setting == "custom_loss": trainer.compute_loss_func = object()
    if setting == "aux": trainer.aux_loss_enabled = True
    if setting == "virtual": trainer.num_virtual_tokens = 2
    if setting == "eval": model.eval()
    if setting == "inputs": inputs["shift_labels"] = inputs["labels"]
    if setting == "distributed": trainer.accelerator.num_processes = 2
    if setting == "trainable_head": model.lm_head.weight.requires_grad_(True)
    if setting == "head_hook": model.lm_head.register_forward_hook(lambda *args: None)
    assert not patch._eligible_trainer(trainer, model, inputs, _Causal)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs pinned packages and CUDA/T4")
def test_pinned_qwen_trainer_and_soup_seam_with_real_streamed_head(monkeypatch):
    """One actual Qwen layer; one model, no downloaded weights or TRL dataset."""
    from importlib import metadata

    for name, version in (("soup-cli", "0.77.0"), ("transformers", "5.16.1"), ("trl", "0.29.1")):
        try:
            if metadata.version(name) != version:
                pytest.skip("integration targets the pinned package versions")
        except metadata.PackageNotFoundError:
            pytest.skip("integration requires the pinned training packages")
    from soup_cli.trainer.rewind_hf import row_losses_from_token_losses
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    from trl.trainer.sft_trainer import SFTTrainer

    torch.manual_seed(313)
    config = Qwen2Config(vocab_size=151936, hidden_size=1536, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=12,
                         num_key_value_heads=2, tie_word_embeddings=True,
                         attention_dropout=0.0, use_cache=False)
    config._attn_implementation = "sdpa"
    with torch.device("cuda"):
        model = Qwen2ForCausalLM(config)
    model = model.half().train()
    model.requires_grad_(False)
    projection = model.model.layers[0].self_attn.q_proj.weight
    projection.requires_grad_(True)
    inputs = {"input_ids": torch.tensor([[3, 6, 8]], device="cuda"),
              "labels": torch.tensor([[-100, 6, 8]], device="cuda"),
              "attention_mask": torch.ones((1, 3), device="cuda", dtype=torch.long)}
    denominator = torch.tensor(2, device="cuda")

    native_forward = Qwen2ForCausalLM.forward
    if getattr(native_forward, patch._MARKER, False):
        native_forward = native_forward.__wrapped__
    native_compute = SFTTrainer.compute_loss
    if getattr(native_compute, patch._MARKER, False):
        native_compute = native_compute.__wrapped__
    # Restore any pre-existing installed hooks when this test finishes.
    monkeypatch.setattr(Qwen2ForCausalLM, "forward", native_forward)
    monkeypatch.setattr(SFTTrainer, "compute_loss", native_compute)
    with torch.autocast("cuda", dtype=torch.float16):
        expected = native_forward(model, **inputs, use_cache=False,
                                  num_items_in_batch=denominator)
    expected_loss = expected.loss.detach()
    native_logits = expected.logits.detach().float()
    (expected.loss * 32).backward()
    expected_grad = projection.grad.detach().clone()
    projection.grad = None
    del expected

    trainer = SimpleNamespace(
        model=model, label_smoother=None, compute_loss_func=None,
        model_accepts_loss_kwargs=True, num_virtual_tokens=0, aux_loss_enabled=False,
        accelerator=SimpleNamespace(num_processes=1, parallelism_config=None,
                                    gather_for_metrics=lambda value: value),
        args=SimpleNamespace(loss_type="nll", use_liger_kernel=False, padding_free=False,
                             n_gpu=1, average_tokens_across_devices=False),
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
        _total_train_tokens=0,
    )
    assert patch._eligible_trainer(trainer, model, inputs, Qwen2ForCausalLM)
    assert patch.patch_streamed_head()
    installed_forward = Qwen2ForCausalLM.forward
    installed_compute = SFTTrainer.compute_loss
    assert patch.patch_streamed_head()
    assert Qwen2ForCausalLM.forward is installed_forward
    assert SFTTrainer.compute_loss is installed_compute
    with torch.autocast("cuda", dtype=torch.float16):
        actual_loss, output = installed_compute(
            trainer, model, dict(inputs), return_outputs=True,
            num_items_in_batch=denominator)
    assert output.logits is None
    torch.testing.assert_close(actual_loss, expected_loss, rtol=3e-5, atol=3e-5)
    (actual_loss * 32).backward()
    torch.testing.assert_close(projection.grad, expected_grad, rtol=6e-3, atol=2e-4)
    step = trainer._soup_chunked_step
    row_loss, row_count = row_losses_from_token_losses(step.token_loss, inputs["labels"])
    expected_row = F.cross_entropy(native_logits[:, :-1].reshape(-1, 151936),
                                   inputs["labels"][:, 1:].reshape(-1))
    assert row_count == [2]
    assert row_loss[0] == pytest.approx(float(expected_row), rel=3e-5, abs=3e-5)
    log_probs = native_logits.log_softmax(-1)
    expected_entropy = float(-(log_probs.exp() * log_probs).sum(-1).mean())
    assert trainer._metrics["train"]["entropy"] == pytest.approx([expected_entropy], abs=3e-5)
    assert trainer._metrics["train"]["num_tokens"] == [3]
    predictions = native_logits[:, :-1].argmax(-1)
    expected_accuracy = float((predictions == inputs["labels"][:, 1:]).float().mean())
    assert trainer._metrics["train"]["mean_token_accuracy"] == [expected_accuracy]
