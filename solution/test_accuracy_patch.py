"""CPU checks of the guarded TRL rewrite, independent of CUDA and TRL imports.

The simulated accelerated path uses PyTorch argmax as a proxy for the separately
tested Triton operation. It validates integration, not T4 kernel execution.
"""

from collections import defaultdict
import copy
import inspect
import sys
import textwrap
from types import FunctionType, ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from solution import accuracy_patch


def entropy_from_logits(logits):
    log_probs = logits.log_softmax(-1)
    return -(log_probs.exp() * log_probs).sum(-1)


class _Accelerator:
    def gather_for_metrics(self, value):
        return value


class _BaseTrainer:
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        logits = model(inputs["values"])
        scores = logits[:, self.num_virtual_tokens :]
        if "shift_labels" in inputs:
            labels = inputs["shift_labels"]
        else:
            scores = scores[:, :-1]
            labels = inputs["labels"][:, 1:]
        loss = torch.nn.functional.cross_entropy(
            scores.reshape(-1, scores.size(-1)), labels.reshape(-1),
            ignore_index=-100, reduction="sum" if num_items_in_batch is not None else "mean",
        )
        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch
        outputs = SimpleNamespace(loss=loss, logits=logits)
        return (loss, outputs) if return_outputs else loss


class _SFTTrainer(_BaseTrainer):
    def __init__(self, virtual_tokens=0):
        self.accelerator = _Accelerator()
        self.num_virtual_tokens = virtual_tokens
        self._metrics = defaultdict(lambda: defaultdict(list))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # A small fixture retaining the pinned method's relevant control flow,
        # zero-argument super(), live entropy alias, and accuracy expressions.
        mode = "train" if model.training else "eval"
        labels = inputs["labels"] if "shift_labels" not in inputs else None
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        with torch.no_grad():
            token_entropy = entropy_from_logits(outputs.logits)
            if self.num_virtual_tokens > 0:
                token_entropy = token_entropy[:, self.num_virtual_tokens :]
            mask = inputs["attention_mask"]
            entropy = (token_entropy * mask).sum() / mask.sum()
            self._metrics[mode]["entropy"].append(
                self.accelerator.gather_for_metrics(entropy).mean().item()
            )
            self._metrics[mode]["num_tokens"].append(int(mask.sum().item()))
        with torch.no_grad():
            if "shift_labels" in inputs:
                shift_logits = outputs.logits.contiguous()
                shift_labels = inputs["shift_labels"]
            else:
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            if self.num_virtual_tokens > 0:
                shift_logits = shift_logits[:, self.num_virtual_tokens :, :]
            predictions = shift_logits.argmax(dim=-1)
            mask = shift_labels != -100
            correct = ((predictions == shift_labels) & mask).sum()
            count = mask.sum()
            correct = self.accelerator.gather_for_metrics(correct)
            count = self.accelerator.gather_for_metrics(count)
            accuracy = (correct.sum() / count.sum()).item() if count.sum() > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)
        return (loss, outputs) if return_outputs else loss


def _kernel_proxy(monkeypatch):
    calls = []
    module = ModuleType("solution.argmax_ops")

    def argmax_last_dim(value):
        assert not torch.is_grad_enabled()
        calls.append((tuple(value.shape), value.stride(), value.is_contiguous()))
        return value.argmax(dim=-1)

    module.argmax_last_dim = argmax_last_dim
    monkeypatch.setitem(sys.modules, module.__name__, module)
    # Deliberately simulate the CUDA eligibility predicate on small CPU data.
    # No test here claims that these CPU tensors actually use the CUDA kernel.
    monkeypatch.setattr(accuracy_patch, "_supported_logits", lambda value: True)
    return calls


@pytest.mark.parametrize("explicit_shift", [False, True])
@pytest.mark.parametrize("virtual_tokens", [0, 2])
@pytest.mark.parametrize("return_outputs", [False, True])
def test_rewrite_preserves_loss_gradients_outputs_and_metrics(
    monkeypatch, explicit_shift, virtual_tokens, return_outputs
):
    native = _SFTTrainer.compute_loss
    rewritten = accuracy_patch._rewrite_compute_loss(native)
    assert rewritten is not None
    calls = _kernel_proxy(monkeypatch)
    torch.manual_seed(157)
    model = torch.nn.Linear(5, 19)
    reference_model = copy.deepcopy(model)
    trainer = _SFTTrainer(virtual_tokens)
    reference_trainer = _SFTTrainer(virtual_tokens)
    values = torch.randn((2, 7 + virtual_tokens, 5), requires_grad=True)
    reference_values = values.detach().clone().requires_grad_()
    labels = torch.randint(19, (2, 7))
    labels[:, :2] = -100
    inputs = {"values": values, "labels": labels,
              "attention_mask": torch.ones((2, 7), dtype=torch.long)}
    if explicit_shift:
        inputs["shift_labels"] = torch.nn.functional.pad(labels, (0, 1), value=-100)[:, 1:]
    reference_inputs = dict(inputs, values=reference_values)
    expected = native(reference_trainer, reference_model, reference_inputs,
                      return_outputs=return_outputs, num_items_in_batch=13)
    actual = rewritten(trainer, model, inputs,
                       return_outputs=return_outputs, num_items_in_batch=13)
    if return_outputs:
        actual_loss, actual_outputs = actual
        expected_loss, expected_outputs = expected
        assert actual_outputs.loss is actual_loss
        assert actual_outputs.logits.dtype == expected_outputs.logits.dtype == torch.float32
        assert actual_outputs.logits.shape == expected_outputs.logits.shape
        assert actual_outputs.logits.is_contiguous()
        torch.testing.assert_close(actual_outputs.logits, expected_outputs.logits, rtol=0, atol=0)
    else:
        actual_loss, expected_loss = actual, expected
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    actual_loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(values.grad, reference_values.grad, rtol=0, atol=0)
    for parameter, expected_parameter in zip(model.parameters(), reference_model.parameters()):
        torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=0, atol=0)
    assert dict(trainer._metrics["train"]) == dict(reference_trainer._metrics["train"])
    assert len(calls) == 1
    # The unshifted-label branch now passes the original strided slice directly
    # to argmax; the full FP32 outputs themselves were never replaced.
    if not explicit_shift:
        assert not calls[0][2]


def test_rewrite_keeps_live_globals_defaults_and_original_super_closure(monkeypatch):
    native = _SFTTrainer.compute_loss
    rewritten = accuracy_patch._rewrite_compute_loss(native)
    assert rewritten is not None
    assert rewritten.__globals__ is native.__globals__
    assert rewritten.__closure__ is native.__closure__
    assert rewritten.__defaults__ is native.__defaults__
    assert rewritten.__kwdefaults__ is native.__kwdefaults__
    assert rewritten.__annotations__ is native.__annotations__
    assert rewritten.__code__.co_firstlineno == native.__code__.co_firstlineno
    assert inspect.signature(rewritten) == inspect.signature(native)
    calls = []

    def replacement_entropy(logits):
        calls.append(logits)
        return torch.full(logits.shape[:-1], 3.0, dtype=logits.dtype)

    # Rebinding the alias AFTER rewriting must be visible to the new method.
    monkeypatch.setitem(native.__globals__, "entropy_from_logits", replacement_entropy)
    model = torch.nn.Linear(3, 7)
    model.eval()
    trainer = _SFTTrainer()
    inputs = {"values": torch.randn(2, 4, 3), "labels": torch.zeros(2, 4, dtype=torch.long),
              "attention_mask": torch.ones(2, 4)}
    loss, outputs = rewritten(trainer, model, inputs, return_outputs=True)
    assert torch.isfinite(loss)
    assert len(calls) == 1 and calls[0] is outputs.logits
    assert trainer._metrics["eval"]["entropy"] == [3.0]


def test_cpu_helpers_keep_native_contiguous_and_argmax_behavior():
    logits = torch.arange(2 * 4 * 7, dtype=torch.float32).reshape(2, 4, 7)
    assert not accuracy_patch._supported_logits(logits)
    view = accuracy_patch._shifted_view(logits)
    assert view.is_contiguous()
    torch.testing.assert_close(view, logits[:, :-1].contiguous(), rtol=0, atol=0)
    torch.testing.assert_close(accuracy_patch._accuracy_argmax(view), view.argmax(-1), rtol=0, atol=0)


@pytest.mark.parametrize("case", ["clone_missing", "argmax_changed", "duplicate_clone",
                                  "not_no_grad", "extra_observer", "decorator"])
def test_unknown_source_is_rejected_without_global_mutation(monkeypatch, case):
    native = _SFTTrainer.compute_loss
    source = textwrap.dedent(inspect.getsource(native))
    if case == "clone_missing":
        source = source.replace("outputs.logits[..., :-1, :].contiguous()",
                                "outputs.logits[..., :-1, :]")
    elif case == "argmax_changed":
        source = source.replace("shift_logits.argmax(dim=-1)", "shift_logits.argmax(dim=2)")
    elif case == "duplicate_clone":
        line = "            shift_logits = outputs.logits[..., :-1, :].contiguous()"
        assert line in source
        source = source.replace(line, line + "\n" + line)
    elif case == "not_no_grad":
        source = source.replace("with torch.no_grad():", "with nullcontext():")
    elif case == "extra_observer":
        line = "        predictions = shift_logits.argmax(dim=-1)"
        assert line in source
        source = source.replace(line, "        observed = shift_logits\n" + line)
    elif case == "decorator":
        source = "@unknown_decorator\n" + source
    before = dict(native.__globals__)
    monkeypatch.setattr(accuracy_patch.inspect, "getsource", lambda value: source)
    assert accuracy_patch._rewrite_compute_loss(native) is None
    assert native.__globals__.keys() == before.keys()
    assert all(native.__globals__[key] is value for key, value in before.items())


def test_signature_closure_and_namespace_collision_guards(monkeypatch):
    native = _SFTTrainer.compute_loss
    wrong_defaults = FunctionType(native.__code__, native.__globals__, native.__name__,
                                  (True, None), native.__closure__)
    assert accuracy_patch._rewrite_compute_loss(wrong_defaults) is None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return inputs

    assert accuracy_patch._rewrite_compute_loss(compute_loss) is None
    collision = object()
    monkeypatch.setitem(native.__globals__, accuracy_patch._OPS_GLOBAL, collision)
    assert accuracy_patch._rewrite_compute_loss(native) is None
    assert native.__globals__[accuracy_patch._OPS_GLOBAL] is collision


def _fake_trl(monkeypatch):
    root = ModuleType("trl")
    root.__path__ = []
    trainer = ModuleType("trl.trainer")
    trainer.__path__ = []
    sft = ModuleType("trl.trainer.sft_trainer")
    sft.SFTTrainer = _SFTTrainer
    trainer.sft_trainer = sft
    root.trainer = trainer
    for module in (root, trainer, sft):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    # Record the method before the public installer mutates the class.
    monkeypatch.setattr(_SFTTrainer, "compute_loss", _SFTTrainer.compute_loss)


def test_public_installer_is_version_guarded_and_idempotent(monkeypatch):
    _fake_trl(monkeypatch)
    native = _SFTTrainer.compute_loss
    monkeypatch.setattr(accuracy_patch.metadata, "version", lambda name: "0.29.2")
    assert not accuracy_patch.patch_sft_accuracy()
    assert _SFTTrainer.compute_loss is native
    monkeypatch.setattr(accuracy_patch.metadata, "version", lambda name: "0.29.1")
    assert accuracy_patch.patch_sft_accuracy()
    installed = _SFTTrainer.compute_loss
    assert installed is not native
    assert accuracy_patch.patch_sft_accuracy()
    assert _SFTTrainer.compute_loss is installed
    assert installed.__wrapped__ is native


def test_published_trl_method_passes_structural_guard():
    pytest.importorskip("trl")
    from importlib import metadata

    if metadata.version("trl") != "0.29.1":
        pytest.skip("source acceptance targets the pinned TRL 0.29.1 method")
    from trl.trainer.sft_trainer import SFTTrainer

    original = SFTTrainer.compute_loss
    rewritten = accuracy_patch._rewrite_compute_loss(original)
    assert rewritten is not None, "published TRL method failed the accuracy rewrite guards"
    assert rewritten.__globals__ is original.__globals__
    assert rewritten.__closure__ is original.__closure__
    assert rewritten.__defaults__ is original.__defaults__
    assert inspect.signature(rewritten) == inspect.signature(original)
