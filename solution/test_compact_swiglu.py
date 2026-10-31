"""Validate compact cache values, FP32 gradients, and the NF4 provenance guard.

CUDA cases must be executed on the target T4. A CPU skip is not a GPU pass.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA (target: NVIDIA T4)"
)


def _half_origin(shape, device):
    return torch.randn(shape, device=device, dtype=torch.float16).float().requires_grad_()


@requires_cuda
@pytest.mark.parametrize("shape", [(3, 1001), (2, 31, 8960)])
def test_compact_swiglu_matches_example_exactly(shape):
    from solution.compact_swiglu import half_origin_swiglu
    from solution.swiglu import swiglu

    torch.manual_seed(83)
    gate, up = _half_origin(shape, "cuda"), _half_origin(shape, "cuda")
    ref_gate = gate.detach().clone().requires_grad_()
    ref_up = up.detach().clone().requires_grad_()
    upstream = torch.randn(shape, device="cuda", dtype=torch.float32)
    actual = half_origin_swiglu(gate, up)
    expected = swiglu(ref_gate, ref_up)
    actual.backward(upstream)
    expected.backward(upstream)
    assert actual.dtype == torch.float32
    assert gate.grad.dtype == up.grad.dtype == torch.float32
    for actual_tensor, expected_tensor in (
        (actual, expected), (gate.grad, ref_gate.grad), (up.grad, ref_up.grad)
    ):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


@requires_cuda
def test_compact_swiglu_matches_torch_with_arbitrary_upstream():
    from solution.compact_swiglu import half_origin_swiglu

    torch.manual_seed(89)
    gate, up = _half_origin((7, 4103), "cuda"), _half_origin((7, 4103), "cuda")
    ref_gate = gate.detach().clone().requires_grad_()
    ref_up = up.detach().clone().requires_grad_()
    upstream = torch.randn_like(gate) * 3.1
    expected = torch.nn.functional.silu(ref_gate) * ref_up
    actual = half_origin_swiglu(gate, up)
    expected.backward(upstream)
    actual.backward(upstream)
    for actual_tensor, expected_tensor in (
        (actual, expected), (gate.grad, ref_gate.grad), (up.grad, ref_up.grad)
    ):
        # Same pointwise arithmetic as the shipped example, including its
        # approximate sigmoid; storage compression adds no approximation.
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-4, atol=1e-4)


@requires_cuda
def test_compact_swiglu_saves_half_cache_without_mutating_inputs():
    from solution.compact_swiglu import half_origin_swiglu

    gate, up = _half_origin((5, 8960), "cuda"), _half_origin((5, 8960), "cuda")
    before_gate, before_up = gate.detach().clone(), up.detach().clone()
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        actual = half_origin_swiglu(gate, up)
    assert len(saved) == 2
    assert all(tensor.dtype == torch.float16 for tensor in saved)
    cache_bytes = sum(tensor.numel() * tensor.element_size() for tensor in saved)
    assert cache_bytes == (gate.numel() + up.numel()) * 2
    torch.testing.assert_close(saved[0].float(), before_gate, rtol=0, atol=0)
    torch.testing.assert_close(saved[1].float(), before_up, rtol=0, atol=0)
    actual.backward(torch.randn_like(actual))
    torch.testing.assert_close(gate, before_gate, rtol=0, atol=0)
    torch.testing.assert_close(up, before_up, rtol=0, atol=0)


def test_compact_swiglu_cpu_fallback_preserves_arbitrary_fp32_values():
    from solution.compact_swiglu import half_origin_swiglu

    gate = torch.tensor([[-0.1234567, 0.9876543]], requires_grad=True)
    up = torch.tensor([[0.3456789, -1.234567]], requires_grad=True)
    actual = half_origin_swiglu(gate, up)
    expected = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.tensor([[0.72345, -0.34567]])
    actual_grads = torch.autograd.grad(actual, (gate, up), upstream)
    expected_grads = torch.autograd.grad(expected, (gate, up), upstream)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def _guard_fixture():
    class FrozenLinear:
        def __init__(self):
            self.compute_dtype = torch.float16
            self.compute_type_is_set = True
            self.weight = SimpleNamespace(requires_grad=False, quant_state=object())
            self.bias = None
            self._forward_hooks = {}
            self._forward_pre_hooks = {}

    x = SimpleNamespace(is_cuda=True, dtype=torch.float32)
    mlp = SimpleNamespace(
        config=SimpleNamespace(hidden_act="silu"),
        gate_proj=FrozenLinear(), up_proj=FrozenLinear(),
    )
    return mlp, x, FrozenLinear


def test_compact_mlp_guard_requires_half_compute_and_frozen_provenance():
    from solution.compact_swiglu import _eligible_mlp

    mlp, x, FrozenLinear = _guard_fixture()
    assert _eligible_mlp(mlp, x, FrozenLinear)
    mlp.gate_proj.compute_dtype = torch.float32
    assert not _eligible_mlp(mlp, x, FrozenLinear)
    mlp.gate_proj.compute_dtype = torch.float16
    mlp.up_proj.weight.requires_grad = True
    assert not _eligible_mlp(mlp, x, FrozenLinear)
    mlp.up_proj.weight.requires_grad = False
    mlp.up_proj._forward_hooks[0] = object()
    assert not _eligible_mlp(mlp, x, FrozenLinear)


@pytest.mark.parametrize("case", [
    "subclass", "bias", "quant_state", "forward_override", "local_pre_hook",
    "compute_type_unset", "half_input", "cpu_input", "other_activation",
])
def test_compact_mlp_guard_rejects_unproven_layer_results(case):
    from solution.compact_swiglu import _eligible_mlp

    mlp, x, FrozenLinear = _guard_fixture()
    if case == "subclass":
        class OtherLinear(FrozenLinear):
            pass
        mlp.gate_proj = OtherLinear()
    elif case == "bias":
        mlp.gate_proj.bias = object()
    elif case == "quant_state":
        mlp.gate_proj.weight.quant_state = None
    elif case == "forward_override":
        mlp.gate_proj.forward = lambda value: value
    elif case == "local_pre_hook":
        mlp.up_proj._forward_pre_hooks[0] = object()
    elif case == "compute_type_unset":
        mlp.gate_proj.compute_type_is_set = False
    elif case == "half_input":
        x.dtype = torch.float16
    elif case == "cpu_input":
        x.is_cuda = False
    elif case == "other_activation":
        mlp.config.hidden_act = "gelu"
    assert not _eligible_mlp(mlp, x, FrozenLinear)


@pytest.mark.parametrize("hook_name", ["_global_forward_hooks", "_global_forward_pre_hooks"])
def test_compact_mlp_guard_rejects_global_hooks(monkeypatch, hook_name):
    from torch.nn.modules import module as module_hooks
    from solution.compact_swiglu import _eligible_mlp

    mlp, x, FrozenLinear = _guard_fixture()
    assert _eligible_mlp(mlp, x, FrozenLinear)
    monkeypatch.setitem(getattr(module_hooks, hook_name), -1, object())
    assert not _eligible_mlp(mlp, x, FrozenLinear)


def test_compact_mlp_patch_is_idempotent_and_preserves_qwen_cpu_fallback(monkeypatch):
    pytest.importorskip("bitsandbytes")
    modeling = pytest.importorskip("transformers.models.qwen2.modeling_qwen2")
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from solution.compact_swiglu import patch_compact_qwen2_mlp

    # Keep the class method that existed before this test, including any
    # example patch installed by another integration test.
    before = modeling.Qwen2MLP.forward
    monkeypatch.setattr(modeling.Qwen2MLP, "forward", before)
    config = Qwen2Config(
        vocab_size=97, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
    )
    mlp = modeling.Qwen2MLP(config)
    x = torch.randn((2, 7, 32), requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    expected = mlp(ref_x)
    patch_compact_qwen2_mlp()
    installed = modeling.Qwen2MLP.forward
    patch_compact_qwen2_mlp()
    assert modeling.Qwen2MLP.forward is installed
    actual = mlp(x)
    upstream = torch.randn_like(actual)
    actual_grad, = torch.autograd.grad(actual, (x,), upstream)
    expected_grad, = torch.autograd.grad(expected, (ref_x,), upstream)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@requires_cuda
def test_linear4bit_fp32_outputs_are_lossless_half_values():
    bnb = pytest.importorskip("bitsandbytes")
    from solution.compact_swiglu import half_origin_swiglu
    from solution.swiglu import swiglu

    torch.manual_seed(97)
    gate_layer = bnb.nn.Linear4bit(
        64, 256, bias=False, compute_dtype=torch.float16,
        quant_type="nf4", compress_statistics=True,
    ).cuda()
    up_layer = bnb.nn.Linear4bit(
        64, 256, bias=False, compute_dtype=torch.float16,
        quant_type="nf4", compress_statistics=True,
    ).cuda()
    x = torch.randn((2, 17, 64), device="cuda", requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.float16):
        gate, up = gate_layer(x), up_layer(x)
        assert gate.dtype == up.dtype == torch.float32
        torch.testing.assert_close(gate, gate.half().float(), rtol=0, atol=0)
        torch.testing.assert_close(up, up.half().float(), rtol=0, atol=0)
        actual = half_origin_swiglu(gate, up)
        expected = swiglu(gate_layer(ref_x), up_layer(ref_x))
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=0, atol=0)
