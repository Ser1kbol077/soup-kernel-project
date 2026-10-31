"""Frozen MLP cast boundaries, native fallback, and real NF4 integration.

CPU metadata/proxy checks never execute CUDA. Numerical CUDA cases must also
be run on the target T4; skipped tests do not establish GPU correctness.
"""

import importlib.machinery
from math import prod
import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from solution import mlp_casts

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/T4")


class _TensorMetadata:
    def __init__(self, shape, dtype, *, requires_grad=False, contiguous=True):
        self.shape, self.dtype = shape, dtype
        self.ndim, self.device, self.is_cuda = len(shape), "cuda:0", True
        self.requires_grad, self.contiguous = requires_grad, contiguous

    def numel(self):
        return prod(self.shape)

    def is_contiguous(self):
        return self.contiguous


@pytest.fixture
def guarded_metadata(monkeypatch):
    """Replace this module's torch name only, preserving global torch.Tensor."""
    class FrozenLinear:
        def __init__(self, shape):
            self.compute_dtype, self.compute_type_is_set = torch.float16, True
            self.bias = None
            self.weight = _TensorMetadata(((prod(shape) + 1) // 2, 1), torch.uint8)
            self.weight.quant_state = SimpleNamespace(
                shape=shape, dtype=torch.float16, quant_type="nf4"
            )

    proxy = SimpleNamespace(
        Tensor=_TensorMetadata, float32=torch.float32, float16=torch.float16,
        uint8=torch.uint8, version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(get_device_capability=lambda device: (7, 5)),
        is_grad_enabled=lambda: True, is_autocast_enabled=lambda device: False,
        get_autocast_dtype=lambda device: torch.float16,
    )
    monkeypatch.setattr(mlp_casts, "torch", proxy)
    mlp = SimpleNamespace(config=SimpleNamespace(hidden_act="silu"), training=True)
    for name, shape in zip(mlp_casts._PROJECTIONS, mlp_casts._SHAPES):
        setattr(mlp, name, FrozenLinear(shape))
    x = _TensorMetadata((131, mlp_casts.HIDDEN), torch.float32, requires_grad=True)
    return mlp, x, FrozenLinear, proxy


@pytest.mark.parametrize("leading", [(128,), (131,), (2, 128)])
def test_mlp_cast_guard_admits_t4_training_metadata_only(guarded_metadata, leading):
    mlp, x, linear_type, _ = guarded_metadata
    x.shape, x.ndim = (*leading, mlp_casts.HIDDEN), len(leading) + 1
    assert mlp_casts._eligible_mlp(mlp, x, linear_type)


@pytest.mark.parametrize("case", [
    "down_backward_hook", "mlp_backward_hook", "forward_override",
    "offload_hook", "trainable_down", "down_bias", "bf16_autocast",
    "bf16_weight_origin", "other_gpu", "untracked_input", "noncontiguous_input",
])
def test_mlp_cast_guard_rejects_changed_observable_contract(guarded_metadata, case):
    mlp, x, linear_type, proxy = guarded_metadata
    if case == "down_backward_hook":
        mlp.down_proj._backward_hooks = {0: object()}
    elif case == "mlp_backward_hook":
        mlp._backward_hooks = {0: object()}
    elif case == "forward_override":
        mlp.up_proj.forward = lambda value: value
    elif case == "offload_hook":
        mlp.down_proj._hf_hook = object()
    elif case == "trainable_down":
        mlp.down_proj.weight.requires_grad = True
    elif case == "down_bias":
        mlp.down_proj.bias = object()
    elif case == "bf16_autocast":
        proxy.is_autocast_enabled = lambda device: True
        proxy.get_autocast_dtype = lambda device: torch.bfloat16
    elif case == "bf16_weight_origin":
        mlp.gate_proj.weight.quant_state.dtype = torch.bfloat16
    elif case == "other_gpu":
        proxy.cuda.get_device_capability = lambda device: (8, 0)
    elif case == "untracked_input":
        x.requires_grad = False
    elif case == "noncontiguous_input":
        x.contiguous = False
    assert not mlp_casts._eligible_mlp(mlp, x, linear_type)


@pytest.mark.parametrize("hook_name", mlp_casts._GLOBAL_HOOKS)
def test_mlp_cast_guard_retains_global_observing_hooks(
    guarded_metadata, monkeypatch, hook_name
):
    mlp, x, linear_type, _ = guarded_metadata
    monkeypatch.setitem(getattr(mlp_casts.module_hooks, hook_name), -943, object())
    assert not mlp_casts._eligible_mlp(mlp, x, linear_type)


def _dependency_proxies(monkeypatch, *, version="0.50.2", compact=True):
    """Exercise the real installer without requiring the optional train stack."""
    class NativeMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(4, 7, bias=False)
            self.up_proj = torch.nn.Linear(4, 7, bias=False)
            self.down_proj = torch.nn.Linear(7, 4, bias=False)

        def forward(self, x):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    NativeMLP.forward._soup_compact_patch = compact
    modules = {}
    for name in ["bitsandbytes", "bitsandbytes.functional", "transformers",
                 "transformers.models", "transformers.models.qwen2",
                 "transformers.models.qwen2.modeling_qwen2"]:
        module = ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
    modules["bitsandbytes"].__version__ = version
    modules["bitsandbytes"].nn = SimpleNamespace(Linear4bit=object)
    modules["bitsandbytes"].functional = modules["bitsandbytes.functional"]
    modules["bitsandbytes.functional"].dequantize_4bit = lambda *args: None
    modules["transformers"].models = modules["transformers.models"]
    modules["transformers.models"].qwen2 = modules["transformers.models.qwen2"]
    modules["transformers.models.qwen2"].modeling_qwen2 = modules[
        "transformers.models.qwen2.modeling_qwen2"
    ]
    modules["transformers.models.qwen2.modeling_qwen2"].Qwen2MLP = NativeMLP
    return NativeMLP


def test_mlp_cast_installer_is_idempotent_and_cpu_fallback_keeps_autograd(monkeypatch):
    native_class = _dependency_proxies(monkeypatch)
    original = native_class.forward
    mlp = native_class().double()
    mlp_casts.patch_cast_elided_qwen2_mlp()
    installed = native_class.forward
    assert getattr(installed, "_soup_mlp_cast_patch", False)
    mlp_casts.patch_cast_elided_qwen2_mlp()
    assert native_class.forward is installed

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU fallback must not enter the CUDA custom Function")

    monkeypatch.setattr(mlp_casts._CastElidedMLP, "apply", staticmethod(forbidden))
    x = torch.randn((2, 3, 4), dtype=torch.float64, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    actual, expected = mlp(x), original(mlp, ref_x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    first, = torch.autograd.grad(actual.square().sum(), x, create_graph=True)
    ref_first, = torch.autograd.grad(expected.square().sum(), ref_x, create_graph=True)
    torch.testing.assert_close(first, ref_first, rtol=0, atol=0)
    second, = torch.autograd.grad(first.sum(), x)
    ref_second, = torch.autograd.grad(ref_first.sum(), ref_x)
    torch.testing.assert_close(second, ref_second, rtol=0, atol=0)


@pytest.mark.parametrize("version,compact", [("0.49.0", True), ("0.50.2", False)])
def test_mlp_cast_installer_requires_pinned_stack_and_existing_compact_patch(
    monkeypatch, version, compact
):
    native_class = _dependency_proxies(monkeypatch, version=version, compact=compact)
    original = native_class.forward
    mlp_casts.patch_cast_elided_qwen2_mlp()
    assert native_class.forward is original


def test_mlp_cast_backward_adds_independently_rounded_branches_in_fp32(monkeypatch):
    gate = torch.zeros((2, 3), dtype=torch.float16)
    packed = torch.zeros((3, 1), dtype=torch.uint8)
    gate_dx = torch.tensor([[2048., 1., -2048.]], dtype=torch.float16)
    up_dx = torch.tensor([[1., -1., -1.]], dtype=torch.float16)
    remaining = [torch.zeros_like(gate), gate_dx, up_dx]
    monkeypatch.setattr(mlp_casts, "_input_grad", lambda *args: remaining.pop(0))

    class PointwiseStub:
        def __getitem__(self, grid):
            def run(grad, gate, up, dgate, dup, n, **kwargs):
                dgate.zero_()
                dup.zero_()
            return run

    monkeypatch.setattr(mlp_casts, "example_swiglu_bwd_kernel", PointwiseStub())
    ctx = SimpleNamespace(saved_tensors=(gate, gate, packed, packed, packed),
                          states=(None, None, None), dequantize=None)
    with torch.no_grad():
        actual, _, _ = mlp_casts._CastElidedMLP.backward.__wrapped__(ctx, gate.float())
    expected = gate_dx.float() + up_dx.float()
    incorrectly_rounded = (gate_dx + up_dx).float()
    assert expected[0, 0].item() == 2049
    assert incorrectly_rounded[0, 0].item() == 2048
    assert actual.dtype == torch.float32 and not remaining
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.fixture(scope="module")
def real_nf4_mlp():
    bnb = pytest.importorskip("bitsandbytes")
    assert bnb.__version__ == "0.50.2", "requires the challenge's pinned bitsandbytes"
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    torch.manual_seed(947)
    config = Qwen2Config(hidden_size=1536, intermediate_size=8960, hidden_act="silu",
                         num_attention_heads=12, num_key_value_heads=2, num_hidden_layers=1)
    with torch.device("meta"):
        mlp = Qwen2MLP(config)
    for name, (outputs, inputs) in zip(mlp_casts._PROJECTIONS, mlp_casts._SHAPES):
        layer = bnb.nn.Linear4bit(inputs, outputs, bias=False, compute_dtype=torch.float16,
                                 quant_type="nf4", compress_statistics=True).half().cuda()
        assert not layer.weight.requires_grad
        assert layer.weight.quant_state.dtype == torch.float16
        setattr(mlp, name, layer)
    return mlp.train()


@pytest.fixture
def native_nf4_backward(monkeypatch):
    from bitsandbytes.autograd._functions import MatMul4Bit

    backward = MatMul4Bit.backward
    while getattr(backward, "_soup_mlp_dx_patch", False):
        backward = backward.__wrapped__
    monkeypatch.setattr(MatMul4Bit, "backward", staticmethod(backward))


def _reference_mlp(mlp, x):
    from solution.swiglu import swiglu
    return mlp.down_proj(swiglu(mlp.gate_proj(x), mlp.up_proj(x)))


@requires_cuda
@pytest.mark.parametrize("shape", [(128, 1536), (131, 1536), (2, 128, 1536), (4, 512, 1536)])
@pytest.mark.parametrize("scale", [1., 128., 65536.])
def test_mlp_cast_real_nf4_exact_forward_and_input_gradient(
    real_nf4_mlp, native_nf4_backward, shape, scale
):
    import bitsandbytes as bnb
    from bitsandbytes.functional import dequantize_4bit

    torch.manual_seed(953)
    x = (torch.randn(shape, device="cuda", dtype=torch.float32) * .1).requires_grad_()
    ref_x = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.float16):
        if torch.cuda.get_device_capability(x.device) == (7, 5):
            assert mlp_casts._eligible_mlp(real_nf4_mlp, x, bnb.nn.Linear4bit)
        expected = _reference_mlp(real_nf4_mlp, ref_x)
        actual = mlp_casts._CastElidedMLP.apply(x, real_nf4_mlp, dequantize_4bit)
    # Bound the large loss-scale probe so the output FP16 cast cannot overflow.
    upstream = (torch.rand_like(expected) - .5) * (.125 * scale)
    expected_dx, = torch.autograd.grad(expected, ref_x, upstream)
    actual_dx, = torch.autograd.grad(actual, x, upstream)
    assert actual.dtype == actual_dx.dtype == torch.float32
    assert torch.isfinite(actual).all() and torch.isfinite(actual_dx).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_dx, expected_dx, rtol=0, atol=0)
    assert all(getattr(real_nf4_mlp, name).weight.grad is None for name in mlp_casts._PROJECTIONS)


@requires_cuda
def test_mlp_cast_saved_cache_repeated_backward_strided_upstream_and_input_preservation(
    real_nf4_mlp, native_nf4_backward
):
    from bitsandbytes.functional import dequantize_4bit

    torch.manual_seed(967)
    x = (torch.randn((2, 64, 1536), device="cuda") * .1).requires_grad_()
    ref_x = x.detach().clone().requires_grad_()
    before_x = x.detach().clone()
    weights = [getattr(real_nf4_mlp, name).weight for name in mlp_casts._PROJECTIONS]
    before_weights = [weight.detach().clone() for weight in weights]
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with torch.autocast("cuda", dtype=torch.float16):
        expected = _reference_mlp(real_nf4_mlp, ref_x)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            actual = mlp_casts._CastElidedMLP.apply(x, real_nf4_mlp, dequantize_4bit)
    assert len(saved) == 5
    assert all(t.dtype == torch.float16 and t.shape == (2, 64, 8960) for t in saved[:2])
    assert all(t.dtype == torch.uint8 and t.shape == w.shape for t, w in zip(saved[2:], weights))
    assert sum(t.numel() * t.element_size() for t in saved) == (
        2 * 128 * 8960 * 2 + sum(w.numel() * w.element_size() for w in weights)
    )
    for retain in [True, False]:
        # Dense column-major upstream exercises native matmul stride handling.
        upstream = torch.randn((1536, 128), device="cuda").t().reshape(2, 64, 1536)
        assert not upstream.is_contiguous()
        expected_dx, = torch.autograd.grad(expected, ref_x, upstream, retain_graph=retain)
        actual_dx, = torch.autograd.grad(actual, x, upstream, retain_graph=retain)
        torch.testing.assert_close(actual_dx, expected_dx, rtol=0, atol=0)
    torch.testing.assert_close(x, before_x, rtol=0, atol=0)
    for weight, before in zip(weights, before_weights):
        torch.testing.assert_close(weight, before, rtol=0, atol=0)


@requires_cuda
def test_mlp_cast_patch_uses_real_t4_custom_function_and_is_idempotent(
    real_nf4_mlp, native_nf4_backward, monkeypatch
):
    import bitsandbytes as bnb
    if torch.cuda.get_device_capability() != (7, 5):
        pytest.skip("the production dispatch is intentionally T4-only")
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    from solution.compact_swiglu import patch_compact_qwen2_mlp
    from solution.swiglu import patch_qwen2_mlp

    monkeypatch.setattr(Qwen2MLP, "forward", Qwen2MLP.forward)
    patch_qwen2_mlp()
    patch_compact_qwen2_mlp()
    mlp_casts.patch_cast_elided_qwen2_mlp()
    installed = Qwen2MLP.forward
    mlp_casts.patch_cast_elided_qwen2_mlp()
    assert Qwen2MLP.forward is installed
    x = torch.randn((131, 1536), device="cuda", requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.float16):
        assert mlp_casts._eligible_mlp(real_nf4_mlp, x, bnb.nn.Linear4bit)
        expected = _reference_mlp(real_nf4_mlp, ref_x)
        actual = real_nf4_mlp(x)
    assert actual.grad_fn.__class__ is mlp_casts._CastElidedMLP._backward_cls
    upstream = torch.randn_like(actual)
    actual_dx, = torch.autograd.grad(actual, x, upstream)
    expected_dx, = torch.autograd.grad(expected, ref_x, upstream)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_dx, expected_dx, rtol=0, atol=0)
