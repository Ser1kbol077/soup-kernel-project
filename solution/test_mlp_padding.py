"""No cross-sequence targets, unchanged valid gradients and untouched dropout RNG."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from solution import mlp_padding as packing


class _MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(5, 11, bias=False)
        self.up = torch.nn.Linear(5, 11, bias=False)
        self.down = torch.nn.Linear(11, 5, bias=False)
        self.requires_grad_(False)

    def forward(self, hidden):
        return self.down(torch.nn.functional.silu(self.gate(hidden)) * self.up(hidden))


def test_frozen_mlp_packing_preserves_observed_values_and_gradients():
    torch.manual_seed(419)
    module = _MLP().double()
    hidden = torch.randn(2, 5, 5, dtype=torch.float64, requires_grad=True)
    reference = hidden.detach().clone().requires_grad_()
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]])
    labels = torch.tensor([[-100, 3, -100, -100, -100], [-100, 2, 4, 5, -100]])
    rng = torch.get_rng_state().clone()
    with packing.active_mlp_rows(mask, labels) as rows:
        assert rows.indices.numel() == 6
        actual = packing._packed_call(_MLP.forward, module, hidden, rows)
    assert packing._ACTIVE.get() is None
    assert torch.equal(rng, torch.get_rng_state())
    expected = module(reference)
    torch.testing.assert_close(actual[mask.bool()], expected[mask.bool()], rtol=1e-12, atol=1e-12)
    (actual * mask[..., None]).square().sum().backward()
    (expected * mask[..., None]).square().sum().backward()
    torch.testing.assert_close(hidden.grad, reference.grad, rtol=1e-12, atol=1e-12)


def test_supervised_query_outside_mask_is_retained_without_cross_row_shift():
    mask = torch.tensor([[1, 0, 0], [1, 0, 0]])
    labels = torch.tensor([[-100, -100, 4], [5, -100, -100]])
    with packing.active_mlp_rows(mask, labels) as rows:
        assert torch.equal(rows.indices, torch.tensor([0, 1, 3]))


def test_context_is_restored_after_nested_failure_and_unpadded_calls():
    labels = torch.full((1, 3), -100, dtype=torch.long)
    with packing.active_mlp_rows(torch.tensor([[1, 0, 0]]), labels) as outer:
        with pytest.raises(RuntimeError):
            with packing.active_mlp_rows(torch.ones_like(labels), labels) as inner:
                assert inner is None
                raise RuntimeError("forward failed")
        assert packing._ACTIVE.get() is outer
    assert packing._ACTIVE.get() is None


def test_cpu_and_missing_selection_do_not_activate_production_wrapper():
    module = _MLP()
    module.config = SimpleNamespace(hidden_act="silu")
    hidden = torch.randn(2, 3, 1536)
    assert not packing._eligible(module, hidden, None)
    assert not packing._eligible(module, hidden, packing._Rows((2, 3), torch.tensor([0, 1])))


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_gather_scatter_matches_independent_fp64_vjp_on_kept_and_dropped_rows(noncontiguous):
    """Differentiate the explicit SwiGLU formula, independently of autograd's MLP."""
    torch.manual_seed(431)
    module = _MLP().double()
    hidden = (torch.randn(7, 2, 5, dtype=torch.float64).transpose(0, 1)
              if noncontiguous else torch.randn(2, 7, 5, dtype=torch.float64))
    hidden.requires_grad_()
    reference = hidden.detach().clone().requires_grad_()
    mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0]])
    labels = torch.tensor([[-100, 2, 3, -100, -100, -100, 4],
                           [-100, 2, 3, 4, -100, -100, -100]])
    targets = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
    observed = (mask != 0) | (targets != -100)
    upstream = torch.randn_like(hidden) * observed[..., None]
    before = torch.get_rng_state().clone()

    with packing.active_mlp_rows(mask, labels) as rows:
        assert rows is not None and rows.indices.numel() == int(observed.sum())
        packed = packing._packed_call(_MLP.forward, module, hidden, rows)
    dense = module(reference)
    with torch.no_grad():
        gate = hidden @ module.gate.weight.t()
        up = hidden @ module.up.weight.t()
        sigmoid = gate.sigmoid()
        independent_output = (gate * sigmoid * up) @ module.down.weight.t()
        activated_grad = upstream @ module.down.weight
        gate_grad = activated_grad * up * sigmoid * (1 + gate * (1 - sigmoid))
        up_grad = activated_grad * gate * sigmoid
        independent_grad = gate_grad @ module.gate.weight + up_grad @ module.up.weight

    packed_grad = torch.autograd.grad(packed, hidden, upstream, retain_graph=True)[0]
    repeated = torch.autograd.grad(packed, hidden, upstream)[0]
    dense_grad = torch.autograd.grad(dense, reference, upstream)[0]
    assert hidden.is_contiguous() != noncontiguous
    assert torch.equal(before, torch.get_rng_state())
    assert packing._ACTIVE.get() is None
    torch.testing.assert_close(packed[observed], independent_output[observed], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(dense, independent_output, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(packed_grad[observed], independent_grad[observed], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(dense_grad[observed], independent_grad[observed], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(repeated, packed_grad, rtol=0, atol=0)
    assert torch.count_nonzero(packed[~observed]) == 0
    assert torch.count_nonzero(packed_grad[~observed]) == 0
    assert torch.count_nonzero(dense_grad[~observed]) == 0
    assert all(parameter.grad is None for parameter in module.parameters())


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_t4_production_wrapper_preserves_original_tensor_and_never_packs(monkeypatch, dtype):
    """CUDA metadata only: an otherwise eligible SM75 call must keep its GEMM M."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    class Linear4bit(torch.nn.Module):
        def __init__(self, input_width, output_width):
            super().__init__()
            self.in_features, self.out_features = input_width, output_width
            self.bias = None
            self.weight = torch.nn.Parameter(torch.empty(1, device="cuda"), requires_grad=False)

    class QwenMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_act="silu")
            self.gate_proj = Linear4bit(1536, 8960)
            self.up_proj = Linear4bit(1536, 8960)
            self.down_proj = Linear4bit(8960, 1536)

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))

    def forbidden(*args, **kwargs):
        raise AssertionError("T4 fallback must never enter packed MLP")

    monkeypatch.setattr(packing, "_packed_call", forbidden)
    calls, sentinel = [], object()

    def original(self, values):
        calls.append(values)
        return sentinel

    with FakeTensorMode(allow_fallback_kernels=False):
        module = QwenMLP().train()
        hidden = torch.empty((2, 128, 1536), device="cuda", dtype=dtype, requires_grad=True)
        rows = packing._Rows((2, 128), torch.empty(161, device="cuda", dtype=torch.long))
        wrapped = packing._make_forward(original, QwenMLP, Linear4bit)
        token = packing._ACTIVE.set(rows)
        try:
            assert not packing._eligible(module, hidden, rows, Linear4bit)
            # Establish that the architecture guard, rather than unrelated
            # missing metadata, rejects this production-shaped candidate.
            monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
            assert packing._eligible(module, hidden, rows, Linear4bit)
            monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
            assert wrapped(module, hidden) is sentinel
            assert packing._ACTIVE.get() is rows  # the head can still use these indices
        finally:
            packing._ACTIVE.reset(token)
    assert len(calls) == 1 and calls[0] is hidden
    assert tuple(calls[0].shape) == (2, 128, 1536)
    assert calls[0].stride() == hidden.stride()
    assert packing._ACTIVE.get() is None
