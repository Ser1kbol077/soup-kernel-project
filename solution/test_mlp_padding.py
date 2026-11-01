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
