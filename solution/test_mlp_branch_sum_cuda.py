"""Actual T4 checks for the frozen MLP's FP16-branch/FP32-sum boundary.

CPU skips do not establish CUDA kernel correctness or performance.
"""

from importlib import metadata

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from solution.test_mlp_casts import (  # noqa: F401 - imported pytest fixtures
    _reference_mlp,
    native_nf4_backward,
    real_nf4_mlp,
)


requires_t4 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5),
    reason="requires NVIDIA T4",
)


def _assert_finite_fp32_bits(actual, expected):
    assert actual.dtype == expected.dtype == torch.float32
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    # Include signed zero: this is the existing FP32 branch addition exactly.
    assert torch.equal(actual.contiguous().view(torch.int32), expected.contiguous().view(torch.int32))


@requires_t4
@pytest.mark.parametrize("shape", [(1,), (1025,), (3, 5, 1536), (4, 512, 1536)])
@pytest.mark.parametrize("scale", [2. ** -14, 1., 2048.])
def test_t4_branch_sum_exact_fp32_random_tails_scales_rng_and_inputs(shape, scale):
    from solution import mlp_branch_sum as branch_sum

    torch.manual_seed(157)
    gate = torch.randn(shape, device="cuda", dtype=torch.float16) * scale
    up = torch.randn_like(gate) * scale
    before_gate, before_up = gate.clone(), up.clone()
    expected = gate.float() + up.float()
    assert branch_sum._supported(gate, up)
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    actual = branch_sum.sum_input_branches(gate, up)
    repeated = branch_sum.sum_input_branches(gate, up)

    _assert_finite_fp32_bits(actual, expected)
    _assert_finite_fp32_bits(repeated, expected)
    assert actual.is_contiguous() and not actual.requires_grad
    assert actual.data_ptr() != gate.data_ptr() and actual.data_ptr() != up.data_ptr()
    torch.testing.assert_close(gate, before_gate, rtol=0, atol=0)
    torch.testing.assert_close(up, before_up, rtol=0, atol=0)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())


@requires_t4
def test_t4_branch_sum_preserves_fp16_subnormals_cancellation_and_fp32_overflow_margin():
    from solution.mlp_branch_sum import sum_input_branches

    gate = torch.tensor([
        0., -0., 2. ** -24, -(2. ** -24), 2. ** -14, -(2. ** -14),
        1., -1., 2048., -2048., 65504., -65504.,
    ], device="cuda", dtype=torch.float16)
    up = torch.tensor([
        0., -0., -(2. ** -24), 2. ** -24, 2. ** -24, -(2. ** -24),
        2. ** -11, -(2. ** -11), 1., -1., 65504., -65504.,
    ], device="cuda", dtype=torch.float16)
    expected = gate.float() + up.float()
    prematurely_rounded = (gate + up).float()
    actual = sum_input_branches(gate, up)

    _assert_finite_fp32_bits(actual, expected)
    assert actual[8].item() == 2049. and prematurely_rounded[8].item() == 2048.
    assert actual[-2].item() == 131008. and actual[-1].item() == -131008.
    assert torch.isinf(prematurely_rounded[-2:]).all()


@requires_t4
def test_t4_branch_sum_keeps_nonfinite_detection():
    from solution.mlp_branch_sum import sum_input_branches

    gate = torch.tensor([float("inf"), -float("inf"), float("nan"), 1.],
                        device="cuda", dtype=torch.float16)
    up = torch.tensor([1., float("inf"), 2., float("nan")],
                      device="cuda", dtype=torch.float16)
    expected = gate.float() + up.float()
    actual = sum_input_branches(gate, up)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(torch.isfinite(actual), torch.isfinite(expected))
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))


@requires_t4
@pytest.mark.parametrize("requires_grad", [False, True])
def test_t4_strided_branch_inputs_keep_native_fallback_and_autograd(monkeypatch, requires_grad):
    from solution import mlp_branch_sum as branch_sum

    class ForbiddenKernel:
        def __getitem__(self, grid):
            raise AssertionError("strided branches must keep native cast/add")

    monkeypatch.setattr(branch_sum, "mlp_branch_sum_kernel", ForbiddenKernel())
    torch.manual_seed(159)
    gate = torch.randn((5, 7, 2), device="cuda", dtype=torch.float16)[..., 0]
    up = torch.randn((7, 5), device="cuda", dtype=torch.float16).t()
    gate = gate.detach().requires_grad_(requires_grad)
    up = up.detach().requires_grad_(requires_grad)
    assert not gate.is_contiguous() and not up.is_contiguous()
    assert not branch_sum._supported(gate, up)
    before_gate, before_up = gate.detach().clone(), up.detach().clone()
    upstream = torch.randn((5, 7), device="cuda", dtype=torch.float32)
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    actual = branch_sum.sum_input_branches(gate, up)
    expected = gate.float() + up.float()
    _assert_finite_fp32_bits(actual, expected)
    assert actual.requires_grad == requires_grad
    if requires_grad:
        actual_grad = torch.autograd.grad(actual, (gate, up), upstream)
        expected_grad = torch.autograd.grad(expected, (gate, up), upstream)
        for got, wanted in zip(actual_grad, expected_grad):
            torch.testing.assert_close(got, wanted, rtol=0, atol=0)
    torch.testing.assert_close(gate, before_gate, rtol=0, atol=0)
    torch.testing.assert_close(up, before_up, rtol=0, atol=0)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())


@requires_t4
def test_t4_real_nf4_cast_elided_mlp_launches_one_branch_sum_and_keeps_exact_vjp(
        real_nf4_mlp, native_nf4_backward, monkeypatch):
    for name, version in (("bitsandbytes", "0.50.2"), ("transformers", "5.16.1")):
        if metadata.version(name) != version:
            pytest.skip("requires the fixed training package versions")
    from bitsandbytes.functional import dequantize_4bit
    from solution import mlp_branch_sum as branch_sum
    from solution import mlp_casts

    original_kernel = branch_sum.mlp_branch_sum_kernel
    launches = []

    class WatchedKernel:
        def __getitem__(self, grid):
            launch = original_kernel[grid]

            def run(gate, up, output, *args, **kwargs):
                launches.append((gate.shape, gate.dtype, up.dtype, output.dtype))
                return launch(gate, up, output, *args, **kwargs)
            return run

    monkeypatch.setattr(branch_sum, "mlp_branch_sum_kernel", WatchedKernel())
    torch.manual_seed(163)
    hidden = (torch.randn((2, 128, 1536), device="cuda") * .1).requires_grad_()
    reference = hidden.detach().clone().requires_grad_()
    before_hidden = hidden.detach().clone()
    weights = [getattr(real_nf4_mlp, name).weight for name in mlp_casts._PROJECTIONS]
    before_weights = [weight.detach().clone() for weight in weights]
    upstream = (torch.rand_like(hidden) - .5) * 16
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    with torch.autocast("cuda", dtype=torch.float16):
        expected = _reference_mlp(real_nf4_mlp, reference)
        actual = mlp_casts._CastElidedMLP.apply(hidden, real_nf4_mlp, dequantize_4bit)
    expected_dx, = torch.autograd.grad(expected, reference, upstream)
    assert not launches  # the native graph must not use the new sum helper
    actual_dx, = torch.autograd.grad(actual, hidden, upstream)

    assert launches == [(hidden.shape, torch.float16, torch.float16, torch.float32)]
    _assert_finite_fp32_bits(actual, expected)
    _assert_finite_fp32_bits(actual_dx, expected_dx)
    torch.testing.assert_close(hidden, before_hidden, rtol=0, atol=0)
    for weight, before in zip(weights, before_weights):
        torch.testing.assert_close(weight, before, rtol=0, atol=0)
        assert weight.grad is None
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
