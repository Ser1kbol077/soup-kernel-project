"""Compare RMSNorm with Qwen2's native operation order and autograd.

Run on NVIDIA T4: python -m pytest solution/test_rmsnorm.py -q
GPU tests are explicitly skipped without CUDA; skips do not validate kernels.
"""

import pytest

torch = pytest.importorskip("torch")
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA (target: NVIDIA T4)"
)


def _reference(hidden_states, weight, eps):
    # Includes the cast BEFORE multiplying by weight, as Qwen2 does.
    values = hidden_states.float()
    normalized = values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + eps)
    return weight * normalized.to(hidden_states.dtype)


def _compare(hidden_states, weight, eps, *, exact=False, strided_grad=False):
    from solution.rmsnorm import rms_norm

    ref_input = hidden_states.detach().clone().requires_grad_()
    ref_weight = weight.detach().clone().requires_grad_(weight.requires_grad)
    actual = rms_norm(hidden_states, weight, eps)
    expected = _reference(ref_input, ref_weight, eps)
    if strided_grad:
        upstream = torch.randn(
            (*actual.shape, 2), device=actual.device, dtype=actual.dtype
        )[..., 0]
        assert not upstream.is_contiguous()
    else:
        upstream = torch.randn_like(expected)
    actual.backward(upstream)
    expected.backward(upstream)
    rtol, atol = (0, 0) if exact else (3e-5, 3e-6)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(hidden_states.grad, ref_input.grad, rtol=rtol, atol=atol)
    if weight.requires_grad:
        torch.testing.assert_close(weight.grad, ref_weight.grad, rtol=rtol, atol=atol)


@requires_cuda
@pytest.mark.parametrize("shape", [(1536,), (2, 13, 1536), (3, 1537), (11, 65)])
@pytest.mark.parametrize("eps", [1e-6, 1e-3])
def test_rmsnorm_fp32_forward_backward(shape, eps):
    torch.manual_seed(77)
    hidden_states = torch.randn(shape, device="cuda", requires_grad=True)
    # Nonuniform and negative weights exercise the backward weight factor.
    weight = torch.randn(shape[-1], device="cuda")
    before = hidden_states.detach().clone()
    before_weight = weight.clone()
    _compare(hidden_states, weight, eps)
    torch.testing.assert_close(hidden_states.detach(), before, rtol=0, atol=0)
    torch.testing.assert_close(weight, before_weight, rtol=0, atol=0)


@requires_cuda
def test_rmsnorm_strided_upstream_gradient():
    torch.manual_seed(79)
    hidden_states = torch.randn((2, 7, 1536), device="cuda", requires_grad=True)
    weight = torch.randn(1536, device="cuda")
    _compare(hidden_states, weight, 1e-6, strided_grad=True)


@requires_cuda
@pytest.mark.parametrize("scale", [0.0, 1e-5, 1e3])
def test_rmsnorm_fp32_magnitude(scale):
    torch.manual_seed(83)
    hidden_states = (torch.randn((7, 1536), device="cuda") * scale).requires_grad_()
    weight = torch.randn(1536, device="cuda")
    _compare(hidden_states, weight, 1e-6)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("train_weight", [False, True])
def test_rmsnorm_native_fallback_and_weight_gradient(device, dtype, train_weight):
    torch.manual_seed(89)
    # A strided input also tests dispatch fallback even for frozen FP32 weights.
    hidden_states = torch.randn((2, 65, 3), device=device, dtype=dtype)
    hidden_states = hidden_states.transpose(1, 2).detach().requires_grad_()
    weight = torch.randn(65, device=device, dtype=torch.float32, requires_grad=train_weight)
    _compare(hidden_states, weight, 1e-6, exact=True)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rmsnorm_contiguous_low_precision_cast_before_weight(dtype):
    torch.manual_seed(97)
    hidden_states = torch.randn((3, 1536), device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(1536, device="cuda", dtype=torch.float32)
    _compare(hidden_states, weight, 1e-6, exact=True)


@requires_cuda
def test_rmsnorm_contiguous_trainable_weight_fallback():
    torch.manual_seed(101)
    hidden_states = torch.randn((3, 1536), device="cuda", requires_grad=True)
    weight = torch.randn(1536, device="cuda", requires_grad=True)
    _compare(hidden_states, weight, 1e-6, exact=True)


def test_qwen2_rmsnorm_patch_idempotent_and_native_cpu():
    pytest.importorskip("transformers")
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    from solution.rmsnorm import patch_qwen2_rmsnorm

    patch_qwen2_rmsnorm()
    patched = Qwen2RMSNorm.forward
    patch_qwen2_rmsnorm()
    assert Qwen2RMSNorm.forward is patched
    module = Qwen2RMSNorm(1536)
    hidden_states = torch.randn((2, 1536), requires_grad=True)
    expected = _reference(hidden_states, module.weight, module.variance_epsilon)
    torch.testing.assert_close(module(hidden_states), expected, rtol=0, atol=0)


@requires_cuda
def test_qwen2_rmsnorm_patch_gpu_forward_backward():
    pytest.importorskip("transformers")
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    from solution.rmsnorm import _FrozenRMSNorm, patch_qwen2_rmsnorm

    patch_qwen2_rmsnorm()
    module = Qwen2RMSNorm(1536).to(device="cuda", dtype=torch.float32)
    module.weight.requires_grad_(False)
    hidden_states = torch.randn((2, 17, 1536), device="cuda", requires_grad=True)
    reference_states = hidden_states.detach().clone().requires_grad_()
    expected = _reference(reference_states, module.weight, module.variance_epsilon)
    actual = module(hidden_states)
    assert actual.grad_fn.__class__ is _FrozenRMSNorm._backward_cls
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(hidden_states.grad, reference_states.grad, rtol=3e-5, atol=3e-6)
