"""Exercise architecture dispatch on CPU without compiling or launching CUDA.

FakeTensorMode supplies CUDA tensor metadata only. These tests establish that
SM75 retains the native operation; they do not establish T4 kernel correctness
or performance. Run separately from the CUDA tests: pytest this file -q.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
from torch._subclasses.fake_tensor import FakeTensorMode

from solution import mlp_backward, mlp_dx


class _ForbiddenKernel:
    def __getitem__(self, grid):
        raise AssertionError("SM75 must use native matmul, never the Triton GEMM")


def _nf4_context(grad_shape, *, device="cuda:0"):
    """The pinned bnb backward metadata, without importing bitsandbytes."""
    outputs = grad_shape[-1]
    inputs = 1536 if outputs == 8960 else 8960
    packed = torch.empty(
        ((outputs * inputs + 1) // 2, 1), device=device, dtype=torch.uint8
    )
    grad = torch.empty(grad_shape, device=device, dtype=torch.float16)
    ctx = SimpleNamespace(
        is_empty=False,
        dtype_A=torch.float32,
        dtype_B=torch.uint8,
        needs_input_grad=(True, False, False, False, False),
        state=SimpleNamespace(
            shape=(outputs, inputs), quant_type="nf4", dtype=torch.float16
        ),
        tensors=(None, packed),
    )
    return ctx, grad


@pytest.mark.parametrize("shape", [(129, 65), (3, 43, 65)])
def test_sm75_helper_calls_native_matmul_once_without_launch(monkeypatch, shape):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    # If dispatch regresses, reach our kernel sentinel without entering CUDA.
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(mlp_dx, "mlp_input_grad_kernel", _ForbiddenKernel())
    calls = []

    with FakeTensorMode(allow_fallback_kernels=False), torch.no_grad():
        grad = torch.empty(shape, device="cuda:0", dtype=torch.float16)
        weight = torch.empty((65, 129), device="cuda:0", dtype=torch.float16)
        native_result = torch.empty(
            (*shape[:-1], 129), device="cuda:0", dtype=torch.float16
        )

        def native_matmul(left, right):
            calls.append((left, right))
            return native_result

        monkeypatch.setattr(torch, "matmul", native_matmul)
        actual = mlp_dx.mlp_input_grad(grad, weight)

    assert actual is native_result
    assert len(calls) == 1
    assert calls[0][0] is grad and calls[0][1] is weight


@pytest.mark.parametrize("capability,expected", [((7, 5), False), ((8, 0), True)])
def test_helper_architecture_gate_retains_ampere_path(monkeypatch, capability, expected):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    with FakeTensorMode(allow_fallback_kernels=False), torch.no_grad():
        grad = torch.empty((128, 1536), device="cuda:0", dtype=torch.float16)
        weight = torch.empty((1536, 8960), device="cuda:0", dtype=torch.float16)
        assert mlp_dx._supported(grad, weight) is expected


@pytest.mark.parametrize("capability,expected", [((7, 5), False), ((8, 0), True)])
@pytest.mark.parametrize("outputs", [8960, 1536])
@pytest.mark.parametrize("leading", [(128,), (2, 64)])
def test_nf4_backward_architecture_dispatch(
    monkeypatch, capability, expected, outputs, leading
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    monkeypatch.setattr(torch.version, "hip", None)
    with FakeTensorMode(allow_fallback_kernels=False), torch.no_grad():
        ctx, grad = _nf4_context((*leading, outputs))
        assert mlp_backward._eligible_context(ctx, grad) is expected


@pytest.mark.parametrize("leading", [(128,), (2, 64)])
def test_nf4_dispatch_requires_cuda_and_backward_grad_mode(monkeypatch, leading):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    monkeypatch.setattr(torch.version, "hip", None)
    with FakeTensorMode(allow_fallback_kernels=False):
        cuda_ctx, cuda_grad = _nf4_context((*leading, 8960))
        cpu_ctx, cpu_grad = _nf4_context((*leading, 8960), device="cpu")
        with torch.enable_grad():
            assert not mlp_backward._eligible_context(cuda_ctx, cuda_grad)
        with torch.no_grad():
            assert not mlp_backward._eligible_context(cpu_ctx, cpu_grad)
            assert mlp_backward._eligible_context(cuda_ctx, cuda_grad)


@pytest.mark.parametrize("track_graph", [False, True])
def test_cpu_native_dispatch_preserves_values_and_autograd(monkeypatch, track_graph):
    def forbidden_capability_query(device):
        raise AssertionError("CPU fallback must not query CUDA")

    monkeypatch.setattr(torch.cuda, "get_device_capability", forbidden_capability_query)
    monkeypatch.setattr(mlp_dx, "mlp_input_grad_kernel", _ForbiddenKernel())
    grad = torch.linspace(-0.5, 0.75, 42, dtype=torch.float64).reshape(2, 3, 7)
    grad = grad.requires_grad_()
    weight = torch.linspace(-0.25, 0.5, 77, dtype=torch.float64).reshape(7, 11)
    weight = weight.requires_grad_()
    ref_grad = grad.detach().clone().requires_grad_()
    ref_weight = weight.detach().clone().requires_grad_()
    native_matmul = torch.matmul
    calls = []

    def observed_matmul(left, right):
        calls.append((left, right))
        return native_matmul(left, right)

    monkeypatch.setattr(torch, "matmul", observed_matmul)
    with torch.set_grad_enabled(track_graph):
        actual = mlp_dx.mlp_input_grad(grad, weight)
        expected = native_matmul(ref_grad, ref_weight)
    assert len(calls) == 1
    assert calls[0][0] is grad and calls[0][1] is weight
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.requires_grad is track_graph
    if track_graph:
        actual_grads = torch.autograd.grad(
            actual.square().sum(), (grad, weight), create_graph=True
        )
        expected_grads = torch.autograd.grad(
            expected.square().sum(), (ref_grad, ref_weight), create_graph=True
        )
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
        actual_second = torch.autograd.grad(
            sum(value.sum() for value in actual_grads), (grad, weight)
        )
        expected_second = torch.autograd.grad(
            sum(value.sum() for value in expected_grads), (ref_grad, ref_weight)
        )
        for actual_grad, expected_grad in zip(actual_second, expected_second):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
