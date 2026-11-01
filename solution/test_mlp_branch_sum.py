"""FP32 accumulation contracts and single-GPU-safe dispatch metadata."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from solution import mlp_branch_sum as branches


@pytest.mark.parametrize("shape", [(1,), (7,), (1023,), (1024,), (1025,), (2, 3, 1536)])
def test_cpu_sum_matches_independent_fp64_expected_and_preserves_inputs(shape):
    torch.manual_seed(157)
    gate = torch.randn(shape, dtype=torch.float16)
    up = torch.randn(shape, dtype=torch.float16)
    before = gate.clone(), up.clone()
    rng = torch.get_rng_state().clone()
    actual = branches.sum_input_branches(gate, up)
    expected = (gate.double() + up.double()).float()
    assert actual.dtype == torch.float32 and actual.shape == gate.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.data_ptr() != gate.data_ptr() and actual.data_ptr() != up.data_ptr()
    assert torch.equal(torch.get_rng_state(), rng)
    for value, saved in zip((gate, up), before):
        torch.testing.assert_close(value, saved, rtol=0, atol=0)


def test_fp16_rounding_cancellation_extrema_and_subnormal_inputs():
    smallest = 2. ** -24
    gate = torch.tensor([2048., 65504., -65504., 1., -1., smallest, -smallest, 0.],
                        dtype=torch.float16)
    up = torch.tensor([1., 65504., 65504., -1., 1., smallest, smallest, -0.],
                      dtype=torch.float16)
    actual = branches.sum_input_branches(gate, up)
    torch.testing.assert_close(actual, (gate.double() + up.double()).float(), rtol=0, atol=0)
    assert actual[0].item() == 2049.
    assert (gate + up)[0].item() == 2048.
    assert actual[1].item() == 131008. and torch.isfinite(actual).all()
    assert torch.isinf((gate + up)[1])
    assert actual[2].item() == actual[3].item() == actual[4].item() == actual[6].item() == 0.
    assert actual[5].item() == 2. ** -23


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "strided"])
def test_native_fallback_keeps_layout_values_and_higher_order_autograd(dtype, layout):
    torch.manual_seed(163)
    gate, up = (torch.randn(3, 8, dtype=dtype) for _ in range(2))
    if layout == "transpose":
        gate, up = gate.t(), up.t()
    elif layout == "strided":
        gate, up = gate[:, ::2], up[:, ::2]
    gate.requires_grad_()
    up.requires_grad_()
    reference_gate = gate.detach().clone().requires_grad_()
    reference_up = up.detach().clone().requires_grad_()
    before = gate.detach().clone(), up.detach().clone()
    actual = branches.sum_input_branches(gate, up)
    expected = reference_gate.float() + reference_up.float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradients = torch.autograd.grad(actual.square().sum(), (gate, up), create_graph=True)
    references = torch.autograd.grad(expected.square().sum(), (reference_gate, reference_up),
                                    create_graph=True)
    for got, wanted in zip(gradients, references):
        assert got.dtype == wanted.dtype == dtype
        torch.testing.assert_close(got, wanted, rtol=0, atol=0)
    second = torch.autograd.grad(sum(value.sum() for value in gradients), (gate, up))
    reference_second = torch.autograd.grad(sum(value.sum() for value in references),
                                          (reference_gate, reference_up))
    for got, wanted in zip(second, reference_second):
        torch.testing.assert_close(got, wanted, rtol=0, atol=0)
    for value, saved in zip((gate, up), before):
        torch.testing.assert_close(value, saved, rtol=0, atol=0)


def test_native_fallback_preserves_empty_and_broadcast_addition():
    empty = branches.sum_input_branches(torch.empty(0, 3), torch.empty(0, 3))
    assert empty.shape == (0, 3) and empty.dtype == torch.float32
    gate, up = torch.arange(6.).reshape(2, 3), torch.tensor([1., -2., 3.])
    torch.testing.assert_close(branches.sum_input_branches(gate, up), gate + up, rtol=0, atol=0)


@pytest.fixture
def cuda_metadata_guard(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensor

    calls = []
    proxy = SimpleNamespace(
        Tensor=FakeTensor, float16=torch.float16, strided=torch.strided,
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(get_device_capability=lambda device: calls.append(device) or (7, 5)),
    )
    monkeypatch.setattr(branches, "torch", proxy)
    if branches.triton is None:
        monkeypatch.setattr(branches, "triton", object())
    return branches._supported, proxy, calls


@pytest.mark.parametrize("shape", [(1,), (1025,), (2, 3, 1536)])
def test_t4_guard_accepts_only_metadata_without_cuda_execution(cuda_metadata_guard, shape):
    from torch._subclasses.fake_tensor import FakeTensorMode

    supported, _, calls = cuda_metadata_guard
    with FakeTensorMode(allow_fallback_kernels=False):
        gate, up = (torch.empty(shape, device="cuda", dtype=torch.float16) for _ in range(2))
        assert supported(gate, up)
    assert calls == [gate.device]


@pytest.mark.parametrize("case", [
    "cpu_gate", "cpu_up", "fp32_gate", "fp32_up", "shape", "empty",
    "strided_gate", "strided_up", "grad_gate", "grad_up", "hip", "other_gpu",
])
def test_guard_rejects_unsupported_metadata_before_device_query(cuda_metadata_guard, case):
    from torch._subclasses.fake_tensor import FakeTensorMode

    supported, proxy, calls = cuda_metadata_guard
    with FakeTensorMode(allow_fallback_kernels=False):
        gate, up = (torch.empty((3, 1025), device="cuda", dtype=torch.float16) for _ in range(2))
        if case == "cpu_gate":
            gate = torch.empty((3, 1025), dtype=torch.float16, device="cpu")
        elif case == "cpu_up":
            up = torch.empty((3, 1025), dtype=torch.float16, device="cpu")
        elif case == "fp32_gate":
            gate = gate.float()
        elif case == "fp32_up":
            up = up.float()
        elif case == "shape":
            up = torch.empty((3, 1024), device="cuda", dtype=torch.float16)
        elif case == "empty":
            gate, up = (torch.empty(0, device="cuda", dtype=torch.float16) for _ in range(2))
        elif case == "strided_gate":
            gate = torch.empty_strided((3, 1025), (2050, 2), device="cuda", dtype=torch.float16)
        elif case == "strided_up":
            up = torch.empty_strided((3, 1025), (2050, 2), device="cuda", dtype=torch.float16)
        elif case == "grad_gate":
            gate.requires_grad_()
        elif case == "grad_up":
            up.requires_grad_()
        elif case == "hip":
            proxy.version.hip = "test"
        else:
            proxy.cuda.get_device_capability = lambda device: calls.append(device) or (8, 0)
        assert not supported(gate, up)
    assert len(calls) == (1 if case == "other_gpu" else 0)
