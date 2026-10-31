"""Original MLP input-gradient GEMM and its frozen-NF4 integration.

Run on NVIDIA T4: python -m pytest solution/test_mlp_dx.py -q
CUDA skips do not establish correctness or performance on the target GPU.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA (target: NVIDIA T4)"
)


@requires_cuda
@pytest.mark.parametrize("rows", [17, 129, 256])
@pytest.mark.parametrize("inner,columns", [(1536, 8960), (8960, 1536)])
def test_mlp_dx_qwen_shapes_match_fp32_accumulation(rows, inner, columns):
    from solution.mlp_dx import mlp_input_grad

    torch.manual_seed(107)
    grad = torch.randn((rows, inner), device="cuda", dtype=torch.float16) * 0.125
    weight = torch.randn((inner, columns), device="cuda", dtype=torch.float16) * 0.125
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        actual = mlp_input_grad(grad, weight)
        # FP64 reference avoids depending on the runtime's TF32 policy for
        # FP32 matmul. The kernel accumulates FP16 products in FP32 and rounds
        # its result to FP16. Also compare with the operation replaced in bnb.
        reference = (grad.double() @ weight.double()).half()
        native = grad @ weight
    assert actual.shape == (rows, columns)
    assert actual.dtype == torch.float16
    assert torch.isfinite(actual).all()
    # FP16 has a relative spacing of about 9.77e-4. Two spacings plus a
    # small absolute allowance cover different FP32 reduction orders near
    # cancellation; this does not allow lower-precision input multiplication.
    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(actual, native, rtol=2e-3, atol=2e-3)


@requires_cuda
@pytest.mark.parametrize("shape", [(33, 65), (3, 11, 65)])
def test_mlp_dx_tail_tiles_and_input_preservation(shape):
    from solution.mlp_dx import mlp_input_grad

    torch.manual_seed(109)
    grad = torch.randn(shape, device="cuda", dtype=torch.float16)
    weight = torch.randn((65, 129), device="cuda", dtype=torch.float16)
    before_grad, before_weight = grad.clone(), weight.clone()
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        actual = mlp_input_grad(grad, weight)
        expected = (grad.double() @ weight.double()).half()
    assert actual.shape == (*shape[:-1], 129)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(grad, before_grad, rtol=0, atol=0)
    torch.testing.assert_close(weight, before_weight, rtol=0, atol=0)


@requires_cuda
def test_mlp_dx_diagonal_and_zero_columns_are_exact():
    from solution.mlp_dx import mlp_input_grad

    torch.manual_seed(113)
    grad = torch.randn((33, 65), device="cuda", dtype=torch.float16)
    weight = torch.zeros((65, 129), device="cuda", dtype=torch.float16)
    weight[:, :65] = torch.eye(65, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        actual = mlp_input_grad(grad, weight)
    torch.testing.assert_close(actual[:, :65], grad, rtol=0, atol=0)
    assert torch.count_nonzero(actual[:, 65:]) == 0


def _compare_native_autograd(device, dtype):
    from solution.mlp_dx import mlp_input_grad

    torch.manual_seed(127)
    grad = (torch.randn((2, 3, 7), device=device, dtype=dtype) * 0.125).requires_grad_()
    weight = (torch.randn((7, 11), device=device, dtype=dtype) * 0.125).requires_grad_()
    ref_grad = grad.detach().clone().requires_grad_()
    ref_weight = weight.detach().clone().requires_grad_()
    actual = mlp_input_grad(grad, weight)
    expected = ref_grad @ ref_weight
    assert actual.requires_grad
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    first_actual = torch.autograd.grad(actual.square().sum(), (grad, weight), create_graph=True)
    first_expected = torch.autograd.grad(
        expected.square().sum(), (ref_grad, ref_weight), create_graph=True
    )
    for actual_grad, expected_grad in zip(first_actual, first_expected):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    second_actual = torch.autograd.grad(
        sum(value.sum() for value in first_actual), (grad, weight)
    )
    second_expected = torch.autograd.grad(
        sum(value.sum() for value in first_expected), (ref_grad, ref_weight)
    )
    for actual_grad, expected_grad in zip(second_actual, second_expected):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_mlp_dx_cpu_fallback_retains_first_and_second_gradients():
    _compare_native_autograd("cpu", torch.float64)


@requires_cuda
def test_mlp_dx_cuda_tracked_inputs_retain_first_and_second_gradients():
    _compare_native_autograd("cuda", torch.float16)


@requires_cuda
@pytest.mark.parametrize("case", ["fp32", "strided_grad", "strided_weight"])
def test_mlp_dx_unsupported_inputs_use_native_matmul(case):
    from solution.mlp_dx import mlp_input_grad

    torch.manual_seed(131)
    dtype = torch.float32 if case == "fp32" else torch.float16
    grad = torch.randn((33, 65), device="cuda", dtype=dtype)
    weight = torch.randn((65, 129), device="cuda", dtype=dtype)
    if case == "strided_grad":
        grad = torch.randn((33, 130), device="cuda", dtype=dtype)[:, ::2]
        assert not grad.is_contiguous()
    if case == "strided_weight":
        weight = torch.randn((129, 65), device="cuda", dtype=dtype).t()
        assert not weight.is_contiguous()
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        actual = mlp_input_grad(grad, weight)
        expected = grad @ weight
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _native_bnb_backward(monkeypatch):
    pytest.importorskip("bitsandbytes")
    from bitsandbytes.autograd._functions import MatMul4Bit

    original = MatMul4Bit.backward
    while getattr(original, "_soup_mlp_dx_patch", False):
        original = original.__wrapped__
    # Other tests may have called solution.apply(). Obtain the true baseline
    # before constructing either graph; restore the previous method afterward.
    monkeypatch.setattr(MatMul4Bit, "backward", staticmethod(original))
    return MatMul4Bit, original


def test_mlp_dx_bnb_patch_is_idempotent(monkeypatch):
    from solution.mlp_backward import patch_mlp_input_grad

    cls, original = _native_bnb_backward(monkeypatch)
    patch_mlp_input_grad()
    installed = cls.backward
    assert getattr(installed, "_soup_mlp_dx_patch", False)
    assert installed.__wrapped__ is original
    patch_mlp_input_grad()
    assert cls.backward is installed


@requires_cuda
@pytest.mark.parametrize("in_features,out_features", [(1536, 8960), (8960, 1536)])
def test_mlp_dx_real_nf4_layer_preserves_forward_and_input_gradient(
    monkeypatch, in_features, out_features
):
    bnb = pytest.importorskip("bitsandbytes")
    from solution import mlp_backward

    _native_bnb_backward(monkeypatch)
    torch.manual_seed(137)
    # Quantize half weights, as in the fixed model, and retain its FP32
    # activation / FP16 compute / FP32 returned-gradient cast boundaries.
    layer = bnb.nn.Linear4bit(
        in_features, out_features, bias=False, compute_dtype=torch.float16,
        quant_type="nf4", compress_statistics=True,
    ).half().cuda()
    assert not layer.weight.requires_grad
    x = torch.randn((2, 64, in_features), device="cuda", dtype=torch.float32)
    x = x.requires_grad_()
    ref_x = x.detach().clone().requires_grad_()
    before_x, before_weight = x.detach().clone(), layer.weight.detach().clone()
    with torch.autocast("cuda", dtype=torch.float16):
        expected = layer(ref_x)
    upstream = torch.randn_like(expected)
    expected.backward(upstream)

    calls = []
    original_gemm = mlp_backward.mlp_input_grad

    def observed_gemm(grad, weight):
        calls.append((tuple(grad.shape), tuple(weight.shape)))
        return original_gemm(grad, weight)

    monkeypatch.setattr(mlp_backward, "mlp_input_grad", observed_gemm)
    mlp_backward.patch_mlp_input_grad()
    with torch.autocast("cuda", dtype=torch.float16):
        actual = layer(x)
    actual.backward(upstream)

    uses_custom_gemm = torch.cuda.get_device_capability(x.device) >= (8, 0)
    expected_calls = [((2, 64, out_features), (out_features, in_features))]
    # Triton 3.6.0 uses FMA rather than MMA on SM75. On T4 the complete
    # original bnb backward must run, without even calling our GEMM helper.
    assert calls == (expected_calls if uses_custom_gemm else [])
    assert actual.dtype == expected.dtype == torch.float32
    assert x.grad.dtype == ref_x.grad.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.square().mean(), expected.square().mean(), rtol=0, atol=0)
    tolerance = 2e-3 if uses_custom_gemm else 0
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(x, before_x, rtol=0, atol=0)
    torch.testing.assert_close(layer.weight, before_weight, rtol=0, atol=0)
    assert layer.weight.grad is None
