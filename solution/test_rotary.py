"""RoPE analytic VJPs, native fallbacks, and CUDA dispatch metadata.

CUDA numerical checks require NVIDIA T4; CPU/FakeTensor passes do not execute
the Triton kernels. Trig halves deliberately differ to catch an invalid
inverse-rotation backward that assumes the usual repeated sin/cos values.
"""

import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def _values(shape, layout="contiguous", *, device="cpu", dtype=torch.float64):
    if layout == "transpose":
        # A dense BTHD allocation exposed as BHTD, matching attention inputs.
        result = torch.randn((shape[0], shape[2], shape[1], shape[3]),
                             device=device, dtype=dtype).transpose(1, 2)
    elif layout == "strided":
        result = torch.randn((*shape[:-1], shape[-1] * 2),
                             device=device, dtype=dtype)[..., ::2]
    else:
        result = torch.randn(shape, device=device, dtype=dtype)
    return result.requires_grad_()


def _trig(batch, tokens, width, *, device="cpu", dtype=torch.float64, train=False):
    cos = torch.randn(batch, tokens, width, device=device, dtype=dtype) * .3
    sin = torch.randn_like(cos) * .4
    half = width // 2
    cos[..., :half] += .75
    cos[..., half:] -= .4
    sin[..., :half] += .2
    sin[..., half:] -= .7
    return cos.requires_grad_(train), sin.requires_grad_(train)


def _analytic_output(x, cos, sin, unsqueeze_dim=1):
    c, s = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    return torch.cat((first * c[..., :half] - second * s[..., :half],
                      second * c[..., half:] + first * s[..., half:]), -1)


def _analytic_vjp(grad, cos, sin, unsqueeze_dim=1):
    c, s = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    half = grad.shape[-1] // 2
    first, second = grad[..., :half], grad[..., half:]
    return torch.cat((first * c[..., :half] + second * s[..., half:],
                      second * c[..., half:] - first * s[..., :half]), -1)


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "strided"])
@pytest.mark.parametrize("trig_batch", [1, 2])
def test_cpu_fp64_matches_independent_rotation_and_repeated_strided_vjp(layout, trig_batch):
    from solution.rotary import rotary_pos_emb

    torch.manual_seed(1201)
    q, k = _values((2, 12, 5, 8), layout), _values((2, 2, 5, 8), layout)
    cos, sin = _trig(trig_batch, 5, 8)
    before = [value.detach().clone() for value in (q, k, cos, sin)]
    cpu_rng = torch.get_rng_state().clone()
    actual_q, actual_k = rotary_pos_emb(q, k, cos, sin)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    for actual, x in ((actual_q, q), (actual_k, k)):
        torch.testing.assert_close(actual, _analytic_output(x, cos, sin), rtol=1e-12, atol=1e-12)
    grad_q = torch.randn((*q.shape, 2), dtype=q.dtype)[..., 0]
    grad_k = torch.randn((*k.shape, 2), dtype=k.dtype)[..., 0]
    assert not grad_q.is_contiguous() and not grad_k.is_contiguous()
    for retain in (True, False):
        dq, dk = torch.autograd.grad((actual_q, actual_k), (q, k), (grad_q, grad_k),
                                     retain_graph=retain)
        torch.testing.assert_close(dq, _analytic_vjp(grad_q, cos, sin), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(dk, _analytic_vjp(grad_k, cos, sin), rtol=1e-12, atol=1e-12)
    for value, saved in zip((q, k, cos, sin), before):
        torch.testing.assert_close(value, saved, rtol=0, atol=0)


@pytest.mark.parametrize("trig_batch", [1, 2])
def test_cpu_trainable_trig_keeps_broadcast_gradients_and_native_double_backward(trig_batch):
    from solution.rotary import rotary_pos_emb

    torch.manual_seed(1213)
    q, k = _values((2, 3, 5, 8)), _values((2, 2, 5, 8))
    cos, sin = _trig(trig_batch, 5, 8, train=True)
    oq, ok = rotary_pos_emb(q, k, cos, sin)
    gq, gk = torch.randn_like(q), torch.randn_like(k)
    dq, dk, dc, ds = torch.autograd.grad((oq, ok), (q, k, cos, sin), (gq, gk), create_graph=True)
    expected_cos = (gq * q).sum(1) + (gk * k).sum(1)
    half = q.shape[-1] // 2
    rotated_q = torch.cat((-q[..., half:], q[..., :half]), -1)
    rotated_k = torch.cat((-k[..., half:], k[..., :half]), -1)
    expected_sin = (gq * rotated_q).sum(1) + (gk * rotated_k).sum(1)
    if trig_batch == 1:
        expected_cos = expected_cos.sum(0, keepdim=True)
        expected_sin = expected_sin.sum(0, keepdim=True)
    torch.testing.assert_close(dc, expected_cos, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(ds, expected_sin, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(dq, _analytic_vjp(gq, cos, sin), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(dk, _analytic_vjp(gk, cos, sin), rtol=1e-12, atol=1e-12)
    # Native fallback remains ordinary autograd, including mixed derivatives.
    mixed_cos, mixed_sin = torch.autograd.grad(dq.sum() + dk.sum(), (cos, sin))
    torch.testing.assert_close(mixed_cos, torch.autograd.grad(
        _analytic_vjp(gq, cos, sin).sum() + _analytic_vjp(gk, cos, sin).sum(),
        cos, retain_graph=True)[0], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(mixed_sin, torch.autograd.grad(
        _analytic_vjp(gq, cos, sin).sum() + _analytic_vjp(gk, cos, sin).sum(),
        sin)[0], rtol=1e-12, atol=1e-12)


def test_cpu_nondefault_broadcast_axis_keeps_native_forward_and_backward():
    from solution.rotary import rotary_pos_emb

    q, k = _values((2, 5, 3, 8)), _values((2, 5, 2, 8))
    cos, sin = _trig(2, 5, 8)
    oq, ok = rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
    gq, gk = torch.randn_like(q), torch.randn_like(k)
    dq, dk = torch.autograd.grad((oq, ok), (q, k), (gq, gk))
    for actual, expected in ((oq, _analytic_output(q, cos, sin, 2)),
                             (ok, _analytic_output(k, cos, sin, 2)),
                             (dq, _analytic_vjp(gq, cos, sin, 2)),
                             (dk, _analytic_vjp(gk, cos, sin, 2))):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_installer_version_guard_idempotence_and_exact_fallback_arguments():
    from solution.rotary import patch_qwen2_rotary

    calls, sentinel = [], object()

    def original(q, k, cos, sin, unsqueeze_dim=1):
        calls.append((q, k, cos, sin, unsqueeze_dim))
        return sentinel

    module = SimpleNamespace(apply_rotary_pos_emb=original)
    patch_qwen2_rotary(module, version="5.16.0")
    assert module.apply_rotary_pos_emb is original
    patch_qwen2_rotary(module, version="5.16.1")
    installed = module.apply_rotary_pos_emb
    assert installed is not original
    assert inspect.signature(installed) == inspect.signature(original)
    patch_qwen2_rotary(module, version="5.16.1")
    assert module.apply_rotary_pos_emb is installed
    q, k, cos, sin = (object() for _ in range(4))
    assert installed(q, k, cos=cos, sin=sin, unsqueeze_dim=2) is sentinel
    assert calls == [(q, k, cos, sin, 2)]


def _cuda_metadata(layout="transpose", *, tokens=5, trig_batch=2):
    q = torch.empty((2, 12, tokens, 128), device="cuda", dtype=torch.float32)
    k = torch.empty((2, 2, tokens, 128), device="cuda", dtype=torch.float32)
    if layout == "transpose":
        q = torch.empty((2, tokens, 12, 128), device="cuda").transpose(1, 2)
        k = torch.empty((2, tokens, 2, 128), device="cuda").transpose(1, 2)
    cos = torch.empty((trig_batch, tokens, 128), device="cuda")
    return q, k, cos, torch.empty_like(cos)


@pytest.fixture
def t4_metadata_guard(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensor
    from solution import rotary

    # Production deliberately accepts exact Tensor types. Replace only this
    # module's torch name to inspect FakeTensor metadata without changing that
    # production policy or globally replacing torch.Tensor.
    proxy = SimpleNamespace(Tensor=FakeTensor, float32=torch.float32, strided=torch.strided,
                            version=SimpleNamespace(hip=None), cuda=SimpleNamespace(
                                get_device_capability=lambda device: (7, 5)))
    monkeypatch.setattr(rotary, "torch", proxy)
    if rotary.triton is None:
        monkeypatch.setattr(rotary, "triton", object())
    return rotary._supported, proxy


@pytest.mark.parametrize("layout", ["contiguous", "transpose"])
@pytest.mark.parametrize("trig_batch", [1, 2])
def test_t4_guard_accepts_production_attention_layout_metadata(t4_metadata_guard, layout, trig_batch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    supported, _ = t4_metadata_guard
    with FakeTensorMode(allow_fallback_kernels=False):
        q, k, cos, sin = _cuda_metadata(layout, trig_batch=trig_batch)
        assert supported(q, k, cos, sin, 1)


@pytest.mark.parametrize("case", [
    "one_token", "fp16_query", "fp16_key", "fp16_trig", "trainable_cos", "trainable_sin",
    "other_gpu", "query_heads", "key_heads", "head_width", "token_mismatch",
    "batch_mismatch", "trig_tokens", "trig_batch", "trig_strided", "query_strided",
    "key_strided", "other_axis", "different_device",
])
def test_t4_guard_rejects_changed_attention_contract(t4_metadata_guard, case):
    from torch._subclasses.fake_tensor import FakeTensorMode

    supported, proxy = t4_metadata_guard
    with FakeTensorMode(allow_fallback_kernels=False):
        q, k, cos, sin = _cuda_metadata(tokens=1 if case == "one_token" else 5)
        axis = 1
        if case == "fp16_query":
            q = q.half()
        elif case == "fp16_key":
            k = k.half()
        elif case == "fp16_trig":
            cos, sin = cos.half(), sin.half()
        elif case == "trainable_cos":
            cos.requires_grad_()
        elif case == "trainable_sin":
            sin.requires_grad_()
        elif case == "other_gpu":
            proxy.cuda.get_device_capability = lambda device: (8, 0)
        elif case == "query_heads":
            q = torch.empty((2, 11, 5, 128), device="cuda")
        elif case == "key_heads":
            k = torch.empty((2, 3, 5, 128), device="cuda")
        elif case == "head_width":
            q = torch.empty((2, 12, 5, 64), device="cuda")
            k = torch.empty((2, 2, 5, 64), device="cuda")
            cos, sin = torch.empty((2, 5, 64), device="cuda"), torch.empty((2, 5, 64), device="cuda")
        elif case == "token_mismatch":
            k = torch.empty((2, 2, 4, 128), device="cuda")
        elif case == "batch_mismatch":
            k = torch.empty((1, 2, 5, 128), device="cuda")
        elif case == "trig_tokens":
            cos, sin = torch.empty((2, 4, 128), device="cuda"), torch.empty((2, 4, 128), device="cuda")
        elif case == "trig_batch":
            cos = torch.empty((3, 5, 128), device="cuda")
            sin = torch.empty_like(cos)
        elif case == "trig_strided":
            cos = torch.empty_strided((2, 5, 128), (1280, 256, 2), device="cuda")
        elif case == "query_strided":
            q = torch.empty_strided((2, 12, 5, 128), (15360, 1280, 256, 2), device="cuda")
        elif case == "key_strided":
            k = torch.empty_strided((2, 2, 5, 128), (2560, 1280, 256, 2), device="cuda")
        elif case == "other_axis":
            axis = 2
        elif case == "different_device":
            sin = torch.empty((2, 5, 128), device="cuda:1")
        assert not supported(q, k, cos, sin, axis)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5),
                    reason="requires NVIDIA T4")
@pytest.mark.parametrize("layout", ["contiguous", "transpose"])
@pytest.mark.parametrize("trig_batch", [1, 2])
def test_t4_kernel_matches_fp64_analytic_reference_and_preserves_input_layout(layout, trig_batch):
    from solution.rotary import _supported, rotary_pos_emb

    torch.manual_seed(1229)
    q = _values((2, 12, 5, 128), layout, device="cuda", dtype=torch.float32)
    k = _values((2, 2, 5, 128), layout, device="cuda", dtype=torch.float32)
    cos, sin = _trig(trig_batch, 5, 128, device="cuda", dtype=torch.float32)
    before = [value.detach().clone() for value in (q, k, cos, sin)]
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    assert _supported(q, k, cos, sin, 1)
    oq, ok = rotary_pos_emb(q, k, cos, sin)
    assert oq.stride() == q.stride() and ok.stride() == k.stride()
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    for out, x in ((oq, q), (ok, k)):
        expected = _analytic_output(x.detach().cpu().double(), cos.cpu().double(), sin.cpu().double())
        torch.testing.assert_close(out.cpu().double(), expected, rtol=2e-6, atol=2e-6)
    gq, gk = torch.randn((*q.shape, 2), device="cuda")[..., 0], torch.randn(
        (*k.shape, 2), device="cuda")[..., 0]
    for retain in (True, False):
        dq, dk = torch.autograd.grad((oq, ok), (q, k), (gq, gk), retain_graph=retain)
        for grad, upstream in ((dq, gq), (dk, gk)):
            expected = _analytic_vjp(upstream.cpu().double(), cos.cpu().double(), sin.cpu().double())
            torch.testing.assert_close(grad.cpu().double(), expected, rtol=2e-6, atol=2e-6)
    for value, saved in zip((q, k, cos, sin), before):
        torch.testing.assert_close(value, saved, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5),
                    reason="requires NVIDIA T4")
def test_t4_key_tail_and_unused_output_backward_with_broadcast_upstream():
    from solution.rotary import _supported, rotary_pos_emb

    # Key has 1*2*3*64=384 pairs: the final BLOCK256 program is partially live.
    q = _values((1, 12, 3, 128), "transpose", device="cuda", dtype=torch.float32)
    k = _values((1, 2, 3, 128), "transpose", device="cuda", dtype=torch.float32)
    cos, sin = _trig(1, 3, 128, device="cuda", dtype=torch.float32)
    assert _supported(q, k, cos, sin, 1)
    oq, ok = rotary_pos_emb(q, k, cos, sin)
    for output, source in ((oq, q), (ok, k)):
        expected = _analytic_output(source.cpu().double(), cos.cpu().double(), sin.cpu().double())
        torch.testing.assert_close(output.cpu().double(), expected, rtol=2e-6, atol=2e-6)
    for output, source, retain in ((oq, q, True), (ok, k, False)):
        upstream = torch.tensor(.375, device="cuda").expand_as(source)
        gradient, = torch.autograd.grad(output, source, upstream, retain_graph=retain)
        expected = _analytic_vjp(upstream.cpu().double(), cos.cpu().double(), sin.cpu().double())
        torch.testing.assert_close(gradient.cpu().double(), expected, rtol=2e-6, atol=2e-6)
