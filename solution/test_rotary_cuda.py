"""Execute the fused FP32 rotary embedding and its VJP on NVIDIA T4.

These tests require actual SM75 CUDA execution. Skips do not validate kernels.
"""

from importlib import metadata

import pytest

torch = pytest.importorskip("torch")
requires_t4 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5),
    reason="requires NVIDIA T4",
)


def _native(q, k, cos, sin, unsqueeze_dim=1):
    """The pinned Qwen expression, with separate FP32 products and addition."""
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)

    def rotate_half(value):
        first, second = value.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def _inputs(batch, length, transposed, trig_batch):
    def projection(heads):
        shape = ((batch, length, heads, 128) if transposed
                 else (batch, heads, length, 128))
        values = torch.randn(shape, device="cuda", dtype=torch.float32)
        return (values.transpose(1, 2) if transposed else values).detach().requires_grad_()

    q, k = projection(12), projection(2)
    # Deliberately different halves: a correct VJP must use the opposite
    # half's sine rather than assume Qwen's usual repeated trig halves.
    cos_batch, sin_batch = trig_batch if isinstance(trig_batch, tuple) else (trig_batch, trig_batch)
    cos = torch.randn((cos_batch, length, 128), device="cuda", dtype=torch.float32)
    sin = torch.randn((sin_batch, length, 128), device="cuda", dtype=torch.float32)
    return q, k, cos, sin


def _strided_upstream(tensor):
    result = torch.randn((*tensor.shape, 2), device=tensor.device, dtype=tensor.dtype)[..., 0]
    assert not result.is_contiguous()
    return result


@requires_t4
@pytest.mark.parametrize("batch,length,transposed,trig_batch", [
    (1, 7, False, 1),
    (4, 128, True, 4),
    (4, 512, False, 1),
    (1, 128, True, 1),
    (4, 7, True, (1, 4)),
    (4, 128, False, (4, 1)),
])
def test_t4_fp32_rotary_exact_values_vjp_layout_rng_and_saved_tensors(
        batch, length, transposed, trig_batch):
    from solution import rotary

    torch.manual_seed(149)
    q, k, cos, sin = _inputs(batch, length, transposed, trig_batch)
    q_ref = q.detach().clone().requires_grad_()
    k_ref = k.detach().clone().requires_grad_()
    assert rotary._supported(q, k, cos, sin, 1)
    originals = [tensor.detach().clone() for tensor in (q, k, cos, sin)]
    upstream = (_strided_upstream(q), _strided_upstream(k))
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        actual = rotary.rotary_pos_emb(q, k, cos, sin)
    expected = _native(q_ref, k_ref, cos, sin)
    # Saving either large projection would erase the intended memory benefit.
    assert len(saved) == 2 and saved[0] is cos and saved[1] is sin
    for got, wanted, source in zip(actual, expected, (q, k)):
        assert got.dtype == wanted.dtype == torch.float32
        assert got.shape == wanted.shape == source.shape
        assert got.stride() == wanted.stride() == source.stride()
        torch.testing.assert_close(got, wanted, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual, (q, k), upstream, retain_graph=True)
    repeated = torch.autograd.grad(actual, (q, k), upstream)
    expected_grad = torch.autograd.grad(expected, (q_ref, k_ref), upstream)
    for got, again, wanted in zip(actual_grad, repeated, expected_grad):
        assert got.stride() == wanted.stride()
        torch.testing.assert_close(got, wanted, rtol=0, atol=0)
        torch.testing.assert_close(again, got, rtol=0, atol=0)
    for tensor, original in zip((q, k, cos, sin), originals):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())


@requires_t4
@pytest.mark.parametrize("used_output", [0, 1])
def test_t4_rotary_unused_projection_keeps_none_gradient(used_output):
    from solution import rotary

    torch.manual_seed(150)
    q, k, cos, sin = _inputs(1, 7, True, 1)
    references = [tensor.detach().clone().requires_grad_() for tensor in (q, k)]
    actual = rotary.rotary_pos_emb(q, k, cos, sin)
    expected = _native(*references, cos, sin)
    upstream = _strided_upstream(actual[used_output])
    actual_grad = torch.autograd.grad(actual[used_output], (q, k), upstream, allow_unused=True)
    expected_grad = torch.autograd.grad(
        expected[used_output], tuple(references), upstream, allow_unused=True,
    )
    assert actual_grad[1 - used_output] is expected_grad[1 - used_output] is None
    torch.testing.assert_close(actual_grad[used_output], expected_grad[used_output], rtol=0, atol=0)


@requires_t4
def test_t4_pinned_qwen_rotary_patch_keeps_sdpa_values_layout_and_vjp(monkeypatch):
    try:
        if metadata.version("transformers") != "5.16.1":
            pytest.skip("requires transformers==5.16.1")
    except metadata.PackageNotFoundError:
        pytest.skip("requires the fixed training packages")
    from transformers.models.qwen2 import modeling_qwen2
    from solution import rotary

    native = modeling_qwen2.apply_rotary_pos_emb
    while getattr(native, "_soup_rotary_patch", False):
        native = native.__wrapped__
    monkeypatch.setattr(modeling_qwen2, "apply_rotary_pos_emb", native)
    assert rotary.patch_qwen2_rotary()
    installed = modeling_qwen2.apply_rotary_pos_emb
    rotary.patch_qwen2_rotary()
    assert modeling_qwen2.apply_rotary_pos_emb is installed

    torch.manual_seed(151)
    hidden = torch.randn((2, 128, 1536), device="cuda", requires_grad=True)
    reference = hidden.detach().clone().requires_grad_()
    weights = [torch.randn((heads * 128, 1536), device="cuda") / (1536 ** 0.5)
               for heads in (12, 2, 2)]
    cos = torch.randn((2, 128, 128), device="cuda")
    sin = torch.randn_like(cos)
    originals = [tensor.detach().clone() for tensor in (hidden, *weights, cos, sin)]

    def projections(values):
        return tuple(torch.nn.functional.linear(values, weight)
                     .view(2, 128, heads, 128).transpose(1, 2)
                     for weight, heads in zip(weights, (12, 2, 2)))

    q, k, value = projections(hidden)
    q_ref, k_ref, value_ref = projections(reference)
    projection_gradients = {}

    def observe_layout(name):
        def hook(grad):
            projection_gradients[name] = grad
        return hook

    for tensor, name in ((q, "q"), (k, "k"), (value, "v"),
                         (q_ref, "q_ref"), (k_ref, "k_ref"), (value_ref, "v_ref")):
        tensor.register_hook(observe_layout(name))
    upstream = _strided_upstream(q)
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    actual_q, actual_k = installed(q, k, cos, sin)
    expected_q, expected_k = native(q_ref, k_ref, cos, sin)
    assert actual_q.dtype == expected_q.dtype and actual_k.dtype == expected_k.dtype
    assert actual_q.stride() == expected_q.stride()
    assert actual_k.stride() == expected_k.stride()

    def attention(query, key, values):
        # The production attention has 12 query heads and 2 KV heads. Repeat
        # before SDPA exactly as the Qwen SDPA integration does on T4.
        key = key.repeat_interleave(6, dim=1)
        values = values.repeat_interleave(6, dim=1)
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, values, dropout_p=0.0, is_causal=True,
        )

    actual = attention(actual_q, actual_k, value)
    expected = attention(expected_q, expected_k, value_ref)
    actual_grad = torch.autograd.grad(actual, hidden, upstream)[0]
    expected_grad = torch.autograd.grad(expected, reference, upstream)[0]
    assert actual.dtype == expected.dtype
    assert actual.stride() == expected.stride()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    for name in ("q", "k", "v"):
        got, wanted = projection_gradients[name], projection_gradients[name + "_ref"]
        assert got.dtype == wanted.dtype
        assert got.stride() == wanted.stride()
        torch.testing.assert_close(got, wanted, rtol=0, atol=0)
    for tensor, original in zip((hidden, *weights, cos, sin), originals):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)
    assert all(weight.grad is None for weight in weights)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
