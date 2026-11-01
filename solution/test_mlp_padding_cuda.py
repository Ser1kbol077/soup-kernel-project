"""Actual Qwen NF4 MLP: same-shape T4 fallback, gradients and unchanged RNG."""

from importlib import metadata

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5),
    reason="requires Kaggle T4",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_production_qwen_nf4_mlp_padding_values_gradients_and_rng(monkeypatch, dtype):
    """The production T4 path must preserve the complete preceding MLP call."""
    for name, version in (("bitsandbytes", "0.50.2"), ("transformers", "5.16.1")):
        try:
            if metadata.version(name) != version:
                pytest.skip("requires the fixed package versions")
        except metadata.PackageNotFoundError:
            pytest.skip("requires the fixed training packages")
    import bitsandbytes as bnb
    from bitsandbytes import functional as bnb_functions
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    from solution import mlp_padding as packing
    from solution.compact_swiglu import patch_compact_qwen2_mlp
    from solution.mlp_casts import patch_cast_elided_qwen2_mlp
    from solution.nf4_decode import patch_nf4_decode
    from solution.swiglu import patch_qwen2_mlp

    # Register restoration before installers mutate their module/class slots.
    monkeypatch.setattr(Qwen2MLP, "forward", Qwen2MLP.forward)
    monkeypatch.setattr(bnb_functions, "dequantize_4bit", bnb_functions.dequantize_4bit)
    existing = Qwen2MLP.forward
    if getattr(existing, "_soup_mlp_padding_patch", False):
        Qwen2MLP.forward = existing.__wrapped__
    patch_nf4_decode()
    patch_qwen2_mlp()
    patch_compact_qwen2_mlp()
    patch_cast_elided_qwen2_mlp()
    unpacked_forward = Qwen2MLP.forward
    unpacked_calls = []

    def watched_unpacked(owner, values):
        unpacked_calls.append(values)
        return unpacked_forward(owner, values)

    # Observe the wrapper's actual fallback target, rather than only _eligible.
    monkeypatch.setattr(Qwen2MLP, "forward", watched_unpacked)
    assert packing.patch_qwen2_mlp_padding()

    torch.manual_seed(743)
    config = Qwen2Config(vocab_size=151936, hidden_size=1536, intermediate_size=8960,
                         num_hidden_layers=28, num_attention_heads=12,
                         num_key_value_heads=2, hidden_act="silu")
    with torch.device("meta"):
        module = Qwen2MLP(config)
    for name, input_width, output_width in (
            ("gate_proj", 1536, 8960), ("up_proj", 1536, 8960),
            ("down_proj", 8960, 1536)):
        projection = bnb.nn.Linear4bit(
            input_width, output_width, bias=False, compute_dtype=torch.float16,
            compress_statistics=True, quant_type="nf4",
        ).to(dtype=torch.float16).to("cuda")
        setattr(module, name, projection)
    module.train().requires_grad_(False)
    hidden = torch.randn(2, 128, 1536, device="cuda", dtype=dtype, requires_grad=True)
    reference = hidden.detach().clone().requires_grad_()
    mask = torch.zeros((2, 128), device="cuda", dtype=torch.long)
    mask[0, :96], mask[1, :64] = 1, 1
    labels = torch.full_like(mask, -100)
    labels[:, 1:8] = 3
    labels[0, 100] = 5  # a supervised query outside the metric mask is retained
    targets = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
    observed = (mask != 0) | (targets != -100)
    dense_grad_output = torch.randn_like(hidden) * 32
    grad_output = dense_grad_output * observed[..., None]
    parameters = [(parameter, parameter.detach().clone()) for parameter in module.parameters()]

    # Native Linear4bit establishes compute_type_is_set on its first call;
    # compare both paths after that one-time initialization has happened.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        unpacked_forward(module, hidden.detach())
    with torch.autocast("cuda", dtype=torch.float16):
        expected = unpacked_forward(module, reference)

    def forbidden_packed(*args, **kwargs):
        raise AssertionError("SM75 must preserve the original MLP GEMM dimensions")

    monkeypatch.setattr(packing, "_packed_call", forbidden_packed)
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    with packing.active_mlp_rows(mask, labels) as rows:
        assert rows is not None and rows.indices.numel() == 161
        assert not packing._eligible(module, hidden, rows, bnb.nn.Linear4bit)
        with torch.autocast("cuda", dtype=torch.float16):
            actual = module(hidden)
    assert len(unpacked_calls) == 1 and unpacked_calls[0] is hidden
    assert tuple(unpacked_calls[0].shape) == (2, 128, 1536)
    assert packing._ACTIVE.get() is None
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    # Both calls now have identical shapes and graphs; exact agreement is
    # required, including ordinary MLP outputs at rows excluded by the mask.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.count_nonzero(actual[~observed]) > 0
    expected.backward(grad_output, retain_graph=True)
    actual.backward(grad_output, retain_graph=True)
    torch.testing.assert_close(hidden.grad[observed], reference.grad[observed], rtol=0, atol=0)
    torch.testing.assert_close(hidden.grad[~observed], reference.grad[~observed], rtol=0, atol=0)
    assert torch.count_nonzero(hidden.grad[~observed]) == 0

    # A true fallback also keeps the Jacobian on excluded rows. The masked
    # training VJP above alone could miss an implementation that zeroed it.
    hidden.grad = reference.grad = None
    expected.backward(dense_grad_output)
    actual.backward(dense_grad_output)
    torch.testing.assert_close(hidden.grad, reference.grad, rtol=0, atol=0)
    assert torch.count_nonzero(hidden.grad[~observed]) > 0
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    assert all(parameter.grad is None for parameter in module.parameters())
    for parameter, saved in parameters:
        torch.testing.assert_close(parameter, saved, rtol=0, atol=0)
