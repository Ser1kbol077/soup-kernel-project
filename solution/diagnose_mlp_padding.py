"""Explicit numerical investigation of the unsafe M=256 -> M=161 T4 packing.

Run after installing the pinned training dependencies:
    python -m solution.diagnose_mlp_padding

This is an offline diagnostic, never imported by ``solution.apply()``. It
forces the old packed operation for measurement, without changing production
dispatch or asserting that its FP16 results satisfy the original tolerance.
"""

import argparse
from importlib import metadata
import json
import math
import platform


def _metrics(actual, reference, *, rtol=1e-2, atol=2e-4):
    """Compute reporting statistics in double precision, including zero refs."""
    import torch

    actual, reference = actual.detach().cpu().double(), reference.detach().cpu().double()
    difference = (actual - reference).abs()
    allowed = atol + rtol * reference.abs()
    nonzero = reference != 0
    nonfinite = ~torch.isfinite(actual) | ~torch.isfinite(reference)

    def scalar(value):
        value = float(value)
        return value if math.isfinite(value) else None

    return {
        "elements": difference.numel(),
        "mismatches": int(((difference > allowed) | nonfinite).sum()),
        "different": int((actual != reference).sum()),
        "max_abs": scalar(difference.max()) if difference.numel() else 0.,
        "rms_abs": scalar(difference.square().mean().sqrt()) if difference.numel() else 0.,
        "max_relative_nonzero_reference": (
            scalar((difference[nonzero] / reference[nonzero].abs()).max())
            if bool(nonzero.any()) else 0.
        ),
        "different_at_zero_reference": int(((actual != reference) & ~nonzero).sum()),
        "nonfinite": int(nonfinite.sum()),
        "rtol": rtol,
        "atol": atol,
    }


def _by_rows(actual, reference, observed, *, output=False):
    # CPU row masks also allow comparison against a CPU FP64 reference.
    actual, reference, observed = actual.detach().cpu(), reference.detach().cpu(), observed.cpu()
    tolerances = {"rtol": 5e-3, "atol": 3e-4} if output else {}
    return {name: _metrics(actual[mask], reference[mask], **tolerances)
            for name, mask in (("kept", observed), ("dropped", ~observed))}


def _fp64_reference(hidden, upstream, weights, device):
    """Independent analytic SwiGLU VJP, with no FP16 activation/GEMM rounding.

    The reference treats the same native-decoded NF4 weights as fixed real
    matrices. It measures numerical accuracy of both finite-precision paths;
    it does not redefine bitsandbytes' cast-backward semantics.
    """
    import torch

    with torch.no_grad():
        x, grad = hidden.detach().to(device=device, dtype=torch.float64), upstream.to(
            device=device, dtype=torch.float64)
        gate_weight, up_weight, down_weight = [weight.to(device=device, dtype=torch.float64)
                                              for weight in weights]
        gate, up = x @ gate_weight.t(), x @ up_weight.t()
        sigmoid = gate.sigmoid()
        output = (gate * sigmoid * up) @ down_weight.t()
        dactivated = grad @ down_weight
        dgate = dactivated * up * sigmoid * (1 + gate * (1 - sigmoid))
        dup = dactivated * gate * sigmoid
        dx = dgate @ gate_weight + dup @ up_weight
        return output.cpu(), dx.cpu()


def _gemm_shape_probes(hidden, upstream, weights, indices):
    """Change only M, with identical selected operands and decoded weights."""
    import torch
    import torch.nn.functional as F
    import triton

    from solution.swiglu import BLOCK, example_swiglu_bwd_kernel, swiglu

    result = {}

    def probe(name, operand, weight, *, forward):
        compact = operand.index_select(0, indices)
        full = F.linear(operand, weight) if forward else operand @ weight
        packed = F.linear(compact, weight) if forward else compact @ weight
        result[name] = _metrics(packed, full.index_select(0, indices), rtol=0, atol=0)
        return full

    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        x = hidden.detach().reshape(-1, hidden.shape[-1]).half()
        grad = upstream.reshape_as(x).half()
        gate_weight, up_weight, down_weight = weights
        gate = probe("forward_gate", x, gate_weight, forward=True)
        up = probe("forward_up", x, up_weight, forward=True)
        activated = swiglu(gate, up)
        probe("forward_down", activated, down_weight, forward=True)
        dactivated = probe("backward_down", grad, down_weight, forward=False)
        dgate, dup = torch.empty_like(gate), torch.empty_like(up)
        n = gate.numel()
        example_swiglu_bwd_kernel[(triton.cdiv(n, BLOCK),)](
            dactivated, gate, up, dgate, dup, n, BLOCK=BLOCK)
        probe("backward_gate", dgate, gate_weight, forward=False)
        probe("backward_up", dup, up_weight, forward=False)
    return result


def run(reference_device):
    import torch

    required = {"bitsandbytes": "0.50.2", "transformers": "5.16.1"}
    for package, version in required.items():
        if metadata.version(package) != version:
            raise RuntimeError(f"Requires {package}=={version}")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError("This diagnostic requires NVIDIA T4 (SM75)")

    import bitsandbytes as bnb
    from bitsandbytes import functional as bnb_functions
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    from solution import mlp_padding as packing
    from solution.compact_swiglu import patch_compact_qwen2_mlp
    from solution.mlp_casts import _eligible_mlp, patch_cast_elided_qwen2_mlp
    from solution.nf4_decode import patch_nf4_decode
    from solution.swiglu import patch_qwen2_mlp

    native_decode = bnb_functions.dequantize_4bit
    while getattr(native_decode, "_soup_nf4_decode_patch", False):
        native_decode = native_decode.__wrapped__
    patch_nf4_decode()
    patch_qwen2_mlp()
    patch_compact_qwen2_mlp()
    patch_cast_elided_qwen2_mlp()
    unpacked_forward = Qwen2MLP.forward
    packing.patch_qwen2_mlp_padding()

    torch.manual_seed(743)
    config = Qwen2Config(vocab_size=151936, hidden_size=1536, intermediate_size=8960,
                         num_hidden_layers=28, num_attention_heads=12,
                         num_key_value_heads=2, hidden_act="silu")
    with torch.device("meta"):
        module = Qwen2MLP(config)
    for name, inputs, outputs in (("gate_proj", 1536, 8960), ("up_proj", 1536, 8960),
                                  ("down_proj", 8960, 1536)):
        layer = bnb.nn.Linear4bit(inputs, outputs, bias=False, compute_dtype=torch.float16,
                                 compress_statistics=True, quant_type="nf4").half().cuda()
        setattr(module, name, layer)
    module.train().requires_grad_(False)
    hidden = torch.randn(2, 128, 1536, device="cuda")
    mask = torch.zeros((2, 128), device="cuda", dtype=torch.long)
    mask[0, :96], mask[1, :64] = 1, 1
    labels = torch.full_like(mask, -100)
    labels[:, 1:8], labels[0, 100] = 3, 5
    targets = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
    observed = (mask != 0) | (targets != -100)
    upstream = torch.randn_like(hidden) * observed[..., None] * 32
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        unpacked_forward(module, hidden)

    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    values, gradients, paths = {}, {}, {}
    with packing.active_mlp_rows(mask, labels) as rows:
        assert rows is not None and rows.indices.numel() == 161
        for name in ("dense", "forced_packed", "production_fallback"):
            x = hidden.detach().clone().requires_grad_()
            with torch.autocast("cuda", dtype=torch.float16):
                if name == "forced_packed":
                    output = packing._packed_call(unpacked_forward, module, x, rows)
                elif name == "dense":
                    output = unpacked_forward(module, x)
                else:
                    output = module(x)
                compact = x.reshape(1, -1, 1536).index_select(1, rows.indices)
                paths[name] = _eligible_mlp(module, compact if name == "forced_packed" else x,
                                            bnb.nn.Linear4bit)
            dx, = torch.autograd.grad(output, x, upstream)
            values[name], gradients[name] = output.detach(), dx.detach()

        decode_metrics, weights = {}, []
        with torch.no_grad():
            for name in ("gate_proj", "up_proj", "down_proj"):
                packed = getattr(module, name).weight
                weight = native_decode(packed, packed.quant_state)
                weights.append(weight)
                decode_metrics[name] = _metrics(
                    bnb_functions.dequantize_4bit(packed, packed.quant_state), weight, rtol=0, atol=0)
        probes = _gemm_shape_probes(hidden, upstream, weights, rows.indices)

    ref_output, ref_dx = _fp64_reference(hidden, upstream, weights, reference_device)
    comparisons = {}
    for name in ("forced_packed", "production_fallback"):
        comparisons[f"{name}_vs_dense"] = {
            "output": _by_rows(values[name], values["dense"], observed, output=True),
            "dx": _by_rows(gradients[name], gradients["dense"], observed),
        }
    for name in values:
        comparisons[f"{name}_vs_fp64"] = {
            "output": _by_rows(values[name], ref_output, observed, output=True),
            "dx": _by_rows(gradients[name], ref_dx, observed),
        }
    return {
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                        "packages": required, "reference_device": reference_device,
                        "allow_fp16_reduced_precision_reduction":
                            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction},
        "seed": 743, "full_rows": 256, "kept_rows": 161,
        "cast_elided_path_eligible": paths,
        "cpu_rng_unchanged": torch.equal(cpu_rng, torch.get_rng_state()),
        "cuda_rng_unchanged": torch.equal(cuda_rng, torch.cuda.get_rng_state()),
        "nf4_decode_vs_native": decode_metrics,
        "identical_operand_fp16_gemm_shape_probes": probes,
        "comparisons": comparisons,
        "reference_semantics": "Real analytic FP64 SwiGLU with native NF4-decoded weights; "
                               "does not reproduce intermediate FP16 casts/rounding.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-device", choices=("cpu", "cuda"), default="cpu",
                        help="Device for the independent FP64 reference (default: CPU)")
    args = parser.parse_args()
    print(json.dumps(run(args.reference_device), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
