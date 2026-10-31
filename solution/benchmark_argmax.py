"""Optional T4 microbenchmark; run only after the official training has exited.

This measures a synthetic accuracy operation, not soup train or rule compliance.
No allocator, precision, training setting or compiler-cache setting is changed.
"""

import argparse
import json
import time

import torch
import triton

from .argmax_ops import argmax_last_dim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This microbenchmark requires CUDA; no CPU timings are substituted.")
    if torch.cuda.get_device_capability(0) != (7, 5) or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Run on the target NVIDIA T4.")

    torch.manual_seed(223)
    logits = torch.randn((4, 512, 151936), device="cuda:0", dtype=torch.float32)
    shifted = logits[:, :-1, :]

    def native():
        return shifted.contiguous().argmax(dim=-1)

    def candidate():
        return argmax_last_dim(shifted)

    with torch.no_grad():
        torch.cuda.synchronize()
        first_call = time.perf_counter()
        actual = candidate()
        torch.cuda.synchronize()
        first_call_s = time.perf_counter() - first_call
        torch.testing.assert_close(actual, native(), rtol=0, atol=0)

        def measure(operation):
            for _ in range(3):
                operation()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            allocated = torch.cuda.memory_allocated()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(args.repeats):
                result = operation()
            end.record()
            torch.cuda.synchronize()
            return {
                "cuda_ms_per_call": begin.elapsed_time(end) / args.repeats,
                "extra_peak_allocated_mib": (torch.cuda.max_memory_allocated() - allocated) / 2**20,
                "prediction_shape": list(result.shape),
            }

        print(json.dumps({
            "kind": "synthetic accuracy microbenchmark, not training",
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "triton": triton.__version__,
            "repeats": args.repeats,
            "first_candidate_call_wall_s_including_possible_compilation": first_call_s,
            "native_contiguous_argmax": measure(native),
            "strided_triton_argmax": measure(candidate),
        }, indent=2))


if __name__ == "__main__":
    main()
