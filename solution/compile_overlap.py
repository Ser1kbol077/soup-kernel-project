"""Compile our fixed-task kernels while the model loads, without launching them.

Triton's documented JIT ``warmup`` accepts MockTensor pointer descriptions and
sets ``warmup=True``: compilation runs, execution does not. The same process,
device and existing compiler cache are used; all elapsed time remains inside
the harness wall clock. No Torch random draws, allocator changes or external
processes are involved. The worker is joined before Trainer.train starts.
Missing variants or compilation failures retain normal first-use compilation.
"""

import atexit
from functools import wraps
import threading
import time


STATE = {"thread": None, "started": False, "joined": False, "seconds": 0.0,
         "compiled": [], "errors": []}
_LOCK = threading.Lock()
_MARKER = "_soup_compile_overlap_patch"


def _compile_specs(torch, mock_type):
    """Descriptions match the pointers/constexprs of actual fixed-task calls.

    Partial statistics are sliced from contiguous arrays. Tail chunk sizes
    modulo four change pointer alignment, so compile those useful variants too.
    The mock pointer addresses are only specialization metadata; none is read.
    """
    from . import nf4_decode, rmsnorm, streamed_head, vocab_ops

    class Pointer(mock_type):
        def __init__(self, dtype, byte_offset=0):
            super().__init__(dtype)
            self.byte_offset = byte_offset

        def data_ptr(self):
            return self.byte_offset

    def ptr(dtype, offset=0):
        return Pointer(dtype, offset)

    f16, f32, i32, i64 = torch.float16, torch.float32, torch.int32, torch.int64
    vocab, parts = 151936, 38
    options = {"num_warps": 4, "enable_fp_fusion": False}
    for rows in (streamed_head.CHUNK_ROWS, 1, 2, 3):
        partial = [ptr(f32, index * rows * parts * 4) for index in range(4)]
        scalar = [ptr(f32, index * rows * 4) for index in range(5)]
        maximum, logsum, loss, entropy, bound = scalar
        logits, labels, indices, prediction = ptr(f16), ptr(i64), ptr(i32), ptr(i64)
        yield (f"head_tile_tail{rows % 4}", streamed_head.head_statistics_tile_kernel,
               (logits, labels, partial[0], partial[1], partial[2], indices, partial[3],
                vocab, parts, streamed_head.VOCAB_TILE), (rows, parts), options)
        yield (f"head_merge_tail{rows % 4}", streamed_head.head_statistics_merge_kernel,
               (logits, labels, partial[0], partial[1], partial[2], indices, partial[3],
                maximum, logsum, loss, entropy, prediction, bound,
                vocab, parts, -100, 64), (rows,), options)
        yield (f"head_gradient_tail{rows % 4}", vocab_ops.vocab_ce_backward_kernel,
               (logits, labels, maximum, logsum, ptr(f32), logits,
                vocab, vocab, -100, vocab_ops.VOCAB_TILE), (rows, parts), options)
    yield ("nf4_decode", nf4_decode.nf4_double_decode_kernel,
           (ptr(torch.uint8), ptr(torch.uint8), ptr(f32), ptr(f32), ptr(f32),
            ptr(f32), ptr(f16), 1536, nf4_decode.BLOCKS_PER_PROGRAM), (1,), options)
    yield ("rmsnorm_forward", rmsnorm.rmsnorm_row_forward_kernel,
           (ptr(f32), ptr(f32), ptr(f32), ptr(f32), 1536, 1e-6, 2048), (1,), options)
    yield ("rmsnorm_backward", rmsnorm.rmsnorm_row_backward_kernel,
           (ptr(f32), ptr(f32), ptr(f32), ptr(f32), ptr(f32), 1536, 2048), (1,), options)


def _build_kernels():
    import torch
    from triton.runtime.jit import MockTensor

    with torch.cuda.device(0):
        for name, kernel, args, grid, options in _compile_specs(torch, MockTensor):
            try:
                kernel.warmup(*args, grid=grid, **options)
                STATE["compiled"].append(name)
            except Exception as error:
                STATE["errors"].append({"kernel": name, "error": repr(error)})


def _start_worker(worker):
    """Start once, including after a previous worker finished or failed."""
    with _LOCK:
        if STATE["started"]:
            return False
        STATE["started"] = True

        def work():
            started = time.perf_counter()
            try:
                worker()
            except Exception as error:
                STATE["errors"].append({"kernel": "worker", "error": repr(error)})
            finally:
                STATE["seconds"] = round(time.perf_counter() - started, 3)

        thread = threading.Thread(target=work, name="soup-kernel-compile", daemon=True)
        STATE["thread"] = thread
        thread.start()
        # Preflight/regression processes may install patches without ever
        # constructing a trainer. Finish before Torch/Triton module teardown.
        atexit.register(join_compile_overlap)
        return True


def join_compile_overlap():
    """Wait for compilation before training; misses remain ordinary JIT calls."""
    thread = STATE["thread"]
    if thread is None:
        return False
    thread.join()
    with _LOCK:
        first_join = not STATE["joined"]
        STATE["joined"] = True
    if first_join:
        print(f"[solution] compile overlap: {len(STATE['compiled'])} warmups, "
              f"{STATE['seconds']:.3f} s, {len(STATE['errors'])} misses", flush=True)
    return True


def _install_join(trainer_cls):
    original = trainer_cls.train
    if getattr(original, _MARKER, False):
        return False

    @wraps(original)
    def train(self, *args, **kwargs):
        join_compile_overlap()
        return original(self, *args, **kwargs)

    setattr(train, _MARKER, True)
    trainer_cls.train = train
    return True


def start_compile_overlap(trainer_cls=None):
    """Enable only on the challenge's single visible T4, before model loading."""
    try:
        import torch

        if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
                or torch.cuda.get_device_capability(0) != (7, 5)):
            return False
        if trainer_cls is None:
            from transformers import Trainer

            trainer_cls = Trainer
        _install_join(trainer_cls)
        return _start_worker(_build_kernels)
    except Exception as error:
        STATE["errors"].append({"kernel": "setup", "error": repr(error)})
        return False
