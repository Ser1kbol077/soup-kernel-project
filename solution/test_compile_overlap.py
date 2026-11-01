"""Compilation worker lifecycle and warmup specialization descriptions."""

import inspect
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from solution import compile_overlap as overlap


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch):
    # Another integration test may have installed solution.apply() on CUDA.
    # Finish that real worker before replacing its shared diagnostic state.
    overlap.join_compile_overlap()
    state = {"thread": None, "started": False, "joined": False, "seconds": 0.0,
             "compiled": [], "errors": [], "exit_callbacks": []}
    monkeypatch.setattr(overlap, "STATE", state)
    monkeypatch.setattr(overlap, "atexit", SimpleNamespace(register=state["exit_callbacks"].append))
    yield state
    thread = state["thread"]
    if thread is not None:
        thread.join(timeout=3)
        assert not thread.is_alive()


def test_worker_starts_once_and_trainer_waits_before_entering_train(isolated_state):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def worker():
        entered.set()
        assert release.wait(timeout=2)
        finished.set()

    class Trainer:
        def train(self, value, *, resume=False):
            assert finished.is_set()
            return value, resume

    signature = inspect.signature(Trainer.train)
    assert overlap._install_join(Trainer)
    installed = Trainer.train
    assert not overlap._install_join(Trainer)
    assert Trainer.train is installed
    assert inspect.signature(Trainer.train) == signature
    assert overlap._start_worker(worker)
    assert entered.wait(timeout=2)
    assert not overlap._start_worker(lambda: pytest.fail("must not start twice"))
    assert isolated_state["exit_callbacks"] == [overlap.join_compile_overlap]
    assert not isolated_state["joined"]
    release.set()
    assert Trainer().train(7, resume=True) == (7, True)
    assert isolated_state["joined"]
    assert isolated_state["errors"] == []


def test_worker_failure_is_reported_and_native_training_still_runs(isolated_state, capsys):
    def broken():
        raise RuntimeError("compile miss")

    class Trainer:
        def train(self):
            return "native fallback"

    overlap._install_join(Trainer)
    overlap._start_worker(broken)
    assert Trainer().train() == "native fallback"
    assert isolated_state["errors"][0]["kernel"] == "worker"
    assert "compile miss" in isolated_state["errors"][0]["error"]
    assert "1 misses" in capsys.readouterr().out
    assert Trainer().train() == "native fallback"
    assert capsys.readouterr().out == ""


def test_process_exit_joins_a_worker_without_a_trainer(isolated_state):
    finished = threading.Event()
    assert overlap._start_worker(finished.set)
    assert isolated_state["exit_callbacks"] == [overlap.join_compile_overlap]
    isolated_state["exit_callbacks"][0]()
    assert finished.is_set() and isolated_state["joined"]


def test_join_without_started_worker_is_a_noop():
    assert not overlap.join_compile_overlap()


@pytest.mark.parametrize("import_fails", [False, True])
def test_worker_starts_before_lazy_trainer_import_and_remains_joinable(
    monkeypatch, isolated_state, import_fails
):
    entered, finished = threading.Event(), threading.Event()
    order = []

    def worker():
        order.append("compile")
        entered.set()
        finished.set()

    class Trainer:
        def train(self):
            assert finished.is_set()
            return "trained"

    class LazyTransformers(ModuleType):
        def __getattr__(self, name):
            if name != "Trainer":
                raise AttributeError(name)
            assert entered.wait(timeout=2), "compilation must start before Trainer import"
            assert isolated_state["started"]
            assert isolated_state["exit_callbacks"] == [overlap.join_compile_overlap]
            order.append("trainer_import")
            if import_fails:
                raise RuntimeError("Trainer unavailable")
            return Trainer

    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                           get_device_capability=lambda device: (7, 5))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setitem(sys.modules, "transformers", LazyTransformers("transformers"))
    monkeypatch.setattr(overlap, "_build_kernels", worker)
    assert overlap.start_compile_overlap() is (not import_fails)
    assert order[:2] == ["compile", "trainer_import"]
    assert isolated_state["started"] and not isolated_state["joined"]
    if import_fails:
        assert isolated_state["errors"] == [{"kernel": "setup", "error": "RuntimeError('Trainer unavailable')"}]
        isolated_state["exit_callbacks"][0]()
    else:
        installed = Trainer.train
        assert not overlap.start_compile_overlap(Trainer)
        assert Trainer.train is installed
        assert Trainer().train() == "trained"
        assert isolated_state["errors"] == []
    assert isolated_state["joined"]


def test_specifications_cover_tail_pointer_alignment_without_real_tensors(monkeypatch):
    import solution

    def kernel(name):
        return SimpleNamespace(name=name)

    head = SimpleNamespace(CHUNK_ROWS=256, VOCAB_TILE=4096,
                           head_statistics_tile_kernel=kernel("tile"),
                           head_statistics_merge_kernel=kernel("merge"))
    nf4 = SimpleNamespace(BLOCKS_PER_PROGRAM=16, nf4_double_decode_kernel=kernel("nf4"))
    norm = SimpleNamespace(rmsnorm_row_forward_kernel=kernel("norm_fwd"),
                           rmsnorm_row_backward_kernel=kernel("norm_bwd"))
    vocab = SimpleNamespace(VOCAB_TILE=4096, vocab_ce_backward_kernel=kernel("gradient"))
    swiglu = SimpleNamespace(BLOCK=1024, example_swiglu_fwd_kernel=kernel("swiglu_fwd"),
                             example_swiglu_bwd_kernel=kernel("swiglu_bwd"))
    for name, module in (("nf4_decode", nf4), ("rmsnorm", norm),
                         ("streamed_head", head), ("swiglu", swiglu), ("vocab_ops", vocab)):
        monkeypatch.setattr(solution, name, module, raising=False)
    dtypes = SimpleNamespace(float16="half", float32="float", int32="int", int64="long", uint8="byte")

    class Mock:
        def __init__(self, dtype):
            self.dtype = dtype

    specs = list(overlap._compile_specs(dtypes, Mock))
    assert len(specs) == 20
    tile_specs = [spec for spec in specs if spec[1].name == "tile"]
    merge_specs = [spec for spec in specs if spec[1].name == "merge"]
    assert [spec[0] for spec in tile_specs] == ["head_tile_tail0", "head_tile_tail1",
                                               "head_tile_tail2", "head_tile_tail3"]
    # The partial Sum pointer is aligned only for even-sized tails.
    assert [spec[2][3].data_ptr() % 16 for spec in tile_specs] == [0, 8, 0, 8]
    # LogSum lies one row-array past RowMax: every size modulo four matters.
    assert [spec[2][8].data_ptr() % 16 for spec in merge_specs] == [0, 4, 8, 12]
    for name, _, _, _, options in specs:
        assert options == ({} if name.startswith("example_swiglu") else
                           {"num_warps": 4, "enable_fp_fusion": False})
    pointwise = [(name, kernel.name, tuple(arg.dtype for arg in args[:-2]), args[-2:], grid)
                 for name, kernel, args, grid, _ in specs if name.startswith("example_swiglu")]
    assert pointwise == [
        ("example_swiglu_forward_fp32", "swiglu_fwd", ("float", "float", "float"), (8960, 1024), (9,)),
        ("example_swiglu_forward_fp16", "swiglu_fwd", ("half", "half", "half"), (8960, 1024), (9,)),
        ("example_swiglu_backward_fp32", "swiglu_bwd", ("float",) * 5, (8960, 1024), (9,)),
        ("example_swiglu_backward_compact", "swiglu_bwd", ("float", "half", "half", "float", "float"),
         (8960, 1024), (9,)),
        ("example_swiglu_backward_fp16", "swiglu_bwd", ("half",) * 5, (8960, 1024), (9,)),
    ]


def test_cpu_setup_does_not_install_or_start_worker(isolated_state):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CPU guard test")

    class Trainer:
        def train(self):
            return 3

    original = Trainer.train
    assert not overlap.start_compile_overlap(Trainer)
    assert Trainer.train is original
    assert not isolated_state["started"]


def test_cuda_warmup_does_not_launch_or_advance_torch_rng(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA warmup runtime requires T4")
    from solution.streamed_head import head_statistics_tile_kernel

    before_cuda = torch.cuda.get_rng_state().clone()
    before_cpu = torch.random.get_rng_state().clone()
    calls = []
    original = head_statistics_tile_kernel.run

    def checked(*args, **kwargs):
        calls.append(kwargs.get("warmup"))
        assert kwargs.get("warmup") is True
        return original(*args, **kwargs)

    monkeypatch.setattr(head_statistics_tile_kernel, "run", checked)
    from triton.runtime.jit import MockTensor

    tile = next(spec for spec in overlap._compile_specs(torch, MockTensor)
                if spec[0] == "head_tile_tail0")
    _, fn, args, grid, options = tile
    fn.warmup(*args, grid=grid, **options)
    assert calls == [True]
    assert torch.equal(torch.random.get_rng_state(), before_cpu)
    assert torch.equal(torch.cuda.get_rng_state(), before_cuda)
