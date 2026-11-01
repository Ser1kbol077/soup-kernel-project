"""Activation snapshots, parameter aliases, native fallback, and CUDA gradients."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from solution import activation_offload as offload


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_dense_stride_policy_accepts_transpose_and_rejects_overlap_or_gaps():
    assert offload._dense_strides((3, 4), (4, 1))
    assert offload._dense_strides((4, 3), (1, 4))
    assert offload._dense_strides((1, 3, 4), (999, 4, 1))
    assert not offload._dense_strides((3, 4), (0, 1))
    assert not offload._dense_strides((3, 4), (8, 2))
    assert not offload._dense_strides((3, 4), (-4, 1))


def test_cpu_saved_tensors_keep_native_values_and_gradients():
    module = torch.nn.Linear(7, 5)
    x = torch.randn(9, 7, requires_grad=True)
    reference = module(x).square().sum()
    expected = torch.autograd.grad(reference, (x, module.weight, module.bias))
    hooks = offload.SavedTensorOffload(module, min_bytes=1)
    with torch.autograd.graph.saved_tensors_hooks(hooks.pack, hooks.unpack):
        actual = module(x).square().sum()
    got = torch.autograd.grad(actual, (x, module.weight, module.bias))
    assert hooks.bytes == 0
    for first, second in zip(expected, got):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_storage_identity_excludes_nonzero_offset_parameter_views():
    module = torch.nn.Linear(7, 5)
    hooks = offload.SavedTensorOffload(module, min_bytes=1)
    view = module.weight.detach().flatten()[1:]
    assert offload._storage_key(view) in hooks._resident


def test_host_slab_reuse_waits_for_generation_lease_and_preserves_dense_strides(monkeypatch):
    import gc

    allocations = []

    def allocate(nbytes):
        allocations.append(nbytes)
        return torch.empty(nbytes, dtype=torch.uint8)

    monkeypatch.setattr(offload, "_allocate_host", allocate)
    pool = offload._PinnedPool(block_bytes=1024)
    source = torch.randn(8, 4).t()
    first = pool.acquire()
    first_view = first.allocate(source)
    first_view.copy_(source)
    assert first_view.stride() == source.stride()
    second = pool.acquire()
    second_view = second.allocate(source)
    assert first_view.data_ptr() != second_view.data_ptr()
    second_view.zero_()
    torch.testing.assert_close(first_view, source, rtol=0, atol=0)
    first_pointer = first_view.data_ptr()
    del first
    gc.collect()
    third = pool.acquire()
    assert third.allocate(source).data_ptr() == first_pointer
    assert allocations == [1024, 1024]


def test_fragmented_host_blocks_consolidate_only_at_an_idle_graph_boundary(monkeypatch):
    import gc

    monkeypatch.setattr(offload, "_allocate_host", lambda n: torch.empty(n, dtype=torch.uint8))
    pool = offload._PinnedPool(block_bytes=1024)
    first = pool.acquire()
    source = torch.randn(192)
    views = [first.allocate(source) for _ in range(3)]
    for index, view in enumerate(views):
        view.fill_(index)
    assert len(first.arena.blocks) == 3
    arena = first.arena
    del first
    gc.collect()
    next_generation = pool.acquire()
    assert next_generation.arena is arena
    assert len(arena.blocks) == 1 and arena.blocks[0].numel() == 4096
    # The consolidated capacity fits all the old requests without growing.
    for _ in range(3):
        next_generation.allocate(source)
    assert len(arena.blocks) == 1


def test_idle_host_arena_waits_only_for_outstanding_copies_and_never_reuses_a_live_lease():
    import gc

    class Event:
        def __init__(self, done):
            self.done, self.queries, self.waits = done, 0, 0

        def query(self):
            self.queries += 1
            return self.done

        def synchronize(self):
            self.waits += 1
            self.done = True

    pool = offload._PinnedPool(block_bytes=1024)
    first = pool.acquire()
    arena = first.arena
    complete, pending = Event(True), Event(False)
    arena.record_event(complete)
    arena.record_event(pending)
    arena.record_event(pending)  # batch events are deduplicated
    second = pool.acquire()
    assert second.arena is not arena
    assert complete.queries == pending.queries == 0
    assert complete.waits == pending.waits == 0
    del first
    gc.collect()
    reused = pool.acquire()
    assert reused.arena is arena
    assert complete.queries == pending.queries == 1
    assert complete.waits == 0 and pending.waits == 1
    assert not arena.events
    del reused
    gc.collect()
    again = pool.acquire()
    assert again.arena is arena
    assert pending.waits == 1  # no stale wait on the next normal generation


def test_installer_is_idempotent_and_cpu_eval_fallback_is_unchanged():
    class Decoder(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(3.0))

        def forward(self, hidden_states, multiplier=1):
            return hidden_states * self.scale * multiplier

    config = SimpleNamespace(model_type="qwen2", vocab_size=151936, hidden_size=1536,
                             intermediate_size=8960, num_hidden_layers=28)
    module = SimpleNamespace(Qwen2DecoderLayer=Decoder)
    assert offload.patch_qwen2_activation_offload(module)
    assert not offload.patch_qwen2_activation_offload(module)
    layer = Decoder(config)
    x = torch.randn(2, 3, 1536, requires_grad=True)
    assert layer._soup_activation_offload_target
    torch.testing.assert_close(layer(x, multiplier=2), x * 6)
    layer.eval()
    torch.testing.assert_close(layer(x), x * 3)
    config.vocab_size = 23
    assert not Decoder(config)._soup_activation_offload_target


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("transpose", [False, True])
def test_cuda_snapshot_roundtrip_preserves_dtype_shape_stride_and_values(dtype, transpose):
    source = torch.randn(128, 256, dtype=dtype, device="cuda")
    if transpose:
        source = source.t()
    hooks = offload.SavedTensorOffload(torch.nn.Module(), min_bytes=1)
    payload = hooks.pack(source)
    assert isinstance(payload, offload._HostSavedTensor)
    assert payload.host.device.type == "cpu" and payload.host.is_pinned()
    for _ in range(2):  # retain_graph / repeated saved-tensor access
        restored = hooks.unpack(payload)
        assert restored.shape == source.shape and restored.stride() == source.stride()
        assert restored.dtype == source.dtype and restored.device == source.device
        torch.testing.assert_close(restored, source, rtol=0, atol=0)


@requires_cuda
def test_cuda_autograd_matches_native_and_parameters_are_not_offloaded():
    torch.manual_seed(91)
    module = torch.nn.Linear(256, 256, device="cuda")
    x = torch.randn(128, 256, device="cuda", requires_grad=True)
    expected_loss = module(x).sin().square().mean()
    expected = torch.autograd.grad(expected_loss, (x, module.weight, module.bias))
    hooks = offload.SavedTensorOffload(module, min_bytes=1)
    assert not hooks._eligible(module.weight.t())
    with torch.autograd.graph.saved_tensors_hooks(hooks.pack, hooks.unpack):
        actual_loss = module(x).sin().square().mean()
    assert hooks.bytes > 0
    got = torch.autograd.grad(actual_loss, (x, module.weight, module.bias), retain_graph=True)
    repeated = torch.autograd.grad(actual_loss, x)[0]
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for first, second in zip(expected, got):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(repeated, expected[0], rtol=0, atol=0)


@requires_cuda
def test_cuda_weak_cache_budget_and_source_release():
    import gc
    import weakref

    source = torch.randn(512, 512, device="cuda")
    expected = source.clone()
    nbytes = source.numel() * source.element_size()
    hooks = offload.SavedTensorOffload(torch.nn.Module(), min_bytes=1, max_bytes=nbytes)
    payload = hooks.pack(source)
    assert hooks.pack(source) is payload
    source.add_(1)
    assert torch.is_tensor(hooks.pack(source))  # changed version needs another snapshot; budget full
    ref = weakref.ref(source)
    del source
    gc.collect()
    assert ref() is None  # a payload must not keep the GPU activation alive
    torch.testing.assert_close(hooks.unpack(payload), expected, rtol=0, atol=0)


@requires_cuda
def test_cuda_copy_event_orders_restore_on_a_different_stream():
    source = torch.randn(128, 256, device="cuda")
    hooks = offload.SavedTensorOffload(torch.nn.Module(), min_bytes=1)
    payload = hooks.pack(source)
    other = torch.cuda.Stream()
    with torch.cuda.stream(other):
        restored = hooks.unpack(payload)
        result = restored.square()
    torch.cuda.current_stream().wait_stream(other)
    torch.testing.assert_close(result, source.square(), rtol=0, atol=0)


@requires_cuda
def test_cuda_layer_batch_releases_source_refs_and_restores_exact_snapshot():
    source = torch.randn(256, 128, device="cuda")
    expected = source.clone()
    step = offload._Step(source.device)
    batch = offload._LayerBatch(step, 0)
    hooks = offload.SavedTensorOffload(torch.nn.Module(), min_bytes=1, batch=batch)
    payload = hooks.pack(source)
    assert payload.ready is None and batch.sources
    batch.flush()
    assert payload.ready is batch.ready
    del source
    replacement = torch.empty_like(expected).fill_(17)
    step.drain(2, finish=True)
    assert not batch.sources and not step.pending
    torch.testing.assert_close(hooks.unpack(payload), expected, rtol=0, atol=0)
    assert torch.all(replacement == 17)


@requires_cuda
def test_cuda_batched_copies_prefetch_previous_layer_and_preserve_all_gradients():
    torch.manual_seed(113)
    first = torch.nn.Linear(128, 128, device="cuda")
    second = torch.nn.Linear(128, 128, device="cuda")
    x = torch.randn(64, 128, device="cuda", requires_grad=True)
    reference = second(first(x).sin()).sin().square().mean()
    leaves = (x, first.weight, first.bias, second.weight, second.bias)
    expected = torch.autograd.grad(reference, leaves)
    step = offload._Step(x.device)
    batch0, batch1 = offload._LayerBatch(step, 0), offload._LayerBatch(step, 1)
    hooks0 = offload.SavedTensorOffload(first, min_bytes=1, batch=batch0)
    hooks1 = offload.SavedTensorOffload(second, min_bytes=1, batch=batch1)
    with torch.autograd.graph.saved_tensors_hooks(hooks0.pack, hooks0.unpack):
        hidden = first(x).sin()
    batch0.flush()
    with torch.autograd.graph.saved_tensors_hooks(hooks1.pack, hooks1.unpack):
        output = second(hidden).sin()
    batch1.flush()
    step.drain(2, finish=True)
    before = offload.STATS["prefetched_layers"]
    got = torch.autograd.grad(output.square().mean(), leaves, retain_graph=True)
    assert offload.STATS["prefetched_layers"] > before
    repeated = torch.autograd.grad(output.square().mean(), x)[0]
    for actual, wanted in zip(got, expected):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    torch.testing.assert_close(repeated, expected[0], rtol=0, atol=0)


@requires_cuda
def test_cuda_host_pool_reuses_allocations_after_backward_but_not_for_a_live_graph():
    import gc

    module = torch.nn.Linear(128, 128, device="cuda")
    pool = offload._PinnedPool(block_bytes=1 << 20)

    def build(x):
        hooks = offload.SavedTensorOffload(module, min_bytes=1, pool=pool)
        with torch.autograd.graph.saved_tensors_hooks(hooks.pack, hooks.unpack):
            return module(x).sin().square().mean()

    x = torch.randn(64, 128, device="cuda", requires_grad=True)
    first = build(x)
    allocation_count = offload.STATS["host_allocations"]
    expected = torch.autograd.grad(module(x).sin().square().mean(), x)[0]
    first.backward()
    torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)
    del first
    gc.collect()
    second = build(x)
    assert offload.STATS["host_allocations"] == allocation_count
    # The live second graph leases its slab; a third graph must use another.
    third = build(x * 2)
    assert offload.STATS["host_allocations"] > allocation_count
    torch.autograd.grad(second, x)
    torch.autograd.grad(third, x)


@requires_cuda
def test_t4_decoder_wrapper_activates_and_preserves_parameter_and_input_gradients():
    if torch.cuda.get_device_capability() != (7, 5):
        pytest.skip("production wrapper targets T4")

    class Decoder(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.5, device="cuda"))

        def forward(self, hidden_states):
            return hidden_states.sin() * self.scale

    config = SimpleNamespace(model_type="qwen2", vocab_size=151936, hidden_size=1536,
                             intermediate_size=8960, num_hidden_layers=28)
    native = Decoder(config)
    x = torch.randn(4, 64, 1536, device="cuda", requires_grad=True)
    expected_loss = native(x).square().mean()
    expected = torch.autograd.grad(expected_loss, (x, native.scale))
    assert offload.patch_qwen2_activation_offload(SimpleNamespace(Qwen2DecoderLayer=Decoder))
    patched = Decoder(config)
    before = offload.STATS["offloaded_tensors"]
    loss = patched(x).square().mean()
    assert offload.STATS["offloaded_tensors"] > before
    got = torch.autograd.grad(loss, (x, patched.scale))
    for first, second in zip(expected, got):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    before = offload.STATS["offloaded_tensors"]
    with torch.no_grad():
        patched(x)
    assert offload.STATS["offloaded_tensors"] == before
