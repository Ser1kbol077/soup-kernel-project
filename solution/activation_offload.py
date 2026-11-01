"""Save large decoder activations in pinned CPU RAM without changing their values.

Large saved tensors use reusable pinned host slabs. Decoder-layer copies are
batched onto a side stream, while the next layer computes. Backward restores
the current layer and prefetches the preceding layer on another side stream.
Generation leases keep retained graphs from reusing each other's host storage.
No input values, dtypes, Trainer flags, or allocator options are changed.
"""

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import weakref

import torch


MIN_BYTES = 512 << 10
MAX_LAYER_BYTES = 256 << 20
HOST_BLOCK_BYTES = 16 << 20
STATS = {"offloaded_tensors": 0, "offloaded_bytes": 0, "restored_tensors": 0,
         "host_allocations": 0, "host_allocated_bytes": 0, "prefetched_layers": 0}
_RESIDENT_ACTIVATIONS = ContextVar("soup_resident_decoder_activations", default=False)


@contextmanager
def resident_decoder_activations(enabled=True):
    """Keep the original decoder graph on GPU within a bounded caller scope."""
    token = _RESIDENT_ACTIVATIONS.set(enabled)
    try:
        yield
    finally:
        _RESIDENT_ACTIVATIONS.reset(token)


def _allocate_host(nbytes):
    STATS["host_allocations"] += 1
    STATS["host_allocated_bytes"] += nbytes
    return torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=True)


class _Arena:
    def __init__(self, block_bytes):
        self.block_bytes = block_bytes
        self.blocks = []
        self.offsets = []
        self.owner = None
        self.used = 0
        self.events = {}

    def record_event(self, event):
        # A layer batch shares its event across many entries of this slab.
        self.events[id(event)] = event

    def wait_for_transfers(self):
        # Graph lifetime and CUDA-copy lifetime differ when a partial graph is
        # discarded. A dead lease alone does not authorize overwriting pinned
        # memory that an upload stream may still be reading.
        for event in self.events.values():
            if not event.query():
                event.synchronize()
        self.events.clear()

    def reset(self):
        self.wait_for_transfers()
        # At a graph boundary fragmented first-use blocks can become one
        # reusable high-water slab. Geometric capacity avoids growing a new
        # pinned block for every slightly longer dynamic sequence length.
        if len(self.blocks) > 1:
            size = max(self.block_bytes, 1 << max(0, self.used - 1).bit_length())
            self.blocks = [_allocate_host(size)]
        self.offsets = [0] * len(self.blocks)
        self.used = 0

    def allocate(self, shape, stride, dtype, nbytes):
        for index, (block, offset) in enumerate(zip(self.blocks, self.offsets)):
            if block.numel() - offset >= nbytes:
                break
        else:
            index = len(self.blocks)
            size = max(self.block_bytes, 1 << max(0, nbytes - 1).bit_length())
            block, offset = _allocate_host(size), 0
            self.blocks.append(block)
            self.offsets.append(0)
        self.offsets[index] = (offset + nbytes + 511) & ~511
        self.used += (nbytes + 511) & ~511
        return block[offset:offset + nbytes].view(dtype).as_strided(shape, stride)


class _Lease:
    def __init__(self, arena):
        self.arena = arena

    def allocate(self, tensor):
        return self.arena.allocate(tensor.shape, tensor.stride(), tensor.dtype,
                                   tensor.numel() * tensor.element_size())


class _PinnedPool:
    """One layer's high-water host buffers, reused only after its graph releases them."""
    def __init__(self, block_bytes=HOST_BLOCK_BYTES):
        self.block_bytes = block_bytes
        self.arenas = []

    def acquire(self):
        for arena in self.arenas:
            if arena.owner is None or arena.owner() is None:
                break
        else:
            arena = _Arena(self.block_bytes)
            self.arenas.append(arena)
        arena.reset()
        lease = _Lease(arena)
        arena.owner = weakref.ref(lease)
        return lease


_STREAMS = {}


class _Step:
    def __init__(self, device):
        if device not in _STREAMS:
            with torch.cuda.device(device):
                _STREAMS[device] = (torch.cuda.Stream(device=device), torch.cuda.Stream(device=device))
        self.down, self.up = _STREAMS[device]
        self.layers = {}
        self.pending = deque()
        self.backward_seen = set()

    def drain(self, index, finish=False):
        while self.pending:
            batch = self.pending[0]
            done = batch.ready.query()
            if not done and not finish and batch.index > index - 2:
                break
            if not done:
                batch.ready.synchronize()
            batch.sources.clear()
            self.pending.popleft()

    def backward(self, index):
        if index in self.backward_seen:
            return
        self.backward_seen.add(index)
        previous = self.layers.get(index - 1)
        previous = previous() if previous is not None else None
        if previous is not None:
            previous.prefetch()
            STATS["prefetched_layers"] += 1


class _LayerBatch:
    def __init__(self, step, index):
        self.step, self.index = step, index
        self.entries = []
        self.sources = []
        self.ready = None
        step.layers[index] = weakref.ref(self)

    def flush(self):
        if not self.sources:
            return
        stream = self.step.down
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for payload, source, version in self.sources:
                if source._version != version:
                    raise RuntimeError("a saved decoder activation changed before its offload copy")
                source.record_stream(stream)
                payload.host.copy_(source, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(stream)
        self.ready = ready
        for payload, _, _ in self.sources:
            payload.ready = ready
            payload.lease.arena.record_event(ready)
        self.step.pending.append(self)

    def prefetch(self):
        live = [entry() for entry in self.entries]
        live = [entry for entry in live if entry is not None and entry.restored is None
                and (entry.restored_ref is None or entry.restored_ref() is None)]
        if not live:
            return
        compute = torch.cuda.current_stream(live[0].device)
        stream = self.step.up
        stream.wait_stream(compute)
        stream.wait_event(self.ready)
        # Allocate on the consuming stream, then record the upload stream too.
        buffers = [torch.empty_strided(entry.host.shape, entry.host.stride(),
                                      dtype=entry.host.dtype, device=entry.device) for entry in live]
        with torch.cuda.stream(stream):
            for entry, buffer in zip(live, buffers):
                buffer.copy_(entry.host, non_blocking=True)
                buffer.record_stream(stream)
            ready = torch.cuda.Event()
            ready.record(stream)
        for entry, buffer in zip(live, buffers):
            entry.restored, entry.restored_ready = buffer, ready
            entry.lease.arena.record_event(ready)


# The fixed task has one sequential decoder stack per process. Individual
# batches/leases remain owned by their graph, so retained graphs stay valid.
_FORWARD = {"step": None}


def _storage_key(tensor):
    """Identify parameter aliases even when a saved view has a different offset."""
    return tensor.device, tensor.untyped_storage().data_ptr()


def _dense_strides(shape, stride):
    """Accept contiguous tensors and dense permutations, without overlapping views."""
    expected = 1
    for step, size in sorted((st, n) for n, st in zip(shape, stride) if n > 1):
        if step != expected:
            return False
        expected *= size
    return True


@dataclass
class _HostSavedTensor:
    host: object
    device: object
    ready: object
    lease: object = None
    batch: object = None
    restored: object = None
    restored_ref: object = None
    restored_ready: object = None


class SavedTensorOffload:
    """Per-forward hooks. Payloads own host snapshots, never the source GPU tensor.

    A weak identity cache avoids copying an identical tensor more than once in
    one layer. Its version is part of the lookup so an intervening in-place
    update cannot reuse an older snapshot.
    """

    def __init__(self, module, min_bytes=MIN_BYTES, max_bytes=MAX_LAYER_BYTES,
                 *, pool=None, batch=None):
        self.min_bytes = min_bytes
        self.max_bytes = max_bytes
        self.bytes = 0
        self._cache = {}
        self.lease = (pool or _PinnedPool()).acquire()
        self.batch = batch
        self._resident = {
            _storage_key(t)
            for t in (*module.parameters(), *module.buffers())
            if t.layout == torch.strided and t.device.type != "meta" and t.numel()
        }

    def _eligible(self, tensor):
        return (
            type(tensor) is torch.Tensor
            and tensor.is_cuda
            and tensor.layout == torch.strided
            and not tensor.is_quantized
            and not tensor.is_conj()
            and not tensor.is_neg()
            and tensor.numel() * tensor.element_size() >= self.min_bytes
            and _dense_strides(tensor.shape, tensor.stride())
            and _storage_key(tensor) not in self._resident
        )

    def pack(self, tensor):
        if not self._eligible(tensor):
            return tensor.detach()
        version = tensor._version
        cached = self._cache.get(id(tensor))
        if cached is not None and cached[0]() is tensor and cached[1] == version:
            return cached[2]
        nbytes = tensor.numel() * tensor.element_size()
        if self.bytes + nbytes > self.max_bytes:
            return tensor.detach()
        host = self.lease.allocate(tensor)
        payload = _HostSavedTensor(host, tensor.device, None, self.lease, self.batch)
        if self.batch is None:
            # Standalone hooks snapshot immediately, including callers that
            # subsequently mutate a source tensor on its current stream.
            with torch.cuda.device(tensor.device):
                stream = torch.cuda.current_stream(tensor.device)
                tensor.record_stream(stream)
                host.copy_(tensor.detach(), non_blocking=True)
                payload.ready = torch.cuda.Event()
                payload.ready.record(stream)
                self.lease.arena.record_event(payload.ready)
        else:
            self.batch.entries.append(weakref.ref(payload))
            self.batch.sources.append((payload, tensor.detach(), version))
        self._cache[id(tensor)] = weakref.ref(tensor), version, payload
        self.bytes += nbytes
        STATS["offloaded_tensors"] += 1
        STATS["offloaded_bytes"] += nbytes
        return payload

    @staticmethod
    def unpack(payload):
        if not isinstance(payload, _HostSavedTensor):
            return payload
        with torch.cuda.device(payload.device):
            stream = torch.cuda.current_stream(payload.device)
            if payload.batch is not None:
                batch = payload.batch
                existing = payload.restored if payload.restored is not None else (
                    payload.restored_ref() if payload.restored_ref is not None else None)
                if existing is None:
                    batch.prefetch()
                    existing = payload.restored
                stream.wait_event(payload.restored_ready)
                existing.record_stream(stream)
                payload.restored = None
                payload.restored_ref = weakref.ref(existing)
                batch.step.backward(batch.index)
                STATS["restored_tensors"] += 1
                return existing
            stream.wait_event(payload.ready)
            restored = torch.empty_strided(
                payload.host.shape, payload.host.stride(),
                dtype=payload.host.dtype, device=payload.device,
            )
            restored.copy_(payload.host, non_blocking=True)
            finished = torch.cuda.Event()
            finished.record(stream)
            payload.lease.arena.record_event(finished)
        STATS["restored_tensors"] += 1
        return restored


def _target_config(config):
    return (
        getattr(config, "model_type", None) == "qwen2"
        and getattr(config, "vocab_size", None) == 151936
        and getattr(config, "hidden_size", None) == 1536
        and getattr(config, "intermediate_size", None) == 8960
        and getattr(config, "num_hidden_layers", None) == 28
    )


def _supported_call(layer, hidden_states):
    return (
        getattr(layer, "_soup_activation_offload_target", False)
        and layer.training
        and torch.is_grad_enabled()
        and type(hidden_states) is torch.Tensor
        and hidden_states.is_cuda
        and hidden_states.ndim == 3
        and hidden_states.shape[-1] == 1536
        and hidden_states.dtype in (torch.float16, torch.float32)
        and torch.cuda.get_device_capability(hidden_states.device) == (7, 5)
        and not hasattr(layer, "_hf_hook")
    )


def patch_qwen2_activation_offload(modeling_qwen2=None):
    """Install before model construction; other models, devices and eval stay native."""
    if modeling_qwen2 is None:
        from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2DecoderLayer
    original_init, original_forward = cls.__init__, cls.forward
    if getattr(original_forward, "_soup_activation_offload_patch", False):
        return False

    @wraps(original_init)
    def init(self, config, *args, **kwargs):
        original_init(self, config, *args, **kwargs)
        self._soup_activation_offload_target = _target_config(config)
        self._soup_offload_index = kwargs.get("layer_idx", args[0] if args else None)
        self._soup_offload_layers = getattr(config, "num_hidden_layers", 0)
        self._soup_offload_pool = _PinnedPool()

    @wraps(original_forward)
    def forward(self, hidden_states, *args, **kwargs):
        if _RESIDENT_ACTIVATIONS.get() or not _supported_call(self, hidden_states):
            return original_forward(self, hidden_states, *args, **kwargs)
        index = self._soup_offload_index
        # An isolated layer (including tests) owns its own short-lived step.
        isolated = index is None
        if isolated or index == 0 or _FORWARD["step"] is None:
            _FORWARD["step"] = _Step(hidden_states.device)
        step = _FORWARD["step"]
        index = 0 if isolated else index
        step.drain(index)
        batch = _LayerBatch(step, index)
        hooks = SavedTensorOffload(self, pool=self._soup_offload_pool, batch=batch)
        try:
            with torch.autograd.graph.saved_tensors_hooks(hooks.pack, hooks.unpack):
                result = original_forward(self, hidden_states, *args, **kwargs)
            batch.flush()
        except BaseException:
            batch.sources.clear()
            step.drain(index, finish=True)
            _FORWARD["step"] = None
            raise
        if isolated or index == self._soup_offload_layers - 1:
            step.drain(index, finish=True)
            _FORWARD["step"] = None
        return result

    forward._soup_activation_offload_patch = True
    cls.__init__, cls.forward = init, forward
    return True
