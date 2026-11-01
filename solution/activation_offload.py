"""Save large decoder activations in pinned CPU RAM without changing their values.

Only the fixed Qwen2 training path is wrapped. The model still runs the same
forward and backward kernels with the same Trainer settings. Saved parameter
and buffer aliases stay on the GPU; other large, dense saved tensors are copied
to the host and restored on demand. This initial implementation has no backward
prefetch: its purpose is a reviewable memory-saving candidate for the T4, whose
end-to-end time must still be measured.

Transfers use the current compute stream, rather than a second CUDA stream.
The source allocation is recorded on that stream before it may be released.
An event orders a later restore even if autograd unpacks on another stream.
No input values, dtypes, optimizer settings, or allocator options are changed.
"""

from dataclasses import dataclass
from functools import wraps
import weakref

import torch


MIN_BYTES = 512 << 10
MAX_LAYER_BYTES = 256 << 20
STATS = {"offloaded_tensors": 0, "offloaded_bytes": 0, "restored_tensors": 0}


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


@dataclass(frozen=True)
class _HostSavedTensor:
    host: object
    device: object
    ready: object


class SavedTensorOffload:
    """Per-forward hooks. Payloads own host snapshots, never the source GPU tensor.

    A weak identity cache avoids copying an identical tensor more than once in
    one layer. Its version is part of the lookup so an intervening in-place
    update cannot reuse an older snapshot.
    """

    def __init__(self, module, min_bytes=MIN_BYTES, max_bytes=MAX_LAYER_BYTES):
        self.min_bytes = min_bytes
        self.max_bytes = max_bytes
        self.bytes = 0
        self._cache = {}
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
        host = torch.empty_strided(
            tensor.shape, tensor.stride(), dtype=tensor.dtype,
            device="cpu", pin_memory=True,
        )
        with torch.cuda.device(tensor.device):
            stream = torch.cuda.current_stream(tensor.device)
            # record_stream protects the GPU allocation when its last Python
            # reference disappears before this queued device-to-host copy.
            tensor.record_stream(stream)
            host.copy_(tensor.detach(), non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(stream)
        payload = _HostSavedTensor(host, tensor.device, ready)
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
            stream.wait_event(payload.ready)
            restored = torch.empty_strided(
                payload.host.shape, payload.host.stride(),
                dtype=payload.host.dtype, device=payload.device,
            )
            restored.copy_(payload.host, non_blocking=True)
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

    @wraps(original_forward)
    def forward(self, hidden_states, *args, **kwargs):
        if not _supported_call(self, hidden_states):
            return original_forward(self, hidden_states, *args, **kwargs)
        hooks = SavedTensorOffload(self)
        with torch.autograd.graph.saved_tensors_hooks(hooks.pack, hooks.unpack):
            return original_forward(self, hidden_states, *args, **kwargs)

    forward._soup_activation_offload_patch = True
    cls.__init__, cls.forward = init, forward
    return True
