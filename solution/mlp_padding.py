"""Avoid frozen MLP work on padding without changing attention or LoRA RNG.

The decoder keeps its original batch layout and attention mask. Only the
token-wise, dropout-free frozen MLP is gathered and scattered. Positions used
by supervised loss are retained even if their attention mask is zero.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps

import torch
import torch.nn.functional as F


_ACTIVE = ContextVar("soup_mlp_valid_rows", default=None)


@dataclass(frozen=True)
class _Rows:
    shape: tuple
    indices: object


@contextmanager
def active_mlp_rows(attention_mask, labels, enabled=True):
    rows = None
    if (enabled and isinstance(labels, torch.Tensor) and labels.ndim == 2
            and isinstance(attention_mask, torch.Tensor)
            and attention_mask.shape == labels.shape
            and attention_mask.device == labels.device):
        targets = F.pad(labels, (0, 1), value=-100)[..., 1:]
        observed = (attention_mask != 0) | (targets != -100)
        indices = observed.reshape(-1).nonzero(as_tuple=False).flatten()
        if 0 < indices.numel() < labels.numel():
            rows = _Rows(tuple(labels.shape), indices)
    token = _ACTIVE.set(rows)
    try:
        yield rows
    finally:
        _ACTIVE.reset(token)


def _packed_call(original, module, hidden, rows):
    width = hidden.shape[-1]
    compact = hidden.reshape(-1, width).index_select(0, rows.indices).unsqueeze(0)
    computed = original(module, compact)
    result = computed.new_zeros((hidden.numel() // width, width))
    result.index_copy_(0, rows.indices, computed.reshape(-1, width))
    return result.view_as(hidden)


def _eligible(module, hidden, rows, linear4bit_type=None):
    if (rows is None or not isinstance(hidden, torch.Tensor) or not hidden.is_cuda
            or not module.training or not torch.is_grad_enabled()
            or hidden.ndim != 3 or tuple(hidden.shape[:2]) != rows.shape
            or hidden.shape[-1] != 1536 or hidden.device != rows.indices.device
            or getattr(getattr(module, "config", None), "hidden_act", None) != "silu"
            or hasattr(module, "_hf_hook")
            or module._forward_hooks or module._forward_pre_hooks):
        return False
    for name, shape in (("gate_proj", (1536, 8960)), ("up_proj", (1536, 8960)),
                        ("down_proj", (8960, 1536))):
        projection = getattr(module, name, None)
        if (linear4bit_type is None or type(projection) is not linear4bit_type
                or (projection.in_features, projection.out_features) != shape
                or getattr(projection, "bias", None) is not None
                or "forward" in projection.__dict__
                or hasattr(projection, "_hf_hook")
                or getattr(projection, "_forward_hooks", None)
                or getattr(projection, "_forward_pre_hooks", None)
                or any(p.requires_grad for p in projection.parameters())):
            return False
    return True


def patch_qwen2_mlp_padding():
    """Install after existing MLP optimizations, leaving unrelated calls native."""
    from transformers.models.qwen2 import modeling_qwen2
    from bitsandbytes.nn import Linear4bit

    cls = modeling_qwen2.Qwen2MLP
    original = cls.forward
    if getattr(original, "_soup_mlp_padding_patch", False):
        return False

    @wraps(original)
    def forward(self, hidden_states):
        rows = _ACTIVE.get()
        if type(self) is not cls or not _eligible(self, hidden_states, rows, Linear4bit):
            return original(self, hidden_states)
        return _packed_call(original, self, hidden_states, rows)

    forward._soup_mlp_padding_patch = True
    cls.forward = forward
    return True
