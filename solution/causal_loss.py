"""Retain Transformers' causal-loss contract without a dense logits.float().

The original LOSS_MAPPING and ForCausalLMLoss callable remain untouched: Trainer
uses their identity to count shifted labels. Only matching Qwen2 instances get
a guarded instance loss function through the supported loss_function setter.
"""

from functools import wraps

import torch
import torch.nn.functional as F


def make_causal_loss(original, expected_vocab_size=151936):
    from .vocab_ops import cross_entropy_fp32

    @wraps(original)
    def causal_loss(logits, labels, vocab_size, num_items_in_batch=None,
                    ignore_index=-100, shift_labels=None, **kwargs):
        targets = labels if shift_labels is None else shift_labels
        if (not isinstance(logits, torch.Tensor) or not logits.is_cuda
                or logits.dtype not in (torch.float16, torch.float32)
                or logits.ndim < 2 or not logits.is_contiguous()
                or vocab_size != expected_vocab_size
                or logits.size(-1) != vocab_size
                or not isinstance(targets, torch.Tensor)
                or targets.dtype != torch.long):
            return original(logits, labels, vocab_size,
                            num_items_in_batch=num_items_in_batch,
                            ignore_index=ignore_index, shift_labels=shift_labels, **kwargs)

        if shift_labels is None:
            padded = F.pad(labels, (0, 1), value=ignore_index)
            shift_labels = padded[..., 1:].contiguous()
        flat_logits = logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1).to(logits.device)
        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss = cross_entropy_fp32(flat_logits, flat_labels,
                                  reduction=reduction, ignore_index=ignore_index)
        if num_items_in_batch is not None:
            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(loss.device)
            loss = loss / num_items_in_batch
        return loss

    causal_loss._soup_causal_loss_patch = True
    return causal_loss


def patch_qwen2_causal_loss():
    """Install before model construction; leave other loss functions native."""
    from transformers.loss import loss_utils
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2ForCausalLM
    original_init = cls.__init__
    if getattr(original_init, "_soup_causal_init_patch", False):
        return

    @wraps(original_init)
    def init(self, config, *args, **kwargs):
        original_init(self, config, *args, **kwargs)
        if (getattr(config, "vocab_size", None) == 151936
                and self.loss_function is loss_utils.ForCausalLMLoss):
            self.loss_function = make_causal_loss(self.loss_function)

    init._soup_causal_init_patch = True
    cls.__init__ = init
