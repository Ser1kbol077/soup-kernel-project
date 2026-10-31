"""Narrow integration points for the pinned Soup/Transformers/TRL stack.

Label shifting, Trainer bookkeeping, metrics aggregation and the rewind recorder
remain in their original callers. Only the vocabulary reductions are replaced.
"""

from functools import wraps


def patch_vocab_operations():
    """Install idempotent, guarded vocabulary-operation wrappers.

    Imports stay here so importing ``solution`` does not eagerly load optional
    training dependencies. Other model vocabularies, devices and dtypes use the
    original functions, including Soup's CPU regression-test paths.
    """
    import torch
    from soup_cli.trainer import rewind_hf
    from transformers.loss import loss_utils
    from trl.trainer import sft_trainer
    from trl.trainer import utils as trl_utils

    from .vocab_ops import cross_entropy, entropy_from_logits

    def supported(logits, dimensions):
        return (
            isinstance(logits, torch.Tensor)
            and logits.is_cuda
            and logits.dtype == torch.float32
            and logits.ndim == dimensions
            and logits.size(-1) == 151936
            and logits.stride(-1) == 1
        )

    original_ce = loss_utils.fixed_cross_entropy
    if not getattr(original_ce, "_soup_vocab_patch", False):
        @wraps(original_ce)
        def fixed_cross_entropy(
            source, target, num_items_in_batch=None, ignore_index=-100, **kwargs
        ):
            if not supported(source, 2) or not isinstance(target, torch.Tensor):
                return original_ce(
                    source, target, num_items_in_batch=num_items_in_batch,
                    ignore_index=ignore_index, **kwargs,
                )
            reduction = "sum" if num_items_in_batch is not None else "mean"
            loss = cross_entropy(
                source, target, reduction=reduction, ignore_index=ignore_index
            )
            if reduction == "sum":
                if torch.is_tensor(num_items_in_batch):
                    num_items_in_batch = num_items_in_batch.to(loss.device)
                loss = loss / num_items_in_batch
            return loss

        fixed_cross_entropy._soup_vocab_patch = True
        loss_utils.fixed_cross_entropy = fixed_cross_entropy

    def wrap_entropy(original):
        @wraps(original)
        def entropy(logits, chunk_size=128):
            if (not isinstance(logits, torch.Tensor)
                    or logits.ndim < 2
                    or not isinstance(chunk_size, int)
                    or chunk_size <= 0
                    or not supported(logits, logits.ndim)
                    or not logits.is_contiguous()
                    or (torch.is_grad_enabled() and logits.requires_grad)):
                return original(logits, chunk_size=chunk_size)
            return entropy_from_logits(logits, chunk_size=chunk_size)

        entropy._soup_vocab_patch = True
        return entropy

    # SFTTrainer imports the utility by value, so both names must be patched.
    for module in (trl_utils, sft_trainer):
        original = module.entropy_from_logits
        if not getattr(original, "_soup_vocab_patch", False):
            module.entropy_from_logits = wrap_entropy(original)

    original_row_losses = rewind_hf.row_losses
    if not getattr(original_row_losses, "_soup_vocab_patch", False):
        @wraps(original_row_losses)
        def row_losses(logits, labels):
            if (not supported(logits, 3)
                    or not isinstance(labels, torch.Tensor)
                    or labels.ndim != 2
                    or labels.shape != logits.shape[:2]
                    or labels.dtype != torch.long
                    or labels.device != logits.device):
                return original_row_losses(logits, labels)
            row_loss = []
            row_tokens = []
            with torch.no_grad():
                shift_labels = labels[:, 1:]
                for index in range(shift_labels.size(0)):
                    targets = shift_labels[index]
                    token_ce = cross_entropy(
                        logits[index, :-1, :].detach(), targets,
                        reduction="none", ignore_index=-100,
                    ).float()
                    mask = (targets != -100).to(token_ce.dtype)
                    tokens = mask.sum()
                    total = (token_ce * mask).sum()
                    row_loss.append(float(total / tokens.clamp(min=1.0)))
                    row_tokens.append(int(tokens.item()))
            return row_loss, row_tokens

        row_losses._soup_vocab_patch = True
        rewind_hf.row_losses = row_losses
