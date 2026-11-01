"""Connect our bounded-memory head to the pinned Qwen2/TRL/Soup training path.

Only an eligible SFT training call requests logits-free execution. Generation,
evaluation, custom losses and unsupported models keep their installed forwards.
Transformers' Trainer still owns loss scaling; Soup's existing ChunkedStep seam
owns rewind recording and its existing helper logs TRL's metrics.
"""

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from importlib import metadata

import torch


_MARKER = "_soup_streamed_head_patch"
_REQUEST = ContextVar("soup_streamed_head_request", default=None)
_INPUT_KEYS = frozenset({"input_ids", "labels", "attention_mask", "position_ids",
                         "use_cache", "_prediction_loss_only", "logits_to_keep"})


@dataclass
class _Request:
    model: object
    stats: object = None


def _base_model(model):
    get_base = getattr(model, "get_base_model", None)
    return get_base() if callable(get_base) else model


def _eligible_trainer(trainer, model, inputs, causal_cls):
    """Guard before bypassing TRL's logits-reading compute_loss method."""
    if (not getattr(model, "training", False) or not torch.is_grad_enabled()
            or not isinstance(inputs, dict) or not set(inputs).issubset(_INPUT_KEYS)
            or getattr(trainer, "label_smoother", None) is not None
            or getattr(trainer, "compute_loss_func", None) is not None
            or getattr(trainer, "num_virtual_tokens", 0) != 0
            or getattr(trainer, "aux_loss_enabled", False)):
        return False
    args = getattr(trainer, "args", None)
    accelerator = getattr(trainer, "accelerator", None)
    if (getattr(args, "loss_type", None) != "nll"
            or getattr(args, "use_liger_kernel", False)
            or getattr(args, "padding_free", False)
            or getattr(args, "n_gpu", 1) > 1
            or getattr(accelerator, "num_processes", 1) != 1
            or getattr(accelerator, "parallelism_config", None) is not None):
        return False
    causal = _base_model(model)
    config = getattr(causal, "config", None)
    head = getattr(causal, "lm_head", None)
    if (type(causal) is not causal_cls
            or getattr(config, "vocab_size", None) != 151936
            or getattr(config, "hidden_size", None) != 1536
            or getattr(config, "output_router_logits", False)
            or type(head) is not torch.nn.Linear
            or head.bias is not None or head.weight.requires_grad
            or tuple(head.weight.shape) != (151936, 1536)
            or not head.weight.is_cuda
            or head.weight.dtype not in (torch.float16, torch.float32)
            or head._forward_hooks or head._forward_pre_hooks):
        return False
    labels, ids, mask = (inputs.get(key) for key in ("labels", "input_ids", "attention_mask"))
    if (not all(isinstance(t, torch.Tensor) for t in (labels, ids, mask))
            or labels.ndim != 2 or not labels.numel()
            or labels.dtype != torch.long or ids.dtype != torch.long
            or ids.shape != labels.shape or mask.shape != labels.shape
            or not labels.is_cuda or ids.device != labels.device
            or mask.device != labels.device or head.weight.device != labels.device
            or inputs.get("use_cache", False) not in (False, None)
            or type(inputs.get("logits_to_keep", 0)) is not int
            or inputs.get("logits_to_keep", 0) != 0):
        return False
    return True


def _eligible_forward(model, request, labels, past_key_values, inputs_embeds,
                      logits_to_keep, kwargs):
    return (
        request is not None and request.model is model
        and model.training and torch.is_grad_enabled()
        and isinstance(labels, torch.Tensor) and labels.is_cuda
        and labels.dtype == torch.long and labels.ndim == 2
        and past_key_values is None and inputs_embeds is None
        and type(logits_to_keep) is int and logits_to_keep == 0
        and kwargs.get("return_dict", True) is not False
        and not kwargs.get("output_attentions", False)
        and not kwargs.get("output_hidden_states", False)
        and kwargs.get("shift_labels") is None
        and (not torch.is_autocast_enabled("cuda")
             or torch.get_autocast_dtype("cuda") == torch.float16)
    )


def _active_head_loss(head_loss, hidden, weight, labels, denominator, attention_mask, rows=None):
    """Skip head rows neither observed by metrics nor used by supervised loss.

    Shift labels before gathering: concatenated batch rows must never acquire
    next-token targets from their neighbours. Decoder/dropout execution is
    unchanged, so this introduces no extra RNG draws or attention changes.
    """
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != labels.shape:
        return head_loss(hidden, weight, labels, num_items_in_batch=denominator,
                         attention_mask=attention_mask)
    targets = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
    if rows is None:
        observed = (attention_mask != 0) | (targets != -100)
        indices = observed.reshape(-1).nonzero(as_tuple=False).flatten()
    else:
        indices = rows.indices
    if indices.numel() == 0 or indices.numel() == labels.numel():
        return head_loss(hidden, weight, labels, num_items_in_batch=denominator,
                         attention_mask=attention_mask)
    packed = hidden.reshape(-1, hidden.shape[-1]).index_select(0, indices).unsqueeze(0)
    packed_targets = targets.reshape(-1).index_select(0, indices).unsqueeze(0)
    loss, compact = head_loss(packed, weight, None, num_items_in_batch=denominator,
                              shift_labels=packed_targets)
    stats = {}
    for name, value in compact.items():
        full = value.new_zeros(labels.numel())
        full.index_copy_(0, indices, value.reshape(-1))
        stats[name] = full.view_as(labels)
    return loss, stats


def _resident_decoder_call(model, labels):
    # The bounded head removes vocabulary-sized retained activations. Spend
    # that memory saving on the fixed task's decoder instead of PCIe copies.
    # Larger calls retain the existing offload policy until measured on T4.
    return (1 <= labels.shape[0] <= 4 and 1 <= labels.shape[1] <= 512
            and not getattr(model.model, "gradient_checkpointing", False))


def _make_forward(original, head_loss, output_type):
    @wraps(original)
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, logits_to_keep=0, **kwargs):
        request = _REQUEST.get()
        if not _eligible_forward(self, request, labels, past_key_values,
                                 inputs_embeds, logits_to_keep, kwargs):
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, labels=labels,
                            use_cache=use_cache, logits_to_keep=logits_to_keep, **kwargs)
        from .mlp_padding import active_mlp_rows
        from .activation_offload import resident_decoder_activations

        with active_mlp_rows(attention_mask, labels,
                             enabled=not getattr(self.model, "gradient_checkpointing", False)) as rows:
            with resident_decoder_activations(enabled=_resident_decoder_call(self, labels)):
                decoder = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                     position_ids=position_ids, past_key_values=past_key_values,
                                     inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
            loss, stats = _active_head_loss(
                head_loss, decoder.last_hidden_state, self.lm_head.weight, labels,
                kwargs.get("num_items_in_batch"), attention_mask, rows,
            )
        request.stats = stats
        return output_type(loss=loss, logits=None,
                           past_key_values=getattr(decoder, "past_key_values", None),
                           hidden_states=getattr(decoder, "hidden_states", None),
                           attentions=getattr(decoder, "attentions", None))

    setattr(forward, _MARKER, True)
    return forward


def _step_from_stats(stats, labels, step_type):
    targets = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
    labelled = targets != -100
    correct = ((stats["predictions"] == targets) & labelled).sum()
    return step_type(token_entropy=stats["entropy"], token_loss=stats["token_loss"],
                     correct=correct, labelled=labelled.sum())


def _make_compute_loss(original, trainer_compute_loss, causal_cls, step_type,
                       log_metrics, step_attr, entropy_from_logits):
    @wraps(original)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not _eligible_trainer(self, model, inputs, causal_cls):
            # A stale chunked record must never be mistaken for this step.
            setattr(self, step_attr, None)
            return original(self, model, inputs, return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)
        labels = inputs["labels"]
        request = _Request(_base_model(model))
        token = _REQUEST.set(request)
        setattr(self, step_attr, None)
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        try:
            loss, outputs = trainer_compute_loss(
                self, model, inputs, return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
        finally:
            _REQUEST.reset(token)
        with torch.no_grad():
            if request.stats is not None:
                step = _step_from_stats(request.stats, labels, step_type)
                setattr(self, step_attr, step)
            else:
                # A model wrapper may have requested an unsupported forward.
                # Its real logits remain usable: log once, without a second
                # model call (which would consume another dropout mask).
                logits = outputs.logits
                targets = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
                labelled = targets != -100
                step = step_type(token_entropy=entropy_from_logits(logits), token_loss=None,
                                 correct=((logits.argmax(-1) == targets) & labelled).sum(),
                                 labelled=labelled.sum())
            log_metrics(self, "train", inputs, step)
        return (loss, outputs) if return_outputs else loss

    setattr(compute_loss, _MARKER, True)
    return compute_loss


def patch_streamed_head():
    """Enable only the verified package contracts, before constructing a trainer."""
    try:
        if any(metadata.version(name) != version for name, version in (
            ("soup-cli", "0.77.0"), ("transformers", "5.16.1"), ("trl", "0.29.1")
        )):
            return False
    except metadata.PackageNotFoundError:
        return False
    from soup_cli.utils.stream_loss_chunk import ChunkedStep, STEP_ATTR, _log_trl_metrics
    from transformers import Trainer
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    from trl.trainer.sft_trainer import SFTTrainer
    from trl.trainer.utils import entropy_from_logits

    from .streamed_head import linear_head_loss

    if (getattr(Qwen2ForCausalLM.forward, _MARKER, False)
            and getattr(SFTTrainer.compute_loss, _MARKER, False)):
        return True
    if (getattr(Qwen2ForCausalLM.forward, _MARKER, False)
            or getattr(SFTTrainer.compute_loss, _MARKER, False)):
        return False
    forward = _make_forward(Qwen2ForCausalLM.forward, linear_head_loss, CausalLMOutputWithPast)
    compute = _make_compute_loss(SFTTrainer.compute_loss, Trainer.compute_loss,
                                Qwen2ForCausalLM, ChunkedStep, _log_trl_metrics,
                                STEP_ATTR, entropy_from_logits)
    Qwen2ForCausalLM.forward = forward
    SFTTrainer.compute_loss = compute
    return True
