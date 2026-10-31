"""Remove TRL's temporary shifted-logits copy solely for token accuracy.

The pinned SFTTrainer method remains the implementation of loss, entropy,
token counting and metric aggregation. Its two accuracy statements are rewritten
only after checking their structure. Model outputs and their FP32 dtype remain
unchanged, and the new argmax operates under the original no-grad context.
"""

import ast
import functools
from importlib import metadata
import inspect
import textwrap
from types import FunctionType, SimpleNamespace

import torch


_OPS_GLOBAL = "_soup_strided_accuracy_ops_v1"
_PATCH_MARKER = "_soup_strided_accuracy_patch"
_CLONE_SOURCE = "shift_logits = outputs.logits[..., :-1, :].contiguous()"
_ARGMAX_SOURCE = "predictions = shift_logits.argmax(dim=-1)"


def _supported_logits(logits):
    return (
        type(logits) is torch.Tensor
        and logits.is_cuda
        and logits.dtype == torch.float32
        and logits.ndim == 3
        and logits.numel() > 0
        and logits.size(-1) == 151936
        and logits.stride(-1) == 1
    )


def _shifted_view(logits):
    if _supported_logits(logits):
        return logits[..., :-1, :]
    return logits[..., :-1, :].contiguous()


def _accuracy_argmax(logits):
    if _supported_logits(logits):
        from .argmax_ops import argmax_last_dim

        return argmax_last_dim(logits)
    return logits.argmax(dim=-1)


_OPS = SimpleNamespace(shifted_view=_shifted_view, argmax=_accuracy_argmax)


def _tree(node):
    return ast.dump(node, include_attributes=False)


def _has_signature(original):
    parameters = list(inspect.signature(original, follow_wrapped=False).parameters.values())
    names = ["self", "model", "inputs", "return_outputs", "num_items_in_batch"]
    return (
        [parameter.name for parameter in parameters] == names
        and all(parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                for parameter in parameters)
        and all(parameter.default is inspect.Parameter.empty for parameter in parameters[:3])
        and parameters[3].default is False
        and parameters[4].default is None
        and original.__defaults__ == (False, None)
        and not original.__kwdefaults__
    )


def _ancestors(node, parents):
    while node in parents:
        node = parents[node]
        yield node


def _is_no_grad(node):
    return (
        isinstance(node, ast.With)
        and len(node.items) == 1
        and node.items[0].optional_vars is None
        and _tree(node.items[0].context_expr) == _tree(ast.parse("torch.no_grad()", mode="eval").body)
    )


def _rewrite_compute_loss(original):
    """Return a surgically rewritten method, or None for an unknown contract.

    Building the code inside a temporary class creates the implicit __class__
    free variable needed by zero-argument super(). The function then receives
    the *original* closure and live globals, rather than the temporary class.
    """
    if not isinstance(original, FunctionType):
        return None
    if getattr(original, _PATCH_MARKER, None) is _OPS:
        return original
    if (original.__name__ != "compute_loss"
            or original.__code__.co_freevars != ("__class__",)
            or original.__closure__ is None or len(original.__closure__) != 1):
        return None
    try:
        if not isinstance(original.__closure__[0].cell_contents, type) or not _has_signature(original):
            return None
        source = textwrap.dedent(inspect.getsource(original))
        module = ast.parse(source)
    except (OSError, TypeError, ValueError, SyntaxError):
        return None
    if (len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef)
            or module.body[0].name != "compute_loss" or module.body[0].decorator_list):
        return None
    method = module.body[0]
    if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
           for statement in method.body for node in ast.walk(statement)):
        return None
    clone_tree = _tree(ast.parse(_CLONE_SOURCE).body[0])
    argmax_tree = _tree(ast.parse(_ARGMAX_SOURCE).body[0])
    clones = [node for node in ast.walk(method) if _tree(node) == clone_tree]
    argmaxes = [node for node in ast.walk(method) if _tree(node) == argmax_tree]
    if len(clones) != 1 or len(argmaxes) != 1:
        return None
    clone, argmax = clones[0], argmaxes[0]
    parents = {child: node for node in ast.walk(method) for child in ast.iter_child_nodes(node)}
    no_grad = next((node for node in _ancestors(clone, parents) if _is_no_grad(node)), None)
    if no_grad is None or no_grad not in set(_ancestors(argmax, parents)):
        return None
    shift_test = _tree(ast.parse("'shift_labels' in inputs", mode="eval").body)
    shift_branch = next(
        (node for node in _ancestors(clone, parents)
         if isinstance(node, ast.If) and _tree(node.test) == shift_test), None
    )
    if shift_branch is None or not any(clone in ast.walk(node) for node in shift_branch.orelse):
        return None
    # The copied tensor has no other observer: virtual-token slicing and argmax
    # are its only reads in the pinned method. Reject additional uses/aliases.
    expected_stores = {
        clone_tree,
        _tree(ast.parse("shift_logits = outputs.logits.contiguous()").body[0]),
        _tree(ast.parse("shift_logits = shift_logits[:, self.num_virtual_tokens :, :]").body[0]),
    }
    assignments = [node for node in ast.walk(method) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "shift_logits"
                           for target in node.targets)]
    loads = [node for node in ast.walk(method) if isinstance(node, ast.Name)
             and node.id == "shift_logits" and isinstance(node.ctx, ast.Load)]
    stores = [node for node in ast.walk(method) if isinstance(node, ast.Name)
              and node.id == "shift_logits" and isinstance(node.ctx, ast.Store)]
    if (len(assignments) != 3 or {_tree(node) for node in assignments} != expected_stores
            or len(loads) != 2 or len(stores) != 3):
        return None
    existing = original.__globals__.get(_OPS_GLOBAL, _OPS)
    if existing is not _OPS:
        return None

    class Rewrite(ast.NodeTransformer):
        def visit_Assign(self, node):
            if node is clone:
                replacement = ast.parse(
                    f"shift_logits = {_OPS_GLOBAL}.shifted_view(outputs.logits)"
                ).body[0]
                return ast.copy_location(replacement, node)
            if node is argmax:
                replacement = ast.parse(
                    f"predictions = {_OPS_GLOBAL}.argmax(shift_logits)"
                ).body[0]
                return ast.copy_location(replacement, node)
            return self.generic_visit(node)

    method = Rewrite().visit(method)
    # Annotation objects are restored from the original function; do not
    # evaluate source annotations while constructing the temporary method.
    method.returns = None
    for argument in method.args.args:
        argument.annotation = None
    if hasattr(method, "type_params"):
        method.type_params = []
    # Preserve the installed method's line numbers for useful training traces.
    ast.increment_lineno(method, original.__code__.co_firstlineno - method.lineno)
    carrier = ast.ClassDef(name="_SoupAccuracyCodeCarrier", bases=[], keywords=[],
                           body=[method], decorator_list=[])
    ast.copy_location(carrier, method)
    if hasattr(method, "type_params"):
        carrier.type_params = []
    rewritten_module = ast.fix_missing_locations(ast.Module(body=[carrier], type_ignores=[]))
    namespace = {}
    try:
        exec(compile(rewritten_module, original.__code__.co_filename, "exec"),
             original.__globals__, namespace)
        template = namespace[carrier.name].compute_loss
        if template.__code__.co_freevars != original.__code__.co_freevars:
            return None
        rewritten = FunctionType(template.__code__, original.__globals__, original.__name__,
                                 original.__defaults__, original.__closure__)
        functools.update_wrapper(rewritten, original)
        rewritten.__kwdefaults__ = original.__kwdefaults__
        rewritten.__annotations__ = original.__annotations__
    except (SyntaxError, TypeError, ValueError, NameError):
        return None
    setattr(rewritten, _PATCH_MARKER, _OPS)
    # No module mutation occurs until every source/closure check has succeeded.
    original.__globals__[_OPS_GLOBAL] = _OPS
    return rewritten


def patch_sft_accuracy():
    """Install only on the verified TRL 0.29.1 method; return whether active."""
    try:
        if metadata.version("trl") != "0.29.1":
            return False
    except metadata.PackageNotFoundError:
        return False
    from trl.trainer import sft_trainer

    cls = sft_trainer.SFTTrainer
    original = cls.compute_loss
    rewritten = _rewrite_compute_loss(original)
    if rewritten is None:
        return False
    cls.compute_loss = rewritten
    return True
