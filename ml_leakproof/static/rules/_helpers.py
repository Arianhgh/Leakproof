"""Shared helpers for static rules."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from typing import Any

from ...core.context import StaticContext
from ...core.models import Finding, Location


def location_of(node: ast.AST, ctx: StaticContext, *, label_suffix: str = "") -> Location:
    line = getattr(node, "lineno", None)
    col = getattr(node, "col_offset", None)
    end_line = getattr(node, "end_lineno", None)
    end_col = getattr(node, "end_col_offset", None)
    label = None
    if ctx.is_notebook and line is not None and ctx.line_map:
        cell = ctx.line_map.get(line)
        if cell:
            label = f"notebook cell {cell[0]} line {cell[1]}{label_suffix}"
    return Location(
        file=ctx.module_path,
        line=line,
        col=col,
        end_line=end_line,
        end_col=end_col,
        context_label=label,
    )


def has_kwarg(call: ast.Call, name: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
        if kw.arg is None:  # **kwargs -> cannot prove absence
            return None
    return None


def kwarg_is_true(value: ast.expr | None) -> bool:
    return isinstance(value, ast.Constant) and value.value is True


def kwarg_is_false(value: ast.expr | None) -> bool:
    return isinstance(value, ast.Constant) and value.value is False


def notebook_note(ctx: StaticContext) -> str:
    if ctx.is_notebook:
        return " (notebook analyzed as concatenated cells; out-of-order execution not modeled)"
    return ""


def first(it: Iterable[Finding]) -> Finding | None:
    for f in it:
        return f
    return None


_IMPUTER_CLASSES = {"SimpleImputer", "KNNImputer", "IterativeImputer"}


def text_has_hint(value: str, hints: tuple[str, ...]) -> bool:
    """Token-aware hint matching; avoids substring hits like 'validated' -> 'date'."""
    low = value.lower()
    tokens = set(re.findall(r"[a-z0-9]+", low))
    for hint in hints:
        h = hint.lower().strip("_")
        if hint.startswith("_") and hint in low:
            return True
        if h in tokens:
            return True
    return False


def node_has_hint(node: ast.AST, hints: tuple[str, ...]) -> bool:
    """Return whether a single expression contains a relevant lexical hint."""

    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            if text_has_hint(sub.value, hints):
                return True
        elif isinstance(sub, ast.Name) and text_has_hint(sub.id, hints):
            return True
        elif isinstance(sub, ast.Attribute) and text_has_hint(sub.attr, hints):
            return True
    return False


def input_has_related_hint(
    ctx: StaticContext,
    node: ast.AST,
    input_vars: Iterable[str],
    input_binding_versions: dict[str, int],
    hints: tuple[str, ...],
    *,
    scope_id: int,
) -> bool:
    """Find a hint on the operation or on the current input binding.

    The previous implementation searched the complete module.  That made an
    unrelated ``groups`` or ``date`` example affect every split in a notebook.
    This bounded check only follows the operation's input names and the
    versioned assignments that produced them.
    """

    if node_has_hint(node, hints):
        return True
    inputs = {name for name in input_vars if name and name != "?"}
    if not inputs:
        return False
    line = getattr(node, "lineno", 0)
    for target, version, value in ctx.dataflow.assignment_records_for_scope(scope_id):
        if getattr(value, "lineno", 0) > line:
            continue
        parent = target.split("[", 1)[0]
        expected = input_binding_versions.get(parent) or input_binding_versions.get(target)
        if expected is not None and version != expected:
            continue
        value_names = set(_expr_names(value))
        if parent in inputs or target in inputs:
            if node_has_hint(value, hints):
                return True
        # A group/time column assignment derived from the same input object is
        # relevant even when the helper variable is not passed to the split.
        if text_has_hint(parent, hints) or text_has_hint(target, hints):
            if inputs & value_names:
                return True
    return False


def _expr_names(node: ast.AST) -> list[str]:
    return [sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)]


def classify_full_fit(fit: Any, ctx: StaticContext) -> str | None:
    """Return the rule id responsible for a leaky fit call, or None if clean.

    Ensures exactly one of P002/S004/S003/P003/P001/S001 fires per fit call.
    """
    df = ctx.dataflow
    if df.disconnected_fit_transform_demo(fit):
        return None
    if df.leak_taint(fit) is None:
        return None
    # A pipeline component is safe to fit as part of Pipeline.fit, but the
    # pipeline's own fit still has to be checked.  Fitting that owner on a
    # held-out/full input is the same split violation as fitting a transformer
    # directly; only the component-level delegated fits are exempt.
    if fit.in_pipeline:
        if fit.class_name and ctx.adapters.is_pipeline_constructor(fit.class_name):
            # A standalone exploratory ``pipe.fit(full_data)`` is not a
            # leakage finding unless the fitted pipeline is subsequently used
            # for evaluation.  Held-out arguments remain an explicit
            # violation even when no later consumer is visible.
            if any(taint.is_eval_side for taint in fit.arg_taints):
                return "S001"
            if df.fit_output_used_for_evaluation(fit):
                return "S001"
            return None
        return None
    cls = fit.class_name
    # concat of train+test feeding a fit -> P002 (most specific)
    if df.fit_arg_is_concat(fit):
        return "P002"
    if fit.is_resampler:
        return "S003"
    if fit.is_feature_selector:
        return "S004"
    if cls in _IMPUTER_CLASSES:
        return "P003"
    if fit.is_transformer and not ctx.adapters.is_stateless_transformer(cls or ""):
        return "P001" if fit.method == "fit_transform" else "S001"
    return None
