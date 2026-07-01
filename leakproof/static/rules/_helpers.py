"""Shared helpers for static rules."""

from __future__ import annotations

import ast
from collections.abc import Iterable

from ...core.context import StaticContext
from ...core.models import Finding


def location_of(node: ast.AST, ctx: StaticContext, *, label_suffix: str = ""):
    from ...core.models import Location

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


def classify_full_fit(fit, ctx: StaticContext) -> str | None:
    """Return the rule id responsible for a leaky fit call, or None if clean.

    Ensures exactly one of P002/S004/S003/P003/P001/S001 fires per fit call.
    """
    df = ctx.dataflow
    if fit.in_pipeline:
        return None
    if df.leak_taint(fit) is None:
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
    if fit.is_transformer:
        return "P001" if fit.method == "fit_transform" else "S001"
    return None
