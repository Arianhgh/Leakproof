"""Temporal static rule: TM002 (look-ahead features)."""

from __future__ import annotations

import ast
from collections.abc import Iterable

from ...core.context import StaticContext
from ...core.models import Category, Finding, Fix, Layer, Severity
from ...core.references import KAPOOR_NARAYANAN_2023
from ...core.registry import register
from ...core.rule import StaticRule
from ._helpers import location_of, notebook_note


@register
class TM002(StaticRule):
    id = "TM002"
    name = "look-ahead feature across split"
    category = Category.TEMPORAL
    severity = Severity.HIGH
    layers = (Layer.STATIC,)
    references = (KAPOOR_NARAYANAN_2023,)
    rationale = (
        "A negative shift pulls future values into the present row, and a rolling/expanding "
        "window computed before splitting can summarize rows that belong to the test period. "
        "Compute time features within each split's window only."
    )

    def check(self, ctx: StaticContext) -> Iterable[Finding]:
        for node in ast.walk(ctx.tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            method = node.func.attr
            if method == "shift" and self._negative_shift(node):
                yield self._make(node, ctx, "shift() with a negative period pulls future values backward")
            elif method in ("rolling", "expanding") and self._on_full(node, ctx):
                yield self._make(
                    node,
                    ctx,
                    f"{method}() window computed on pre-split data may cross the split boundary",
                )

    def _negative_shift(self, node: ast.Call) -> bool:
        for arg in [*node.args, *(kw.value for kw in node.keywords if kw.arg == "periods")]:
            if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                return True
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int) and arg.value < 0:
                return True
        return False

    def _on_full(self, node: ast.Call, ctx: StaticContext) -> bool:
        # the receiver chain references a FULL/unknown frame and there is a split later
        if not ctx.dataflow.has_split():
            return False
        from ..dataflow import Taint

        if not isinstance(node.func, ast.Attribute):
            return False
        for sub in ast.walk(node.func.value):
            if isinstance(sub, ast.Name) and ctx.dataflow.taint_of(sub.id) is Taint.FULL:
                return True
        return False

    def _make(self, node: ast.AST, ctx: StaticContext, why: str) -> Finding:
        return Finding(
            rule_id=self.id,
            category=self.category,
            severity=self.severity,
            layer=Layer.STATIC,
            message=f"Possible look-ahead feature: {why}." + notebook_note(ctx),
            location=location_of(node, ctx),
            confidence=0.55,
            references=self.references,
            fix=Fix(summary="Compute time-window features per split, never across the boundary.", autofixable=False),
            evidence={},
        )
