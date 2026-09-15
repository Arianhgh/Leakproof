"""Temporal static rule: TM002 (look-ahead features)."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from typing import cast

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
        "A negative shift pulls future values into the present row, and a centered rolling "
        "window can summarize rows across a split boundary. Trailing, past-only windows and "
        "explicit future-target construction require separate review."
    )

    def check(self, ctx: StaticContext) -> Iterable[Finding]:
        for node in ast.walk(ctx.tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            method = node.func.attr
            if (
                method == "shift"
                and self._negative_shift(node)
                and not self._constructs_future_target(node, ctx)
            ):
                yield self._make(node, ctx, "shift() with a negative period pulls future values backward")
            elif method == "rolling" and self._future_window(node, ctx):
                yield self._make(
                    node,
                    ctx,
                    "a centered rolling window can include future rows across the split boundary",
                )

    def _negative_shift(self, node: ast.Call) -> bool:
        for arg in [*node.args, *(kw.value for kw in node.keywords if kw.arg == "periods")]:
            if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                return True
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int) and arg.value < 0:
                return True
        return False

    def _on_full(self, node: ast.Call, ctx: StaticContext) -> bool:
        # The receiver chain references a FULL/unknown frame and there is a
        # split in the module.  This is used only for explicitly future-looking
        # windows; a trailing rolling/expanding window is past-only and can be
        # valid across a chronological train/test boundary.
        if not ctx.dataflow.has_split():
            return False
        from ..dataflow import Taint

        if not isinstance(node.func, ast.Attribute):
            return False
        for sub in ast.walk(node.func.value):
            if isinstance(sub, ast.Name) and ctx.dataflow.taint_of(sub.id) is Taint.FULL:
                return True
        return False

    def _future_window(self, node: ast.Call, ctx: StaticContext) -> bool:
        if not self._on_full(node, ctx):
            return False
        # pandas' default rolling window is trailing.  Only an explicit
        # centered window provides enough evidence here to claim future-row
        # contamination; arbitrary custom indexers remain outside the proof
        # boundary and are left for review by the user.
        for keyword in node.keywords:
            if keyword.arg == "center":
                return isinstance(keyword.value, ast.Constant) and keyword.value.value is True
        return False

    def _constructs_future_target(self, node: ast.Call, ctx: StaticContext) -> bool:
        """Skip the legitimate pattern that constructs a future label.

        ``frame["future_target"] = frame["target"].shift(-1)`` is target
        construction, not a feature that will be supplied to a model.  We only
        suppress the finding when the negative shift is inside an assignment
        whose destination is explicitly target/label/outcome-like; a value
        assigned to ``future_feature`` remains a finding.
        """

        target_words = ("target", "label", "outcome", "response", "y")
        for parent in ast.walk(ctx.tree):
            if not isinstance(parent, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                continue
            value = getattr(parent, "value", None)
            if not isinstance(value, ast.expr):
                continue
            if not any(sub is node for sub in ast.walk(value)):
                continue
            destinations: list[str] = []
            targets: list[ast.AST]
            if isinstance(parent, (ast.AnnAssign, ast.NamedExpr)):
                targets = [cast(ast.AST, parent.target)]
            else:
                targets = [cast(ast.AST, target) for target in parent.targets]
            for assignment_target in targets:
                for sub in ast.walk(assignment_target):
                    if isinstance(sub, ast.Name):
                        destinations.append(sub.id.lower())
                    elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        destinations.append(sub.value.lower())
            return any(
                any(
                    part == word or part.endswith(f"_{word}") or part.startswith(f"{word}_")
                    for word in target_words
                )
                for destination in destinations
                for part in destination.replace("-", "_").split("_")
            )
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
            advisory_only=True,
            fix=Fix(summary="Compute time-window features per split, never across the boundary.", autofixable=False),
            evidence={},
        )
