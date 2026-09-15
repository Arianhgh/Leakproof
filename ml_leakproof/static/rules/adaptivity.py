"""Adaptivity / test-set-reuse static rules: T002, T003."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from typing import Any

from ...core.context import StaticContext
from ...core.models import Category, Finding, Fix, Layer, Location, Severity
from ...core.references import DWORK_2015_REUSABLE_HOLDOUT, KAPOOR_NARAYANAN_2023
from ...core.registry import register
from ...core.rule import RuntimeRule, StaticRule
from ..dataflow import Taint
from ._helpers import location_of, notebook_note


@register
class T002(StaticRule, RuntimeRule):
    id = "T002"
    name = "early stopping / selection on test"
    category = Category.ADAPTIVITY
    severity = Severity.HIGH
    layers = (Layer.STATIC, Layer.RUNTIME)
    references = (KAPOOR_NARAYANAN_2023,)

    def on_event(self, event: Any, ctx: Any) -> Iterable[Finding]:
        from ...runtime.taint import EventKind

        if event.kind is not EventKind.FIT:
            return
        memberships = event.payload.get("eval_membership", [])
        test_rows = sum(item.get("membership", {}).get("test", 0) for item in memberships)
        if not test_rows:
            return
        cls = event.payload.get("class", "estimator")
        yield Finding(
            rule_id=self.id,
            category=self.category,
            severity=self.severity,
            layer=Layer.RUNTIME,
            message=(
                f"At runtime, `{cls}.fit(...)` received {test_rows} test row(s) as an "
                "early-stopping/evaluation set; use validation data instead."
            ),
            location=Location(
                context_label=f"{cls}.fit @ {event.call_site} run #{event.run_ordinal}"
            ),
            confidence=0.9,
            references=self.references,
            evidence={"test_rows_seen": test_rows, "class": cls},
        )
    rationale = (
        "Using the test split as an eval_set for early stopping (or to pick the best epoch) "
        "selects the model against the data you report on. Use a separate validation split."
    )

    def check(self, ctx: StaticContext) -> Iterable[Finding]:
        for node in ast.walk(ctx.tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr not in ("fit", "train"):
                continue
            eval_set = self._eval_set_kwarg(node)
            if eval_set is None:
                continue
            if self._references_eval_side(eval_set, ctx):
                yield Finding(
                    rule_id=self.id,
                    category=self.category,
                    severity=self.severity,
                    layer=Layer.STATIC,
                    message=(
                        "Early stopping / eval_set uses the test split; use a dedicated "
                        "validation split for model selection and keep test untouched."
                        + notebook_note(ctx)
                    ),
                    location=location_of(node, ctx),
                    confidence=0.7,
                    references=self.references,
                    fix=Fix(summary="Pass a validation split to eval_set, not X_test/y_test.", autofixable=False),
                    evidence={},
                )

    def _eval_set_kwarg(self, node: ast.Call) -> ast.expr | None:
        for kw in node.keywords:
            if kw.arg in ("eval_set", "validation_data"):
                return kw.value
        return None

    def _references_eval_side(self, expr: ast.expr, ctx: StaticContext) -> bool:
        # Early stopping on a dedicated validation split is correct; only the
        # TEST split is leakage.
        for sub in ast.walk(expr):
            if isinstance(sub, ast.Name) and ctx.dataflow.taint_of(sub.id) is Taint.TEST:
                return True
        return False


@register
class T003(StaticRule):
    id = "T003"
    name = "best-of-N seeds reported"
    category = Category.ADAPTIVITY
    severity = Severity.MEDIUM
    layers = (Layer.STATIC,)
    references = (DWORK_2015_REUSABLE_HOLDOUT, KAPOOR_NARAYANAN_2023)
    rationale = (
        "Looping over seeds/configs and keeping only the maximum score reports a biased "
        "best-case rather than the distribution. Report mean ± std across runs."
    )

    _SEED_HINTS = ("seed", "random_state", "config", "trial", "run")

    def check(self, ctx: StaticContext) -> Iterable[Finding]:
        for node in ast.walk(ctx.tree):
            if not isinstance(node, ast.For):
                continue
            if not self._loops_over_seeds(node):
                continue
            if self._keeps_max(node):
                yield Finding(
                    rule_id=self.id,
                    category=self.category,
                    severity=self.severity,
                    layer=Layer.STATIC,
                    message=(
                        "A loop over seeds/configs appears to keep only the best score; report "
                        "the distribution (mean ± std), not the maximum." + notebook_note(ctx)
                    ),
                    location=location_of(node, ctx),
                    confidence=0.4,
                    references=self.references,
                    advisory_only=True,
                    fix=Fix(summary="Collect all scores and report mean/std (and CI).", autofixable=False),
                    evidence={},
                )

    def _loops_over_seeds(self, node: ast.For) -> bool:
        target_names = [n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)]
        if any(any(h in t.lower() for h in self._SEED_HINTS) for t in target_names):
            return True
        # iterating a variable named seeds/configs
        for sub in ast.walk(node.iter):
            if isinstance(sub, ast.Name) and any(h in sub.id.lower() for h in self._SEED_HINTS):
                return True
        return False

    def _keeps_max(self, node: ast.For) -> bool:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == "max":
                return True
            # best = score if score > best
            if isinstance(sub, ast.Compare) and any(
                isinstance(op, (ast.Gt, ast.GtE, ast.Lt, ast.LtE)) for op in sub.ops
            ):
                names = {n.id.lower() for n in ast.walk(sub) if isinstance(n, ast.Name)}
                if any("best" in n for n in names):
                    return True
        return False
