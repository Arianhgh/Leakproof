"""Metric static rules: M002, M003, M004."""

from __future__ import annotations

import ast
from collections.abc import Iterable

from ...core.context import RuntimeContext, StaticContext
from ...core.models import Category, Finding, Fix, Layer, Severity
from ...core.references import KAPOOR_NARAYANAN_2023, SKLEARN_PITFALLS
from ...core.registry import register
from ...core.rule import RuntimeRule, StaticRule
from ...runtime.taint import RuntimeEvent
from ..dataflow import Taint
from ._helpers import location_of, notebook_note


def _arg_taint(node: ast.Call, index: int, ctx: StaticContext) -> Taint:
    # Call records capture the lexical scope and binding versions while the
    # call is visited.  Looking up names after the whole file has been walked
    # lets a module binding shadow a function parameter, or lets a later
    # reassignment rewrite the meaning of an earlier metric call.
    record = next(
        (call for call in ctx.dataflow.calls if call.node is node),
        None,
    )
    if record is not None:
        if index < len(node.args) and index < len(record.arg_taints):
            return record.arg_taints[index]
        keyword_names = (
            ("y_true", "y", "target", "labels")
            if index == 0
            else ("y_pred", "pred", "prediction", "output", "scores")
        )
        observed = [
            record.keyword_taints[name]
            for name in keyword_names
            if name in record.keyword_taints
        ]
        if observed:
            return ctx.dataflow._merge_taints(observed)
    return Taint.UNKNOWN


@register
class M004(StaticRule, RuntimeRule):
    id = "M004"
    name = "metric computed on training data"
    category = Category.METRIC
    severity = Severity.MEDIUM
    layers = (Layer.STATIC, Layer.RUNTIME)
    references = (SKLEARN_PITFALLS,)
    rationale = (
        "Reporting a score on the training data measures memorization, not generalization. "
        "Evaluate on the held-out split."
    )

    def check(self, ctx: StaticContext) -> Iterable[Finding]:
        metrics = ctx.adapters.metrics
        for node in ast.walk(ctx.tree):
            if not isinstance(node, ast.Call):
                continue
            fname = ctx.dataflow.resolve_name(node.func)
            tail = fname.rsplit(".", 1)[-1] if fname else None
            if tail in metrics:
                y_true_idx, _ = metrics[tail]
                if _arg_taint(node, y_true_idx, ctx) is Taint.TRAIN:
                    yield self._make(node, ctx, tail)
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr == "score":
                if any(_arg_taint(node, i, ctx) is Taint.TRAIN for i in range(2)):
                    yield self._make(node, ctx, "score")

    def _make(self, node: ast.AST, ctx: StaticContext, what: str) -> Finding:
        return Finding(
            rule_id=self.id,
            category=self.category,
            severity=self.severity,
            layer=Layer.STATIC,
            message=(
                f"`{what}(...)` is computed on training data; report metrics on the held-out "
                f"split instead." + notebook_note(ctx)
            ),
            location=location_of(node, ctx),
            confidence=0.7,
            references=self.references,
            fix=Fix(summary="Evaluate on X_test/y_test, not the training split.", autofixable=False),
            evidence={"what": what},
        )

    def on_event(self, event: RuntimeEvent, ctx: RuntimeContext) -> Iterable[Finding]:
        from ...core.models import Location
        from ...runtime.taint import EventKind

        if event.kind is not EventKind.SCORE:
            return
        seen = set(event.payload.get("seen", []))
        if not seen:
            return
        train_overlap = len(seen & ctx.taint.train_hashes)
        eval_overlap = len(seen & ctx.taint.eval_side())
        if train_overlap > 0 and eval_overlap == 0:
            cls = event.payload.get("class", "estimator")
            yield Finding(
                rule_id=self.id,
                category=self.category,
                severity=self.severity,
                layer=Layer.RUNTIME,
                message=(
                    f"At runtime, `{cls}.score(...)` was computed on training rows; report "
                    f"metrics on the held-out split."
                ),
                location=Location(
                    context_label=f"{cls}.score @ {event.call_site} run #{event.run_ordinal}"
                ),
                confidence=0.9,
                references=self.references,
                evidence={"class": cls, "train_rows_seen": train_overlap},
            )


@register
class M002(StaticRule):
    id = "M002"
    name = "decision threshold tuned on test"
    category = Category.METRIC
    severity = Severity.MEDIUM
    layers = (Layer.STATIC,)
    references = (KAPOOR_NARAYANAN_2023,)
    rationale = (
        "Selecting a classification threshold from a curve computed on the test labels tunes "
        "the decision rule on the data you report on. Pick the threshold on a validation split."
    )

    _CURVE_FUNCS = {"roc_curve", "precision_recall_curve", "det_curve"}

    def check(self, ctx: StaticContext) -> Iterable[Finding]:
        for node in ast.walk(ctx.tree):
            if not isinstance(node, ast.Call):
                continue
            fname = ctx.dataflow.resolve_name(node.func)
            tail = fname.rsplit(".", 1)[-1] if fname else None
            if tail not in self._CURVE_FUNCS:
                continue
            call = next(
                (
                    record
                    for record in ctx.dataflow.calls
                    if record.node is node
                ),
                None,
            )
            if (
                _arg_taint(node, 0, ctx) is Taint.TEST
                and call is not None
                and self._selects_threshold_from_curve(ctx, call)
            ):
                yield Finding(
                    rule_id=self.id,
                    category=self.category,
                    severity=self.severity,
                    layer=Layer.STATIC,
                    message=(
                        f"`{tail}` is computed on the test labels, suggesting the "
                        f"decision threshold is tuned on test. Tune it on a validation split."
                        + notebook_note(ctx)
                    ),
                    location=location_of(node, ctx),
                    confidence=0.5,
                    references=self.references,
                    fix=Fix(summary="Select the threshold on validation, evaluate on test once.", autofixable=False),
                    evidence={"func": tail},
                )

    def _selects_threshold_from_curve(self, ctx: StaticContext, call: object) -> bool:
        """Require a threshold choice derived from curve metrics.

        Plotting a test ROC/PR curve, or marking a fixed default threshold,
        does not prove test-set tuning.  This bounded data-flow check follows
        reductions of the curve metrics into an index and then into the
        returned threshold array.
        """

        output_bindings = list(getattr(call, "output_binding_versions", {}).items())
        if len(output_bindings) < 2:
            return False
        metric_vars = set(output_bindings[:-1])
        threshold_vars = set(output_bindings[-1:])
        derived = set(metric_vars)
        scope_id = getattr(call, "scope_id", 0)
        line = getattr(call, "line", 0)
        dependencies = ctx.dataflow.assignment_dependencies_for_scope(scope_id)
        assignments = [
            (target, version, value)
            for target, version, value in ctx.dataflow.assignment_records_for_scope(scope_id)
            if getattr(value, "lineno", 0) > line
        ]
        assignments.sort(key=lambda item: getattr(item[2], "lineno", 0))
        for target, version, _value in assignments:
            binding_dependencies = {
                (name, binding_version)
                for name, binding_version in dependencies.get((target, version), {}).items()
            }
            if threshold_vars & binding_dependencies and derived & binding_dependencies:
                return True
            if derived & binding_dependencies:
                derived.add((target.split("[", 1)[0], version))
        return False


@register
class M003(StaticRule):
    advisory_only = True
    id = "M003"
    name = "no variance/CI across folds or seeds"
    category = Category.METRIC
    severity = Severity.LOW
    layers = (Layer.STATIC,)
    references = (KAPOOR_NARAYANAN_2023,)
    rationale = (
        "A single point estimate hides run-to-run variance. Report spread (std / CI) across "
        "CV folds or seeds."
    )

    def check(self, ctx: StaticContext) -> Iterable[Finding]:
        cv_score_vars = self._cv_score_vars(ctx)
        mean_nodes = self._cv_mean_nodes(ctx, cv_score_vars)
        cv_results_mean_nodes = self._cv_results_mean_nodes(ctx)
        if not mean_nodes and not cv_results_mean_nodes:
            return
        if self._reports_spread(ctx, cv_score_vars):
            return
        node = mean_nodes[0] if mean_nodes else cv_results_mean_nodes[0]
        yield Finding(
            rule_id=self.id,
            category=self.category,
            severity=self.severity,
            layer=Layer.STATIC,
            message=(
                "Cross-validation scores are summarized by mean only; also report the "
                "standard deviation / confidence interval." + notebook_note(ctx)
            ),
            location=location_of(node, ctx),
            confidence=0.4,
            references=self.references,
            advisory_only=self.advisory_only,
            fix=Fix(summary="Report scores.std() alongside scores.mean().", autofixable=False),
            evidence={},
        )

    def _cv_score_vars(self, ctx: StaticContext) -> set[str]:
        out: set[str] = set()
        for node in ast.walk(ctx.tree):
            if not isinstance(node, ast.Assign):
                continue
            if not isinstance(node.value, ast.Call):
                continue
            tail = self._call_tail(node.value, ctx)
            if tail not in {"cross_val_score", "cross_validate"}:
                continue
            for target in node.targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name):
                        out.add(sub.id)
        return out

    def _cv_mean_nodes(self, ctx: StaticContext, cv_score_vars: set[str]) -> list[ast.AST]:
        out: list[ast.AST] = []
        for node in ast.walk(ctx.tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr != "mean":
                continue
            receiver = node.func.value
            if isinstance(receiver, ast.Call) and self._call_tail(receiver, ctx) in {
                "cross_val_score",
                "cross_validate",
            }:
                out.append(node)
            elif isinstance(receiver, ast.Name) and receiver.id in cv_score_vars:
                out.append(node)
        return out

    def _cv_results_mean_nodes(self, ctx: StaticContext) -> list[ast.AST]:
        out: list[ast.AST] = []
        for node in ast.walk(ctx.tree):
            if isinstance(node, ast.Constant) and node.value == "mean_test_score":
                out.append(node)
        return out

    def _reports_spread(self, ctx: StaticContext, cv_score_vars: set[str]) -> bool:
        for node in ast.walk(ctx.tree):
            if isinstance(node, ast.Constant) and node.value == "std_test_score":
                return True
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ("std", "var", "sem", "quantile"):
                    receiver = node.func.value
                    if isinstance(receiver, ast.Name) and receiver.id in cv_score_vars:
                        return True
                    if isinstance(receiver, ast.Call) and self._call_tail(receiver, ctx) in {
                        "cross_val_score",
                        "cross_validate",
                    }:
                        return True
            if isinstance(node, ast.Call):
                tail = self._call_tail(node, ctx)
                if tail in {"std", "var", "sem", "quantile"}:
                    return True
        return False

    def _call_tail(self, node: ast.Call, ctx: StaticContext) -> str | None:
        fname = ctx.dataflow.resolve_name(node.func)
        return fname.rsplit(".", 1)[-1] if fname else None
