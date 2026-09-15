"""Cross-validation static rules: C001..C006."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from typing import Any

from ...core.context import StaticContext
from ...core.models import Category, Finding, Fix, Layer, Location, Severity
from ...core.references import KAPOOR_NARAYANAN_2023, SKLEARN_CV, SKLEARN_PITFALLS
from ...core.registry import register
from ...core.rule import RuntimeRule, StaticRule
from ..dataflow import Taint
from ._helpers import (
    input_has_related_hint,
    location_of,
    notebook_note,
)

_GROUP_HINTS = (
    "group",
    "patient",
    "user",
    "subject",
    "session",
    "customer",
    "entity",
    "account",
    "device",
)
_TIME_HINTS = ("date", "time", "timestamp", "datetime", "_dt")


_OUTER_CV_CALLS = {"cross_val_score", "cross_validate", "cross_val_predict"}


def _associated_splitter_use(ctx: StaticContext, splitter: Any) -> list[Any]:
    outputs = set(splitter.output_binding_versions)
    uses: list[Any] = []
    for call in ctx.dataflow.calls:
        if call.scope_id != splitter.scope_id:
            continue
        if call.func_name == "split" and call.receiver_var in outputs:
            if call.receiver_binding_version == splitter.output_binding_versions.get(
                call.receiver_var
            ):
                uses.append(call)
        if call.func_name in _OUTER_CV_CALLS and "groups" in call.keywords:
            cv_value = call.keywords.get("cv")
            if cv_value is None:
                continue
            cv_names = {
                node.id for node in ast.walk(cv_value) if isinstance(node, ast.Name)
            }
            if cv_names & outputs or splitter.node in ast.walk(cv_value):
                uses.append(call)
    return uses


def _has_group_signal(ctx: StaticContext, call: Any) -> bool:
    if input_has_related_hint(
        ctx,
        call.node,
        call.arg_vars,
        call.arg_binding_versions,
        _GROUP_HINTS,
        scope_id=call.scope_id,
    ):
        return True
    if not ctx.adapters.is_cv_splitter(call.func_name):
        return False
    return any(
        input_has_related_hint(
            ctx,
            use.node,
            use.arg_vars,
            use.arg_binding_versions,
            _GROUP_HINTS,
            scope_id=use.scope_id,
        )
        for use in _associated_splitter_use(ctx, call)
    )


def _has_temporal_signal(ctx: StaticContext, call: Any) -> bool:
    if input_has_related_hint(
        ctx,
        call.node,
        call.arg_vars,
        call.arg_binding_versions,
        _TIME_HINTS,
        scope_id=call.scope_id,
    ):
        return True
    if not ctx.adapters.is_cv_splitter(call.func_name):
        return False
    return any(
        input_has_related_hint(
            ctx,
            use.node,
            use.arg_vars,
            use.arg_binding_versions,
            _TIME_HINTS,
            scope_id=use.scope_id,
        )
        for use in _associated_splitter_use(ctx, call)
    )


@register
class C001(StaticRule, RuntimeRule):
    id = "C001"
    name = "preprocessing fit outside CV loop"
    category = Category.CV
    severity = Severity.HIGH
    layers = (Layer.STATIC, Layer.RUNTIME)
    references = (SKLEARN_PITFALLS, SKLEARN_CV)
    rationale = (
        "Fitting a transformer once on all data and reusing it across CV folds leaks "
        "validation-fold statistics into training. Put the transformer in a Pipeline "
        "passed to cross_val_score/GridSearchCV so it is refit per fold."
    )

    def on_event(self, event: Any, ctx: Any) -> Iterable[Finding]:
        """Report a transformer fit reused by a runtime CV/search call.

        The hook layer marks the active CV context on fit events.  If a backend
        cannot expose that context, no runtime finding is invented; static
        analysis remains the source of truth for the unsupported path.
        """

        from ...runtime.taint import EventKind
        from ...static.adapters import load_adapters
        if event.kind is not EventKind.CV:
            return
        adapters = ctx.scratch.setdefault("_adapters", load_adapters())
        reused_classes = [
            value for value in event.payload.get("reused_fit_classes", []) if isinstance(value, str)
        ]
        if reused_classes and all(adapters.is_stateless_transformer(value) for value in reused_classes):
            return
        reused = event.payload.get("reused_fit_ordinals", [])
        if not reused:
            return
        yield Finding(
            rule_id=self.id,
            category=self.category,
            severity=self.severity,
            layer=Layer.RUNTIME,
            message=(
                f"A preprocessing result was created outside the CV folds and reused by "
                f"`{event.payload.get('function', 'cross-validation')}`. Put learned "
                "preprocessing inside the CV Pipeline."
            ),
            location=Location(context_label=f"{event.payload.get('function', 'CV')} @ {event.call_site}"),
            confidence=0.9,
            references=self.references,
            evidence={
                "reused_fit_ordinals": list(reused),
                "function": event.payload.get("function"),
            },
        )

    def check(self, ctx: StaticContext) -> Iterable[Finding]:
        if not ctx.dataflow.uses_cv_utility():
            return
        for fit in ctx.dataflow.fit_calls:
            if fit.in_pipeline:
                continue
            if not (fit.is_transformer or fit.is_feature_selector):
                continue
            if fit.class_name and ctx.adapters.is_stateless_transformer(fit.class_name):
                continue
            # Only a fit whose learned result reaches this file's CV/search
            # consumer is a C001 observation.  Disconnected demonstrations
            # in educational notebooks are not leakage.
            if not ctx.dataflow.fit_output_used_by_cv(fit):
                continue
            cls = fit.class_name or "transformer"
            yield Finding(
                rule_id=self.id,
                category=self.category,
                severity=self.severity,
                layer=Layer.STATIC,
                message=(
                    f"`{cls}.{fit.method}(...)` is fit once outside the cross-validation loop; "
                    f"wrap it in a Pipeline passed to the CV utility so it refits per fold."
                    + notebook_note(ctx)
                ),
                location=location_of(fit.node, ctx),
                confidence=0.7,
                references=self.references,
                fix=Fix(
                    summary="Move the transformer into a Pipeline used by cross_val_score.",
                    autofixable=False,
                ),
                evidence={"class": cls, "method": fit.method},
            )


@register
class C002(StaticRule):
    id = "C002"
    name = "plain KFold on grouped data"
    category = Category.CV
    severity = Severity.HIGH
    layers = (Layer.STATIC,)
    references = (SKLEARN_CV,)
    rationale = (
        "When the same entity (patient/user/session) has multiple rows, a plain KFold or "
        "train_test_split can place rows from one entity on both sides of the split. Use "
        "GroupKFold/StratifiedGroupKFold."
    )

    def check(self, ctx: StaticContext) -> Iterable[Finding]:
        seen: set[int] = set()
        for c in ctx.dataflow.calls:
            if not (
                ctx.adapters.is_cv_splitter(c.func_name) or c.func_name == "train_test_split"
            ):
                continue
            if ctx.adapters.is_group_cv_splitter(c.func_name):
                continue
            if id(c.node) in seen:
                continue
            seen.add(id(c.node))
            if not _has_group_signal(ctx, c):
                continue
            yield Finding(
                rule_id=self.id,
                category=self.category,
                severity=self.severity,
                layer=Layer.STATIC,
                message=(
                    f"`{c.func_name}` is used while grouped/entity data appears present; rows "
                    f"from one group may straddle the split. Use GroupKFold/StratifiedGroupKFold."
                    + notebook_note(ctx)
                ),
                location=location_of(c.node, ctx),
                confidence=0.5,
                references=self.references,
                advisory_only=True,
                fix=Fix(summary="Switch to GroupKFold and pass groups=.", autofixable=False),
                evidence={"splitter": c.func_name},
            )


@register
class C003(StaticRule):
    id = "C003"
    name = "plain KFold on time series"
    category = Category.CV
    severity = Severity.HIGH
    layers = (Layer.STATIC,)
    references = (SKLEARN_CV,)
    rationale = (
        "Random KFold on time-ordered data trains on the future to predict the past. Use "
        "TimeSeriesSplit."
    )

    def check(self, ctx: StaticContext) -> Iterable[Finding]:
        seen: set[int] = set()
        for c in ctx.dataflow.calls:
            if not ctx.adapters.is_cv_splitter(c.func_name):
                continue
            if ctx.adapters.is_temporal_cv_splitter(c.func_name):
                continue
            if id(c.node) in seen:
                continue
            seen.add(id(c.node))
            if not _has_temporal_signal(ctx, c):
                continue
            yield Finding(
                rule_id=self.id,
                category=self.category,
                severity=self.severity,
                layer=Layer.STATIC,
                message=(
                    f"`{c.func_name}` shuffles time-ordered data across folds; use TimeSeriesSplit."
                    + notebook_note(ctx)
                ),
                location=location_of(c.node, ctx),
                confidence=0.5,
                references=self.references,
                advisory_only=True,
                fix=Fix(summary="Replace with TimeSeriesSplit.", autofixable=False),
                evidence={"splitter": c.func_name},
            )


@register
class C004(StaticRule, RuntimeRule):
    id = "C004"
    name = "tuning on the test set"
    category = Category.CV
    severity = Severity.HIGH
    layers = (Layer.STATIC, Layer.RUNTIME)
    references = (KAPOOR_NARAYANAN_2023, SKLEARN_PITFALLS)
    rationale = (
        "Fitting a hyperparameter search directly on the held-out test split selects "
        "hyperparameters using the data you report results on."
    )

    def check(self, ctx: StaticContext) -> Iterable[Finding]:
        for fit in ctx.dataflow.fit_calls:
            cls = fit.class_name
            if not (cls and ctx.adapters.is_search_constructor(cls)):
                # also catch grid.fit where grid is a search object
                cls = ctx.dataflow.class_of(fit.receiver_var) if fit.receiver_var else None
                if not (cls and ctx.adapters.is_search_constructor(cls)):
                    continue
            if any(t is Taint.TEST for t in fit.arg_taints):
                yield Finding(
                    rule_id=self.id,
                    category=self.category,
                    severity=self.severity,
                    layer=Layer.STATIC,
                    message=(
                        f"`{cls}` is fit on the test/validation split; tune on training/CV folds "
                        f"and touch the test set once, at the end." + notebook_note(ctx)
                    ),
                    location=location_of(fit.node, ctx),
                    confidence=0.8,
                    references=self.references,
                    fix=Fix(summary="Fit the search on X_train; evaluate on X_test once.", autofixable=False),
                    evidence={"search": cls},
                )

    def on_event(self, event: Any, ctx: Any) -> Iterable[Finding]:
        from ...runtime.detect import emit_fit_leak, fit_saw_eval_rows
        from ...static.adapters import load_adapters

        overlap = fit_saw_eval_rows(event, ctx)
        if not overlap:
            return
        adapters = ctx.scratch.setdefault("_adapters", load_adapters())
        if not adapters.is_search_constructor(event.payload.get("class", "")):
            return
        yield from emit_fit_leak(
            self, event, ctx, overlap=overlap,
            what="the hyperparameter search is tuned on held-out rows",
        )


@register
class C005(StaticRule):
    id = "C005"
    name = "model selection without nested CV"
    category = Category.CV
    severity = Severity.MEDIUM
    layers = (Layer.STATIC,)
    references = (KAPOOR_NARAYANAN_2023, SKLEARN_CV)
    rationale = (
        "Reporting a hyperparameter search's best_score_ as model performance reuses the "
        "selection folds for evaluation, biasing the estimate upward. Use nested CV."
    )

    def check(self, ctx: StaticContext) -> Iterable[Finding]:
        # search object present?
        search_calls = [c for c in ctx.dataflow.calls if ctx.adapters.is_search_constructor(c.func_name) and c.func_name in ("GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV", "HalvingRandomSearchCV")]
        if not search_calls:
            return
        # ``best_estimator_`` is a legitimate fitted model-selection result on
        # its own.  Only treating ``best_score_`` as an evaluation claim keeps
        # the detector from flagging ordinary downstream use of the selected
        # estimator.
        uses_best_score = False
        wraps_in_outer_cv = False
        for node in ast.walk(ctx.tree):
            if isinstance(node, ast.Attribute) and node.attr == "best_score_":
                uses_best_score = True
            if isinstance(node, ast.Call):
                fname = ctx.dataflow.resolve_name(node.func)
                tail = fname.rsplit(".", 1)[-1] if fname else None
                if tail in ("cross_val_score", "cross_validate"):
                    # is a search object passed in? heuristic: any arg is a Name bound to a search ctor
                    for a in node.args:
                        if not isinstance(a, ast.Name):
                            continue
                        cls = ctx.dataflow.class_of(a.id)
                        if cls and ctx.adapters.is_search_constructor(cls):
                            wraps_in_outer_cv = True
        if uses_best_score and not wraps_in_outer_cv:
            search_fits = [
                fit
                for fit in ctx.dataflow.fit_calls
                if fit.method == "fit"
                and fit.class_name
                and ctx.adapters.is_search_constructor(fit.class_name)
            ]
            # In a source module/notebook, a demonstrated independent holdout
            # is a valid alternative to nested CV for the selected model.  The
            # notebook analyzer does not reconstruct execution state, so use
            # the module-level holdout evidence conservatively rather than
            # attributing a later rebinding to a different search.
            if any(ctx.dataflow.search_has_holdout_evaluation(fit) for fit in search_fits):
                return
            c = search_calls[0]
            yield Finding(
                rule_id=self.id,
                category=self.category,
                severity=self.severity,
                layer=Layer.STATIC,
                message=(
                    "Hyperparameter search results are reported without an outer (nested) CV; "
                    "best_score_ reuses the selection folds. Wrap the search in cross_val_score."
                    + notebook_note(ctx)
                ),
                location=location_of(c.node, ctx),
                confidence=0.45,
                references=self.references,
                advisory_only=True,
                fix=Fix(summary="Use nested CV: cross_val_score(search, X, y).", autofixable=False),
                evidence={"search": c.func_name},
            )


@register
class C006(StaticRule):
    id = "C006"
    name = "target/mean encoding without fold isolation"
    category = Category.CV
    severity = Severity.HIGH
    layers = (Layer.STATIC,)
    references = (KAPOOR_NARAYANAN_2023, SKLEARN_PITFALLS)
    rationale = (
        "Computing target/mean/count encodings over data spanning the evaluation split lets "
        "each row's encoding depend on its own (and the held-out rows') target. Compute "
        "encodings within CV folds (out-of-fold)."
    )

    _AGG = {"mean", "count", "sum", "median", "std", "nunique"}

    def check(self, ctx: StaticContext) -> Iterable[Finding]:
        # TargetEncoder fit on full data:
        for fit in ctx.dataflow.fit_calls:
            if fit.class_name == "TargetEncoder" and ctx.dataflow.leak_taint(fit) is not None and not fit.in_pipeline:
                yield self._make(fit.node, ctx, "TargetEncoder")
        # manual groupby-mean encoding idiom: df.groupby(col)[target].transform('mean')
        for node in ast.walk(ctx.tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr not in ("transform", "agg", "apply"):
                continue
            # arg is an aggregation string
            agg_str = None
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                agg_str = node.args[0].value
            if agg_str not in self._AGG:
                continue
            # the chain must contain a groupby
            if self._chain_has_groupby(node.func.value):
                yield self._make(node, ctx, f"groupby(...).transform('{agg_str}')")

    def _chain_has_groupby(self, node: ast.expr) -> bool:
        cur: ast.expr | None = node
        depth = 0
        while isinstance(cur, (ast.Attribute, ast.Call, ast.Subscript)) and depth < 8:
            if isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute) and cur.func.attr == "groupby":
                return True
            cur = cur.func.value if isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute) else getattr(cur, "value", None)
            depth += 1
            if cur is None:
                break
        return False

    def _make(self, node: ast.AST, ctx: StaticContext, what: str) -> Finding:
        return Finding(
            rule_id=self.id,
            category=self.category,
            severity=self.severity,
            layer=Layer.STATIC,
            message=(
                f"`{what}` looks like target/mean encoding computed across the evaluation split; "
                f"compute it out-of-fold (inside CV) to avoid target leakage." + notebook_note(ctx)
            ),
            location=location_of(node, ctx),
            confidence=0.5,
            references=self.references,
            fix=Fix(summary="Compute encodings out-of-fold (e.g. sklearn TargetEncoder in a Pipeline).", autofixable=False),
            evidence={"pattern": what},
        )
