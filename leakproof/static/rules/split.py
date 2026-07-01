"""Split-hygiene static rules: S001, S002, S003, S004."""

from __future__ import annotations

import ast
from collections.abc import Iterable

from ...core.context import StaticContext
from ...core.models import Category, Finding, Fix, Layer, Severity
from ...core.references import KAUFMAN_2012, SKLEARN_CV, SKLEARN_PITFALLS
from ...core.registry import register
from ...core.rule import StaticRule
from ._helpers import has_kwarg, kwarg_is_false, location_of, notebook_note, text_has_hint
from .preprocessing import _FullFitRule


@register
class S001(_FullFitRule):
    id = "S001"
    name = "transformer fit on full data before split"
    category = Category.SPLIT
    severity = Severity.HIGH
    rationale = (
        "A scaler/encoder/PCA fit on data that predates (or crosses) the split learns "
        "parameters from the held-out rows. Fit on the training split only, ideally "
        "inside a Pipeline passed to the cross-validation utility."
    )

    def _make(self, fit, ctx: StaticContext) -> Finding:
        cls = fit.class_name or "transformer"
        return Finding(
            rule_id=self.id,
            category=self.category,
            severity=self.severity,
            layer=Layer.STATIC,
            message=(
                f"`{cls}.{fit.method}(...)` is fit on pre-split (or cross-split) data; fit on "
                f"the training split only." + notebook_note(ctx)
            ),
            location=location_of(fit.node, ctx),
            confidence=0.85,
            references=(SKLEARN_PITFALLS, KAUFMAN_2012),
            fix=Fix(
                summary="Wrap the transformer in a Pipeline, or fit after train_test_split.",
                autofixable=False,
            ),
            evidence={"class": cls, "method": fit.method},
        )


@register
class S003(_FullFitRule):
    id = "S003"
    name = "resampling/augmentation before split"
    category = Category.SPLIT
    severity = Severity.HIGH
    rationale = (
        "SMOTE/over/under-sampling applied to the full dataset before splitting places "
        "synthetic neighbours of test rows into the training set (and vice versa)."
    )

    def _make(self, fit, ctx: StaticContext) -> Finding:
        cls = fit.class_name or "resampler"
        return Finding(
            rule_id=self.id,
            category=self.category,
            severity=self.severity,
            layer=Layer.STATIC,
            message=(
                f"`{cls}.{fit.method}(...)` resamples pre-split data; resample only the training "
                f"split (use an imblearn Pipeline inside CV)." + notebook_note(ctx)
            ),
            location=location_of(fit.node, ctx),
            confidence=0.85,
            references=(SKLEARN_PITFALLS,),
            fix=Fix(summary="Move resampling after the split, train side only.", autofixable=False),
            evidence={"class": cls, "method": fit.method},
        )


@register
class S004(_FullFitRule):
    id = "S004"
    name = "feature selection on full data before split"
    category = Category.SPLIT
    severity = Severity.HIGH
    layers = (Layer.STATIC, Layer.RUNTIME)
    rationale = (
        "SelectKBest/RFE/VarianceThreshold fit on the full feature matrix chooses features "
        "using the test labels/values, a classic source of optimistic bias."
    )

    def _make(self, fit, ctx: StaticContext) -> Finding:
        cls = fit.class_name or "selector"
        return Finding(
            rule_id=self.id,
            category=self.category,
            severity=self.severity,
            layer=Layer.STATIC,
            message=(
                f"`{cls}.{fit.method}(...)` selects features on pre-split data; select inside "
                f"the CV loop / on the training split only." + notebook_note(ctx)
            ),
            location=location_of(fit.node, ctx),
            confidence=0.85,
            references=(SKLEARN_PITFALLS, SKLEARN_CV),
            fix=Fix(summary="Put feature selection in a Pipeline used inside CV.", autofixable=False),
            evidence={"class": cls, "method": fit.method},
        )

    def on_event(self, event, ctx):
        from ...runtime.detect import emit_fit_leak, fit_saw_eval_rows
        from ...static.adapters import load_adapters

        overlap = fit_saw_eval_rows(event, ctx)
        if not overlap:
            return
        adapters = ctx.scratch.setdefault("_adapters", load_adapters())
        if not adapters.is_feature_selector(event.payload.get("class", "")):
            return
        yield from emit_fit_leak(
            self, event, ctx, overlap=overlap,
            what="features are selected using the held-out labels",
        )


@register
class S002(StaticRule):
    id = "S002"
    name = "train_test_split(shuffle=True) on temporal data"
    category = Category.TEMPORAL
    severity = Severity.MEDIUM
    layers = (Layer.STATIC, Layer.DATA)
    references = (SKLEARN_CV,)
    rationale = (
        "Shuffling before splitting time-ordered data lets the model train on the future "
        "and evaluate on the past. Use shuffle=False or TimeSeriesSplit when a datetime "
        "column/index is present."
    )

    _TIME_HINTS = ("date", "time", "timestamp", "datetime", "_dt", "year", "month", "day")

    def check(self, ctx: StaticContext) -> Iterable[Finding]:
        if not self._temporal_signal(ctx):
            return
        for split in ctx.dataflow.splits:
            if split.func_name != "train_test_split":
                continue
            shuffle = has_kwarg(split.node, "shuffle")
            if shuffle is not None and kwarg_is_false(shuffle):
                continue
            yield Finding(
                rule_id=self.id,
                category=self.category,
                severity=self.severity,
                layer=Layer.STATIC,
                message=(
                    "train_test_split shuffles by default but the data appears temporal; "
                    "pass shuffle=False or use TimeSeriesSplit." + notebook_note(ctx)
                ),
                location=location_of(split.node, ctx),
                confidence=0.6,
                references=self.references,
                fix=Fix(summary="Add shuffle=False or switch to TimeSeriesSplit.", autofixable=False),
                evidence={"func": split.func_name},
            )

    def _temporal_signal(self, ctx: StaticContext) -> bool:
        for node in ast.walk(ctx.tree):
            if isinstance(node, ast.Name):
                if text_has_hint(node.id, self._TIME_HINTS):
                    return True
            if isinstance(node, ast.Attribute) and node.attr in ("to_datetime", "resample"):
                return True
            if isinstance(node, ast.Call) and self._call_has_temporal_signal(node, ctx):
                return True
        return False

    def _call_has_temporal_signal(self, node: ast.Call, ctx: StaticContext) -> bool:
        name = ctx.dataflow.resolve_name(node.func) or ""
        tail = name.rsplit(".", 1)[-1]
        if tail in {"to_datetime", "date_range", "resample"}:
            return True
        if tail == "astype" and any(
            isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and "datetime" in arg.value.lower()
            for arg in [*node.args, *(kw.value for kw in node.keywords if kw.arg)]
        ):
            return True
        if tail == "read_csv":
            for kw in node.keywords:
                if kw.arg == "parse_dates":
                    return True
        if tail in {"sort_values", "sort_index"}:
            return self._args_reference_time_columns(node)
        return False

    def _args_reference_time_columns(self, node: ast.Call) -> bool:
        values = list(node.args)
        values.extend(kw.value for kw in node.keywords if kw.arg in {"by", "on", "key"})
        for value in values:
            for sub in ast.walk(value):
                if (
                    isinstance(sub, ast.Constant)
                    and isinstance(sub.value, str)
                    and text_has_hint(sub.value, self._TIME_HINTS)
                ):
                    return True
        return False
