"""Split-hygiene static rules: S001, S002, S003, S004."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ...core.context import RuntimeContext, StaticContext
from ...core.models import Category, Finding, Fix, Layer, Severity
from ...core.references import KAUFMAN_2012, SKLEARN_CV, SKLEARN_PITFALLS
from ...core.registry import register
from ...core.rule import StaticRule
from ._helpers import (
    has_kwarg,
    input_has_related_hint,
    kwarg_is_false,
    location_of,
    notebook_note,
)
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

    def _make(self, fit: Any, ctx: StaticContext) -> Finding:
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

    def on_event(self, event: Any, ctx: Any) -> Iterable[Finding]:
        from ...runtime.detect import emit_fit_leak, fit_saw_eval_rows
        from ...static.adapters import load_adapters

        overlap = fit_saw_eval_rows(event, ctx)
        if not overlap:
            return
        if event.payload.get("concat_sources"):
            return
        adapters = ctx.scratch.setdefault("_adapters", load_adapters())
        cls = event.payload.get("class", "")
        if adapters.is_stateless_transformer(cls):
            return
        if not adapters.is_transformer(cls) or event.payload.get("method") == "fit_transform":
            return
        if cls in {"SimpleImputer", "KNNImputer", "IterativeImputer"}:
            return
        if adapters.is_feature_selector(cls) or adapters.is_resampler(cls):
            return
        yield from emit_fit_leak(
            self,
            event,
            ctx,
            overlap=overlap,
            what="learned transformer state includes held-out rows",
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

    def on_event(self, event: Any, ctx: RuntimeContext) -> Iterable[Finding]:
        from ...runtime.detect import emit_fit_leak, fit_saw_eval_rows
        from ...runtime.taint import EventKind
        from ...static.adapters import load_adapters

        if event.kind is not EventKind.FIT:
            return
        if event.payload.get("method") not in {"fit_resample", "fit"}:
            return
        adapters = ctx.scratch.setdefault("_adapters", load_adapters())
        if not adapters.is_resampler(event.payload.get("class", "")):
            return
        overlap = fit_saw_eval_rows(event, ctx)
        if not overlap or event.payload.get("concat_sources"):
            return
        yield from emit_fit_leak(
            self,
            event,
            ctx,
            overlap=overlap,
            what="resampling was applied across the held-out boundary",
        )

    def _make(self, fit: Any, ctx: StaticContext) -> Finding:
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

    def _make(self, fit: Any, ctx: StaticContext) -> Finding:
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

    def on_event(self, event: Any, ctx: Any) -> Iterable[Finding]:
        from ...runtime.detect import emit_fit_leak, fit_saw_eval_rows
        from ...static.adapters import load_adapters

        overlap = fit_saw_eval_rows(event, ctx)
        if not overlap:
            return
        if event.payload.get("concat_sources"):
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
    advisory_only = True
    id = "S002"
    name = "train_test_split(shuffle=True) on temporal data"
    category = Category.TEMPORAL
    severity = Severity.MEDIUM
    layers = (Layer.STATIC,)
    references = (SKLEARN_CV,)
    rationale = (
        "Shuffling before splitting time-ordered data lets the model train on the future "
        "and evaluate on the past. Use shuffle=False or TimeSeriesSplit when a datetime "
        "column/index is present."
    )

    _TIME_HINTS = ("date", "time", "timestamp", "datetime", "_dt", "year", "month", "day")

    def check(self, ctx: StaticContext) -> Iterable[Finding]:
        for split in ctx.dataflow.splits:
            if split.func_name != "train_test_split":
                continue
            shuffle = has_kwarg(split.node, "shuffle")
            if shuffle is not None and kwarg_is_false(shuffle):
                continue
            if not input_has_related_hint(
                ctx,
                split.node,
                split.input_vars,
                split.input_binding_versions,
                self._TIME_HINTS,
                scope_id=split.scope_id,
            ):
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
                advisory_only=self.advisory_only,
                fix=Fix(summary="Add shuffle=False or switch to TimeSeriesSplit.", autofixable=False),
                evidence={"func": split.func_name},
            )
