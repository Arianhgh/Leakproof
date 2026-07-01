"""Preprocessing-leakage static rules: P001, P002, P003."""

from __future__ import annotations

from collections.abc import Iterable

from ...core.context import RuntimeContext, StaticContext
from ...core.models import Category, Finding, Fix, Layer, Severity
from ...core.references import KAUFMAN_2012, SKLEARN_PITFALLS
from ...core.registry import register
from ...core.rule import RuntimeRule, StaticRule
from ...runtime.taint import RuntimeEvent
from ._helpers import classify_full_fit, location_of, notebook_note

_REFS = (SKLEARN_PITFALLS, KAUFMAN_2012)


class _FullFitRule(StaticRule, RuntimeRule):
    """Base for rules driven by the shared full-fit classifier.

    These rules detect in both the static and runtime layers; subclasses that
    have no runtime detector simply inherit the no-op ``on_event``.
    """

    layers = (Layer.STATIC, Layer.RUNTIME)
    references = _REFS

    def check(self, ctx: StaticContext) -> Iterable[Finding]:
        for fit in ctx.dataflow.fit_calls:
            if classify_full_fit(fit, ctx) != self.id:
                continue
            yield self._make(fit, ctx)

    def on_event(self, event: RuntimeEvent, ctx: RuntimeContext) -> Iterable[Finding]:
        return ()

    def _make(self, fit, ctx: StaticContext) -> Finding:  # pragma: no cover - overridden
        raise NotImplementedError


@register
class P001(_FullFitRule):
    id = "P001"
    name = "fit_transform on full X before split"
    category = Category.PREPROCESSING
    severity = Severity.HIGH
    rationale = (
        "Calling fit_transform on the entire feature matrix before splitting lets "
        "statistics from the test rows (means, variances, components) leak into the "
        "transform applied to training, inflating measured performance."
    )

    def _make(self, fit, ctx: StaticContext) -> Finding:
        cls = fit.class_name or "transformer"
        return Finding(
            rule_id=self.id,
            category=self.category,
            severity=self.severity,
            layer=Layer.STATIC,
            message=(
                f"`{cls}.fit_transform(...)` is fit on pre-split data; learned statistics "
                f"leak from the held-out rows. Fit only on the training split (or use a "
                f"Pipeline inside cross-validation)." + notebook_note(ctx)
            ),
            location=location_of(fit.node, ctx),
            confidence=0.9,
            references=self.references,
            fix=Fix(
                summary="Split first, then fit_transform on X_train and transform X_test.",
                autofixable=False,
            ),
            evidence={"class": cls, "method": fit.method, "taints": [t.value for t in fit.arg_taints + fit.output_taints]},
        )

    def on_event(self, event: RuntimeEvent, ctx: RuntimeContext) -> Iterable[Finding]:
        from ...runtime.detect import emit_fit_leak, fit_saw_eval_rows
        from ...static.adapters import load_adapters

        overlap = fit_saw_eval_rows(event, ctx)
        if not overlap:
            return
        adapters = ctx.scratch.setdefault("_adapters", load_adapters())
        cls = event.payload.get("class", "")
        # P001 owns generic transformers at runtime (imputers -> P003, selectors -> S004)
        if not adapters.is_transformer(cls):
            return
        if cls in {"SimpleImputer", "KNNImputer", "IterativeImputer"}:
            return
        if adapters.is_feature_selector(cls):
            return
        yield from emit_fit_leak(
            self, event, ctx, overlap=overlap,
            what="statistics from the test rows leak into the transform",
        )


@register
class P002(_FullFitRule):
    id = "P002"
    name = "fit on concat([train, test])"
    category = Category.PREPROCESSING
    severity = Severity.HIGH
    rationale = (
        "Fitting a transformer on data concatenated from both train and test splits "
        "directly exposes the model to test-set distribution information."
    )

    def _make(self, fit, ctx: StaticContext) -> Finding:
        cls = fit.class_name or "transformer"
        return Finding(
            rule_id=self.id,
            category=self.category,
            severity=self.severity,
            layer=Layer.STATIC,
            message=(
                f"`{cls}.{fit.method}(...)` is fit on data concatenated across the train/test "
                f"boundary; fit on the training split only." + notebook_note(ctx)
            ),
            location=location_of(fit.node, ctx),
            confidence=0.92,
            references=self.references,
            fix=Fix(summary="Fit on the training frame, not concat([train, test]).", autofixable=False),
            evidence={"class": cls, "method": fit.method},
        )


@register
class P003(_FullFitRule):
    id = "P003"
    name = "imputation statistics from full data"
    category = Category.PREPROCESSING
    severity = Severity.HIGH
    rationale = (
        "Imputation parameters (mean/median/mode/neighbors) learned across the split "
        "encode test-set values into the filled-in training data."
    )

    def _make(self, fit, ctx: StaticContext) -> Finding:
        cls = fit.class_name or "imputer"
        return Finding(
            rule_id=self.id,
            category=self.category,
            severity=self.severity,
            layer=Layer.STATIC,
            message=(
                f"`{cls}.{fit.method}(...)` learns imputation statistics from pre-split data; "
                f"fit the imputer on the training split only." + notebook_note(ctx)
            ),
            location=location_of(fit.node, ctx),
            confidence=0.88,
            references=self.references,
            fix=Fix(summary="Fit the imputer on X_train, then transform X_test.", autofixable=False),
            evidence={"class": cls, "method": fit.method},
        )

    def on_event(self, event: RuntimeEvent, ctx: RuntimeContext) -> Iterable[Finding]:
        from ...runtime.detect import emit_fit_leak, fit_saw_eval_rows

        overlap = fit_saw_eval_rows(event, ctx)
        if not overlap:
            return
        if event.payload.get("class", "") not in {"SimpleImputer", "KNNImputer", "IterativeImputer"}:
            return
        yield from emit_fit_leak(
            self, event, ctx, overlap=overlap,
            what="imputation statistics are learned from the held-out rows",
        )
