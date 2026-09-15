"""Preprocessing-leakage static rules: P001, P002, P003."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ...core.context import RuntimeContext, StaticContext
from ...core.models import Category, Finding, Fix, Layer, Severity
from ...core.references import KAUFMAN_2012, SKLEARN_PITFALLS
from ...core.registry import register
from ...core.rule import RuntimeRule, StaticRule
from ...runtime.taint import RuntimeEvent
from ._helpers import classify_full_fit, location_of, notebook_note

_REFS = (SKLEARN_PITFALLS, KAUFMAN_2012)


def _event_classes(event: RuntimeEvent) -> list[str]:
    classes = [str(event.payload.get("class", ""))]
    classes.extend(str(value) for value in event.payload.get("components", []))
    return [value for value in classes if value]


class _FullFitRule(StaticRule, RuntimeRule):
    """Base for rules driven by the shared full-fit classifier.

    These rules detect in both the static and runtime layers; subclasses that
    have no runtime detector simply inherit the no-op ``on_event``.
    """

    layers = (Layer.STATIC, Layer.RUNTIME)
    references = _REFS

    def check(self, ctx: StaticContext) -> Iterable[Finding]:
        for fit in ctx.dataflow.fit_calls:
            owner = classify_full_fit(fit, ctx)
            if owner != self.id:
                continue
            # When the fitted output is consumed by a CV/search operation,
            # C001 owns the observation.  This keeps a single operation from
            # producing both a generic full-data finding and a more useful
            # outside-the-fold finding.
            if (
                self.id in {"P001", "P002", "P003", "S001", "S003", "S004"}
                and ctx.dataflow.uses_cv_utility()
                and ctx.dataflow.fit_output_used_by_cv(fit)
            ):
                continue
            yield self._make(fit, ctx)

    def on_event(self, event: RuntimeEvent, ctx: RuntimeContext) -> Iterable[Finding]:
        return ()

    def _make(self, fit: Any, ctx: StaticContext) -> Finding:  # pragma: no cover - overridden
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

    def _make(self, fit: Any, ctx: StaticContext) -> Finding:
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
        classes = _event_classes(event)
        if event.payload.get("method") != "fit_transform" and not (
            event.payload.get("method") == "fit" and len(classes) > 1
        ):
            return
        if event.payload.get("concat_sources"):
            return
        # P001 owns generic transformers at runtime (imputers -> P003, selectors -> S004)
        if not adapters.is_transformer(cls) and not any(
            adapters.is_transformer(component) for component in classes[1:]
        ):
            return
        # Normalizer/Binarizer/FunctionTransformer/HashingVectorizer do not
        # learn cross-row statistics.  A call to their fit_transform is not a
        # preprocessing leakage event merely because it receives test rows.
        if adapters.is_stateless_transformer(cls):
            return
        if cls in {"SimpleImputer", "KNNImputer", "IterativeImputer"} or any(
            component in {"SimpleImputer", "KNNImputer", "IterativeImputer"}
            for component in classes[1:]
        ):
            return
        if adapters.is_feature_selector(cls) or any(
            adapters.is_feature_selector(component) for component in classes[1:]
        ):
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

    def _make(self, fit: Any, ctx: StaticContext) -> Finding:
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

    def on_event(self, event: RuntimeEvent, ctx: RuntimeContext) -> Iterable[Finding]:
        from ...runtime.detect import emit_fit_leak, fit_saw_eval_rows
        from ...static.adapters import load_adapters

        overlap = fit_saw_eval_rows(event, ctx)
        if not overlap:
            return
        adapters = ctx.scratch.setdefault("_adapters", load_adapters())
        classes = _event_classes(event)
        if not adapters.is_transformer(event.payload.get("class", "")) and not any(
            adapters.is_transformer(component) for component in classes[1:]
        ):
            return
        if event.payload.get("method") not in {"fit", "fit_transform"}:
            return
        if not event.payload.get("concat_sources"):
            return
        yield from emit_fit_leak(
            self,
            event,
            ctx,
            overlap=overlap,
            what="the fitted input was assembled from multiple split partitions",
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

    def _make(self, fit: Any, ctx: StaticContext) -> Finding:
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
        if event.payload.get("concat_sources"):
            return
        classes = _event_classes(event)
        if not any(
            component in {"SimpleImputer", "KNNImputer", "IterativeImputer"}
            for component in classes
        ):
            return
        yield from emit_fit_leak(
            self, event, ctx, overlap=overlap,
            what="imputation statistics are learned from the held-out rows",
        )
