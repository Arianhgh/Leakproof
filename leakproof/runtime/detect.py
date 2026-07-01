"""Shared runtime detection helpers used by multi-layer rules' on_event."""

from __future__ import annotations

from collections.abc import Iterable

from ..core.context import RuntimeContext
from ..core.models import Finding, Layer, Location
from .taint import EventKind, RuntimeEvent


def fit_saw_eval_rows(event: RuntimeEvent, ctx: RuntimeContext) -> int:
    """Number of eval-side rows a fit event consumed (0 if none/not a fit)."""
    if event.kind is not EventKind.FIT:
        return 0
    seen = set(event.payload.get("seen", []))
    if not seen:
        return 0
    eval_side = ctx.taint.eval_side()
    return len(seen & eval_side)


def emit_fit_leak(
    rule,
    event: RuntimeEvent,
    ctx: RuntimeContext,
    *,
    overlap: int,
    what: str,
) -> Iterable[Finding]:
    cls = event.payload.get("class", "estimator")
    label = f"{cls}.fit @ {event.call_site} run #{event.run_ordinal}"
    yield Finding(
        rule_id=rule.id,
        category=rule.category,
        severity=rule.severity,
        layer=Layer.RUNTIME,
        message=(
            f"At runtime, `{cls}` was fit on {overlap} row(s) that are in the held-out "
            f"split — {what}."
        ),
        location=Location(context_label=label),
        confidence=0.95,
        references=rule.references,
        evidence={"class": cls, "eval_rows_seen": overlap, "call_site": event.call_site},
    )
