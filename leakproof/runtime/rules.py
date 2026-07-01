"""Runtime-only built-in rules: T001 (repeated test evaluation).

Multi-layer rules (P001/P003/S004/C004/M004/T002) implement their runtime
detector on the same class registered in the static layer; importing the static
rules package registers them, so they are not re-registered here.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..core.context import RuntimeContext
from ..core.models import Category, Finding, Layer, Location, Severity
from ..core.references import DWORK_2015_REUSABLE_HOLDOUT
from ..core.registry import register
from ..core.rule import RuntimeRule

# Ensure the multi-layer static+runtime rules are imported/registered.
from ..static import rules as _static_rules  # noqa: F401
from .taint import EventKind, RuntimeEvent, TaintTable


@register
class T001(RuntimeRule):
    id = "T001"
    name = "test set evaluated multiple times"
    category = Category.ADAPTIVITY
    severity = Severity.MEDIUM
    layers = (Layer.RUNTIME,)
    references = (DWORK_2015_REUSABLE_HOLDOUT,)
    rationale = (
        "Scoring against the same held-out split repeatedly within a run is adaptive data "
        "analysis: each peek erodes the validity of the held-out estimate. Touch the test "
        "set once."
    )

    def on_event(self, event: RuntimeEvent, ctx: RuntimeContext) -> Iterable[Finding]:
        if event.kind is not EventKind.SCORE:
            return
        seen = event.payload.get("seen", [])
        if not seen:
            return
        eval_side = ctx.taint.eval_side()
        # only count scores that actually hit the held-out split
        if not (set(seen) & eval_side):
            return
        sig = TaintTable.signature(list(seen))
        counts = ctx.scratch.setdefault("score_signatures", {})
        counts[sig] = counts.get(sig, 0) + 1
        n = counts[sig]
        if n >= 2:
            cls = event.payload.get("class", "estimator")
            yield Finding(
                rule_id=self.id,
                category=self.category,
                severity=self.severity,
                layer=Layer.RUNTIME,
                message=(
                    f"The same held-out split has now been scored {n} times in this run "
                    f"(adaptive overfitting); evaluate the test set once."
                ),
                location=Location(
                    context_label=f"repeated test score @ {event.call_site} run #{event.run_ordinal}"
                ),
                confidence=0.85,
                references=self.references,
                evidence={"times_scored": n, "class": cls},
            )
