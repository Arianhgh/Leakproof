"""pandas adapter — lineage-relevant constructs (concat/merge)."""

from __future__ import annotations

from . import FrameworkAdapter


def adapter() -> FrameworkAdapter:
    return FrameworkAdapter(
        name="pandas",
        # pandas itself doesn't fit estimators, but concat/merge matter for
        # lineage; we surface them via the dataflow concat-detection, not here.
        learn_methods=set(),
        apply_methods=set(),
    )
