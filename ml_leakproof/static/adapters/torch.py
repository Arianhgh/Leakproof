"""PyTorch adapter entries used by determinism rules."""

from __future__ import annotations

from . import FrameworkAdapter


def adapter() -> FrameworkAdapter:
    return FrameworkAdapter(
        name="torch",
        estimators={"Module"},
        learn_methods=set(),
        apply_methods=set(),
        seeded_callables=set(),
    )
