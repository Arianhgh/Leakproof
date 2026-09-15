"""numpy adapter — determinism-relevant seeding."""

from __future__ import annotations

from . import FrameworkAdapter


def adapter() -> FrameworkAdapter:
    return FrameworkAdapter(
        name="numpy",
        learn_methods=set(),
        apply_methods=set(),
    )
