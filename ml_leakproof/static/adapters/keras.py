"""Keras adapter entries."""

from __future__ import annotations

from . import FrameworkAdapter


def adapter() -> FrameworkAdapter:
    return FrameworkAdapter(
        name="keras",
        estimators={"Sequential", "Model"},
        learn_methods=set(),
        apply_methods=set(),
    )
