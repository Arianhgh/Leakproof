"""leakproof — a hybrid (static + data + runtime) leakage and evaluation-rigor checker.

Public API:
    leakproof.check(paths)              -> list[Finding]   (static layer)
    leakproof.audit_data(train, test..) -> list[Finding]   (data layer)
    leakproof.watch()                   -> context manager  (runtime layer)
"""

from __future__ import annotations

from pathlib import Path

from .core.config import Config
from .core.models import (
    Category,
    Finding,
    Fix,
    Layer,
    Location,
    Severity,
)

__version__ = "0.1.0"

__all__ = [
    "check",
    "audit_data",
    "watch",
    "Config",
    "Finding",
    "Severity",
    "Category",
    "Layer",
    "Location",
    "Fix",
    "__version__",
]


def check(paths, config: Config | None = None, *, root: Path | None = None) -> list[Finding]:
    """Run the static layer over the given paths and return findings."""
    from .static.engine import StaticEngine

    cfg = config or Config.load()
    if isinstance(paths, (str, Path)):
        paths = [paths]
    engine = StaticEngine(cfg)
    return engine.run([Path(p) for p in paths], root=root)


def audit_data(
    train,
    test,
    *,
    val=None,
    target: str | None = None,
    group: str | None = None,
    time: str | None = None,
    feature_types: dict | None = None,
    config: Config | None = None,
) -> list[Finding]:
    """Run the data layer over explicit train/test/val frames."""
    from .data.engine import DataEngine
    from .data.input import DataAuditInput

    cfg = config or Config.load()
    audit_input = DataAuditInput(
        train=train,
        test=test,
        val=val,
        target=target,
        group=group,
        time=time,
        feature_types=feature_types,
    )
    return DataEngine(cfg).run(audit_input)


def watch(config: Config | None = None):
    """Context manager that instruments sklearn/pandas to detect leakage at runtime."""
    from .runtime.engine import watch as _watch

    return _watch(config or Config.load())
