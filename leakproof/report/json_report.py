"""Stable, versioned JSON report."""

from __future__ import annotations

import json

from ..core.aggregate import Summary
from ..core.models import Finding

SCHEMA_VERSION = "1.0"


def build(findings: list[Finding], summary: Summary) -> dict:
    return {
        "version": SCHEMA_VERSION,
        "tool": {"name": "leakproof", "version": _tool_version()},
        "summary": summary.to_dict(),
        "findings": [f.to_dict() for f in findings],
    }


def render(findings: list[Finding], summary: Summary) -> str:
    return json.dumps(build(findings, summary), indent=2, default=str)


def _tool_version() -> str:
    try:
        from .. import __version__

        return __version__
    except Exception:  # pragma: no cover
        return "0"
