"""JSON report schema 2.0."""

from __future__ import annotations

import dataclasses
import json
import math
from enum import Enum
from pathlib import Path
from typing import Any, cast

from ..core.aggregate import Summary
from ..core.models import AnalysisResult, Finding

SCHEMA_VERSION = "2.0"


def build(
    findings: list[Finding] | None = None,
    summary: Summary | None = None,
    *,
    result: AnalysisResult | None = None,
) -> dict[str, Any]:
    if result is not None:
        findings = result.findings
        summary = result.summary
    findings = findings or []
    summary = summary or Summary(total=len(findings), displayed=len(findings))
    output: dict[str, Any] = {
        "version": SCHEMA_VERSION,
        "tool": {"name": "ml-leakproof", "version": _tool_version()},
        "summary": summary.to_dict(),
        "findings": [finding.to_dict() for finding in findings],
    }
    if result is not None:
        output.update(
            {
                "diagnostics": [diagnostic.to_dict() for diagnostic in result.diagnostics],
                "coverage": result.coverage.to_dict(),
                "completion": result.completion.value,
                "script_exit_status": result.script_exit_status,
                "metadata": result.metadata,
            }
        )
    else:
        output.update(
            {
                "diagnostics": [],
                "coverage": {},
                "completion": "complete",
                "script_exit_status": None,
                "metadata": {},
            }
        )
    return cast(dict[str, Any], _json_safe(output))


def render(
    findings: list[Finding] | None = None,
    summary: Summary | None = None,
    *,
    result: AnalysisResult | None = None,
) -> str:
    obj = build(findings, summary, result=result)
    text = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)
    validate(obj)
    return text


def validate(obj: dict[str, Any]) -> None:
    """Validate the stable top-level contract and JSON scalar safety."""

    required = {"version", "tool", "summary", "findings", "diagnostics", "coverage", "completion"}
    missing = required - set(obj)
    if missing:
        raise ValueError(f"JSON report missing required keys: {sorted(missing)}")
    if obj["version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported JSON report schema: {obj['version']}")
    if obj["completion"] not in {"complete", "partial", "failed"}:
        raise ValueError("invalid report completion status")
    json.dumps(obj, allow_nan=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        fields = getattr(value, "__dataclass_fields__", {})
        return _json_safe({name: getattr(value, name) for name in fields})
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    # NumPy scalar values are common in evidence assembled by integrations;
    # convert them explicitly rather than relying on json.dumps(default=str).
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"report contains unsupported JSON value: {type(value).__name__}") from exc
    return value


def _tool_version() -> str:
    try:
        from .. import __version__

        return __version__
    except Exception:  # pragma: no cover
        return "0"
