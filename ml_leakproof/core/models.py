"""Stable public models shared by every Leakproof layer.

The models are dependency free.  A scan can therefore report a missing optional
dependency or parser failure without importing pandas, scikit-learn, or an LLM
SDK just to serialize the result.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, cast


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def gate_rank(self) -> int:
        return {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}[self.value]

    @classmethod
    def from_str(cls, value: str) -> Severity:
        if not isinstance(value, str):
            raise ValueError("severity must be a string")
        return cls(value.strip().lower())


class Category(str, Enum):
    SPLIT = "split"
    CV = "cross_validation"
    PREPROCESSING = "preprocessing"
    DATA_OVERLAP = "data_overlap"
    TARGET_LEAKAGE = "target_leakage"
    TEMPORAL = "temporal"
    ADAPTIVITY = "adaptivity"
    METRIC = "metric"
    DETERMINISM = "determinism"


class Layer(str, Enum):
    STATIC = "static"
    DATA = "data"
    RUNTIME = "runtime"


class DiagnosticLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class CompletionStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True)
class Location:
    file: Path | None = None
    line: int | None = None
    col: int | None = None
    end_line: int | None = None
    end_col: int | None = None
    snippet: str | None = None
    context_label: str | None = None
    cell_id: str | None = None
    cell_index: int | None = None
    cell_line: int | None = None

    def short(self) -> str:
        if self.file is not None and self.line is not None:
            label = f" ({self.context_label})" if self.context_label else ""
            return f"{self.file}:{self.line}{label}"
        if self.context_label:
            return self.context_label
        if self.file is not None:
            return str(self.file)
        return "<unknown>"


@dataclass(frozen=True)
class Fix:
    summary: str
    suggested_diff: str | None = None
    autofixable: bool = False


@dataclass(frozen=True)
class Diagnostic:
    """An operational diagnostic, separate from a methodology finding."""

    code: str
    message: str
    level: DiagnosticLevel = DiagnosticLevel.WARNING
    layer: Layer | None = None
    rule_id: str | None = None
    location: Location | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "level": self.level.value,
            "layer": self.layer.value if self.layer else None,
            "rule_id": self.rule_id,
            "location": _location_dict(self.location),
            "details": _jsonable(self.details),
        }


@dataclass
class Coverage:
    """The inputs and checks a run actually inspected."""

    requested_inputs: list[str] = field(default_factory=list)
    analyzed_inputs: list[str] = field(default_factory=list)
    skipped_inputs: list[str] = field(default_factory=list)
    failed_inputs: list[str] = field(default_factory=list)
    executed_checks: list[str] = field(default_factory=list)
    unavailable_checks: list[str] = field(default_factory=list)
    sampled_checks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def requested(self) -> list[str]:
        return self.requested_inputs

    @property
    def analyzed(self) -> list[str]:
        return self.analyzed_inputs

    @property
    def skipped(self) -> list[str]:
        return self.skipped_inputs

    @property
    def failed(self) -> list[str]:
        return self.failed_inputs

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_inputs": list(self.requested_inputs),
            "analyzed_inputs": list(self.analyzed_inputs),
            "skipped_inputs": list(self.skipped_inputs),
            "failed_inputs": list(self.failed_inputs),
            "executed_checks": sorted(set(self.executed_checks)),
            "unavailable_checks": sorted(set(self.unavailable_checks)),
            "sampled_checks": sorted(set(self.sampled_checks)),
            "notes": list(self.notes),
        }


@dataclass
class AnalysisResult:
    """Result returned by result-oriented APIs."""

    findings: list[Finding] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    summary: Any = None
    coverage: Coverage = field(default_factory=Coverage)
    completion: CompletionStatus = CompletionStatus.COMPLETE
    script_exit_status: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.completion is CompletionStatus.COMPLETE

    @property
    def partial(self) -> bool:
        return self.completion is CompletionStatus.PARTIAL

    @property
    def failed(self) -> bool:
        return self.completion is CompletionStatus.FAILED

    def to_dict(self) -> dict[str, Any]:
        summary = self.summary.to_dict() if hasattr(self.summary, "to_dict") else self.summary
        return {
            "findings": [_jsonable(f.to_dict()) for f in self.findings],
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "summary": _jsonable(summary),
            "coverage": self.coverage.to_dict(),
            "completion": self.completion.value,
            "script_exit_status": self.script_exit_status,
            "metadata": _jsonable(self.metadata),
        }


class AnalysisError(RuntimeError):
    """Raised by list-returning APIs when the requested analysis was incomplete."""

    def __init__(self, message: str, result: AnalysisResult):
        super().__init__(message)
        self.result = result
        self.partial_result = result


@dataclass(frozen=True)
class Finding:
    rule_id: str
    category: Category
    severity: Severity
    layer: Layer
    message: str
    location: Location
    fix: Fix | None = None
    confidence: float = 1.0
    references: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)
    advisory_only: bool = False
    # Kept for source compatibility with 0.1 reports.  Gate calculations ignore
    # this cached value and always recompute it from current settings.
    gateable: bool | None = None
    profile: str | None = None

    @property
    def key(self) -> tuple[Any, ...]:
        identity_keys = {
            "feature",
            "features",
            "split_pair",
            "pair",
            "modality",
            "column",
            "group",
            "call_site",
        }
        evidence_identity = tuple(
            (key, _stable_value(value))
            for key, value in sorted(self.evidence.items())
            if key in identity_keys
        )
        return (
            self.rule_id,
            self.layer.value,
            str(self.location.file),
            self.location.line,
            self.location.col,
            self.location.end_line,
            self.location.context_label,
            self.location.cell_id,
            evidence_identity,
        )

    def to_dict(self) -> dict[str, Any]:
        loc = self.location
        out: dict[str, Any] = {
            "rule_id": self.rule_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "layer": self.layer.value,
            "message": self.message,
            "confidence": self.confidence,
            "references": list(self.references),
            "evidence": _jsonable(self.evidence),
            "advisory_only": self.advisory_only,
            "location": _location_dict(loc),
            "fix": (
                {
                    "summary": self.fix.summary,
                    "suggested_diff": self.fix.suggested_diff,
                    "autofixable": self.fix.autofixable,
                }
                if self.fix is not None
                else None
            ),
        }
        if self.gateable is not None:
            out["gateable"] = self.gateable
        if self.profile is not None:
            out["profile"] = self.profile
        return out


def _location_dict(location: Location | None) -> dict[str, Any] | None:
    if location is None:
        return None
    return {
        "file": str(location.file) if location.file is not None else None,
        "line": location.line,
        "col": location.col,
        "end_line": location.end_line,
        "end_col": location.end_col,
        "snippet": location.snippet,
        "context_label": location.context_label,
        "cell_id": location.cell_id,
        "cell_index": location.cell_index,
        "cell_line": location.cell_line,
    }


def _stable_value(value: Any) -> str:
    try:
        return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    except Exception:
        return repr(value)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return _jsonable(
            {
                field.name: getattr(value, field.name)
                for field in dataclasses.fields(cast(Any, value))
            }
        )
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(v) for v in sorted(value, key=repr)]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported JSON value: {type(value).__name__}") from exc
    return value
