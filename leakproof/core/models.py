"""Core data models for leakproof findings."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Severity(str, Enum):
    CRITICAL = "critical"  # near-certain invalid results (e.g., exact train/test overlap)
    HIGH = "high"  # very likely leakage (e.g., scaler fit on full X)
    MEDIUM = "medium"  # probable problem or strong smell
    LOW = "low"  # hygiene / advisory (missing seed)
    INFO = "info"

    @property
    def gate_rank(self) -> int:
        return {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}[self.value]

    @classmethod
    def from_str(cls, value: str) -> Severity:
        return cls(value.strip().lower())


class Category(str, Enum):
    SPLIT = "split"
    CV = "cross_validation"
    PREPROCESSING = "preprocessing"
    DATA_OVERLAP = "data_overlap"
    TARGET_LEAKAGE = "target_leakage"
    TEMPORAL = "temporal"
    ADAPTIVITY = "adaptivity"  # test-set reuse / multiple comparisons
    METRIC = "metric"
    DETERMINISM = "determinism"


class Layer(str, Enum):
    STATIC = "static"
    DATA = "data"
    RUNTIME = "runtime"


@dataclass(frozen=True)
class Location:
    file: Path | None = None
    line: int | None = None
    col: int | None = None
    end_line: int | None = None
    end_col: int | None = None
    snippet: str | None = None
    # for data/runtime findings that aren't tied to a source line:
    context_label: str | None = None  # e.g., "train_test_split @ pipeline.py:42 run #2"

    def short(self) -> str:
        if self.file is not None and self.line is not None:
            return f"{self.file}:{self.line}"
        if self.context_label:
            return self.context_label
        if self.file is not None:
            return str(self.file)
        return "<unknown>"


@dataclass(frozen=True)
class Fix:
    summary: str  # one-line human description
    suggested_diff: str | None = None  # unified diff, optional
    autofixable: bool = False


@dataclass(frozen=True)
class Finding:
    rule_id: str  # e.g., "P001"
    category: Category
    severity: Severity
    layer: Layer
    message: str  # what + why, <= 2 sentences
    location: Location
    fix: Fix | None = None
    confidence: float = 1.0  # 0..1; static heuristics may be < 1
    references: tuple[str, ...] = ()  # doc anchors / paper URLs
    evidence: dict = field(default_factory=dict)  # rule-specific structured data

    @property
    def key(self) -> tuple:
        # used for dedupe across layers
        return (
            self.rule_id,
            str(self.location.file),
            self.location.line,
            self.location.context_label,
        )

    def to_dict(self) -> dict:
        loc = self.location
        return {
            "rule_id": self.rule_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "layer": self.layer.value,
            "message": self.message,
            "confidence": self.confidence,
            "references": list(self.references),
            "evidence": self.evidence,
            "location": {
                "file": str(loc.file) if loc.file is not None else None,
                "line": loc.line,
                "col": loc.col,
                "end_line": loc.end_line,
                "end_col": loc.end_col,
                "snippet": loc.snippet,
                "context_label": loc.context_label,
            },
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
