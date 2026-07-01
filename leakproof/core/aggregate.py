"""Finding aggregation: dedupe, severity gating, summary statistics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .models import Category, Finding, Severity

_SEVERITY_ORDER = [
    Severity.CRITICAL,
    Severity.HIGH,
    Severity.MEDIUM,
    Severity.LOW,
    Severity.INFO,
]


def dedupe(findings: list[Finding]) -> list[Finding]:
    """Drop duplicate findings sharing the same ``key`` across layers.

    Keep the one with the highest confidence (ties broken by severity), so a
    runtime confirmation supersedes a lower-confidence static guess at the same
    site.
    """
    best: dict[tuple, Finding] = {}
    for f in findings:
        existing = best.get(f.key)
        if existing is None:
            best[f.key] = f
            continue
        if (f.confidence, f.severity.gate_rank) > (existing.confidence, existing.severity.gate_rank):
            best[f.key] = f
    return list(best.values())


def sort_findings(findings: list[Finding]) -> list[Finding]:
    def sort_key(f: Finding):
        return (
            str(f.location.file or ""),
            -f.severity.gate_rank,
            f.location.line or 0,
            f.rule_id,
        )

    return sorted(findings, key=sort_key)


@dataclass
class Summary:
    total: int = 0
    by_severity: dict[str, int] = field(default_factory=dict)
    by_category: dict[str, int] = field(default_factory=dict)
    by_layer: dict[str, int] = field(default_factory=dict)
    by_rule: dict[str, int] = field(default_factory=dict)
    gate: str = Severity.HIGH.value
    gated_count: int = 0  # findings at/above gate

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "by_severity": self.by_severity,
            "by_category": self.by_category,
            "by_layer": self.by_layer,
            "by_rule": self.by_rule,
            "gate": self.gate,
            "gated_count": self.gated_count,
        }


def summarize(findings: list[Finding], gate: Severity) -> Summary:
    sev = Counter(f.severity.value for f in findings)
    cat = Counter(f.category.value for f in findings)
    layer = Counter(f.layer.value for f in findings)
    rule = Counter(f.rule_id for f in findings)
    gated = sum(1 for f in findings if f.severity.gate_rank >= gate.gate_rank)
    return Summary(
        total=len(findings),
        by_severity={s.value: sev.get(s.value, 0) for s in _SEVERITY_ORDER if sev.get(s.value, 0)},
        by_category=dict(sorted(cat.items())),
        by_layer=dict(sorted(layer.items())),
        by_rule=dict(sorted(rule.items())),
        gate=gate.value,
        gated_count=gated,
    )


def gate_failed(findings: list[Finding], gate: Severity) -> bool:
    return any(f.severity.gate_rank >= gate.gate_rank for f in findings)


def categories_present(findings: list[Finding]) -> set[Category]:
    return {f.category for f in findings}
