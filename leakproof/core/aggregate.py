"""Finding aggregation and exit-code gating."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace

from .models import Category, Finding, Severity

_SEVERITY_ORDER = [
    Severity.CRITICAL,
    Severity.HIGH,
    Severity.MEDIUM,
    Severity.LOW,
    Severity.INFO,
]


def dedupe(findings: list[Finding]) -> list[Finding]:
    """Keep the strongest finding for each location/rule key."""
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
    gated_count: int = 0  # findings that can fail the configured gate
    gate_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "by_severity": self.by_severity,
            "by_category": self.by_category,
            "by_layer": self.by_layer,
            "by_rule": self.by_rule,
            "gate": self.gate,
            "gated_count": self.gated_count,
            "gate_confidence": self.gate_confidence,
        }


def is_gateable(finding: Finding, gate: Severity, gate_confidence: float = 0.0) -> bool:
    if finding.gateable is not None:
        return finding.gateable
    return (
        finding.severity.gate_rank >= gate.gate_rank
        and finding.confidence >= gate_confidence
    )


def annotate_gateability(
    findings: list[Finding],
    *,
    gate: Severity,
    gate_confidence: float,
    profile: str,
) -> list[Finding]:
    out: list[Finding] = []
    for f in findings:
        gateable = is_gateable(f, gate, gate_confidence)
        evidence = dict(f.evidence)
        reason_codes = list(evidence.get("reason_codes", []))
        if f.severity.gate_rank < gate.gate_rank:
            reason_codes.append("severity-below-gate")
        if f.confidence < gate_confidence:
            reason_codes.append("confidence-below-gate")
        if f.gateable is False:
            reason_codes.append("rule-marked-non-gateable")
        if f.layer.value == "static" and f.confidence < 0.60:
            reason_codes.append("advisory-confidence")
        elif f.layer.value == "static" and f.confidence < 0.85:
            reason_codes.append("probable-static")
        if gateable:
            reason_codes.append("gateable")
        if reason_codes:
            evidence["reason_codes"] = sorted(set(reason_codes))
        out.append(replace(f, gateable=gateable, profile=profile, evidence=evidence))
    return out


def filter_min_confidence(findings: list[Finding], min_confidence: float = 0.0) -> list[Finding]:
    return [f for f in findings if f.confidence >= min_confidence]


def summarize(
    findings: list[Finding],
    gate: Severity,
    gate_confidence: float = 0.0,
) -> Summary:
    sev = Counter(f.severity.value for f in findings)
    cat = Counter(f.category.value for f in findings)
    layer = Counter(f.layer.value for f in findings)
    rule = Counter(f.rule_id for f in findings)
    gated = sum(1 for f in findings if is_gateable(f, gate, gate_confidence))
    return Summary(
        total=len(findings),
        by_severity={s.value: sev.get(s.value, 0) for s in _SEVERITY_ORDER if sev.get(s.value, 0)},
        by_category=dict(sorted(cat.items())),
        by_layer=dict(sorted(layer.items())),
        by_rule=dict(sorted(rule.items())),
        gate=gate.value,
        gated_count=gated,
        gate_confidence=gate_confidence,
    )


def gate_failed(
    findings: list[Finding],
    gate: Severity,
    gate_confidence: float = 0.0,
) -> bool:
    return any(is_gateable(f, gate, gate_confidence) for f in findings)


def categories_present(findings: list[Finding]) -> set[Category]:
    return {f.category for f in findings}
