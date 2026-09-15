"""Deterministic aggregation, presentation filtering, and gate calculation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

from .models import Category, Finding, Severity

_SEVERITY_ORDER = [
    Severity.CRITICAL,
    Severity.HIGH,
    Severity.MEDIUM,
    Severity.LOW,
    Severity.INFO,
]


def dedupe(findings: list[Finding]) -> list[Finding]:
    """Deduplicate identical observations without collapsing distinct evidence."""

    best: dict[tuple[Any, ...], Finding] = {}
    for finding in findings:
        existing = best.get(finding.key)
        if existing is None or (
            finding.confidence,
            finding.severity.gate_rank,
            finding.message,
        ) > (
            existing.confidence,
            existing.severity.gate_rank,
            existing.message,
        ):
            best[finding.key] = finding
    return list(best.values())


def sort_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(
        findings,
        key=lambda f: (
            str(f.location.file or ""),
            f.location.line or 0,
            f.location.col or 0,
            f.rule_id,
            f.layer.value,
            f.location.context_label or "",
        ),
    )


@dataclass
class Summary:
    total: int = 0
    displayed: int | None = None
    hidden: int = 0
    by_severity: dict[str, int] = field(default_factory=dict)
    by_category: dict[str, int] = field(default_factory=dict)
    by_layer: dict[str, int] = field(default_factory=dict)
    by_rule: dict[str, int] = field(default_factory=dict)
    gate: str = Severity.HIGH.value
    gated_count: int = 0
    gate_confidence: float = 0.0
    advisory_count: int = 0
    diagnostics_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "displayed": self.displayed if self.displayed is not None else self.total,
            "hidden": self.hidden,
            "by_severity": self.by_severity,
            "by_category": self.by_category,
            "by_layer": self.by_layer,
            "by_rule": self.by_rule,
            "gate": self.gate,
            "gated_count": self.gated_count,
            "gate_confidence": self.gate_confidence,
            "advisory_count": self.advisory_count,
            "diagnostics_count": self.diagnostics_count,
        }


def is_gateable(finding: Finding, gate: Severity, gate_confidence: float = 0.0) -> bool:
    """Compute gateability from current settings.

    ``Finding.gateable`` is intentionally ignored.  Reports generated under a
    different threshold must not inherit a cached decision from an earlier run.
    Advisory-only rules can be shown at any confidence but never fail a gate.
    """

    return (
        not finding.advisory_only
        and finding.severity.gate_rank >= gate.gate_rank
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
    for finding in findings:
        gateable = is_gateable(finding, gate, gate_confidence)
        evidence = dict(finding.evidence)
        reason_codes = set(evidence.get("reason_codes", []))
        if finding.advisory_only:
            reason_codes.add("advisory-only")
        if finding.severity.gate_rank < gate.gate_rank:
            reason_codes.add("severity-below-gate")
        if finding.confidence < gate_confidence:
            reason_codes.add("confidence-below-gate")
        if finding.layer.value == "static" and finding.confidence < 0.60:
            reason_codes.add("advisory-confidence")
        elif finding.layer.value == "static" and finding.confidence < 0.85:
            reason_codes.add("probable-static")
        if gateable:
            reason_codes.add("gateable")
        if reason_codes:
            evidence["reason_codes"] = sorted(reason_codes)
        out.append(
            replace(finding, gateable=gateable, profile=profile, evidence=evidence)
        )
    return out


def filter_min_confidence(findings: list[Finding], min_confidence: float = 0.0) -> list[Finding]:
    return [finding for finding in findings if finding.confidence >= min_confidence]


def summarize(
    findings: list[Finding],
    gate: Severity,
    gate_confidence: float = 0.0,
    *,
    total_findings: int | None = None,
    hidden_count: int = 0,
    diagnostics_count: int = 0,
    gate_findings: list[Finding] | None = None,
) -> Summary:
    sev = Counter(f.severity.value for f in findings)
    cat = Counter(f.category.value for f in findings)
    layer = Counter(f.layer.value for f in findings)
    rule = Counter(f.rule_id for f in findings)
    total = total_findings if total_findings is not None else len(findings)
    return Summary(
        total=total,
        displayed=len(findings),
        hidden=hidden_count if total_findings is not None else max(0, total - len(findings)),
        by_severity={s.value: sev[s.value] for s in _SEVERITY_ORDER if sev[s.value]},
        by_category=dict(sorted(cat.items())),
        by_layer=dict(sorted(layer.items())),
        by_rule=dict(sorted(rule.items())),
        gate=gate.value,
        gated_count=sum(
            1
            for f in (gate_findings if gate_findings is not None else findings)
            if is_gateable(f, gate, gate_confidence)
        ),
        gate_confidence=gate_confidence,
        advisory_count=sum(1 for f in findings if f.advisory_only),
        diagnostics_count=diagnostics_count,
    )


def gate_failed(
    findings: list[Finding],
    gate: Severity,
    gate_confidence: float = 0.0,
) -> bool:
    return any(is_gateable(finding, gate, gate_confidence) for finding in findings)


def categories_present(findings: list[Finding]) -> set[Category]:
    return {finding.category for finding in findings}
