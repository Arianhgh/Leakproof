from __future__ import annotations

from leakproof.core.aggregate import dedupe, gate_failed, summarize
from leakproof.core.models import Category, Finding, Layer, Location, Severity


def _f(rule_id, sev, conf=1.0, line=1, layer=Layer.STATIC):
    return Finding(
        rule_id=rule_id,
        category=Category.PREPROCESSING,
        severity=sev,
        layer=layer,
        message="m",
        location=Location(file=None, line=line),
        confidence=conf,
    )


def test_dedupe_keeps_higher_confidence():
    a = _f("P001", Severity.HIGH, conf=0.7)
    b = _f("P001", Severity.HIGH, conf=0.95, layer=Layer.RUNTIME)
    out = dedupe([a, b])
    assert len(out) == 1 and out[0].confidence == 0.95


def test_summarize_counts():
    findings = [_f("P001", Severity.HIGH), _f("R001", Severity.LOW, line=2)]
    s = summarize(findings, Severity.HIGH)
    assert s.total == 2
    assert s.by_severity["high"] == 1
    assert s.gated_count == 1


def test_gate_failed():
    assert gate_failed([_f("P001", Severity.HIGH)], Severity.HIGH)
    assert not gate_failed([_f("R001", Severity.LOW)], Severity.HIGH)
