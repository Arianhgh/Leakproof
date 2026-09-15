from __future__ import annotations

from ml_leakproof.core.aggregate import annotate_gateability, dedupe, gate_failed, summarize
from ml_leakproof.core.models import Category, Finding, Layer, Location, Severity


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


def test_dedupe_preserves_distinct_layers():
    a = _f("P001", Severity.HIGH, conf=0.7)
    b = _f("P001", Severity.HIGH, conf=0.95, layer=Layer.RUNTIME)
    out = dedupe([a, b])
    assert len(out) == 2
    assert {item.layer for item in out} == {Layer.STATIC, Layer.RUNTIME}


def test_dedupe_keeps_higher_confidence_for_same_evidence():
    a = _f("P001", Severity.HIGH, conf=0.7)
    b = _f("P001", Severity.HIGH, conf=0.95)
    out = dedupe([a, b])
    assert len(out) == 1 and out[0].confidence == 0.95


def test_summarize_counts():
    findings = [_f("P001", Severity.HIGH), _f("R001", Severity.LOW, line=2)]
    s = summarize(findings, Severity.HIGH)
    assert s.total == 2
    assert s.by_severity["high"] == 1
    assert s.gated_count == 1

    s = summarize([_f("C001", Severity.HIGH, conf=0.7)], Severity.HIGH, 0.75)
    assert s.gated_count == 0


def test_gate_failed():
    assert gate_failed([_f("P001", Severity.HIGH)], Severity.HIGH)
    assert not gate_failed([_f("R001", Severity.LOW)], Severity.HIGH)
    assert not gate_failed([_f("C001", Severity.HIGH, conf=0.7)], Severity.HIGH, 0.75)


def test_annotate_gateability_adds_metadata():
    findings = annotate_gateability(
        [_f("C001", Severity.HIGH, conf=0.7)],
        gate=Severity.HIGH,
        gate_confidence=0.75,
        profile="ci",
    )
    assert findings[0].gateable is False
    assert findings[0].profile == "ci"
    assert "confidence-below-gate" in findings[0].evidence["reason_codes"]
