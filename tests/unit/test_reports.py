from __future__ import annotations

import json

import pytest

from ml_leakproof.core.aggregate import annotate_gateability, summarize
from ml_leakproof.core.models import Category, Finding, Layer, Location, Severity
from ml_leakproof.report import json_report, markdown, sarif, terminal


def _findings():
    return [
        Finding(
            rule_id="P001",
            category=Category.PREPROCESSING,
            severity=Severity.HIGH,
            layer=Layer.STATIC,
            message="leak",
            location=Location(file=None, line=3, snippet="scaler.fit_transform(X)"),
            confidence=0.9,
            references=("https://example.com",),
        ),
        Finding(
            rule_id="D001",
            category=Category.DATA_OVERLAP,
            severity=Severity.CRITICAL,
            layer=Layer.DATA,
            message="overlap",
            location=Location(context_label="train/test overlap"),
        ),
    ]


def test_json_schema():
    f = annotate_gateability(
        _findings(), gate=Severity.HIGH, gate_confidence=0.75, profile="ci"
    )
    s = summarize(f, Severity.HIGH, 0.75)
    obj = json.loads(json_report.render(f, s))
    assert obj["version"] == json_report.SCHEMA_VERSION
    assert len(obj["findings"]) == 2
    assert obj["summary"]["total"] == 2
    assert obj["summary"]["gate_confidence"] == 0.75
    assert obj["findings"][0]["gateable"] is True
    assert obj["findings"][0]["profile"] == "ci"


def test_sarif_structure():
    f = annotate_gateability(
        _findings(), gate=Severity.HIGH, gate_confidence=0.75, profile="ci"
    )
    obj = json.loads(sarif.render(f))
    assert obj["version"] == "2.1.0"
    run = obj["runs"][0]
    assert run["tool"]["driver"]["name"] == "ml-leakproof"
    assert len(run["results"]) == 2
    # rules descriptors present
    ids = {r["id"] for r in run["tool"]["driver"]["rules"]}
    assert {"P001", "D001"} <= ids
    # partial fingerprints for dedupe
    assert all("partialFingerprints" in r for r in run["results"])
    # critical maps to error
    levels = {r["ruleId"]: r["level"] for r in run["results"]}
    assert levels["D001"] == "error"
    props = run["results"][0]["properties"]
    assert "gateable" in props and "profile" in props and "evidence" in props


def test_markdown_and_terminal_render():
    f = annotate_gateability(
        _findings(), gate=Severity.HIGH, gate_confidence=0.75, profile="ci"
    )
    s = summarize(f, Severity.HIGH, 0.75)
    md = markdown.render(f, s)
    assert "Leakproof report" in md
    assert "P001" in md
    assert "gateable" in md
    txt = terminal.render(f, s, color=False)
    assert "P001" in txt and "D001" in txt


def test_json_report_rejects_unsupported_evidence() -> None:
    finding = Finding(
        rule_id="P001",
        category=Category.PREPROCESSING,
        severity=Severity.HIGH,
        layer=Layer.STATIC,
        message="leak",
        location=Location(),
        evidence={"unsupported": object()},
    )
    with pytest.raises(ValueError, match="unsupported JSON value"):
        json_report.render([finding], summarize([finding], Severity.HIGH))
