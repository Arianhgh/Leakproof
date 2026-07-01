from __future__ import annotations

import json

from leakproof.core.aggregate import summarize
from leakproof.core.models import Category, Finding, Layer, Location, Severity
from leakproof.report import json_report, markdown, sarif, terminal


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
    f = _findings()
    s = summarize(f, Severity.HIGH)
    obj = json.loads(json_report.render(f, s))
    assert obj["version"] == json_report.SCHEMA_VERSION
    assert len(obj["findings"]) == 2
    assert obj["summary"]["total"] == 2


def test_sarif_structure():
    f = _findings()
    obj = json.loads(sarif.render(f))
    assert obj["version"] == "2.1.0"
    run = obj["runs"][0]
    assert run["tool"]["driver"]["name"] == "leakproof"
    assert len(run["results"]) == 2
    # rules descriptors present
    ids = {r["id"] for r in run["tool"]["driver"]["rules"]}
    assert {"P001", "D001"} <= ids
    # partial fingerprints for dedupe
    assert all("partialFingerprints" in r for r in run["results"])
    # critical maps to error
    levels = {r["ruleId"]: r["level"] for r in run["results"]}
    assert levels["D001"] == "error"


def test_markdown_and_terminal_render():
    f = _findings()
    s = summarize(f, Severity.HIGH)
    md = markdown.render(f, s)
    assert "leakproof report" in md
    assert "P001" in md
    txt = terminal.render(f, s, color=False)
    assert "P001" in txt and "D001" in txt
