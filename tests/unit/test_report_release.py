"""Portable report locations and rich evidence serialization."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ml_leakproof import AnalysisResult, CompletionStatus, Config, Diagnostic, Location, analyze
from ml_leakproof.core.aggregate import summarize
from ml_leakproof.core.models import Category, Finding, Layer, Severity
from ml_leakproof.report import json_report, markdown, render, sarif


def finding(path):
    return Finding(
        rule_id="vendor-X001",
        category=Category.METRIC,
        severity=Severity.HIGH,
        layer=Layer.STATIC,
        message="review | this",
        location=Location(file=path, line=3, col=2, end_line=3, end_col=8, snippet="  score(x)"),
        evidence={"count": np.int64(2), "value": np.nan, "path": Path("input")},
    )


def test_sarif_unicode_offsets_are_codepoints(tmp_path):
    path = tmp_path / "métrics space.py"
    source = 'from sklearn.model_selection import train_test_split\n名前 = "é"; a,b = train_test_split(X)\n'
    path.write_text(source)
    result = analyze(path, Config(profile="research"))
    target = next(f for f in result.findings if f.rule_id == "R001")
    assert target.location.col == source.splitlines()[1].index("train_test_split")
    report = sarif.build(result.findings, result=result, root=tmp_path)
    run = report["runs"][0]
    assert run["columnKind"] == "unicodeCodePoints"
    loc = run["results"][0]["locations"][0]["physicalLocation"]
    assert loc["artifactLocation"]["uri"] == "m%C3%A9trics%20space.py"
    assert loc["region"]["startColumn"] == target.location.col + 1
    assert run["invocations"][0]["executionSuccessful"]


def test_sarif_same_file_common_root_and_notebook(tmp_path):
    path = tmp_path / "notebook.ipynb"
    a = finding(path)
    a = replace(a, evidence={})
    notebook = replace(
        a,
        location=replace(
            a.location, cell_id="abc", cell_index=2, cell_line=1, context_label="cell 2"
        ),
    )
    result = AnalysisResult(
        findings=[a, notebook],
        completion=CompletionStatus.PARTIAL,
        diagnostics=[Diagnostic(code="LP008", message="skipped cell")],
    )
    report = sarif.build(result.findings, result=result)
    run = report["runs"][0]
    assert not run["invocations"][0]["executionSuccessful"]
    assert (
        run["results"][0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        == "notebook.ipynb"
    )
    assert run["results"][1]["properties"]["notebook"]["cell_id"] == "abc"
    assert "endLine" not in run["results"][1]["locations"][0]["physicalLocation"]["region"]
    assert run["tool"]["driver"]["rules"][0]["id"] == "vendor-X001"
    assert sarif._relative_path(tmp_path.parent / "outside.py", tmp_path) == "outside.py"


def test_json_rich_evidence_and_schema_validation(tmp_path):
    f = finding(tmp_path / "source.py")
    summary = summarize([f], Severity.HIGH)
    result = AnalysisResult(
        findings=[f], summary=summary, metadata={"mode": Severity.HIGH, "where": Location(line=1)}
    )
    report = json.loads(json_report.render(result=result))
    assert report["findings"][0]["evidence"] == {"count": 2, "value": None, "path": "input"}
    for invalid in [{}, {**report, "version": "no"}, {**report, "completion": "unknown"}]:
        with pytest.raises(ValueError):
            json_report.validate(invalid)
    assert json_report._json_safe({"x": {1, 2}})["x"] in ([1, 2], [2, 1])
    with pytest.raises(ValueError):
        render("invalid", [], summary)


def test_markdown_partial_and_empty_reports(tmp_path):
    f = finding(tmp_path / "source.py")
    result = AnalysisResult(
        findings=[f],
        completion=CompletionStatus.PARTIAL,
        diagnostics=[Diagnostic(code="LP008", message="skipped")],
    )
    result.coverage.unavailable_checks = ["D002"]
    result.coverage.notes = ["bounded"]
    summary = summarize([f], Severity.HIGH)
    result.summary = summary
    text = markdown.render([f], summary, result=result)
    assert "partial" in text and "LP008" in text
    assert "No leakage" in markdown.render([], summarize([], Severity.HIGH))


def test_sarif_fingerprint_survives_checkout_relocation(tmp_path):
    a = finding(tmp_path / "checkout-a" / "code.py")
    b = replace(a, location=replace(a.location, file=tmp_path / "checkout-b" / "code.py"))
    first = sarif.build([a], root=tmp_path / "checkout-a")["runs"][0]["results"][0]
    second = sarif.build([b], root=tmp_path / "checkout-b")["runs"][0]["results"][0]
    assert first["partialFingerprints"] == second["partialFingerprints"]
