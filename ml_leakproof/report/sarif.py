"""SARIF 2.1.0 output for GitHub code scanning."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ..core.models import AnalysisResult, Finding, Severity
from ..core.registry import all_rules

SARIF_VERSION = "2.1.0"
SARIF_SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"
_LEVEL = {
    Severity.CRITICAL: "error",
    Severity.HIGH: "error",
    Severity.MEDIUM: "warning",
    Severity.LOW: "note",
    Severity.INFO: "note",
}


def _tool_version() -> str:
    try:
        from .. import __version__

        return __version__
    except Exception:  # pragma: no cover
        return "0"


def _reporting_descriptors(used_rule_ids: set[str]) -> list[dict[str, Any]]:
    rules = all_rules()
    descriptors: list[dict[str, Any]] = []
    for rule_id in sorted(used_rule_ids):
        rule = rules.get(rule_id)
        if rule is None:
            descriptors.append({"id": rule_id, "name": rule_id})
            continue
        descriptor: dict[str, Any] = {
            "id": rule.id,
            "name": rule.name,
            "shortDescription": {"text": rule.name},
            "fullDescription": {"text": rule.rationale or rule.name},
            "help": {"text": rule.rationale or rule.name},
            "defaultConfiguration": {"level": _LEVEL[rule.severity]},
            "properties": {
                "category": rule.category.value,
                "layers": [layer.value for layer in rule.layers],
            },
        }
        if rule.references:
            descriptor["helpUri"] = rule.references[0]
        descriptors.append(descriptor)
    return descriptors


def build(
    findings: list[Finding],
    *,
    result: AnalysisResult | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    root_path = (root or _common_root(findings)).resolve()
    results: list[dict[str, Any]] = []
    for finding in findings:
        location = finding.location
        result_location: dict[str, Any] = {}
        if location.file is not None:
            relative = _relative_path(location.file, root_path)
            region: dict[str, Any] = {}
            # Notebook findings refer to the notebook artifact and cell-relative
            # line; the cell metadata is retained in properties.
            region["startLine"] = location.cell_line or location.line or 1
            if location.col is not None:
                region["startColumn"] = _unicode_column(location.snippet, location.col + 1)
            if location.end_line is not None and not location.cell_line:
                region["endLine"] = location.end_line
            if location.end_col is not None and not location.cell_line:
                region["endColumn"] = location.end_col + 1
            if location.snippet:
                region["snippet"] = {"text": location.snippet}
            result_location["physicalLocation"] = {
                "artifactLocation": {"uri": quote(relative, safe="/._-~")},
                "region": region,
            }
        if location.context_label:
            result_location.setdefault("logicalLocations", [{"fullyQualifiedName": location.context_label}])
        properties: dict[str, Any] = {
            "category": finding.category.value,
            "layer": finding.layer.value,
            "confidence": finding.confidence,
            "gateable": finding.gateable,
            "advisory_only": finding.advisory_only,
            "profile": finding.profile,
            "evidence": finding.evidence,
        }
        if location.cell_id is not None or location.cell_index is not None:
            properties["notebook"] = {
                "cell_id": location.cell_id,
                "cell_index": location.cell_index,
                "cell_line": location.cell_line,
            }
        results.append(
            {
                "ruleId": finding.rule_id,
                "level": _LEVEL[finding.severity],
                "message": {"text": finding.message},
                "locations": [result_location] if result_location else [],
                "partialFingerprints": {"mlLeakproofKey": _fingerprint(finding, root_path)},
                "properties": properties,
            }
        )
    sarif: dict[str, Any] = {
        "$schema": SARIF_SCHEMA,
        "version": SARIF_VERSION,
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "ml-leakproof",
                        "informationUri": "https://github.com/Arianhgh/Leakproof",
                        "version": _tool_version(),
                        "rules": _reporting_descriptors({f.rule_id for f in findings}),
                    }
                },
                "results": results,
            }
        ],
    }
    if result is not None:
        sarif["runs"][0]["invocations"] = [
            {
                "executionSuccessful": result.completion.value == "complete",
                "properties": {
                    "completion": result.completion.value,
                    "script_exit_status": result.script_exit_status,
                    "diagnostics": [diagnostic.to_dict() for diagnostic in result.diagnostics],
                },
            }
        ]
    return sarif


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.name or path.as_posix().lstrip("/")


def _common_root(findings: list[Finding]) -> Path:
    files = [str(f.location.file.resolve()) for f in findings if f.location.file is not None]
    if not files:
        return Path.cwd()
    return Path(os.path.commonpath(files)).parent if len(files) == 1 else Path(os.path.commonpath(files))


def _unicode_column(snippet: str | None, column: int) -> int:
    if snippet is None or column <= 1:
        return max(1, column)
    prefix = snippet[: column - 1]
    # SARIF columns are Unicode code-point columns.  Python source offsets are
    # already code-point based; this explicit conversion documents the contract.
    return len(prefix) + 1


def _fingerprint(finding: Finding, root: Path) -> str:
    file = _relative_path(finding.location.file, root) if finding.location.file else ""
    raw = "|".join(
        str(value)
        for value in (
            finding.rule_id,
            finding.layer.value,
            file,
            finding.location.cell_id or "",
            finding.location.cell_line or finding.location.line or 0,
            finding.location.col or 0,
            finding.location.context_label or "",
            finding.key,
        )
    )
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()


def render(
    findings: list[Finding],
    *,
    result: AnalysisResult | None = None,
    root: Path | None = None,
) -> str:
    return json.dumps(build(findings, result=result, root=root), indent=2, ensure_ascii=False, allow_nan=False)
