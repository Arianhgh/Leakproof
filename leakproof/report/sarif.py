"""SARIF 2.1.0 report for GitHub code scanning."""

from __future__ import annotations

import json

from ..core.models import Finding, Severity
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


def _reporting_descriptors(used_rule_ids: set[str]) -> list[dict]:
    rules = all_rules()
    out: list[dict] = []
    for rid in sorted(used_rule_ids):
        rule = rules.get(rid)
        if rule is None:
            out.append({"id": rid, "name": rid})
            continue
        out.append(
            {
                "id": rule.id,
                "name": rule.name,
                "shortDescription": {"text": rule.name},
                "fullDescription": {"text": rule.rationale or rule.name},
                "helpUri": rule.references[0] if rule.references else None,
                "help": {"text": rule.rationale or rule.name},
                "defaultConfiguration": {"level": _LEVEL[rule.severity]},
                "properties": {
                    "category": rule.category.value,
                    "layers": [layer.value for layer in rule.layers],
                },
            }
        )
    # drop None helpUri keys
    for d in out:
        if d.get("helpUri") is None:
            d.pop("helpUri", None)
    return out


def build(findings: list[Finding]) -> dict:
    used = {f.rule_id for f in findings}
    results = []
    for f in findings:
        loc = f.location
        physical = None
        if loc.file is not None:
            region: dict[str, object] = {}
            if loc.line is not None:
                region["startLine"] = loc.line
            if loc.col is not None:
                region["startColumn"] = loc.col + 1
            if loc.end_line is not None:
                region["endLine"] = loc.end_line
            if loc.snippet:
                region["snippet"] = {"text": loc.snippet}
            physical = {
                "artifactLocation": {"uri": str(loc.file).replace("\\", "/")},
                "region": region or {"startLine": 1},
            }
        location_obj = {}
        if physical is not None:
            location_obj["physicalLocation"] = physical
        if loc.context_label:
            location_obj["message"] = {"text": loc.context_label}
        results.append(
            {
                "ruleId": f.rule_id,
                "level": _LEVEL[f.severity],
                "message": {"text": f.message},
                "locations": [location_obj] if location_obj else [],
                "partialFingerprints": {"leakproofKey": _fingerprint(f)},
                "properties": {
                    "category": f.category.value,
                    "layer": f.layer.value,
                    "confidence": f.confidence,
                },
            }
        )
    return {
        "$schema": SARIF_SCHEMA,
        "version": SARIF_VERSION,
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "leakproof",
                        "informationUri": "https://github.com/leakproof/leakproof",
                        "version": _tool_version(),
                        "rules": _reporting_descriptors(used),
                    }
                },
                "results": results,
            }
        ],
    }


def _fingerprint(f: Finding) -> str:
    import hashlib

    raw = "|".join(str(x) for x in f.key)
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=12).hexdigest()


def render(findings: list[Finding]) -> str:
    return json.dumps(build(findings), indent=2, default=str)
