"""Validate reviewed source scopes from pinned repositories without execution.

Run ``python -m tests.corpus.labeled --output corpus-labeled-results.json``.
Selected cells preserve their original text and order. Explicit line ranges
permit excluding notebook UI setup; the report hashes every inspected excerpt.
These are scoped regression labels, not exhaustive repository ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from ml_leakproof.core.config import Config
from ml_leakproof.core.models import Layer
from ml_leakproof.core.registry import all_rules
from ml_leakproof.corpus import _checkout_pinned, _validate_manifest
from ml_leakproof.static.engine import StaticEngine
from tests.corpus.provenance import ROOT, source_snapshot


def extract_source(path: Path, case: dict) -> str:
    if path.suffix == ".ipynb":
        document = json.loads(path.read_text(encoding="utf-8"))
        indexes = case["cells"]
        if indexes != sorted(set(indexes)):
            raise ValueError("cell indices must be unique and in source order")
        cells = [document["cells"][i] for i in indexes]
        if any(cell["cell_type"] != "code" for cell in cells):
            raise ValueError("review scopes must select code cells")
        source = "\n\n".join("".join(cell["source"]) for cell in cells)
    else:
        source = path.read_text(encoding="utf-8")
    if "line_start" in case:
        source = "\n".join(source.splitlines()[case["line_start"] - 1 : case.get("line_end")])
    return source + "\n"


def run(manifest: Path, workdir: Path, *, offline: bool = False) -> dict:
    import subprocess

    document = json.loads(manifest.read_text(encoding="utf-8"))
    repos = _validate_manifest(document, manifest)
    records = []
    catalog = all_rules()
    for repo in repos:
        checkout = workdir / repo["name"]
        try:
            if offline:
                commit = subprocess.check_output(
                    ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
                ).strip()
                dirty = subprocess.check_output(
                    ["git", "-C", str(checkout), "status", "--porcelain"], text=True
                ).strip()
                if commit != repo["commit"] or dirty:
                    raise ValueError("cached checkout must be clean and match the pinned commit")
            else:
                status = _checkout_pinned(repo["url"], checkout, repo["commit"])
                if status["status"] != "ok":
                    raise ValueError(status["error"])
            for case in repo.get("cases", []):
                path = (checkout / case["path"]).resolve()
                path.relative_to(checkout.resolve())
                if not set(case["select"]) <= set(catalog) or not set(case["expected"]) <= set(
                    case["select"]
                ):
                    raise ValueError("unknown or unselected label rule")
                source = extract_source(path, case)
                with tempfile.TemporaryDirectory() as tmp:
                    scan_path = Path(tmp) / "reviewed.py"
                    scan_path.write_text(source, encoding="utf-8")
                    result = StaticEngine(
                        Config(select=case["select"], layers=[Layer.STATIC], profile="research")
                    ).run_file_result(scan_path)
                observed = {finding.rule_id for finding in result.findings}
                expected = set(case["expected"])
                records.append(
                    {
                        "id": case["id"],
                        "repository": repo["name"],
                        "commit": repo["commit"],
                        "path": case["path"],
                        "cells": case.get("cells"),
                        "line_start": case.get("line_start"),
                        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                        "selected": case["select"],
                        "expected": sorted(expected),
                        "observed": sorted(observed),
                        "missing": sorted(expected - observed),
                        "unexpected": sorted(observed - expected),
                        "completion": result.completion.value,
                        "passed": result.complete and observed == expected,
                        "diagnostics": [
                            {"code": d.code, "message": d.message} for d in result.diagnostics
                        ],
                        "rationale": case["rationale"],
                    }
                )
        except Exception as exc:
            records.append({"repository": repo["name"], "passed": False, "error": str(exc)})
    return {
        "schema_version": "1.0",
        "source_snapshot": source_snapshot(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "repositories": len(repos),
        "cases": records,
        "passed": bool(records) and all(record["passed"] for record in records),
        "scope": "Reviewed source excerpts; labels apply only to selected rules and cells, not whole repositories.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "corpus-labeled.json")
    parser.add_argument("--workdir", type=Path, default=ROOT / ".leakproof-corpus")
    parser.add_argument("--output", type=Path, default=ROOT / "corpus-labeled-results.json")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    report = run(args.manifest, args.workdir, offline=args.offline)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Reviewed corpus: {sum(c['passed'] for c in report['cases'])}/{len(report['cases'])} cases passed"
    )
    raise SystemExit(0 if report["passed"] else 1)
