"""Compute and print per-rule precision/recall on the fixture suite.

Run: ``python -m tests.corpus.metrics``  (from the repo root)

Also used by ``test_benchmark.py`` to gate CI on the clean-set false-positive
rate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Any

import ml_leakproof
from ml_leakproof.core.aggregate import is_gateable
from ml_leakproof.core.config import Config
from tests._harness import (
    CLEAN,
    LEAKY,
    RuleMetrics,
    compute_metrics,
    fixture_kind,
    rule_id_of,
    run_fixture_result,
)
from tests.corpus.provenance import source_snapshot, verify_snapshot

_ROOT = Path(__file__).resolve().parents[2]


def render_table() -> str:
    metrics = compute_metrics()
    rows = []
    rows.append("| Rule | TP | FN | FP | Gateable FP | Precision | Recall |")
    rows.append("|------|----|----|----|-------------|-----------|--------|")
    precisions: list[float] = []
    recalls: list[float] = []
    for rid in sorted(metrics):
        m = metrics[rid]
        precision = "—" if m.precision is None else f"{m.precision:.2f}"
        recall = "—" if m.recall is None else f"{m.recall:.2f}"
        rows.append(
            f"| {rid} | {m.tp} | {m.fn} | {m.fp} | {m.gateable_fp} | "
            f"{precision} | {recall} |"
        )
        if m.precision is not None:
            precisions.append(m.precision)
        if m.recall is not None:
            recalls.append(m.recall)
    macro_p = sum(precisions) / len(precisions) if precisions else None
    macro_r = sum(recalls) / len(recalls) if recalls else None
    precision_text = "—" if macro_p is None else f"{macro_p:.2f}"
    recall_text = "—" if macro_r is None else f"{macro_r:.2f}"
    rows.append(f"| **macro avg** | | | | | **{precision_text}** | **{recall_text}** |")
    return "\n".join(rows)


def clean_false_positive_rate(config: Config | None = None) -> float:
    metrics = compute_metrics(config)
    fp = sum(m.fp for m in metrics.values())
    clean_total = sum(m.fp + m.tn for m in metrics.values())
    return fp / clean_total if clean_total else 0.0


def clean_gateable_false_positive_rate(config: Config | None = None) -> float:
    metrics = compute_metrics(config)
    fp = sum(m.gateable_fp for m in metrics.values())
    clean_total = sum(m.fp + m.tn for m in metrics.values())
    return fp / clean_total if clean_total else 0.0


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_dirty() -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(completed.stdout.strip()) if completed.returncode == 0 else None


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("numpy", "pandas", "sklearn", "scipy", "nbformat", "libcst"):
        try:
            module = __import__(name)
            version = getattr(module, "__version__", None)
            versions[name] = str(version or importlib.metadata.version(name))
        except Exception:
            continue
    return versions


def _corpus_metadata() -> dict[str, Any]:
    path = _ROOT / "corpus-manifest.json"
    if not path.is_file():
        return {"manifest": str(path), "sha256": None, "repositories": []}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        repositories = [
            {key: repo.get(key) for key in ("name", "url", "commit")}
            for repo in document.get("repos", [])
            if isinstance(repo, dict)
        ]
    except (OSError, ValueError, AttributeError):
        repositories = []
    return {
        "manifest": str(path.relative_to(_ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "repositories": repositories,
    }


def build_report(config: Config | None = None) -> dict[str, Any]:
    """Return the machine-readable fixture benchmark used by release review."""

    cfg = (config or Config()).copy()
    metrics = compute_metrics(cfg)
    rules: dict[str, dict[str, Any]] = {}
    for rule_id in sorted(metrics):
        metric = metrics[rule_id]
        rules[rule_id] = {
            "tp": metric.tp,
            "fn": metric.fn,
            "fp": metric.fp,
            "tn": metric.tn,
            "support": metric.tp + metric.fn,
            "clean_support": metric.fp + metric.tn,
            "gateable_fp": metric.gateable_fp,
            "precision": metric.precision,
            "recall": metric.recall,
        }
    labels = []
    for kind, directory in (("positive", LEAKY), ("negative", CLEAN)):
        for path in sorted(directory.glob("*.py")):
            rule_id = rule_id_of(path) if kind == "positive" else None
            observed_result = run_fixture_result(path, cfg)
            observed = observed_result.findings
            labels.append(
                {
                    "case_id": f"{kind}:{path.stem}",
                    "path": str(path.relative_to(_ROOT)),
                    "label": kind,
                    "rule_id": rule_id,
                    "layer": fixture_kind(path.read_text(encoding="utf-8")),
                    "expected_location": None,
                    "evidence_strength": "synthetic_regression_case",
                    "completion": observed_result.completion.value,
                    "diagnostics": [diagnostic.to_dict() for diagnostic in observed_result.diagnostics],
                    "affects_gate": bool(
                        kind == "positive"
                        and rule_id is not None
                        and any(
                            finding.rule_id == rule_id
                            and is_gateable(finding, cfg.fail_on, cfg.gate_confidence)
                            for finding in observed
                        )
                    ),
                }
            )
    return {
        "schema_version": "2.0",
        "tool": {"name": "ml-leakproof", "version": ml_leakproof.__version__},
        "source_commit": _git_commit(),
        "source_dirty": _git_dirty(),
        "source_snapshot": source_snapshot(),
        "configuration": {
            "profile": cfg.profile,
            "fail_on": cfg.fail_on.value,
            "gate_confidence": cfg.gate_confidence,
            "min_confidence": cfg.min_confidence,
        },
        "dependencies": _dependency_versions(),
        "corpus": _corpus_metadata(),
        "fixtures": {
            "leaky_cases": len(list((_ROOT / "tests/fixtures/leaky").glob("*.py"))),
            "clean_cases": len(list((_ROOT / "tests/fixtures/clean").glob("*.py"))),
            "analysis_failures": len(metrics.analysis_failures),
            "analysis_failure_cases": list(metrics.analysis_failures),
        },
        "labels": labels,
        "rules": rules,
        "aggregate": {
            "clean_false_positive_rate": _clean_rate(metrics, gateable=False),
            "clean_gateable_false_positive_rate": _clean_rate(metrics, gateable=True),
        },
        "notes": [
            "This is a paired fixture regression benchmark, not evidence that arbitrary projects are leak-free.",
            "Undefined precision or recall values are serialized as null.",
            "Clean cases are evaluated against every catalog rule, including unrelated false positives.",
            "Incomplete fixture analyses are reported explicitly and do not count as clean evidence.",
        ],
    }


def _clean_rate(metrics: dict[str, RuleMetrics], *, gateable: bool) -> float:
    numerator = sum(
        metric.gateable_fp if gateable else metric.fp for metric in metrics.values()
    )
    denominator = sum(metric.fp + metric.tn for metric in metrics.values())
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the machine-readable benchmark")
    parser.add_argument("--verify", type=Path, help="verify a report matches the current release inputs")
    args = parser.parse_args()
    if args.verify:
        verify_snapshot(json.loads(args.verify.read_text(encoding="utf-8")))
        print("Benchmark source fingerprint verified")
        raise SystemExit(0)
    if args.json:
        print(json.dumps(build_report(), indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print(render_table())
        print(f"\nclean-set false-positive rate: {clean_false_positive_rate():.3f}")
        print(f"clean-set gateable false-positive rate: {clean_gateable_false_positive_rate():.3f}")
