"""Shared fixture-running harness for tests and the benchmark metrics."""

from __future__ import annotations

import importlib.util
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_leakproof.core.aggregate import is_gateable
from ml_leakproof.core.config import Config
from ml_leakproof.core.models import (
    AnalysisResult,
    CompletionStatus,
    Diagnostic,
    DiagnosticLevel,
    Finding,
    Layer,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
LEAKY = FIXTURES / "leaky"
CLEAN = FIXTURES / "clean"


def fixture_kind(source: str) -> str:
    if "def make_input" in source:
        return "data"
    if "def run(" in source:
        return "runtime"
    return "static"


def rule_id_of(path: Path) -> str:
    return path.name.split("__", 1)[0]


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"_fx_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def run_fixture_result(path: Path, config: Config | None = None) -> AnalysisResult:
    """Run a fixture while preserving findings, diagnostics, and completion."""
    cfg = config or Config()
    source = path.read_text(encoding="utf-8")
    kind = fixture_kind(source)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "data":
            from ml_leakproof.data.engine import DataEngine

            module = _load_module(path)
            return DataEngine(cfg).run_result(module.make_input())
        elif kind == "runtime":
            from ml_leakproof.runtime.engine import watch

            module = _load_module(path)
            session = None
            try:
                with watch(cfg) as session:
                    module.run()
                return session.result
            except Exception as exc:
                # A fixture execution failure is an analysis failure, not a
                # clean negative observation.  Keep any finalized runtime
                # findings/diagnostics and add an explicit harness diagnostic.
                result = session.result if session is not None else AnalysisResult()
                result.completion = CompletionStatus.FAILED
                result.coverage.failed_inputs.append(str(path))
                result.diagnostics.append(
                    Diagnostic(
                        code="LP499",
                        message=f"fixture execution failed: {exc}",
                        level=DiagnosticLevel.ERROR,
                        layer=Layer.RUNTIME,
                    )
                )
                return result
        else:
            from ml_leakproof.static.engine import StaticEngine

            return StaticEngine(cfg).run_file_result(path)


def run_fixture_findings(path: Path, config: Config | None = None) -> list[Finding]:
    """Compatibility wrapper returning only findings."""

    return run_fixture_result(path, config).findings


def run_fixture(path: Path, config: Config | None = None) -> set[str]:
    """Return the set of rule ids fired on a fixture, dispatching by kind."""
    return {f.rule_id for f in run_fixture_findings(path, config)}


@dataclass
class RuleMetrics:
    rule_id: str
    tp: int = 0  # leaky fixtures where the rule fired
    fn: int = 0  # leaky fixtures where it did not
    fp: int = 0  # clean fixtures where it fired anyway
    tn: int = 0  # clean fixtures where it correctly stayed silent
    gateable_fp: int = 0  # clean fixture false positives that would fail the gate

    @property
    def precision(self) -> float | None:
        denom = self.tp + self.fp
        return self.tp / denom if denom else None

    @property
    def recall(self) -> float | None:
        denom = self.tp + self.fn
        return self.tp / denom if denom else None


class BenchmarkMetrics(dict[str, RuleMetrics]):
    """Per-rule metrics plus explicit fixture-analysis failure records."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.analysis_failures: list[dict[str, Any]] = []


def compute_metrics(config: Config | None = None) -> BenchmarkMetrics:
    cfg = config or Config()
    from ml_leakproof.core.registry import all_rules

    rule_ids = sorted(all_rules())
    metrics = BenchmarkMetrics({rid: RuleMetrics(rule_id=rid) for rid in rule_ids})

    def m(rid: str) -> RuleMetrics:
        return metrics.setdefault(rid, RuleMetrics(rule_id=rid))

    def run(path: Path, label: str) -> AnalysisResult:
        result = run_fixture_result(path, cfg)
        if result.completion is not CompletionStatus.COMPLETE:
            metrics.analysis_failures.append(
                {
                    "path": str(path.relative_to(Path(__file__).resolve().parent.parent)),
                    "label": label,
                    "completion": result.completion.value,
                    "diagnostics": [diagnostic.to_dict() for diagnostic in result.diagnostics],
                }
            )
        return result

    for path in sorted(LEAKY.glob("*.py")):
        rid = rule_id_of(path)
        fired = {f.rule_id for f in run(path, "positive").findings}
        if rid in fired:
            m(rid).tp += 1
        else:
            m(rid).fn += 1

    for path in sorted(CLEAN.glob("*.py")):
        findings = run(path, "negative").findings
        by_rule: dict[str, list[Finding]] = {}
        for finding in findings:
            by_rule.setdefault(finding.rule_id, []).append(finding)
        # Every clean labeled case is evaluated against every catalog rule. This
        # catches an unrelated false positive, not only one named by the fixture.
        for rid in rule_ids:
            observed = by_rule.get(rid, [])
            if observed:
                m(rid).fp += 1
                if any(is_gateable(f, cfg.fail_on, cfg.gate_confidence) for f in observed):
                    m(rid).gateable_fp += 1
            else:
                m(rid).tn += 1
    return metrics
