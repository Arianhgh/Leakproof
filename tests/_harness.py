"""Shared fixture-running harness for tests and the benchmark metrics."""

from __future__ import annotations

import importlib.util
import warnings
from dataclasses import dataclass
from pathlib import Path

from leakproof.core.config import Config

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


def run_fixture(path: Path, config: Config | None = None) -> set[str]:
    """Return the set of rule ids fired on a fixture, dispatching by kind."""
    cfg = config or Config()
    source = path.read_text(encoding="utf-8")
    kind = fixture_kind(source)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "data":
            from leakproof.data.engine import DataEngine

            module = _load_module(path)
            findings = DataEngine(cfg).run(module.make_input())
        elif kind == "runtime":
            from leakproof.runtime.engine import watch

            module = _load_module(path)
            with watch(cfg) as session:
                module.run()
            findings = session.findings
        else:
            from leakproof.static.engine import StaticEngine

            findings = StaticEngine(cfg).run_file(path)
    return {f.rule_id for f in findings}


@dataclass
class RuleMetrics:
    rule_id: str
    tp: int = 0  # leaky fixtures where the rule fired
    fn: int = 0  # leaky fixtures where it did not
    fp: int = 0  # clean fixtures where it fired anyway
    tn: int = 0  # clean fixtures where it correctly stayed silent

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 1.0


def compute_metrics(config: Config | None = None) -> dict[str, RuleMetrics]:
    cfg = config or Config()
    metrics: dict[str, RuleMetrics] = {}

    def m(rid: str) -> RuleMetrics:
        return metrics.setdefault(rid, RuleMetrics(rule_id=rid))

    for path in sorted(LEAKY.glob("*.py")):
        rid = rule_id_of(path)
        fired = run_fixture(path, cfg)
        if rid in fired:
            m(rid).tp += 1
        else:
            m(rid).fn += 1

    for path in sorted(CLEAN.glob("*.py")):
        own = rule_id_of(path)
        fired = run_fixture(path, cfg)
        # the rule's own clean fixture must stay silent; count FP per firing rule
        if own in fired:
            m(own).fp += 1
        else:
            m(own).tn += 1
    return metrics
