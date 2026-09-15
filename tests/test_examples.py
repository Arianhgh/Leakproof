"""Keep the user-facing examples executable against the public APIs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_leakproof import Config, analyze, analyze_data
from ml_leakproof.core.models import CompletionStatus
from ml_leakproof.runtime.engine import run_script_result

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def test_examples_demonstrate_static_runtime_and_data_layers() -> None:
    config = Config(exclude=[])

    static_result = analyze(EXAMPLES / "leaky_workflow.py", config=config)
    assert static_result.completion is CompletionStatus.COMPLETE
    assert "C001" in {finding.rule_id for finding in static_result.findings}

    clean_result = run_script_result(EXAMPLES / "clean_workflow.py", [], config)
    assert clean_result.completion is CompletionStatus.COMPLETE
    assert not clean_result.findings
    assert not clean_result.diagnostics

    leaky_result = run_script_result(EXAMPLES / "leaky_workflow.py", [], config)
    assert leaky_result.completion is CompletionStatus.COMPLETE
    assert {"C001", "P001"} <= {
        finding.rule_id for finding in leaky_result.findings
    }

    data_result = analyze_data(
        pd.read_csv(EXAMPLES / "data/train.csv"),
        pd.read_csv(EXAMPLES / "data/test.csv"),
        target="target",
        config=config,
    )
    assert data_result.completion is CompletionStatus.COMPLETE
    assert "D001" in {finding.rule_id for finding in data_result.findings}
