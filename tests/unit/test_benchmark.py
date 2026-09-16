"""Gate CI on the clean-set false-positive rate."""

from __future__ import annotations

from tests._harness import compute_metrics, run_fixture_result
from tests.corpus.metrics import (
    build_report,
    clean_false_positive_rate,
    clean_gateable_false_positive_rate,
)

CLEAN_FP_THRESHOLD = 0.02  # at most 2% of clean evaluations may false-positive
GATEABLE_CLEAN_FP_THRESHOLD = 0.01


def test_clean_set_false_positive_rate_under_threshold():
    rate = clean_false_positive_rate()
    assert rate <= CLEAN_FP_THRESHOLD, f"clean-set FP rate {rate:.3f} exceeds {CLEAN_FP_THRESHOLD}"


def test_gateable_clean_set_false_positive_rate_under_threshold():
    rate = clean_gateable_false_positive_rate()
    assert rate <= GATEABLE_CLEAN_FP_THRESHOLD, (
        f"clean-set gateable FP rate {rate:.3f} exceeds {GATEABLE_CLEAN_FP_THRESHOLD}"
    )


def test_every_leaky_fixture_recalled():
    metrics = compute_metrics()
    for rid, m in metrics.items():
        assert m.recall == 1.0, f"{rid} recall {m.recall} < 1.0 on its leaky fixture"


def test_machine_benchmark_report_is_reproducible_metadata():
    report = build_report()
    assert report["schema_version"] == "2.0"
    assert report["tool"]["version"] == "0.2.0rc1"
    assert report["configuration"]["profile"] == "ci"
    assert report["corpus"]["repositories"]
    assert report["fixtures"]["analysis_failures"] == 0
    assert report["rules"]["P001"]["recall"] == 1.0
    assert report["aggregate"]["clean_gateable_false_positive_rate"] == 0.0


def test_fixture_harness_exposes_incomplete_analysis():
    metrics = compute_metrics()
    assert metrics.analysis_failures == []


def test_fixture_harness_preserves_failed_runtime_status(tmp_path):
    fixture = tmp_path / "broken.py"
    fixture.write_text(
        "def run():\n    raise RuntimeError('fixture failure')\n",
        encoding="utf-8",
    )
    result = run_fixture_result(fixture)
    assert result.failed
    assert any(diagnostic.code == "LP499" for diagnostic in result.diagnostics)


def test_benchmark_fingerprint_rejects_stale_or_added_inputs(tmp_path):
    from tests.corpus.provenance import source_snapshot, verify_snapshot

    package = tmp_path / "ml_leakproof"
    package.mkdir()
    source = package / "module.py"
    source.write_text("x = 1")
    report = {"source_snapshot": source_snapshot(tmp_path)}
    verify_snapshot(report, tmp_path)
    source.write_text("x = 2")
    import pytest

    with pytest.raises(ValueError, match="changed"):
        verify_snapshot(report, tmp_path)
    source.write_text("x = 1")
    (package / "added.py").write_text("y = 1")
    with pytest.raises(ValueError, match="changed"):
        verify_snapshot(report, tmp_path)


def test_reviewed_corpus_extraction_is_verbatim_and_bounded(tmp_path):
    import json

    import pytest

    from tests.corpus.labeled import extract_source

    notebook = tmp_path / "scope.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "code", "source": ["x = 1\n", "y = 2"]},
                    {"cell_type": "markdown", "source": ["not code"]},
                    {"cell_type": "code", "source": ["z = 3"]},
                ]
            }
        )
    )
    assert extract_source(notebook, {"cells": [0, 2]}) == "x = 1\ny = 2\n\nz = 3\n"
    assert extract_source(notebook, {"cells": [0], "line_start": 2}) == "y = 2\n"
    with pytest.raises(ValueError):
        extract_source(notebook, {"cells": [2, 0]})
    with pytest.raises(ValueError):
        extract_source(notebook, {"cells": [1]})
