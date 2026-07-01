"""Gate CI on the clean-set false-positive rate."""

from __future__ import annotations

from tests._harness import compute_metrics
from tests.corpus.metrics import clean_false_positive_rate

CLEAN_FP_THRESHOLD = 0.02  # at most 2% of clean evaluations may false-positive


def test_clean_set_false_positive_rate_under_threshold():
    rate = clean_false_positive_rate()
    assert rate <= CLEAN_FP_THRESHOLD, f"clean-set FP rate {rate:.3f} exceeds {CLEAN_FP_THRESHOLD}"


def test_every_leaky_fixture_recalled():
    metrics = compute_metrics()
    for rid, m in metrics.items():
        assert m.recall == 1.0, f"{rid} recall {m.recall} < 1.0 on its leaky fixture"
