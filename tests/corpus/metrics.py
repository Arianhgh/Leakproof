"""Compute and print per-rule precision/recall on the fixture suite.

Run: ``python -m tests.corpus.metrics``  (from the repo root)

Also used by ``test_benchmark.py`` to gate CI on the clean-set false-positive
rate.
"""

from __future__ import annotations

from tests._harness import compute_metrics


def render_table() -> str:
    metrics = compute_metrics()
    rows = []
    rows.append("| Rule | TP | FN | FP | Gateable FP | Precision | Recall |")
    rows.append("|------|----|----|----|-------------|-----------|--------|")
    macro_p = macro_r = 0.0
    for rid in sorted(metrics):
        m = metrics[rid]
        rows.append(
            f"| {rid} | {m.tp} | {m.fn} | {m.fp} | {m.gateable_fp} | "
            f"{m.precision:.2f} | {m.recall:.2f} |"
        )
        macro_p += m.precision
        macro_r += m.recall
    n = len(metrics) or 1
    rows.append(f"| **macro avg** | | | | **{macro_p / n:.2f}** | **{macro_r / n:.2f}** |")
    return "\n".join(rows)


def clean_false_positive_rate() -> float:
    metrics = compute_metrics()
    fp = sum(m.fp for m in metrics.values())
    clean_total = sum(m.fp + m.tn for m in metrics.values())
    return fp / clean_total if clean_total else 0.0


def clean_gateable_false_positive_rate() -> float:
    metrics = compute_metrics()
    fp = sum(m.gateable_fp for m in metrics.values())
    clean_total = sum(m.fp + m.tn for m in metrics.values())
    return fp / clean_total if clean_total else 0.0


if __name__ == "__main__":
    print(render_table())
    print(f"\nclean-set false-positive rate: {clean_false_positive_rate():.3f}")
    print(f"clean-set gateable false-positive rate: {clean_gateable_false_positive_rate():.3f}")
