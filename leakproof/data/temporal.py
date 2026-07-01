"""Temporal range checks across splits."""

from __future__ import annotations

from typing import Any


def time_ranges(df, time_col: str) -> tuple[Any, Any]:
    import pandas as pd

    series = pd.to_datetime(df[time_col], errors="coerce")
    return series.min(), series.max()
