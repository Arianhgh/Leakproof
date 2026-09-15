"""Validated temporal range helpers."""

from __future__ import annotations

from typing import Any


def time_ranges(df: Any, time_col: str) -> tuple[Any, Any]:
    import pandas as pd

    if time_col not in df.columns:
        raise ValueError(f"missing time column: {time_col}")
    original = df[time_col]
    try:
        converted = pd.to_datetime(original, errors="coerce", utc=False)
    except (TypeError, ValueError):
        converted = pd.to_datetime(original, errors="coerce", utc=True)
    invalid = converted.isna() & original.notna()
    if bool(invalid.any()):
        raise ValueError(f"{int(invalid.sum())} timestamp value(s) could not be parsed")
    if converted.empty or converted.dropna().empty:
        return None, None
    try:
        return converted.min(), converted.max()
    except (TypeError, ValueError):
        normalized = pd.to_datetime(original, errors="coerce", utc=True)
        invalid = normalized.isna() & original.notna()
        if bool(invalid.any()):
            raise ValueError(f"{int(invalid.sum())} timestamp value(s) could not be parsed") from None
        return normalized.min(), normalized.max()
