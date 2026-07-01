"""Stable row hashing for exact-overlap detection."""

from __future__ import annotations

import hashlib
from typing import Any


def _normalize_cell(v: Any) -> str:
    if v is None:
        return "\x00"
    try:
        import math

        if isinstance(v, float):
            if math.isnan(v):
                return "\x00nan"
            # normalize -0.0 / integral floats
            if v == int(v):
                return repr(int(v))
            return repr(round(v, 12))
    except Exception:
        pass
    return str(v)


def row_hashes(df, columns: list[str] | None = None, sample_cap: int | None = None):
    import pandas as pd  # noqa: F401

    cols = columns if columns is not None else list(df.columns)
    work = df[cols]
    sampled = False
    if sample_cap is not None and len(work) > sample_cap:
        work = work.head(sample_cap)
        sampled = True

    def hash_row(row) -> str:
        joined = "\x1f".join(_normalize_cell(row[c]) for c in cols)
        return hashlib.blake2b(joined.encode("utf-8"), digest_size=16).hexdigest()

    hashes = work.apply(hash_row, axis=1)
    hashes.attrs["sampled"] = sampled
    return hashes


def overlap(train_hashes, test_hashes) -> tuple[set[str], int]:
    train_set = set(train_hashes.tolist())
    test_list = test_hashes.tolist()
    common = train_set.intersection(test_list)
    count = sum(1 for h in test_list if h in common)
    return common, count
