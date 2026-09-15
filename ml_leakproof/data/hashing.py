"""Collision-resistant, type-aware row canonicalization and hashing."""

from __future__ import annotations

import hashlib
import math
import struct
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        import pandas as pd

        missing = pd.isna(value)
        return bool(missing) if not hasattr(missing, "__len__") else False
    except Exception:
        return False


def canonical_cell(value: Any) -> bytes:
    """Encode values with explicit type tags and length framing.

    Strings are never equated with numbers, delimiters cannot create boundary
    collisions, and floats are encoded without rounding.  Missing values share a
    single representation across pandas nullable dtypes.
    """

    if _is_missing(value):
        return b"N\x00"
    if isinstance(value, bool):
        return b"B" + (b"1" if value else b"0")
    if isinstance(value, int) and not isinstance(value, bool):
        return b"I" + str(value).encode("ascii") + b"\x00"
    if isinstance(value, float):
        if math.isnan(value):
            return b"N\x00"
        if value == 0.0:
            value = 0.0
        return b"F" + struct.pack(">d", value)
    if isinstance(value, Decimal):
        return b"D" + str(value).encode("utf-8") + b"\x00"
    if isinstance(value, (datetime, date, time)):
        return b"T" + value.isoformat().encode("utf-8") + b"\x00"
    if isinstance(value, bytes):
        return b"Y" + len(value).to_bytes(8, "big") + value
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return b"Z" + str(value.dtype).encode("utf-8") + b":" + canonical_cell(value.item())
    except Exception:
        pass
    try:
        import pandas as pd

        if isinstance(value, pd.Timestamp):
            return b"T" + str(value.isoformat()).encode("utf-8") + b"\x00"
        if isinstance(value, pd.Categorical):
            value = str(value)
    except Exception:
        pass
    text = str(value).encode("utf-8", errors="surrogatepass")
    return b"S" + len(text).to_bytes(8, "big") + text


def canonical_row(values: Any) -> bytes:
    parts: list[bytes] = []
    for value in values:
        encoded = canonical_cell(value)
        parts.append(len(encoded).to_bytes(8, "big") + encoded)
    return b"R" + len(parts).to_bytes(8, "big") + b"".join(parts)


def _normalize_cell(value: Any) -> str:
    """Compatibility helper returning a readable, type-marked value."""

    return canonical_cell(value).hex()


def row_hashes(
    df: Any,
    columns: list[str] | None = None,
    sample_cap: int | None = None,
    *,
    chunk_size: int | None = None,
) -> Any:
    """Return one hash per row and retain canonical records for verification.

    ``sample_cap`` is retained for callers that explicitly request sampling, but
    exact-overlap rules pass ``None`` and therefore scan all rows.
    """

    import pandas as pd

    if sample_cap is not None and (
        isinstance(sample_cap, bool) or not isinstance(sample_cap, int) or sample_cap <= 0
    ):
        raise ValueError("sample_cap must be a positive integer")
    if chunk_size is not None and (
        isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer")
    cols = columns if columns is not None else list(df.columns)
    work = df.loc[:, cols]
    sampled = False
    original_indices = list(range(len(work)))
    if sample_cap is not None and len(work) > sample_cap:
        work = work.iloc[:sample_cap]
        original_indices = original_indices[:sample_cap]
        sampled = True
    values: list[str] = []
    canonical: list[bytes] = []
    # Iterate through bounded frames rather than materializing a second full
    # object array. The returned digest/canonical arrays are necessarily kept
    # for collision verification, but peak row-processing memory is bounded.
    step = chunk_size or max(1, len(work))
    for start in range(0, len(work), step):
        chunk = work.iloc[start : start + step]
        for row in chunk.itertuples(index=False, name=None):
            record = canonical_row(row)
            canonical.append(record)
            values.append(hashlib.blake2b(record, digest_size=32).hexdigest())
    hashes = pd.Series(values, index=work.index, dtype="string")
    hashes.attrs["sampled"] = sampled
    hashes.attrs["columns"] = list(cols)
    hashes.attrs["canonical_rows"] = canonical
    hashes.attrs["original_indices"] = original_indices
    return hashes


def overlap_details(train_hashes: Any, test_hashes: Any) -> dict[str, Any]:
    """Return verified row overlap details for two hash series."""

    train_canonical = train_hashes.attrs.get("canonical_rows", [])
    test_canonical = test_hashes.attrs.get("canonical_rows", [])
    train_by_hash: dict[str, dict[bytes, list[int]]] = {}
    train_indices = train_hashes.attrs.get(
        "original_indices", list(range(len(train_hashes)))
    )
    test_indices = test_hashes.attrs.get(
        "original_indices", list(range(len(test_hashes)))
    )
    for position, (digest, record) in enumerate(zip(train_hashes.tolist(), train_canonical)):
        train_by_hash.setdefault(str(digest), {}).setdefault(record, []).append(position)
    common: set[str] = set()
    pairs: list[tuple[Any, Any]] = []
    for position, (digest, record) in enumerate(zip(test_hashes.tolist(), test_canonical)):
        matching = train_by_hash.get(str(digest), {}).get(record)
        if not matching:
            continue
        common.add(str(digest))
        train_position = matching[0]
        pairs.append((train_indices[train_position], test_indices[position]))
    return {"common": common, "count": len(pairs), "pairs": pairs}


def overlap(train_hashes: Any, test_hashes: Any) -> tuple[set[str], int]:
    details = overlap_details(train_hashes, test_hashes)
    return details["common"], details["count"]
