"""Runtime taint tracking: row-identity provenance across split membership."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventKind(str, Enum):
    SPLIT = "split"
    FIT = "fit"
    APPLY = "apply"
    SCORE = "score"


@dataclass
class RuntimeEvent:
    kind: EventKind
    call_site: str  # "file.py:42"
    run_ordinal: int
    payload: dict = field(default_factory=dict)


def row_hashes(data: Any, cap: int = 200_000) -> list[str]:
    """Best-effort stable per-row hashes for array-like / DataFrame input."""
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        return []

    try:
        import pandas as pd

        if isinstance(data, (pd.DataFrame, pd.Series)):
            arr = data.to_numpy()
        else:
            arr = np.asarray(data)
    except Exception:
        try:
            arr = np.asarray(data)
        except Exception:
            return []

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n = arr.shape[0]
    if n > cap:
        arr = arr[:cap]
    out: list[str] = []
    for i in range(arr.shape[0]):
        row = arr[i]
        try:
            b = row.tobytes()
        except Exception:
            b = repr(row.tolist()).encode("utf-8")
        out.append(hashlib.blake2b(b, digest_size=12).hexdigest())
    return out


class TaintTable:
    """Maps row-identity (content hash) to split membership and tracks fits."""

    def __init__(self) -> None:
        self.test_hashes: set[str] = set()
        self.val_hashes: set[str] = set()
        self.train_hashes: set[str] = set()
        # per-split-signature score counts (for T001)
        self.score_counts: dict[str, int] = {}

    def mark_split(self, train: list[str], test: list[str], val: list[str] | None = None) -> None:
        self.train_hashes.update(train)
        self.test_hashes.update(test)
        if val:
            self.val_hashes.update(val)

    def eval_side(self) -> set[str]:
        return self.test_hashes | self.val_hashes

    @staticmethod
    def signature(hashes: list[str]) -> str:
        h = hashlib.blake2b(digest_size=12)
        for x in sorted(hashes):
            h.update(x.encode("ascii"))
        return h.hexdigest()
