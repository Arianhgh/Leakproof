"""Input contract for the data layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DataAuditInput:
    train: Any  # pandas.DataFrame
    test: Any  # pandas.DataFrame
    val: Any | None = None  # pandas.DataFrame | None
    target: str | None = None  # column name
    group: str | None = None  # entity/group id column
    time: str | None = None  # datetime column
    feature_types: dict[str, str] | None = None  # optional hints

    def feature_columns(self) -> list[str]:
        cols = list(self.train.columns)
        drop = {c for c in (self.target, self.group, self.time) if c}
        return [c for c in cols if c not in drop]

    def splits(self) -> list[tuple[str, Any]]:
        out = [("train", self.train), ("test", self.test)]
        if self.val is not None:
            out.append(("val", self.val))
        return out
