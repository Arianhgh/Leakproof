"""Validated input contract for the pandas data-audit layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class DataInputError(ValueError):
    """Raised when explicit split data cannot be audited safely."""


@dataclass
class DataAuditInput:
    train: Any
    test: Any
    val: Any | None = None
    target: str | None = None
    group: str | None = None
    time: str | None = None
    feature_types: dict[str, str] | None = None

    def validate(self) -> list[str]:
        try:
            import pandas as pd
        except Exception as exc:  # pragma: no cover - package-extra isolation
            raise DataInputError("data auditing requires pandas") from exc

        frames = [("train", self.train), ("test", self.test)]
        if self.val is not None:
            frames.insert(1, ("val", self.val))
        for name, frame in frames:
            if not isinstance(frame, pd.DataFrame):
                raise DataInputError(f"{name} must be a pandas.DataFrame")
            if not frame.columns.is_unique:
                duplicates = frame.columns[frame.columns.duplicated()].tolist()
                raise DataInputError(f"{name} has duplicate column names: {duplicates}")
        for label, column in (("target", self.target), ("group", self.group), ("time", self.time)):
            if column is not None and not isinstance(column, str):
                raise DataInputError(f"requested {label} column must be a string")
        columns = set(self.train.columns)
        for name, frame in frames[1:]:
            if set(frame.columns) != columns:
                missing = sorted(columns - set(frame.columns), key=str)
                extra = sorted(set(frame.columns) - columns, key=str)
                raise DataInputError(
                    f"{name} schema does not match train (missing={missing}, extra={extra})"
                )
        for label, column in (("target", self.target), ("group", self.group), ("time", self.time)):
            if column is not None and column not in columns:
                raise DataInputError(f"requested {label} column does not exist: {column}")
        if self.feature_types is not None:
            if not isinstance(self.feature_types, dict):
                raise DataInputError("feature_types must be a mapping of column to modality")
            unknown = sorted(set(self.feature_types) - columns, key=str)
            if unknown:
                raise DataInputError(f"feature_types names missing columns: {unknown}")
            allowed = {"numeric", "categorical", "text", "image", "datetime", "group"}
            invalid = sorted(
                (column, modality)
                for column, modality in self.feature_types.items()
                if not isinstance(column, str) or not isinstance(modality, str) or modality not in allowed
            )
            if invalid:
                raise DataInputError(f"invalid feature_types entries: {invalid}")
        return [name for name, _ in frames]

    def quality_notes(self) -> list[str]:
        """Describe supported data-quality conditions without hiding them.

        These conditions are not automatically failures: exact hashing is
        positional and type-aware, probes impute within folds, and temporal
        parsing is validated separately. Keeping them in coverage makes the
        policy visible to callers and report consumers.
        """

        import numpy as np
        import pandas as pd

        notes: list[str] = []
        for name, frame in self.splits():
            if frame.empty:
                notes.append(f"{name} split is empty; row-dependent checks have no observations")
            missing = int(frame.isna().sum().sum())
            if missing:
                notes.append(f"{name} contains {missing} missing value(s); canonical hashing normalizes them")
            if frame.index.has_duplicates:
                notes.append(
                    f"{name} index contains duplicate labels; checks use positional row identities"
                )
            infinite_columns: list[str] = []
            for column in frame.columns:
                series = frame[column]
                if not pd.api.types.is_numeric_dtype(series.dtype):
                    continue
                try:
                    values = pd.to_numeric(series, errors="coerce").to_numpy(
                        dtype=float, na_value=np.nan
                    )
                except (TypeError, ValueError):
                    continue
                if bool(np.isinf(values).any()):
                    infinite_columns.append(str(column))
            if infinite_columns:
                notes.append(
                    f"{name} contains infinity in column(s) {infinite_columns}; numeric probes treat it as unavailable data"
                )
            if self.time and self.time in frame.columns:
                dtype_text = str(frame[self.time].dtype).lower()
                if "tz" in dtype_text or "timezone" in dtype_text:
                    notes.append(f"{name} uses timezone-aware timestamps; temporal checks normalize to UTC when needed")
        return notes

    def model_feature_columns(self) -> list[str]:
        drop = {column for column in (self.target, self.group, self.time) if column is not None}
        return [column for column in self.train.columns if column not in drop]

    def feature_columns(self) -> list[str]:
        """Backward-compatible alias for model features."""

        return self.model_feature_columns()

    def identity_columns(self) -> list[str]:
        """Columns used for exact identity by default.

        Group and time columns intentionally remain included; only the target is
        excluded unless hash_include/hash_exclude says otherwise.
        """

        return [column for column in self.train.columns if column != self.target]

    def splits(self) -> list[tuple[str, Any]]:
        out = [("train", self.train)]
        if self.val is not None:
            out.append(("val", self.val))
        out.append(("test", self.test))
        return out

    def split_map(self) -> dict[str, Any]:
        return dict(self.splits())
