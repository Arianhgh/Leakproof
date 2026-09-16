"""Built-in dataset-audit rules with complete split-pair coverage."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import Any

from ..core.context import DataContext
from ..core.models import (
    Category,
    Diagnostic,
    DiagnosticLevel,
    Finding,
    Fix,
    Layer,
    Location,
    Severity,
)
from ..core.references import KAPOOR_NARAYANAN_2023, KAUFMAN_2012, SKLEARN_PITFALLS
from ..core.registry import register
from ..core.rule import DataRule


def _loc(label: str) -> Location:
    return Location(context_label=label)


def _same_scalar(left: Any, right: Any) -> bool:
    """Compare target cells without treating two missing values as a conflict."""

    def missing(value: Any) -> bool:
        try:
            import pandas as pd

            result = pd.isna(value)
            return bool(result) if not hasattr(result, "__len__") else False
        except Exception:
            return value is None

    left_missing = missing(left)
    right_missing = missing(right)
    if left_missing or right_missing:
        return left_missing and right_missing
    try:
        result = left == right
        return bool(result) if not hasattr(result, "__len__") else False
    except Exception:
        return repr(left) == repr(right)


def _pairs(ctx: DataContext) -> list[tuple[tuple[str, Any], tuple[str, Any]]]:
    return list(itertools.combinations(ctx.audit_input.splits(), 2))


def _hash_columns(ctx: DataContext) -> list[str]:
    ai = ctx.audit_input
    include = ctx.config.data.hash_include
    if include:
        columns = list(include)
    else:
        columns = ai.identity_columns()
    excluded = set(ctx.config.data.hash_exclude)
    return [column for column in columns if column not in excluded]


def _hashes(ctx: DataContext, split_name: str, frame: Any, columns: list[str]) -> Any:
    from .hashing import row_hashes

    key = "hashes:" + split_name + ":" + "\x1f".join(columns)
    if key not in ctx.cache:
        # Exact and duplicate rules are always complete.  sample_cap is a
        # near-duplicate/probe resource limit, never a silent exact-match cap.
        ctx.cache[key] = row_hashes(
            frame,
            columns,
            sample_cap=None,
            chunk_size=ctx.config.data.hash_chunk_size,
        )
    return ctx.cache[key]


def _sample_rows(frame: Any, cap: int, *, deterministic: bool) -> tuple[Any, list[int], bool]:
    """Bound a similarity probe while retaining original row positions."""

    length = len(frame)
    if length <= cap:
        return frame, list(range(length)), False
    import numpy as np

    generator = np.random.default_rng(0 if deterministic else None)
    positions = sorted(int(value) for value in generator.choice(length, size=cap, replace=False))
    return frame.iloc[positions], positions, True


def _record_unavailable(ctx: DataContext, check: str, message: str) -> None:
    if check not in ctx.coverage.unavailable_checks:
        ctx.coverage.unavailable_checks.append(check)
    ctx.diagnostics.append(
        Diagnostic(
            code="LP330",
            message=message,
            level=DiagnosticLevel.WARNING,
            layer=Layer.DATA,
            rule_id=check,
        )
    )


def _record_partial(ctx: DataContext, check: str, message: str) -> None:
    if check not in ctx.coverage.sampled_checks:
        ctx.coverage.sampled_checks.append(check)
    ctx.diagnostics.append(
        Diagnostic(
            code="LP331",
            message=message,
            level=DiagnosticLevel.WARNING,
            layer=Layer.DATA,
            rule_id=check,
        )
    )


@register
class D001(DataRule):
    id = "D001"
    name = "exact row overlap across splits"
    category = Category.DATA_OVERLAP
    severity = Severity.CRITICAL
    layers = (Layer.DATA,)
    references = (KAUFMAN_2012,)
    rationale = (
        "Identical feature rows in separate split partitions are an observed overlap under "
        "the declared split policy. A matching feature row with a conflicting target is also "
        "reported as overlap evidence."
    )

    def check(self, ctx: DataContext) -> Iterable[Finding]:
        from .hashing import overlap_details

        ai = ctx.audit_input
        columns = _hash_columns(ctx)
        if not columns:
            return
        for (left_name, left), (right_name, right) in _pairs(ctx):
            left_hashes = _hashes(ctx, left_name, left, columns)
            right_hashes = _hashes(ctx, right_name, right, columns)
            details = overlap_details(left_hashes, right_hashes)
            if not details["count"]:
                continue
            conflicts = 0
            examples: list[list[int]] = []
            if ai.target and ai.target in left.columns and ai.target in right.columns:
                for left_pos, right_pos in details["pairs"]:
                    left_target = left.iloc[int(left_pos)][ai.target]
                    right_target = right.iloc[int(right_pos)][ai.target]
                    different = not _same_scalar(left_target, right_target)
                    if different:
                        conflicts += 1
                    if len(examples) < ctx.config.data.max_examples:
                        examples.append([int(left_pos), int(right_pos)])
            denominator = max(1, len(right))
            fraction = details["count"] / denominator
            ctx.cache.setdefault("exact_pairs", {})[(left_name, right_name)] = set(
                details["pairs"]
            )
            yield Finding(
                rule_id=self.id,
                category=self.category,
                severity=self.severity,
                layer=Layer.DATA,
                message=(
                    f"{details['count']} {right_name} rows ({fraction:.1%}) have identical "
                    f"features to {left_name} rows under the declared split policy."
                    + (f" {conflicts} matching feature rows have conflicting targets." if conflicts else "")
                ),
                location=_loc(f"{left_name}/{right_name} exact overlap ({details['count']} rows)"),
                confidence=1.0,
                references=self.references,
                fix=Fix(summary="Remove overlapping rows or revise the split policy.", autofixable=False),
                evidence={
                    "split_pair": [left_name, right_name],
                    "overlap_rows": details["count"],
                    "denominator_rows": len(right),
                    "overlap_fraction": fraction,
                    "unique_overlapping_hashes": len(details["common"]),
                    "conflicting_target_rows": conflicts,
                    "hash_columns": columns,
                    "examples": examples,
                    "observation": "identical feature identity under the declared split policy",
                },
            )


@register
class D003(DataRule):
    id = "D003"
    name = "group/entity in multiple splits"
    category = Category.DATA_OVERLAP
    severity = Severity.CRITICAL
    layers = (Layer.DATA,)
    references = (SKLEARN_PITFALLS,)
    rationale = "Group-aware split policies keep rows belonging to one entity on one side."

    def check(self, ctx: DataContext) -> Iterable[Finding]:
        ai = ctx.audit_input
        if not ai.group:
            return
        for (left_name, left), (right_name, right) in _pairs(ctx):
            left_groups = set(left[ai.group].dropna().unique())
            right_groups = set(right[ai.group].dropna().unique())
            common = left_groups & right_groups
            if not common:
                continue
            sample = [str(value) for value in sorted(common, key=str)[: ctx.config.data.max_examples]]
            yield Finding(
                rule_id=self.id,
                category=self.category,
                severity=self.severity,
                layer=Layer.DATA,
                message=(
                    f"{len(common)} `{ai.group}` group(s) appear in both {left_name} and "
                    f"{right_name}; use a group-aware split policy."
                ),
                location=_loc(f"{left_name}/{right_name} group overlap on '{ai.group}'"),
                confidence=1.0,
                references=self.references,
                fix=Fix(summary="Split by group so one entity cannot straddle the boundary.", autofixable=False),
                evidence={
                    "split_pair": [left_name, right_name],
                    "group": ai.group,
                    "overlapping_groups": len(common),
                    "examples": sample,
                },
            )


@register
class D007(DataRule):
    id = "D007"
    name = "duplicate rows inside held-out splits"
    category = Category.DATA_OVERLAP
    severity = Severity.MEDIUM
    layers = (Layer.DATA,)
    references = (KAUFMAN_2012,)
    rationale = "Internal duplicates reduce the effective held-out sample size."

    def check(self, ctx: DataContext) -> Iterable[Finding]:

        columns = _hash_columns(ctx)
        if not columns:
            return
        for split_name, frame in ctx.audit_input.splits():
            if split_name == "train":
                continue
            hashes = _hashes(ctx, split_name, frame, columns)
            counts = hashes.value_counts()
            duplicate_rows = int((counts[counts > 1] - 1).sum()) if (counts > 1).any() else 0
            if not duplicate_rows:
                continue
            fraction = duplicate_rows / max(1, len(frame))
            yield Finding(
                rule_id=self.id,
                category=self.category,
                severity=self.severity,
                layer=Layer.DATA,
                message=(
                    f"{split_name} contains {duplicate_rows} duplicate row(s) ({fraction:.1%}); "
                    "its effective sample size is smaller than it appears."
                ),
                location=_loc(f"{split_name} duplicates ({duplicate_rows} rows)"),
                confidence=0.9,
                references=self.references,
                fix=Fix(summary=f"Deduplicate the {split_name} split.", autofixable=False),
                evidence={
                    "split": split_name,
                    "duplicate_rows": duplicate_rows,
                    "denominator_rows": len(frame),
                    "fraction": fraction,
                    "hash_columns": columns,
                },
            )


@register
class D002(DataRule):
    advisory_only = True
    id = "D002"
    name = "near-duplicate rows across splits"
    category = Category.DATA_OVERLAP
    severity = Severity.HIGH
    layers = (Layer.DATA,)
    references = (KAUFMAN_2012,)
    rationale = "Similarity is evidence to review, not automatic proof that every result is invalid."

    def check(self, ctx: DataContext) -> Iterable[Finding]:
        from pandas.api import types as ptypes

        from .neardup import image_near_duplicates, numeric_near_duplicates, text_near_duplicates

        ai = ctx.audit_input
        cfg = ctx.config.data
        features = ai.model_feature_columns()
        if not features:
            return
        hints = ai.feature_types or {}
        for (left_name, left), (right_name, right) in _pairs(ctx):
            exact_pairs = ctx.cache.get("exact_pairs", {}).get((left_name, right_name), set())
            if not exact_pairs:
                from .hashing import overlap_details

                exact_pairs = set(
                    overlap_details(
                        _hashes(ctx, left_name, left, _hash_columns(ctx)),
                        _hashes(ctx, right_name, right, _hash_columns(ctx)),
                    )["pairs"]
                )
            modalities: list[tuple[str, Any]] = []
            categorical_columns: list[str] = []
            for column in features:
                modality = hints.get(column)
                if modality == "image" or (modality is None and _looks_like_image_path(left[column])):
                    modalities.append((f"image:{column}", ("image", column)))
                elif modality == "text" or (modality is None and _looks_like_text(left[column])):
                    modalities.append((f"text:{column}", ("text", column)))
                elif modality in {"categorical", "group"}:
                    categorical_columns.append(column)
                elif modality == "datetime":
                    modalities.append((f"datetime:{column}", ("datetime", column)))
            handled_columns = {
                selected
                for _, (kind, selected) in modalities
                if kind in {"image", "text"} and isinstance(selected, str)
            }
            handled_columns.update(categorical_columns)
            numeric = []
            for column in features:
                modality = hints.get(column)
                if modality in {"categorical", "group", "image", "text"}:
                    continue
                if modality in {"numeric", "datetime"} or (
                    modality is None
                    and column not in handled_columns
                    and ptypes.is_numeric_dtype(left[column])
                ):
                    numeric.append(column)
            if numeric:
                modalities.append(("numeric:" + ",".join(numeric), ("numeric", numeric)))
            # Categorical equality is useful as supporting evidence, but one
            # coincidentally matching category must not establish a full-row
            # near duplicate by itself. It is therefore joined to findings
            # produced by an independent numeric/text/image modality below.
            categorical_pairs: set[tuple[int, int]] = set()
            if categorical_columns:
                from .hashing import overlap_details

                categorical_pairs = set(
                    overlap_details(
                        _hashes(ctx, left_name, left, categorical_columns),
                        _hashes(ctx, right_name, right, categorical_columns),
                    )["pairs"]
                )
            if not modalities:
                continue
            sampled_left, left_positions, left_sampled = _sample_rows(
                left,
                cfg.sample_cap,
                deterministic=cfg.deterministic_sampling,
            )
            sampled_right, right_positions, right_sampled = _sample_rows(
                right,
                cfg.sample_cap,
                deterministic=cfg.deterministic_sampling,
            )
            if left_sampled or right_sampled:
                _record_partial(
                    ctx,
                    "D002",
                    f"{left_name}/{right_name} near-duplicate checks sampled at most "
                    f"{cfg.sample_cap} rows per split",
                )
            for label, (kind, selected) in modalities:
                if kind == "image":
                    pairs = image_near_duplicates(
                        sampled_left[selected].astype("string").tolist(),
                        sampled_right[selected].astype("string").tolist(),
                        cfg.imagehash_distance,
                        max_candidates=cfg.max_neardup_candidates,
                        max_pairs=cfg.max_neardup_pairs,
                    )
                elif kind == "text":
                    pairs = text_near_duplicates(
                        sampled_left[selected].astype("string").fillna("").tolist(),
                        sampled_right[selected].astype("string").fillna("").tolist(),
                        cfg.neardup_text_threshold,
                        max_candidates=cfg.max_neardup_candidates,
                        max_pairs=cfg.max_neardup_pairs,
                    )
                elif kind == "numeric":
                    pairs = numeric_near_duplicates(
                        _numeric_matrix(sampled_left, selected),
                        _numeric_matrix(sampled_right, selected),
                        cfg.neardup_distance,
                        max_candidates=cfg.max_neardup_candidates,
                        max_pairs=cfg.max_neardup_pairs,
                    )
                else:
                    # Datetime values are converted to elapsed seconds so
                    # numeric distance remains meaningful across pandas 2/3
                    # and timezone-aware columns.
                    pairs = numeric_near_duplicates(
                        _numeric_matrix(sampled_left, [selected]),
                        _numeric_matrix(sampled_right, [selected]),
                        cfg.neardup_distance,
                        max_candidates=cfg.max_neardup_candidates,
                        max_pairs=cfg.max_neardup_pairs,
                    )
                if pairs.unavailable:
                    _record_unavailable(ctx, "D002", f"{label} near-duplicate check unavailable: {pairs.unavailable}")
                    continue
                if not pairs.complete:
                    _record_partial(ctx, "D002", f"{label} near-duplicate check reached its resource limit")
                if pairs.unreadable:
                    _record_unavailable(ctx, "D002", f"{pairs.unreadable} image value(s) were unreadable for {label}")
                mapped_pairs = [
                    (left_positions[train_pos], right_positions[test_pos], score)
                    for train_pos, test_pos, score in pairs
                    if train_pos < len(left_positions) and test_pos < len(right_positions)
                ]
                filtered = [
                    (train_pos, test_pos, score)
                    for train_pos, test_pos, score in mapped_pairs
                    if (train_pos, test_pos) not in exact_pairs
                ]
                heldout_positions = sorted({test_pos for _, test_pos, _ in filtered})
                if not heldout_positions:
                    continue
                fraction = len(heldout_positions) / max(1, len(right))
                yield Finding(
                    rule_id=self.id,
                    category=self.category,
                    severity=self.severity,
                    layer=Layer.DATA,
                    message=(
                        f"{len(heldout_positions)} {right_name} row(s) are similar to {left_name} "
                        f"rows under {label}; review the split policy."
                    ),
                    location=_loc(f"{left_name}/{right_name} near-duplicate overlap ({label})"),
                    confidence=0.75,
                    references=self.references,
                    advisory_only=self.advisory_only,
                    fix=Fix(summary="Review similar rows and the split policy.", autofixable=False),
                    evidence={
                        "split_pair": [left_name, right_name],
                        "modality": label,
                        "similarity_criterion": (
                            f"text Jaccard >= {cfg.neardup_text_threshold}"
                            if kind == "text"
                            else (f"numeric normalized distance <= {cfg.neardup_distance}" if kind == "numeric" else f"perceptual image-hash distance <= {cfg.imagehash_distance}")
                        ),
                        "similar_rows": len(heldout_positions),
                        "denominator_rows": len(right),
                        "fraction": fraction,
                        "examples": [[int(i), int(j), float(score)] for i, j, score in filtered[: cfg.max_examples]],
                        "comparisons": pairs.comparisons,
                        "complete": pairs.complete,
                        "sampled": bool(pairs.sampled or left_sampled or right_sampled),
                        "source_rows_scanned": len(left_positions),
                        "heldout_rows_scanned": len(right_positions),
                        "categorical_columns": categorical_columns,
                        "categorical_agreement_rows": sum(
                            1
                            for train_pos, test_pos, _ in filtered
                            if (train_pos, test_pos) in categorical_pairs
                        ),
                    },
                )


def _looks_like_image_path(series: Any) -> bool:
    try:
        sample = series.dropna().astype("string").head(20).str.lower()
    except Exception:
        return False
    if sample.empty:
        return False
    suffixes = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff")
    return bool(sample.map(lambda value: str(value).endswith(suffixes)).mean() >= 0.8)


def _looks_like_text(series: Any) -> bool:
    """Infer free text conservatively; low-cardinality strings are categories."""

    try:
        sample = series.dropna().head(200)
    except Exception:
        return False
    if sample.empty:
        return False
    values = sample.tolist()
    if not all(isinstance(value, str) for value in values):
        return False
    unique = len(set(values))
    lengths = [len(value) for value in values]
    words = [len(value.split()) for value in values]
    return bool(
        max(lengths, default=0) >= 24
        or sum(words) / max(1, len(words)) >= 4
        or unique > max(10, len(values) * 0.5)
    )


def _numeric_matrix(frame: Any, columns: list[str]) -> Any:
    """Return finite numeric data for a bounded similarity probe.

    Missing and infinite values are imputed with the finite column median for
    the probe only. Exact identity remains type-aware and never uses this
    lossy representation.
    """

    import numpy as np
    import pandas as pd

    arrays: list[Any] = []
    for column in columns:
        series = frame[column]
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            converted = pd.to_datetime(series, errors="coerce", utc=True)
            values = converted.astype("int64").to_numpy(dtype=float)
            values[converted.isna().to_numpy()] = np.nan
            values /= 1_000_000_000.0
        else:
            converted = pd.to_numeric(series, errors="coerce")
            try:
                values = converted.to_numpy(dtype=float, na_value=np.nan)
            except (TypeError, ValueError):
                try:
                    values = converted.to_numpy(dtype=float)
                except (TypeError, ValueError):
                    values = np.array(
                        [
                            float(value)
                            if value is not None and not pd.isna(value)
                            else np.nan
                            for value in converted
                        ],
                        dtype=float,
                    )
        # pandas may expose a read-only view from ``to_numpy`` (notably for
        # extension dtypes); probes impute in place, so make ownership
        # explicit before normalizing non-finite values.
        values = np.array(values, dtype=float, copy=True)
        values[~np.isfinite(values)] = np.nan
        finite = values[np.isfinite(values)]
        replacement = float(np.median(finite)) if finite.size else 0.0
        values[~np.isfinite(values)] = replacement
        arrays.append(values)
    return np.column_stack(arrays) if arrays else np.empty((len(frame), 0))


@register
class D005(DataRule):
    advisory_only = True
    id = "D005"
    name = "single feature strongly predicts target"
    category = Category.TARGET_LEAKAGE
    severity = Severity.HIGH
    layers = (Layer.DATA,)
    references = (KAUFMAN_2012, KAPOOR_NARAYANAN_2023)
    rationale = "A strong univariate predictor is a proxy-leak hypothesis to verify, not proof by itself."

    def check(self, ctx: DataContext) -> Iterable[Finding]:
        from .target import univariate_scores

        ai = ctx.audit_input
        if not ai.target:
            return
        features = ai.model_feature_columns()
        if not features:
            return
        probe = univariate_scores(
            ai.train,
            ai.target,
            features,
            groups=ai.train[ai.group] if ai.group else None,
            time=ai.train[ai.time] if ai.time else None,
            folds=ctx.config.data.probe_cv_folds,
        )
        for column, reason in probe.metadata.get("unavailable", {}).items():
            _record_unavailable(ctx, "D005", f"target predictivity probe unavailable for {column}: {reason}")
        threshold = ctx.config.data.target_leakage_score
        for column, score in sorted(probe.items(), key=lambda item: -item[1]):
            if score < threshold:
                continue
            yield Finding(
                rule_id=self.id,
                category=self.category,
                severity=self.severity,
                layer=Layer.DATA,
                message=(
                    f"Feature `{column}` has one-feature {probe.metadata.get('metric', 'predictivity')} "
                    f"score {score:.3f}; audit it for a proxy or post-outcome field."
                ),
                location=_loc(f"target predictivity: {column} (score={score:.3f})"),
                confidence=0.7,
                references=self.references,
                advisory_only=self.advisory_only,
                fix=Fix(summary=f"Verify `{column}` is available at prediction time.", autofixable=False),
                evidence={
                    "feature": column,
                    "univariate_score": round(score, 4),
                    "threshold": threshold,
                    **probe.metadata,
                },
            )


@register
class D006(DataRule):
    advisory_only = True
    id = "D006"
    name = "suspiciously high feature-target mutual information"
    category = Category.TARGET_LEAKAGE
    severity = Severity.MEDIUM
    layers = (Layer.DATA,)
    references = (KAUFMAN_2012,)
    rationale = "Mutual information dominance is an advisory signal requiring a data-dictionary review."

    def check(self, ctx: DataContext) -> Iterable[Finding]:
        import statistics

        from .target import mutual_info

        ai = ctx.audit_input
        if not ai.target:
            return
        features = ai.model_feature_columns()
        if len(features) < 3:
            return
        values = mutual_info(ai.train, ai.target, features)
        for column, reason in values.metadata.get("unavailable", {}).items():
            _record_unavailable(ctx, "D006", f"mutual-information probe unavailable for {column}: {reason}")
        if not values:
            _record_unavailable(ctx, "D006", "mutual-information probe was unavailable")
            return
        median = statistics.median(values.values())
        for column, score in sorted(values.items(), key=lambda item: -item[1]):
            other = max((value for name, value in values.items() if name != column), default=0.0)
            if score < max(ctx.config.data.mi_floor, ctx.config.data.mi_dominance_ratio * median):
                continue
            if other and score <= ctx.config.data.mi_dominance_ratio * other:
                continue
            yield Finding(
                rule_id=self.id,
                category=self.category,
                severity=self.severity,
                layer=Layer.DATA,
                message=(
                    f"Feature `{column}` has mutual information {score:.3f}, above the cohort "
                    f"median {median:.3f}; check for a leaked proxy."
                ),
                location=_loc(f"high MI: {column} (MI={score:.3f})"),
                confidence=0.5,
                references=self.references,
                advisory_only=self.advisory_only,
                fix=Fix(summary=f"Audit `{column}` for target leakage.", autofixable=False),
                evidence={
                    "feature": column,
                    "mi": round(score, 4),
                    "cohort_median": round(median, 4),
                    "floor": ctx.config.data.mi_floor,
                    "dominance_ratio": ctx.config.data.mi_dominance_ratio,
                },
            )


@register
class TM001(DataRule):
    id = "TM001"
    name = "temporal overlap across ordered splits"
    category = Category.TEMPORAL
    severity = Severity.HIGH
    layers = (Layer.DATA,)
    references = (KAPOOR_NARAYANAN_2023,)
    rationale = "Chronological train → validation → test boundaries must not overlap."

    def check(self, ctx: DataContext) -> Iterable[Finding]:
        from .temporal import time_ranges

        ai = ctx.audit_input
        if not ai.time:
            return
        ordered = ai.splits()
        ranges: dict[str, tuple[Any, Any]] = {}
        for split_name, frame in ordered:
            try:
                ranges[split_name] = time_ranges(frame, ai.time)
            except ValueError as exc:
                _record_unavailable(ctx, "TM001", f"invalid timestamps in {split_name}: {exc}")
                return
        for (left_name, _), (right_name, _) in zip(ordered, ordered[1:]):
            left_min, left_max = ranges[left_name]
            right_min, right_max = ranges[right_name]
            if left_max is None or right_min is None:
                _record_unavailable(ctx, "TM001", f"{left_name}/{right_name} has no valid timestamps")
                continue
            if left_max >= right_min:
                yield Finding(
                    rule_id=self.id,
                    category=self.category,
                    severity=self.severity,
                    layer=Layer.DATA,
                    message=(
                        f"{left_name} extends to {left_max} while {right_name} starts at {right_min}; "
                        "the chronological split boundary overlaps."
                    ),
                    location=_loc(f"temporal overlap {left_name}/{right_name} on '{ai.time}'"),
                    confidence=0.95,
                    references=self.references,
                    fix=Fix(summary="Split in chronological order so earlier data precedes later data.", autofixable=False),
                    evidence={
                        "split_pair": [left_name, right_name],
                        "time_column": ai.time,
                        "left_min": str(left_min),
                        "left_max": str(left_max),
                        "right_min": str(right_min),
                        "right_max": str(right_max),
                    },
                )


@register
class M001(DataRule):
    advisory_only = True
    id = "M001"
    name = "accuracy may be misleading for an imbalanced target"
    category = Category.METRIC
    severity = Severity.MEDIUM
    layers = (Layer.DATA,)
    references = (SKLEARN_PITFALLS,)
    rationale = "Class imbalance makes an unqualified accuracy report difficult to interpret."

    def check(self, ctx: DataContext) -> Iterable[Finding]:
        ai = ctx.audit_input
        if not ai.target:
            return
        counts = ai.train[ai.target].value_counts(normalize=True, dropna=True)
        if counts.empty:
            return
        majority = float(counts.iloc[0])
        if majority < ctx.config.data.imbalance_ratio or len(counts) > 50:
            return
        yield Finding(
            rule_id=self.id,
            category=self.category,
            severity=self.severity,
            layer=Layer.DATA,
            message=(
                f"The target's majority class is {majority:.1%}; an unqualified accuracy "
                "metric would be dominated by class prevalence. Consider balanced accuracy, "
                "F1, ROC-AUC, or PR-AUC."
            ),
            location=_loc(f"class imbalance ({majority:.1%} majority)"),
            confidence=0.8,
            references=self.references,
            advisory_only=self.advisory_only,
            fix=Fix(summary="Choose a metric appropriate for the class distribution.", autofixable=False),
            evidence={"majority_fraction": majority, "n_classes": int(len(counts)), "metric_supplied": False},
        )
