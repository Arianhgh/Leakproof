"""Built-in data-layer rules: D001, D002, D003, D005, D006, D007, TM001, M001."""

from __future__ import annotations

from collections.abc import Iterable

from ..core.context import DataContext
from ..core.models import Category, Finding, Fix, Layer, Location, Severity
from ..core.references import KAPOOR_NARAYANAN_2023, KAUFMAN_2012, SKLEARN_PITFALLS
from ..core.registry import register
from ..core.rule import DataRule


def _loc(label: str) -> Location:
    return Location(context_label=label)


def _second_highest(mi: dict[str, float], exclude: str) -> float:
    others = [v for k, v in mi.items() if k != exclude]
    return max(others) if others else 0.0


@register
class D001(DataRule):
    id = "D001"
    name = "exact train/test row overlap"
    category = Category.DATA_OVERLAP
    severity = Severity.CRITICAL
    layers = (Layer.DATA,)
    references = (KAUFMAN_2012,)
    rationale = (
        "Identical rows in train and test mean the model is evaluated on examples it trained "
        "on; reported scores are invalid."
    )

    def check(self, ctx: DataContext) -> Iterable[Finding]:
        from .hashing import overlap, row_hashes

        ai = ctx.audit_input
        cols = ai.feature_columns() + ([ai.target] if ai.target else [])
        cap = ctx.config.data.sample_cap
        th = row_hashes(ai.train, cols, sample_cap=cap)
        sh = row_hashes(ai.test, cols, sample_cap=cap)
        common, count = overlap(th, sh)
        if count > 0:
            frac = count / max(1, len(ai.test))
            yield Finding(
                rule_id=self.id,
                category=self.category,
                severity=self.severity,
                layer=Layer.DATA,
                message=(
                    f"{count} test rows ({frac:.1%}) are exact duplicates of training rows; "
                    f"results are invalid until they are removed."
                ),
                location=_loc(f"train/test overlap ({count} rows)"),
                confidence=1.0,
                references=self.references,
                fix=Fix(summary="Deduplicate before splitting; drop overlapping rows from test.", autofixable=False),
                evidence={
                    "overlap_rows": count,
                    "overlap_fraction": frac,
                    "unique_overlapping_hashes": len(common),
                    "sampled": bool(th.attrs.get("sampled") or sh.attrs.get("sampled")),
                },
            )


@register
class D003(DataRule):
    id = "D003"
    name = "group/entity in both splits"
    category = Category.DATA_OVERLAP
    severity = Severity.CRITICAL
    layers = (Layer.DATA,)
    references = (SKLEARN_PITFALLS,)
    rationale = (
        "When the same entity (patient/user/session) appears in both train and test, the model "
        "can memorize entity-specific signal. Split by group."
    )

    def check(self, ctx: DataContext) -> Iterable[Finding]:
        ai = ctx.audit_input
        if not ai.group or ai.group not in ai.train.columns:
            return
        train_groups = set(ai.train[ai.group].dropna().unique())
        test_groups = set(ai.test[ai.group].dropna().unique())
        common = train_groups & test_groups
        if common:
            sample = list(common)[:10]
            yield Finding(
                rule_id=self.id,
                category=self.category,
                severity=self.severity,
                layer=Layer.DATA,
                message=(
                    f"{len(common)} `{ai.group}` group(s) appear in both train and test; use "
                    f"GroupKFold/GroupShuffleSplit to keep groups on one side."
                ),
                location=_loc(f"group overlap on '{ai.group}'"),
                confidence=1.0,
                references=self.references,
                fix=Fix(summary="Split by group so no entity straddles the boundary.", autofixable=False),
                evidence={"group": ai.group, "overlapping_groups": len(common), "sample": [str(s) for s in sample]},
            )


@register
class D007(DataRule):
    id = "D007"
    name = "duplicate rows inflating test"
    category = Category.DATA_OVERLAP
    severity = Severity.MEDIUM
    layers = (Layer.DATA,)
    references = (KAUFMAN_2012,)
    rationale = (
        "Internal duplicate rows in the test set overstate the effective sample size and let a "
        "few patterns dominate the score."
    )

    def check(self, ctx: DataContext) -> Iterable[Finding]:
        from .hashing import row_hashes

        ai = ctx.audit_input
        cols = ai.feature_columns() + ([ai.target] if ai.target else [])
        h = row_hashes(ai.test, cols, sample_cap=ctx.config.data.sample_cap)
        counts = h.value_counts()
        dup_rows = int((counts[counts > 1] - 1).sum()) if (counts > 1).any() else 0
        if dup_rows > 0:
            frac = dup_rows / max(1, len(ai.test))
            yield Finding(
                rule_id=self.id,
                category=self.category,
                severity=self.severity,
                layer=Layer.DATA,
                message=(
                    f"The test set contains {dup_rows} duplicate rows ({frac:.1%}); effective "
                    f"test size is smaller than it appears."
                ),
                location=_loc(f"test duplicates ({dup_rows} rows)"),
                confidence=0.9,
                references=self.references,
                fix=Fix(summary="Deduplicate the test set.", autofixable=False),
                evidence={"duplicate_rows": dup_rows, "fraction": frac},
            )


@register
class D002(DataRule):
    id = "D002"
    name = "near-duplicate across splits"
    category = Category.DATA_OVERLAP
    severity = Severity.HIGH
    layers = (Layer.DATA,)
    references = (KAUFMAN_2012,)
    rationale = (
        "Near-duplicate rows across the split (paraphrased text, augmented images, near-identical "
        "feature vectors) leak almost the same information as exact duplicates."
    )

    def check(self, ctx: DataContext) -> Iterable[Finding]:
        from .neardup import numeric_near_duplicates, text_near_duplicates

        ai = ctx.audit_input
        cfg = ctx.config.data
        feats = ai.feature_columns()
        if not feats:
            return

        text_cols = [c for c in feats if str(ai.train[c].dtype) == "object"]
        numeric_cols = [c for c in feats if c not in text_cols]

        pairs: list[tuple[int, int, float]] = []
        modality = None
        if text_cols:
            col = text_cols[0]
            pairs = text_near_duplicates(
                ai.train[col].astype(str).tolist(),
                ai.test[col].astype(str).tolist(),
                cfg.neardup_text_threshold,
            )
            modality = f"text:{col}"
        elif numeric_cols:
            pairs = numeric_near_duplicates(
                ai.train[numeric_cols].fillna(0.0).to_numpy(),
                ai.test[numeric_cols].fillna(0.0).to_numpy(),
                cfg.neardup_distance,
            )
            modality = "numeric"

        # exclude exact dups (handled by D001) from the count where possible
        if pairs:
            n = len({j for _, j, _ in pairs})
            frac = n / max(1, len(ai.test))
            yield Finding(
                rule_id=self.id,
                category=self.category,
                severity=self.severity,
                layer=Layer.DATA,
                message=(
                    f"{n} test rows ({frac:.1%}) are near-duplicates of training rows "
                    f"({modality}); they leak nearly identical information."
                ),
                location=_loc(f"near-duplicate overlap ({n} rows, {modality})"),
                confidence=0.75,
                references=self.references,
                fix=Fix(summary="Deduplicate near-identical rows before splitting.", autofixable=False),
                evidence={
                    "near_dup_test_rows": n,
                    "fraction": frac,
                    "modality": modality,
                    "examples": [[int(i), int(j), round(s, 4)] for i, j, s in pairs[:5]],
                },
            )


@register
class D005(DataRule):
    id = "D005"
    name = "single feature near-perfectly predicts target"
    category = Category.TARGET_LEAKAGE
    severity = Severity.HIGH
    layers = (Layer.DATA,)
    references = (KAUFMAN_2012, KAPOOR_NARAYANAN_2023)
    rationale = (
        "A lone feature that predicts the target almost perfectly is usually a proxy/leak (an id, "
        "a post-outcome field, or a transformed copy of the label)."
    )

    def check(self, ctx: DataContext) -> Iterable[Finding]:
        from .target import univariate_scores

        ai = ctx.audit_input
        if not ai.target or ai.target not in ai.train.columns:
            return
        feats = ai.feature_columns()
        if not feats:
            return
        scores = univariate_scores(ai.train, ai.target, feats)
        thr = ctx.config.data.target_leakage_score
        suspects = sorted(
            ((c, s) for c, s in scores.items() if s >= thr), key=lambda kv: -kv[1]
        )
        for col, score in suspects:
            yield Finding(
                rule_id=self.id,
                category=self.category,
                severity=self.severity,
                layer=Layer.DATA,
                message=(
                    f"Feature `{col}` alone predicts the target with score {score:.3f}; this is a "
                    f"likely leak (proxy or post-outcome field)."
                ),
                location=_loc(f"target leakage: {col} (score={score:.3f})"),
                confidence=0.7,
                references=self.references,
                fix=Fix(summary=f"Verify `{col}` is available at prediction time; drop if leaked.", autofixable=False),
                evidence={"feature": col, "univariate_score": round(score, 4), "threshold": thr},
            )


@register
class D006(DataRule):
    id = "D006"
    name = "suspiciously high feature-target MI"
    category = Category.TARGET_LEAKAGE
    severity = Severity.MEDIUM
    layers = (Layer.DATA,)
    references = (KAUFMAN_2012,)
    rationale = (
        "A feature whose mutual information with the target is far above the rest of the cohort "
        "may be a leaked proxy. Advisory; confirm against the data dictionary."
    )

    def check(self, ctx: DataContext) -> Iterable[Finding]:
        import statistics

        from .target import mutual_info

        ai = ctx.audit_input
        if not ai.target or ai.target not in ai.train.columns:
            return
        feats = ai.feature_columns()
        if len(feats) < 3:
            return
        mi = mutual_info(ai.train, ai.target, feats)
        # Robust dominance test: a leaked proxy has MI far above an absolute floor
        # AND far above the cohort median. (A plain z-score saturates near
        # sqrt(n-1), so it is unreachable for a handful of features.)
        values = list(mi.values())
        median = statistics.median(values) if values else 0.0
        floor = 0.2  # nats; below this MI is not "suspiciously high"
        for col, m in sorted(mi.items(), key=lambda kv: -kv[1]):
            dominates = m >= max(floor, 4.0 * median) and m > 1.5 * _second_highest(mi, col)
            if dominates:
                yield Finding(
                    rule_id=self.id,
                    category=self.category,
                    severity=self.severity,
                    layer=Layer.DATA,
                    message=(
                        f"Feature `{col}` has mutual information {m:.3f} with the target, far "
                        f"above the rest of the cohort (median {median:.3f}); check for a leaked "
                        f"proxy."
                    ),
                    location=_loc(f"high MI: {col} (MI={m:.3f})"),
                    confidence=0.5,
                    references=self.references,
                    fix=Fix(summary=f"Audit `{col}` for target leakage.", autofixable=False),
                    evidence={"feature": col, "mi": round(m, 4), "cohort_median": round(median, 4)},
                )


@register
class TM001(DataRule):
    id = "TM001"
    name = "train/test temporal overlap"
    category = Category.TEMPORAL
    severity = Severity.HIGH
    layers = (Layer.DATA,)
    references = (KAPOOR_NARAYANAN_2023,)
    rationale = (
        "If the latest training timestamp is at or after the earliest test timestamp, the model "
        "trains on data from the test period — a temporal leak."
    )

    def check(self, ctx: DataContext) -> Iterable[Finding]:
        from .temporal import time_ranges

        ai = ctx.audit_input
        if not ai.time or ai.time not in ai.train.columns:
            return
        tr_min, tr_max = time_ranges(ai.train, ai.time)
        te_min, te_max = time_ranges(ai.test, ai.time)
        if tr_max is None or te_min is None:
            return
        if tr_max >= te_min:
            yield Finding(
                rule_id=self.id,
                category=self.category,
                severity=self.severity,
                layer=Layer.DATA,
                message=(
                    f"Training data extends to {tr_max} but test starts at {te_min}; the splits "
                    f"overlap in time. Split chronologically."
                ),
                location=_loc(f"temporal overlap on '{ai.time}'"),
                confidence=0.95,
                references=self.references,
                fix=Fix(summary="Split by time so train precedes test.", autofixable=False),
                evidence={
                    "train_max": str(tr_max),
                    "test_min": str(te_min),
                    "train_min": str(tr_min),
                    "test_max": str(te_max),
                },
            )


@register
class M001(DataRule):
    id = "M001"
    name = "accuracy on imbalanced target"
    category = Category.METRIC
    severity = Severity.MEDIUM
    layers = (Layer.DATA,)
    references = (SKLEARN_PITFALLS,)
    rationale = (
        "On a heavily imbalanced target, accuracy is dominated by the majority class and can look "
        "high while the model is useless. Prefer balanced accuracy / F1 / AUC / PR-AUC."
    )

    def check(self, ctx: DataContext) -> Iterable[Finding]:
        ai = ctx.audit_input
        if not ai.target or ai.target not in ai.train.columns:
            return
        y = ai.train[ai.target]
        counts = y.value_counts(normalize=True, dropna=True)
        if counts.empty:
            return
        majority = float(counts.iloc[0])
        if majority >= ctx.config.data.imbalance_ratio and len(counts) <= 50:
            yield Finding(
                rule_id=self.id,
                category=self.category,
                severity=self.severity,
                layer=Layer.DATA,
                message=(
                    f"Target is imbalanced (majority class {majority:.1%}); accuracy will be "
                    f"misleading. Report balanced accuracy / F1 / AUC instead."
                ),
                location=_loc(f"class imbalance ({majority:.1%} majority)"),
                confidence=0.8,
                references=self.references,
                fix=Fix(summary="Use balanced_accuracy_score / f1_score / roc_auc_score.", autofixable=False),
                evidence={"majority_fraction": majority, "n_classes": int(len(counts))},
            )
