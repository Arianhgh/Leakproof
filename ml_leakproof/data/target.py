"""Bounded, leakage-aware target predictivity and MI probes."""

from __future__ import annotations

from typing import Any


class ProbeValues(dict[str, float]):
    """Dict-compatible scores with task, metric, and coverage metadata."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.metadata: dict[str, Any] = kwargs.pop("metadata", {})
        super().__init__(*args, **kwargs)


def _is_classification(y: Any) -> bool:
    import pandas as pd

    series = pd.Series(y)
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_object_dtype(series):
        return True
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True
    unique = series.nunique(dropna=True)
    return bool(unique <= max(2, int(0.05 * len(series))) and unique <= 20)


def _make_cv(
    y: Any, *, groups: Any = None, time: Any = None, folds: int = 5
) -> tuple[Any | None, str]:
    import pandas as pd
    from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

    n = len(y)
    if n < 4:
        return None, "fewer than four rows"
    folds = min(int(folds), n)
    if time is not None:
        if len(pd.Series(time).dropna()) != n:
            return None, "invalid timestamps"
        if folds < 3:
            return None, "fewer than two chronological folds"
        return TimeSeriesSplit(n_splits=max(2, folds - 1)), "chronological"
    if groups is not None:
        n_groups = pd.Series(groups).nunique(dropna=True)
        if n_groups < 2:
            return None, "fewer than two groups"
        return GroupKFold(n_splits=min(folds, int(n_groups))), "group"
    if _is_classification(y):
        counts = pd.Series(y).value_counts(dropna=True)
        if len(counts) < 2:
            return None, "classification target has fewer than two classes"
        min_count = int(counts.min())
        if min_count < 2:
            return None, "a class has fewer than two observations"
        folds = min(folds, min_count)
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=0), "stratified"
    if folds < 2:
        return None, "fewer than two folds"
    return KFold(n_splits=folds, shuffle=True, random_state=0), "kfold"


def _preprocessor(series: Any) -> Any:
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if pd.api.types.is_numeric_dtype(series.dtype):
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=1)
    return Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", encoder)])


def univariate_scores(
    df: Any,
    target: str,
    feature_cols: list[str],
    *,
    groups: Any = None,
    time: Any = None,
    folds: int = 5,
) -> ProbeValues:
    """Cross-validated one-feature scores with preprocessing fitted per fold."""

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder

    y = df[target]
    classification = _is_classification(y)
    cv, cv_kind = _make_cv(y, groups=groups, time=time, folds=folds)
    target_series = pd.Series(y)
    if classification:
        counts = target_series.value_counts(normalize=True, dropna=True)
        baseline = 0.5 if target_series.nunique(dropna=True) == 2 else float(counts.max()) if not counts.empty else None
        metric = "roc_auc" if target_series.nunique(dropna=True) == 2 else "accuracy"
    else:
        baseline = 0.0
        metric = "r2"
    result = ProbeValues(metadata={"task": "classification" if classification else "regression", "cv": cv_kind, "folds": getattr(cv, "n_splits", 0), "metric": "roc_auc" if classification and pd.Series(y).nunique() == 2 else ("accuracy" if classification else "r2"), "sample_size": len(df), "unavailable": {}})
    result.metadata.update(
        {
            "metric": metric,
            "baseline": baseline,
            "uncertainty": {},
        }
    )
    if cv is None:
        result.metadata["unavailable"] = {column: cv_kind for column in feature_cols}
        return result
    if classification:
        encoded_y = LabelEncoder().fit_transform(pd.Series(y).astype(str))
        scoring = "roc_auc" if len(np.unique(encoded_y)) == 2 else "accuracy"
    else:
        try:
            encoded_y = np.asarray(y, dtype=float)
        except Exception:
            result.metadata["unavailable"] = {column: "non-numeric regression target" for column in feature_cols}
            return result
        scoring = "r2"

    for column in feature_cols:
        x = df[[column]]
        estimator = LogisticRegression(max_iter=300, random_state=0) if classification else LinearRegression()
        from sklearn.pipeline import Pipeline

        model = Pipeline([("preprocess", _preprocessor(x[column])), ("model", estimator)])
        try:
            kwargs = {"groups": groups} if groups is not None else {}
            scores = cross_val_score(
                model,
                x,
                encoded_y,
                cv=cv,
                scoring=scoring,
                error_score=np.nan,
                **kwargs,
            )
            if not len(scores):
                result.metadata["unavailable"][column] = "probe returned no scores"
            elif not np.isfinite(scores).all():
                result.metadata["unavailable"][column] = (
                    "probe returned non-finite score(s)"
                )
            else:
                result[column] = float(np.mean(scores))
                result.metadata["uncertainty"][column] = {
                    "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
                    "sem": float(np.std(scores, ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else None,
                }
        except Exception as exc:
            result.metadata["unavailable"][column] = f"probe failed: {type(exc).__name__}"
    return result


def mutual_info(
    df: Any,
    target: str,
    feature_cols: list[str],
    *,
    random_state: int = 0,
) -> ProbeValues:
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.preprocessing import OrdinalEncoder

    target_series = pd.Series(df[target])
    classification = _is_classification(target_series)
    work = df[feature_cols].copy()
    discrete: list[bool] = []
    for column in feature_cols:
        series = work[column]
        if not pd.api.types.is_numeric_dtype(series.dtype):
            work[column] = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            ).fit_transform(series.astype("string").fillna("<NA>").to_frame())
            discrete.append(True)
        else:
            work[column] = series.replace([np.inf, -np.inf], np.nan).fillna(series.median())
            discrete.append(False)
    result = ProbeValues(
        metadata={
            "task": "classification" if classification else "regression",
            "metric": "mutual_information",
            "baseline": 0.0,
            "sample_size": len(df),
            "uncertainty": None,
            "unavailable": {},
        }
    )
    try:
        values = work.to_numpy(dtype=float)
        if classification:
            from sklearn.preprocessing import LabelEncoder

            encoded_y = LabelEncoder().fit_transform(target_series.astype(str))
            scores = mutual_info_classif(values, encoded_y, discrete_features=discrete, random_state=random_state)
        else:
            scores = mutual_info_regression(
                values,
                target_series.to_numpy(dtype=float),
                discrete_features=discrete,
                random_state=random_state,
            )
        for column, score in zip(feature_cols, scores):
            if np.isfinite(score):
                result[column] = float(score)
            else:
                result.metadata["unavailable"][column] = "MI returned a non-finite score"
    except Exception as exc:
        result.metadata["unavailable"] = {column: f"MI probe failed: {type(exc).__name__}" for column in feature_cols}
    return result


def zscore(values: dict[str, float]) -> dict[str, float]:
    import numpy as np

    arr = np.array(list(values.values()), dtype=float)
    if arr.size <= 1 or arr.std() == 0:
        return {key: 0.0 for key in values}
    mean, std = arr.mean(), arr.std()
    return {key: float((value - mean) / std) for key, value in values.items()}
