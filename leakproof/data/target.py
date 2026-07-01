"""Target-leakage probes: univariate predictivity + mutual information."""

from __future__ import annotations


def _is_classification(y) -> bool:
    import pandas as pd

    s = pd.Series(y)
    if s.dtype == object or str(s.dtype).startswith("category"):
        return True
    nun = s.nunique(dropna=True)
    return nun <= max(2, int(0.05 * len(s))) and nun <= 20


def univariate_scores(df, target: str, feature_cols: list[str]) -> dict[str, float]:
    """Cross-validated single-feature predictive score per feature (0..1)."""
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import OrdinalEncoder

    y = df[target]
    clf = _is_classification(y)
    scores: dict[str, float] = {}
    n = len(df)
    cv = min(5, max(2, n // 20)) if n >= 10 else 2

    for col in feature_cols:
        x = df[[col]]
        try:
            if x[col].dtype == object or str(x[col].dtype).startswith("category"):
                # encoding a single probe column; not a model-eval pipeline
                x = pd.DataFrame(
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(  # leakproof: ignore[C001]
                        x.astype(str)
                    )
                )
            else:
                x = x.fillna(x.mean(numeric_only=True))
            x = x.to_numpy(dtype=float)
            if np.unique(x).size <= 1:
                scores[col] = 0.0
                continue
            if clf:
                if pd.Series(y).nunique() < 2:
                    scores[col] = 0.0
                    continue
                yy = OrdinalEncoder().fit_transform(pd.DataFrame(y.astype(str))).ravel()  # leakproof: ignore[C001]
                est = LogisticRegression(max_iter=200, random_state=0)
                sc = cross_val_score(est, x, yy, cv=cv, scoring="roc_auc" if pd.Series(yy).nunique() == 2 else "accuracy")
                scores[col] = float(np.mean(sc))
            else:
                est = LinearRegression()
                sc = cross_val_score(est, x, np.asarray(y, dtype=float), cv=cv, scoring="r2")
                scores[col] = float(max(0.0, np.mean(sc)))
        except Exception:
            scores[col] = 0.0
    return scores


def mutual_info(df, target: str, feature_cols: list[str]) -> dict[str, float]:
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.preprocessing import OrdinalEncoder

    y = df[target]
    clf = _is_classification(y)
    X = df[feature_cols].copy()
    discrete = []
    for _i, col in enumerate(feature_cols):
        if X[col].dtype == object or str(X[col].dtype).startswith("category"):
            X[col] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(  # leakproof: ignore[C001]
                X[[col]].astype(str)
            )
            discrete.append(True)
        else:
            X[col] = X[col].fillna(X[col].mean())
            discrete.append(False)
    try:
        Xv = X.to_numpy(dtype=float)
        if clf:
            yy = OrdinalEncoder().fit_transform(pd.DataFrame(y.astype(str))).ravel()  # leakproof: ignore[C001]
            mi = mutual_info_classif(Xv, yy, discrete_features=discrete, random_state=0)
        else:
            mi = mutual_info_regression(
                Xv, np.asarray(y, dtype=float), discrete_features=discrete, random_state=0
            )
        return {c: float(m) for c, m in zip(feature_cols, mi)}
    except Exception:
        return {c: 0.0 for c in feature_cols}


def zscore(values: dict[str, float]) -> dict[str, float]:
    import numpy as np

    arr = np.array(list(values.values()), dtype=float)
    if arr.size <= 1 or arr.std() == 0:
        return {k: 0.0 for k in values}
    mean, std = arr.mean(), arr.std()
    return {k: float((v - mean) / std) for k, v in values.items()}
