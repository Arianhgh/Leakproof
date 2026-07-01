from __future__ import annotations

from pathlib import Path

from leakproof.core.config import Config
from leakproof.static.engine import StaticEngine


def _ids(src: str, tmp_path: Path) -> set[str]:
    path = tmp_path / "case.py"
    path.write_text(src, encoding="utf-8")
    return {f.rule_id for f in StaticEngine(Config()).run_file(path)}


def test_s002_ignores_cross_validated_plot_label(tmp_path):
    src = """
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
plt.ylabel("Cross-Validated Accuracy")
"""
    assert "S002" not in _ids(src, tmp_path)


def test_s002_still_flags_real_temporal_shuffle(tmp_path):
    src = """
import pandas as pd
from sklearn.model_selection import train_test_split
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=4)
"""
    assert "S002" in _ids(src, tmp_path)


def test_r001_kfold_and_logistic_regression_context(tmp_path):
    clean = """
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
KFold(n_splits=5, shuffle=False)
LogisticRegression(solver="lbfgs")
"""
    assert "R001" not in _ids(clean, tmp_path)

    leaky = """
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
KFold(n_splits=5, shuffle=True)
LogisticRegression(solver="liblinear")
"""
    assert "R001" in _ids(leaky, tmp_path)


def test_m003_only_ties_mean_to_cv_scores(tmp_path):
    clean = """
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
y_test.mean()
scores = cross_val_score(LogisticRegression(random_state=0), X, y)
print(scores.std())
"""
    assert "M003" not in _ids(clean, tmp_path)

    leaky = """
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
scores = cross_val_score(LogisticRegression(random_state=0), X, y)
print(scores.mean())
"""
    assert "M003" in _ids(leaky, tmp_path)


def test_demo_fit_transform_not_reported_as_cv_leakage(tmp_path):
    src = """
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
ohe.fit_transform(df[["Sex"]])
column_trans = make_column_transformer((OneHotEncoder(), ["Sex"]), remainder="passthrough")
pipe = make_pipeline(column_trans, LogisticRegression(solver="lbfgs"))
cross_val_score(pipe, X, y, cv=5, scoring="accuracy").mean()
"""
    assert "C001" not in _ids(src, tmp_path)


def test_fit_transform_used_for_evaluation_still_flags(tmp_path):
    src = """
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
cross_val_score(LogisticRegression(solver="lbfgs"), X_scaled, y)
train_test_split(X_scaled, y, random_state=0)
"""
    ids = _ids(src, tmp_path)
    assert "C001" in ids or "P001" in ids


def test_unassigned_fit_transform_before_split_is_ignored(tmp_path):
    src = """
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
StandardScaler().fit_transform(X)
train_test_split(X, y, random_state=0)
"""
    assert "P001" not in _ids(src, tmp_path)
