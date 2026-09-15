from __future__ import annotations

from pathlib import Path

from ml_leakproof.core.config import Config
from ml_leakproof.static.engine import StaticEngine


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


def test_group_and_temporal_hints_do_not_bleed_into_unrelated_splits(tmp_path):
    src = """
from sklearn.model_selection import KFold, train_test_split
import pandas as pd

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
groups = [0, 1, 0, 1]
kfold = KFold(n_splits=2, shuffle=True, random_state=0)
df = pd.read_csv("events.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
"""
    ids = _ids(src, tmp_path)
    assert "C002" not in ids
    assert "S002" not in ids


def test_group_and_temporal_signals_follow_the_split_input(tmp_path):
    grouped = """
import pandas as pd
from sklearn.model_selection import KFold

df = pd.read_csv("patients.csv")
patient_id = df["patient_id"]
kfold = KFold(n_splits=2, shuffle=True, random_state=0)
for train_idx, test_idx in kfold.split(df):
    pass
"""
    assert "C002" in _ids(grouped, tmp_path)

    temporal = """
import pandas as pd
from sklearn.model_selection import KFold

df = pd.read_csv("events.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
kfold = KFold(n_splits=2, shuffle=True, random_state=0)
for train_idx, test_idx in kfold.split(df):
    pass
"""
    assert "C003" in _ids(temporal, tmp_path)


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


def test_m004_uses_metric_call_scope_and_call_time_bindings(tmp_path):
    shadowed_parameter = """
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

y_train, y_test = train_test_split(y, random_state=0)
y = y_train

def report(y, prediction):
    return accuracy_score(y, prediction)
"""
    assert "M004" not in _ids(shadowed_parameter, tmp_path)

    reassigned_after_metric = """
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

y_train, y_test = train_test_split(y, random_state=0)
prediction = model.predict(X_train)
accuracy_score(y_train, prediction)
y_train = y_test
"""
    assert "M004" in _ids(reassigned_after_metric, tmp_path)


def test_m002_does_not_link_reassigned_validation_curve_to_test(tmp_path):
    src = """
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, test_scores)
fpr, tpr, thresholds = roc_curve(y_val, validation_scores)
best = max(range(len(tpr)), key=lambda index: tpr[index] - fpr[index])
chosen_threshold = thresholds[best]
"""
    assert "M002" not in _ids(src, tmp_path)


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


def test_disconnected_fit_demo_not_reported_as_cv_leakage(tmp_path):
    src = """
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

vect = CountVectorizer()
vect.fit(["one two", "three four"])
vect.transform(["one two"])
cross_val_score(LogisticRegression(), X, y)
"""
    assert "C001" not in _ids(src, tmp_path)


def test_inline_fit_output_is_traced_into_cv(tmp_path):
    src = """
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

vect = CountVectorizer().fit(text_train)
X_train = vect.transform(text_train)
cross_val_score(LogisticRegression(), X_train, y_train)
"""
    assert "C001" in _ids(src, tmp_path)


def test_inline_fit_transform_argument_is_traced_into_cv(tmp_path):
    src = """
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

cross_val_score(
    LogisticRegression(), StandardScaler().fit_transform(X), y
)
"""
    assert "C001" in _ids(src, tmp_path)


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


def test_cv_after_outer_holdout_owns_preprocessing_finding(tmp_path):
    src = """
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
cross_val_score(LogisticRegression(random_state=0), X_scaled, y_train)
"""
    ids = _ids(src, tmp_path)
    assert "C001" in ids
    assert "P001" not in ids


def test_unrelated_cv_does_not_relabel_holdout_preprocessing(tmp_path):
    src = """
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)
cross_val_score(model, X_train, y_train)
"""
    ids = _ids(src, tmp_path)
    assert "C001" not in ids
    assert "P001" in ids


def test_unassigned_fit_transform_before_split_is_ignored(tmp_path):
    src = """
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
StandardScaler().fit_transform(X)
train_test_split(X, y, random_state=0)
"""
    assert "P001" not in _ids(src, tmp_path)


def test_pipeline_fit_on_test_data_is_checked(tmp_path):
    src = """
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
pipe = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())])
pipe.fit(X_test, y_test)
"""
    assert "S001" in _ids(src, tmp_path)


def test_exploratory_pipeline_fit_before_later_split_is_not_reported(tmp_path):
    src = """
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

cancer = load_breast_cancer()
pipe = make_pipeline(StandardScaler(), PCA(n_components=2))
pipe.fit(cancer.data)
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0
)
"""
    assert "S001" not in _ids(src, tmp_path)


def test_static_scope_and_reassignment_boundaries_do_not_cross_link(tmp_path):
    separate = """
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def fit_scaler(X):
    scaler = StandardScaler()
    scaler.fit(X)
def split_data(X, y):
    return train_test_split(X, y, random_state=0)
"""
    assert "S001" not in _ids(separate, tmp_path)

    reassigned = """
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X = load_training_data()
scaler = StandardScaler()
scaler.fit(X)
X = load_different_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
"""
    assert "S001" not in _ids(reassigned, tmp_path)


def test_static_transform_preserves_input_split_taint(tmp_path):
    src = """
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = StandardScaler()
scaled = scaler.transform(X_test)
other = MinMaxScaler()
other.fit(scaled)
"""
    assert "S001" in _ids(src, tmp_path)


def test_holdout_evaluation_does_not_require_nested_cv(tmp_path):
    src = """
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
grid = GridSearchCV(SVC(), {"C": [1, 10]}, cv=5)
grid.fit(X_train, y_train)
print(grid.best_score_)
print(grid.score(X_test, y_test))
"""
    assert "C005" not in _ids(src, tmp_path)


def test_threshold_curve_requires_test_derived_selection(tmp_path):
    plot_only = """
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, scores)
plot(precision, recall)
"""
    assert "M002" not in _ids(plot_only, tmp_path)

    selected = """
import numpy as np
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, scores)
best_index = np.argmax(precision[:-1])
best_threshold = thresholds[best_index]
"""
    assert "M002" in _ids(selected, tmp_path)


def test_fixed_kmeans_initialization_needs_no_seed_warning(tmp_path):
    fixed = """
from sklearn.cluster import KMeans

init = X[:3, :]
KMeans(n_clusters=3, init=init, n_init=1).fit(X)
"""
    assert "R001" not in _ids(fixed, tmp_path)

    random = """
from sklearn.cluster import KMeans

KMeans(n_clusters=3, n_init=1).fit(X)
"""
    assert "R001" in _ids(random, tmp_path)
