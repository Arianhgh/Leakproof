"""Generator for the paired leaky/clean fixture suite.

Run with ``python tests/fixtures/_gen.py`` to (re)write all fixtures. Keeping
the definitions in one place makes it easy to audit that every catalog rule has
both a leaky and a clean example, and that clean examples avoid *all* smells
(not just the rule under test) so the clean-set false-positive metric is honest.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
LEAKY = HERE / "leaky"
CLEAN = HERE / "clean"

# Each entry: rule_id -> (slug, leaky_source, clean_source)
STATIC: dict[str, tuple[str, str, str]] = {}


def add(rule_id, slug, leaky, clean):
    STATIC[rule_id] = (slug, leaky.strip() + "\n", clean.strip() + "\n")


# --- P001 ---------------------------------------------------------------
add(
    "P001", "scaler_fit_transform",
    """
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)
""",
    """
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
""",
)

# --- P002 ---------------------------------------------------------------
add(
    "P002", "fit_on_concat",
    """
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("d.csv")
y = df.pop("target")
train, test, y_train, y_test = train_test_split(df, y, random_state=0)
full = pd.concat([train, test])
scaler = StandardScaler()
scaler.fit(full)
""",
    """
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("d.csv")
y = df.pop("target")
train, test, y_train, y_test = train_test_split(df, y, random_state=0)
scaler = StandardScaler()
scaler.fit(train)
""",
)

# --- P003 ---------------------------------------------------------------
add(
    "P003", "imputer_full",
    """
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
imp = SimpleImputer(strategy="mean")
X_imp = imp.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_imp, y, random_state=0)
""",
    """
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
imp = SimpleImputer(strategy="mean")
X_train_i = imp.fit_transform(X_train)
X_test_i = imp.transform(X_test)
""",
)

# --- S001 ---------------------------------------------------------------
add(
    "S001", "scaler_fit_full",
    """
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
scaler = StandardScaler()
scaler.fit(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
""",
    """
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = StandardScaler()
scaler.fit(X_train)
""",
)

# --- S003 ---------------------------------------------------------------
add(
    "S003", "smote_before_split",
    """
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
sm = SMOTE(random_state=0)
X_res, y_res = sm.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, random_state=0)
""",
    """
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
sm = SMOTE(random_state=0)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
""",
)

# --- S004 ---------------------------------------------------------------
add(
    "S004", "selectkbest_full",
    """
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
sel = SelectKBest(k=5)
X_sel = sel.fit_transform(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_sel, y, random_state=0)
""",
    """
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
sel = SelectKBest(k=5)
X_train_s = sel.fit_transform(X_train, y_train)
X_test_s = sel.transform(X_test)
""",
)

# --- S002 ---------------------------------------------------------------
add(
    "S002", "shuffle_on_temporal",
    """
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("d.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
y = df.pop("target")
X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=0)
""",
    """
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("d.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
y = df.pop("target")
X_train, X_test, y_train, y_test = train_test_split(
    df, y, shuffle=False, random_state=0
)
""",
)

# --- C001 ---------------------------------------------------------------
add(
    "C001", "fit_outside_cv",
    """
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X = pd.read_csv("d.csv")
y = X.pop("target")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
scores = cross_val_score(LogisticRegression(), X_scaled, y)
""",
    """
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

X = pd.read_csv("d.csv")
y = X.pop("target")
pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
scores = cross_val_score(pipe, X, y)
""",
)

# --- C002 ---------------------------------------------------------------
add(
    "C002", "kfold_grouped",
    """
import pandas as pd
from sklearn.model_selection import KFold

df = pd.read_csv("d.csv")
patient_id = df["patient_id"]
kf = KFold(n_splits=5, random_state=0, shuffle=True)
for train_idx, test_idx in kf.split(df):
    pass
""",
    """
import pandas as pd
from sklearn.model_selection import GroupKFold

df = pd.read_csv("d.csv")
groups = df["patient_id"]
gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(df, groups=groups):
    pass
""",
)

# --- C003 ---------------------------------------------------------------
add(
    "C003", "kfold_timeseries",
    """
import pandas as pd
from sklearn.model_selection import KFold

df = pd.read_csv("d.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
kf = KFold(n_splits=5, shuffle=True, random_state=0)
for train_idx, test_idx in kf.split(df):
    pass
""",
    """
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv("d.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
tss = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tss.split(df):
    pass
""",
)

# --- C004 ---------------------------------------------------------------
add(
    "C004", "tune_on_test",
    """
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
grid = GridSearchCV(SVC(), {"C": [1, 10]})
grid.fit(X_test, y_test)
""",
    """
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
grid = GridSearchCV(SVC(random_state=0), {"C": [1, 10]})
grid.fit(X_train, y_train)
""",
)

# --- C005 ---------------------------------------------------------------
add(
    "C005", "no_nested_cv",
    """
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

X = pd.read_csv("d.csv")
y = X.pop("target")
grid = GridSearchCV(SVC(), {"C": [1, 10]})
grid.fit(X, y)
print(grid.best_score_)
""",
    """
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score

X = pd.read_csv("d.csv")
y = X.pop("target")
grid = GridSearchCV(SVC(random_state=0), {"C": [1, 10]})
nested = cross_val_score(grid, X, y)
print(nested.mean(), nested.std())
""",
)

# --- C006 ---------------------------------------------------------------
add(
    "C006", "target_mean_encoding",
    """
import pandas as pd

df = pd.read_csv("d.csv")
df["cat_enc"] = df.groupby("cat")["target"].transform("mean")
""",
    """
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("d.csv")
y = df.pop("target")
pipe = make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(random_state=0))
""",
)

# --- TM002 --------------------------------------------------------------
add(
    "TM002", "negative_shift",
    """
import pandas as pd

df = pd.read_csv("d.csv")
df["future_feature"] = df["value"].shift(-1)
""",
    """
import pandas as pd

df = pd.read_csv("d.csv")
df["lag_target"] = df["target"].shift(1)
""",
)

# --- T002 ---------------------------------------------------------------
add(
    "T002", "early_stopping_on_test",
    """
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = XGBClassifier(early_stopping_rounds=10, random_state=0)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
""",
    """
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
X_tr, X_test, y_tr, y_test = train_test_split(X, y, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, random_state=0)
model = XGBClassifier(early_stopping_rounds=10, random_state=0)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
""",
)

# --- T003 ---------------------------------------------------------------
add(
    "T003", "best_of_n_seeds",
    """
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X = pd.read_csv("d.csv")
y = X.pop("target")
best = 0.0
for seed in range(20):
    score = cross_val_score(LogisticRegression(random_state=seed), X, y).mean()
    best = max(best, score)
print(best)
""",
    """
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X = pd.read_csv("d.csv")
y = X.pop("target")
scores = []
for seed in range(20):
    scores.append(cross_val_score(LogisticRegression(random_state=seed), X, y).mean())
print(np.mean(scores), np.std(scores))
""",
)

# --- M002 ---------------------------------------------------------------
add(
    "M002", "threshold_on_test",
    """
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
m = LogisticRegression(random_state=0).fit(X_train, y_train)
probs = m.predict_proba(X_test)[:, 1]
prec, rec, thr = precision_recall_curve(y_test, probs)
""",
    """
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
X_tr, X_test, y_tr, y_test = train_test_split(X, y, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, random_state=0)
m = LogisticRegression(random_state=0).fit(X_train, y_train)
probs = m.predict_proba(X_val)[:, 1]
prec, rec, thr = precision_recall_curve(y_val, probs)
""",
)

# --- M003 ---------------------------------------------------------------
add(
    "M003", "no_variance",
    """
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X = pd.read_csv("d.csv")
y = X.pop("target")
scores = cross_val_score(LogisticRegression(random_state=0), X, y)
print(scores.mean())
""",
    """
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X = pd.read_csv("d.csv")
y = X.pop("target")
scores = cross_val_score(LogisticRegression(random_state=0), X, y)
print(scores.mean(), scores.std())
""",
)

# --- M004 ---------------------------------------------------------------
add(
    "M004", "metric_on_train",
    """
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
m = LogisticRegression(random_state=0).fit(X_train, y_train)
print(accuracy_score(y_train, m.predict(X_train)))
""",
    """
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
m = LogisticRegression(random_state=0).fit(X_train, y_train)
print(accuracy_score(y_test, m.predict(X_test)))
""",
)

# --- R001 ---------------------------------------------------------------
add(
    "R001", "missing_seed",
    """
import pandas as pd
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
X_train, X_test, y_train, y_test = train_test_split(X, y)
""",
    """
import pandas as pd
from sklearn.model_selection import train_test_split

X = pd.read_csv("d.csv")
y = X.pop("target")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
""",
)

# --- R002 ---------------------------------------------------------------
add(
    "R002", "torch_nondeterministic",
    """
import torch

x = torch.randn(10, 3)
model = torch.nn.Linear(3, 1)
out = model(x)
""",
    """
import torch

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
x = torch.randn(10, 3)
model = torch.nn.Linear(3, 1)
out = model(x)
""",
)


# ======================================================================
# Data-layer fixtures (define make_input()).
# ======================================================================
DATA: dict[str, tuple[str, str, str]] = {}


def add_data(rule_id, slug, leaky, clean):
    DATA[rule_id] = (slug, leaky.strip() + "\n", clean.strip() + "\n")


add_data(
    "D001", "exact_overlap",
    """
import pandas as pd
from ml_leakproof.data.input import DataAuditInput

def make_input():
    rows = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [5, 6, 7, 8, 9, 10], "target": [0, 1, 0, 1, 0, 1]})
    train = rows.iloc[[0, 1, 2, 3, 4]].reset_index(drop=True)
    test = rows.iloc[[4, 5]].reset_index(drop=True)  # row 4 overlaps
    return DataAuditInput(train=train, test=test, target="target")
""",
    """
import pandas as pd
from ml_leakproof.data.input import DataAuditInput

def make_input():
    train = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 6, 7, 8, 9], "target": [0, 1, 0, 1, 0]})
    test = pd.DataFrame({"a": [6, 7], "b": [10, 11], "target": [1, 0]})
    return DataAuditInput(train=train, test=test, target="target")
""",
)

add_data(
    "D002", "near_duplicate",
    """
import pandas as pd
import numpy as np
from ml_leakproof.data.input import DataAuditInput

def make_input():
    rng = np.random.RandomState(0)
    base = rng.rand(30, 4)
    train = pd.DataFrame(base, columns=list("abcd"))
    train["target"] = (base[:, 0] > 0.5).astype(int)
    test_base = base[:10] + 1e-6  # near-identical to train rows
    test = pd.DataFrame(test_base, columns=list("abcd"))
    test["target"] = (test_base[:, 0] > 0.5).astype(int)
    return DataAuditInput(train=train, test=test, target="target")
""",
    """
import pandas as pd
import numpy as np
from ml_leakproof.data.input import DataAuditInput

def make_input():
    rng = np.random.RandomState(0)
    train = pd.DataFrame(rng.rand(30, 4), columns=list("abcd"))
    train["target"] = rng.randint(0, 2, 30)
    test = pd.DataFrame(rng.rand(10, 4) + 10.0, columns=list("abcd"))
    test["target"] = rng.randint(0, 2, 10)
    return DataAuditInput(train=train, test=test, target="target")
""",
)

add_data(
    "D003", "group_overlap",
    """
import pandas as pd
from ml_leakproof.data.input import DataAuditInput

def make_input():
    train = pd.DataFrame({"x": [0.5, 0.1, 0.4, 0.2, 0.3, 0.6], "pid": ["p1", "p1", "p2", "p2", "p3", "p3"], "target": [0, 1, 0, 1, 0, 1]})
    test = pd.DataFrame({"x": [0.7, 0.8], "pid": ["p3", "p4"], "target": [0, 1]})  # p3 overlaps
    return DataAuditInput(train=train, test=test, target="target", group="pid")
""",
    """
import pandas as pd
from ml_leakproof.data.input import DataAuditInput

def make_input():
    train = pd.DataFrame({"x": [0.5, 0.1, 0.4, 0.2, 0.3, 0.6], "pid": ["p1", "p1", "p2", "p2", "p3", "p3"], "target": [0, 1, 0, 1, 0, 1]})
    test = pd.DataFrame({"x": [0.7, 0.8], "pid": ["p4", "p5"], "target": [0, 1]})
    return DataAuditInput(train=train, test=test, target="target", group="pid")
""",
)

add_data(
    "D005", "feature_equals_target",
    """
import pandas as pd
import numpy as np
from ml_leakproof.data.input import DataAuditInput

def make_input():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 80)
    train = pd.DataFrame({
        "noise": rng.rand(80),
        "leak": y,  # exact copy of the target
        "target": y,
    })
    test = train.iloc[:20].copy()
    return DataAuditInput(train=train, test=test, target="target")
""",
    """
import pandas as pd
import numpy as np
from ml_leakproof.data.input import DataAuditInput

def make_input():
    rng = np.random.RandomState(0)
    train = pd.DataFrame({
        "f1": rng.rand(80),
        "f2": rng.rand(80),
        "target": rng.randint(0, 2, 80),
    })
    # test rows are well separated from train (offset by +100) so there is no
    # incidental exact/near-duplicate overlap to muddy this clean example.
    test = pd.DataFrame({
        "f1": rng.rand(20) + 100.0,
        "f2": rng.rand(20) + 100.0,
        "target": rng.randint(0, 2, 20),
    })
    return DataAuditInput(train=train, test=test, target="target")
""",
)

add_data(
    "D006", "high_mi_feature",
    """
import pandas as pd
import numpy as np
from ml_leakproof.data.input import DataAuditInput

def make_input():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 200)
    train = pd.DataFrame({
        "f1": rng.rand(200),
        "f2": rng.rand(200),
        "f3": rng.rand(200),
        "proxy": y + rng.normal(0, 0.01, 200),  # near-deterministic proxy
        "target": y,
    })
    test = train.iloc[:40].copy()
    return DataAuditInput(train=train, test=test, target="target")
""",
    """
import pandas as pd
import numpy as np
from ml_leakproof.data.input import DataAuditInput

def make_input():
    rng = np.random.RandomState(0)
    train = pd.DataFrame({
        "f1": rng.rand(200),
        "f2": rng.rand(200),
        "f3": rng.rand(200),
        "f4": rng.rand(200),
        "target": rng.randint(0, 2, 200),
    })
    test = pd.DataFrame({
        "f1": rng.rand(40) + 100.0,
        "f2": rng.rand(40) + 100.0,
        "f3": rng.rand(40) + 100.0,
        "f4": rng.rand(40) + 100.0,
        "target": rng.randint(0, 2, 40),
    })
    return DataAuditInput(train=train, test=test, target="target")
""",
)

add_data(
    "D007", "test_duplicates",
    """
import pandas as pd
from ml_leakproof.data.input import DataAuditInput

def make_input():
    train = pd.DataFrame({"a": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    test = pd.DataFrame({"a": [5, 5, 5, 6, 7], "target": [1, 1, 1, 0, 1]})  # dup rows
    return DataAuditInput(train=train, test=test, target="target")
""",
    """
import pandas as pd
from ml_leakproof.data.input import DataAuditInput

def make_input():
    train = pd.DataFrame({"a": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    test = pd.DataFrame({"a": [5, 6, 7, 8, 9], "target": [1, 0, 1, 0, 1]})
    return DataAuditInput(train=train, test=test, target="target")
""",
)

add_data(
    "TM001", "temporal_overlap",
    """
import pandas as pd
from ml_leakproof.data.input import DataAuditInput

def make_input():
    train = pd.DataFrame({"x": [0.5, 0.1, 0.4, 0.2, 0.3, 0.6, 0.9, 0.8, 0.7, 0.05], "ts": pd.date_range("2020-01-01", periods=10), "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})
    test = pd.DataFrame({"x": [100.0, 101.0], "ts": pd.to_datetime(["2020-01-05", "2021-06-01"]), "target": [1, 0]})
    return DataAuditInput(train=train, test=test, target="target", time="ts")
""",
    """
import pandas as pd
from ml_leakproof.data.input import DataAuditInput

def make_input():
    train = pd.DataFrame({"x": [0.5, 0.1, 0.4, 0.2, 0.3, 0.6, 0.9, 0.8, 0.7, 0.05], "ts": pd.date_range("2020-01-01", periods=10), "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})
    test = pd.DataFrame({"x": [100.0, 101.0], "ts": pd.to_datetime(["2021-01-01", "2021-02-01"]), "target": [1, 0]})
    return DataAuditInput(train=train, test=test, target="target", time="ts")
""",
)

add_data(
    "M001", "imbalanced_accuracy",
    """
import pandas as pd
from ml_leakproof.data.input import DataAuditInput

def make_input():
    train = pd.DataFrame({"f": list(range(100)), "target": [0] * 97 + [1] * 3})
    test = pd.DataFrame({"f": list(range(20)), "target": [0] * 19 + [1]})
    return DataAuditInput(train=train, test=test, target="target")
""",
    """
import pandas as pd
from ml_leakproof.data.input import DataAuditInput

def make_input():
    import numpy as np

    rng = np.random.RandomState(0)
    # balanced target, uncorrelated with the feature (no leakage to find)
    train = pd.DataFrame({"f": rng.rand(100), "target": rng.randint(0, 2, 100)})
    test = pd.DataFrame({"f": rng.rand(20) + 100.0, "target": rng.randint(0, 2, 20)})
    return DataAuditInput(train=train, test=test, target="target")
""",
)


# ======================================================================
# Runtime-layer fixtures (define run()).
# ======================================================================
RUNTIME: dict[str, tuple[str, str, str]] = {}


def add_runtime(rule_id, slug, leaky, clean):
    RUNTIME[rule_id] = (slug, leaky.strip() + "\n", clean.strip() + "\n")


add_runtime(
    "T001", "score_test_twice",
    """
import numpy as np

def run():
    # imports happen inside run() so they resolve to the instrumented functions,
    # mirroring how `ml_leakproof run script.py` executes user code under the hooks.
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    rng = np.random.RandomState(0)
    X = rng.rand(80, 3)
    y = rng.randint(0, 2, 80)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    m = LogisticRegression(random_state=0).fit(X_train, y_train)
    m.score(X_test, y_test)
    m.score(X_test, y_test)  # scored twice -> T001
""",
    """
import numpy as np

def run():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    rng = np.random.RandomState(0)
    X = rng.rand(80, 3)
    y = rng.randint(0, 2, 80)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    m = LogisticRegression(random_state=0).fit(X_train, y_train)
    m.score(X_test, y_test)  # scored once
""",
)

add_runtime(
    "P001", "runtime_fit_on_full",
    """
import numpy as np

def run():
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    rng = np.random.RandomState(0)
    X = rng.rand(60, 4)
    y = rng.randint(0, 2, 60)
    scaler = StandardScaler()
    scaler.fit_transform(X)  # fit on full data before split -> sees the test rows
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
""",
    """
import numpy as np

def run():
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    rng = np.random.RandomState(0)
    X = rng.rand(60, 4)
    y = rng.randint(0, 2, 60)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    scaler = StandardScaler()
    scaler.fit_transform(X_train)  # fit only on the training split
""",
)


def write_all() -> None:
    LEAKY.mkdir(parents=True, exist_ok=True)
    CLEAN.mkdir(parents=True, exist_ok=True)
    for bucket in (STATIC, DATA, RUNTIME):
        for rule_id, (slug, leaky, clean) in bucket.items():
            (LEAKY / f"{rule_id}__{slug}.py").write_text(leaky, encoding="utf-8")
            (CLEAN / f"{rule_id}__{slug}.py").write_text(clean, encoding="utf-8")
    total = len(STATIC) + len(DATA) + len(RUNTIME)
    print(f"wrote {total} paired fixtures ({total * 2} files)")


if __name__ == "__main__":
    write_all()
