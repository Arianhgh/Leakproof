from __future__ import annotations

import ast

from leakproof.static.adapters import load_adapters
from leakproof.static.dataflow import DataFlow, Taint


def _flow(src):
    tree = ast.parse(src)
    return DataFlow(tree, load_adapters(), src.splitlines())


def test_split_outputs_tagged():
    src = "from sklearn.model_selection import train_test_split\n" \
          "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
    df = _flow(src)
    assert df.taint_of("X_train") is Taint.TRAIN
    assert df.taint_of("X_test") is Taint.TEST
    assert df.taint_of("X") is Taint.FULL


def test_concat_bridges_to_full():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "tr, te, ytr, yte = train_test_split(df, y)\n"
        "full = pd.concat([tr, te])\n"
    )
    df = _flow(src)
    assert df.taint_of("full") is Taint.FULL


def test_pipeline_membership_marks_safe():
    src = (
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.pipeline import Pipeline\n"
        "scaler = StandardScaler()\n"
        "pipe = Pipeline([('s', scaler)])\n"
    )
    df = _flow(src)
    assert "scaler" in df.scopes[0].pipelined_vars


def test_tuple_unpack_outputs_tracked():
    src = (
        "from imblearn.over_sampling import SMOTE\n"
        "from sklearn.model_selection import train_test_split\n"
        "X_res, y_res = SMOTE().fit_resample(X, y)\n"
        "a, b, c, d = train_test_split(X_res, y_res)\n"
    )
    df = _flow(src)
    # X_res becomes FULL because it feeds the split
    assert df.taint_of("X_res") is Taint.FULL


def test_fit_transform_output_usage_tracking():
    src = (
        "from sklearn.model_selection import cross_val_score\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "X_scaled = StandardScaler().fit_transform(X)\n"
        "cross_val_score(model, X_scaled, y)\n"
    )
    df = _flow(src)
    fit = df.fit_calls[0]
    assert df.fit_output_used_for_evaluation(fit)


def test_unassigned_fit_transform_marked_as_demo():
    src = (
        "from sklearn.preprocessing import OneHotEncoder\n"
        "ohe = OneHotEncoder()\n"
        "ohe.fit_transform(df[['Sex']])\n"
    )
    df = _flow(src)
    fit = df.fit_calls[0]
    assert df.disconnected_fit_transform_demo(fit)


def test_column_transformer_membership_marks_nested_transformer_safe():
    src = (
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.preprocessing import OneHotEncoder\n"
        "enc = OneHotEncoder()\n"
        "ct = ColumnTransformer([('enc', enc, ['city'])])\n"
    )
    df = _flow(src)
    assert "enc" in df.scopes[0].pipelined_vars
