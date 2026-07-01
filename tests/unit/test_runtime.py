from __future__ import annotations

import numpy as np

from leakproof.core.config import Config
from leakproof.runtime.engine import watch


def test_runtime_fit_on_full_detected():
    with watch(Config()) as session:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(0)
        X = rng.rand(60, 4)
        y = rng.randint(0, 2, 60)
        StandardScaler().fit_transform(X)  # fit before split
        train_test_split(X, y, random_state=0)
    assert "P001" in {f.rule_id for f in session.findings}


def test_runtime_clean_no_findings():
    with watch(Config()) as session:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(0)
        X = rng.rand(60, 4)
        y = rng.randint(0, 2, 60)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        StandardScaler().fit_transform(X_train)
    assert "P001" not in {f.rule_id for f in session.findings}


def test_runtime_repeated_test_scoring():
    with watch(Config()) as session:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        rng = np.random.RandomState(0)
        X = rng.rand(80, 3)
        y = rng.randint(0, 2, 80)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        m = LogisticRegression(random_state=0).fit(X_train, y_train)
        m.score(X_test, y_test)
        m.score(X_test, y_test)
    assert "T001" in {f.rule_id for f in session.findings}


def test_hooks_removed_after_context():
    import sklearn.model_selection as ms

    before = ms.train_test_split
    with watch(Config()):
        pass
    assert ms.train_test_split is before  # patches cleanly removed
