from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml_leakproof.core.config import Config, ConfigError
from ml_leakproof.core.models import Layer
from ml_leakproof.runtime.engine import watch


def _round_trip(value: Any, serializer: str, path: Path) -> Any:
    if serializer == "pickle":
        import pickle

        with path.open("wb") as handle:
            pickle.dump(value, handle)
        with path.open("rb") as handle:
            return pickle.load(handle)
    if serializer == "joblib":
        import joblib

        joblib.dump(value, path)
        return joblib.load(path)
    raise AssertionError(f"unsupported serializer: {serializer}")


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
    assert session.result.complete
    assert not session.diagnostics


def test_runtime_duplicate_values_use_split_positions_not_content():
    with watch(Config()) as session:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        X = np.zeros((80, 4))
        y = np.zeros(80)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0
        )
        StandardScaler().fit(X_train)
    assert "S001" not in {finding.rule_id for finding in session.findings}
    assert session.result.complete
    assert not session.diagnostics


def test_runtime_tracks_mutated_and_combined_split_arrays():
    from sklearn.preprocessing import StandardScaler

    X = np.arange(240, dtype=float).reshape(120, 2)
    cases = {
        "replacement": "S001",
        "combination": "P002",
    }
    for operation, expected_rule in cases.items():
        with watch(Config()) as session:
            from sklearn.model_selection import train_test_split

            X_train, X_test = train_test_split(X, test_size=0.5, random_state=0)
            if operation == "replacement":
                prepared = X_train.copy()
                prepared[:] = X_test
            else:
                prepared = X_train + X_test
            StandardScaler().fit(prepared)
        assert expected_rule in {finding.rule_id for finding in session.findings}
        assert session.result.complete
        assert not session.diagnostics


def test_runtime_tracks_every_array_returned_by_split():
    with watch(Config()) as session:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        X = np.zeros((80, 4))
        X2 = np.arange(80 * 4, dtype=float).reshape(80, 4)
        X_train, X_test, X2_train, X2_test = train_test_split(
            X, X2, test_size=0.25, random_state=0
        )
        StandardScaler().fit(X2_test)
    assert "S001" in {finding.rule_id for finding in session.findings}
    assert session.result.complete
    assert not session.diagnostics


def test_runtime_clean_concat_defaults_are_complete():
    with watch(Config()) as session:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(1)
        X = rng.rand(60, 4)
        y = rng.randint(0, 2, 60)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        StandardScaler().fit(X_train)
        np.concatenate([X_train, X_test])
        np.hstack([X_train[:, :2], X_train[:, 2:]])
        np.vstack([X_train, X_test])
    assert session.result.complete
    assert not session.diagnostics


def test_runtime_cv_tracks_separate_fit_and_transform_only_when_consumed():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(2)
    X = rng.rand(120, 4)
    y = rng.randint(0, 2, 120)
    with watch(Config()) as session:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        scaler = StandardScaler().fit(X_train)
        X_scaled = scaler.transform(X_train)
        cross_val_score(LogisticRegression(random_state=0), X_scaled, y_train, cv=3)
    assert "C001" in {finding.rule_id for finding in session.findings}
    assert session.result.complete
    assert not session.diagnostics

    with watch(Config()) as session:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        scaler = StandardScaler().fit(X_train)
        scaler.transform(X_test)  # computed but not consumed by CV
        cross_val_score(LogisticRegression(random_state=0), X_train, y_train, cv=3)
    assert "C001" not in {finding.rule_id for finding in session.findings}
    assert session.result.complete
    assert not session.diagnostics


def test_runtime_cv_tracks_native_copy_of_transformed_output():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(4)
    X = rng.rand(120, 4)
    y = rng.randint(0, 2, 120)
    with watch(Config()) as session:
        transformed = StandardScaler().fit_transform(X).copy()
        cross_val_score(LogisticRegression(random_state=0), transformed, y, cv=3)
    assert "C001" in {finding.rule_id for finding in session.findings}
    assert session.result.complete
    assert not session.diagnostics


def test_runtime_does_not_link_equal_independent_array():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    X = np.zeros((120, 4))
    y = (np.arange(120) // 10) % 2
    with watch(Config()) as session:
        transformed = StandardScaler().fit_transform(X)
        independent = np.zeros_like(transformed)
        assert type(independent) is np.ndarray
        assert np.array_equal(transformed, independent)
        assert not np.shares_memory(transformed, independent)
        cross_val_score(LogisticRegression(random_state=0), independent, y, cv=3)
    assert "C001" not in {finding.rule_id for finding in session.findings}
    assert session.result.complete
    assert not session.diagnostics


def test_runtime_cv_tracks_dtype_cast_of_transformed_output():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(5)
    X = rng.rand(120, 4)
    y = (np.arange(120) // 10) % 2
    with watch(Config()) as session:
        transformed = StandardScaler().fit_transform(X).astype(np.float32)
        cross_val_score(LogisticRegression(random_state=0), transformed, y, cv=3)
    assert "C001" in {finding.rule_id for finding in session.findings}
    assert session.result.complete
    assert not session.diagnostics


def test_runtime_cv_tracks_modified_copy_of_transformed_output():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(6)
    X = rng.rand(120, 4)
    y = (np.arange(120) // 10) % 2
    with watch(Config()) as session:
        transformed = StandardScaler().fit_transform(X).copy()
        transformed[0, 0] += 1.0
        cross_val_score(LogisticRegression(random_state=0), transformed, y, cv=3)
    assert "C001" in {finding.rule_id for finding in session.findings}
    assert session.result.complete
    assert not session.diagnostics


def test_runtime_unsupported_numpy_propagation_is_partial():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(7)
    X = rng.rand(120, 4)
    y = (np.arange(120) // 10) % 2
    with watch(Config()) as session:
        transformed = StandardScaler().fit_transform(X)
        unsupported = np.take(transformed, np.arange(0, len(X), 2), axis=0)
        cross_val_score(LogisticRegression(random_state=0), unsupported, y[::2], cv=3)
    assert "LP417" in {diagnostic.code for diagnostic in session.diagnostics}
    assert not session.result.complete


def test_runtime_stateless_normalizer_is_not_leakage():
    with watch(Config()) as session:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import Normalizer

        rng = np.random.RandomState(3)
        X = rng.rand(60, 4)
        y = rng.randint(0, 2, 60)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        Normalizer().fit_transform(X_test)
    assert not session.findings
    assert session.result.complete
    assert not session.diagnostics

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    with watch(Config()) as session:
        transformed = Normalizer().fit_transform(X)
        cross_val_score(LogisticRegression(random_state=0), transformed, y, cv=3)
    assert not session.findings
    assert session.result.complete
    assert not session.diagnostics


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


@pytest.mark.parametrize("serializer", ("pickle", "joblib"))
def test_tracked_split_arrays_round_trip_after_watch(
    serializer: str, tmp_path: Path
):
    with watch(Config()):
        from sklearn.model_selection import train_test_split

        X = np.arange(160, dtype=float).reshape(80, 2)
        y = np.arange(80) % 2
        X_train, X_test, _, _ = train_test_split(X, y, random_state=0)

    restored_train, restored_test = _round_trip(
        (X_train, X_test), serializer, tmp_path / f"split.{serializer}"
    )
    assert type(restored_train) is np.ndarray
    assert type(restored_test) is np.ndarray
    assert np.array_equal(restored_train, X_train)
    assert np.array_equal(restored_test, X_test)

    with watch(Config()) as fresh_session:
        assert fresh_session.taint.lineage_for(restored_train) is None
        assert fresh_session.taint.lineage_for(restored_test) is None


@pytest.mark.parametrize("serializer", ("pickle", "joblib"))
def test_fitted_knn_round_trip_after_watch(serializer: str, tmp_path: Path):
    from sklearn.neighbors import KNeighborsClassifier

    rng = np.random.RandomState(8)
    X = rng.rand(80, 4)
    y = np.arange(80) % 2
    with watch(Config()):
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=0)
        assert type(X_train) is not np.ndarray
        model = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)

    expected = model.predict(X_test)
    restored = _round_trip(model, serializer, tmp_path / f"model.{serializer}")
    assert isinstance(restored, KNeighborsClassifier)
    assert np.array_equal(restored.predict(X_test), expected)
    assert type(restored._fit_X) is np.ndarray

    with watch(Config()) as fresh_session:
        assert fresh_session.taint.lineage_for(restored._fit_X) is None


def test_runtime_watch_requires_runtime_layer_before_hooks():
    import sklearn.model_selection as ms

    before = ms.train_test_split
    with pytest.raises(ConfigError, match="runtime layer"):
        with watch(Config(layers=[Layer.STATIC])):
            pass
    assert ms.train_test_split is before


def test_concurrent_watch_sessions_keep_hooks_until_last_exit():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Event

    from sklearn.preprocessing import StandardScaler

    entered = Barrier(2)
    first_exited = Event()
    X = np.arange(160, dtype=float).reshape(80, 2)

    def first_session() -> None:
        with watch(Config()):
            entered.wait(timeout=10)
        first_exited.set()

    def second_session() -> Any:
        with watch(Config()) as session:
            entered.wait(timeout=10)
            assert first_exited.wait(timeout=10)
            from sklearn.model_selection import train_test_split

            _, X_test = train_test_split(X, test_size=0.25, random_state=0)
            StandardScaler().fit(X_test)
        return session

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(first_session)
        second = pool.submit(second_session)
        first.result(timeout=30)
        session = second.result(timeout=30)

    assert "S001" in {finding.rule_id for finding in session.findings}
    assert session.result.complete
    assert not session.diagnostics


def test_runtime_split_helpers_snapshot_rng_and_support_shapes():
    from ml_leakproof.runtime.hooks import _clone_random_state, _row_count, _split_indices

    assert _clone_random_state(7) == 7
    assert _clone_random_state(None) is not None
    random_state = np.random.RandomState(7)
    clone = _clone_random_state(random_state)
    assert isinstance(clone, np.random.RandomState)
    assert np.array_equal(clone.rand(4), random_state.rand(4))
    generator = np.random.default_rng(7)
    generator_clone = _clone_random_state(generator)
    assert isinstance(generator_clone, np.random.Generator)
    assert np.array_equal(generator_clone.random(4), generator.random(4))

    assert _row_count(np.zeros((2, 3))) == 2
    assert _row_count([1, 2]) == 2
    assert _row_count(object()) is None
    assert _split_indices(
        np.arange(8),
        {"shuffle": False},
        train_len=3,
        test_len=2,
        random_state=None,
    ) == {"train": [0, 1, 2], "test": [3, 4]}
    assert _split_indices(
        np.arange(8),
        {"stratify": np.array([0, 1] * 4)},
        train_len=3,
        test_len=2,
        random_state=0,
    ) is not None
    assert _split_indices(
        np.arange(8),
        {},
        train_len=8,
        test_len=8,
        random_state=0,
    ) is None


def test_numpy_views_keep_registered_row_lineage():
    from ml_leakproof.runtime.taint import TaintTable

    data = np.arange(30).reshape(10, 3)
    table = TaintTable()
    table.register_split(train=data[:6], test=data[6:], source=data)
    view = data[6:, :2]
    lineage = table.lineage_for(view)
    assert lineage is not None
    assert table.membership(lineage.rows) == {"train": 0, "test": 4, "val": 0, "eval": 4}


def test_ambiguous_duplicate_copy_split_reports_partial_lineage():
    from ml_leakproof.runtime.taint import TaintTable

    source = np.zeros((8, 3))
    train = source[:4].copy()
    test = source[4:].copy()
    table = TaintTable()
    table.register_split(
        train=train,
        test=test,
        source=source,
    )
    assert [item["code"] for item in table.diagnostics()] == ["LP415"]
    train_lineage = table.lineage_for(train)
    test_lineage = table.lineage_for(test)
    assert train_lineage is not None and not train_lineage.supported
    assert test_lineage is not None and not test_lineage.supported


def test_pandas_reset_index_and_one_dimensional_hstack_keep_lineage():
    import pandas as pd

    from ml_leakproof.runtime.engine import watch

    frame = pd.DataFrame({"value": np.arange(8)}, index=np.arange(100, 108))
    with watch(Config()) as session:
        train = frame.iloc[:5]
        test = frame.iloc[5:]
        session.register_split(train=train, test=test)
        reset = test.reset_index(drop=True)
        reset_lineage = session.taint.lineage_for(reset)
        assert reset_lineage is not None
        assert session.taint.membership(reset_lineage.rows) == {
            "train": 0,
            "test": 3,
            "val": 0,
            "eval": 3,
        }
        reset_with_index = test.reset_index()
        reset_with_index_lineage = session.taint.lineage_for(reset_with_index)
        assert reset_with_index_lineage is not None
        assert session.taint.membership(reset_with_index_lineage.rows) == {
            "train": 0,
            "test": 3,
            "val": 0,
            "eval": 3,
        }

        values = np.arange(8)
        train_values = values[:5]
        test_values = values[5:]
        session.register_split(train=train_values, test=test_values)
        mixed = np.hstack([train_values, test_values])
        mixed_lineage = session.taint.lineage_for(mixed)
        assert mixed_lineage is not None
        assert session.taint.membership(mixed_lineage.rows) == {
            "train": 5,
            "test": 3,
            "val": 0,
            "eval": 3,
        }
