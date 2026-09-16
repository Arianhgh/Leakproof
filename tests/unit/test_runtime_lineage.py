"""Provenance across real selections, assembly, and explicit registration."""

import numpy as np
import pandas as pd
import pytest

from ml_leakproof import Config, watch
from ml_leakproof.runtime.taint import TaintTable


@pytest.mark.parametrize(
    "key,positions",
    [
        (slice(None, None, -1), [3, 2, 1, 0]),
        ([2, 0], [2, 0]),
        (np.array([True, False, True, False]), [0, 2]),
        ((slice(1, 3), slice(None)), [1, 2]),
        ((Ellipsis, slice(None)), [0, 1, 2, 3]),
    ],
)
def test_numpy_selection_retains_exact_rows(key, positions):
    source = np.arange(8).reshape(4, 2)
    table = TaintTable()
    train, test = source[:2], source[2:]
    table.register_split(train=train, test=test, source=source)
    tracked = table.wrap_numpy_result(source)
    result = tracked[key]
    assert table.lineage_for(result).rows == tuple(
        table.lineage_for(source).rows[i] for i in positions
    )


def test_explicit_ids_capture_namespaces_and_validation():
    table = TaintTable()
    train, test, val = [[1], [2]], [[3]], [[4]]
    pending = table.capture(train)
    table.register_split(
        train=train, test=test, val=val, row_ids={"train": ["a", "b"], "test": ["c"], "val": ["d"]}
    )
    rows = table.resolve_capture(pending)
    assert table.membership(rows) == {"train": 2, "test": 0, "val": 0, "eval": 0}
    assert len(table.eval_side()) == 2 and len(table.val_hashes) == 1
    before = table.lineage_for(test).rows
    table.register_split(train=[[1]], test=[[3]], row_ids={"train": ["a"], "test": ["c"]})
    assert table.lineage_for(test).rows == before
    assert table.signature(rows) == table.signature(reversed(rows))
    invalid = TaintTable()
    invalid.register_split(
        train=train,
        test=test,
        row_ids={"train": ["too short"], "unknown": []},
        row_indices={"other": []},
    )
    assert len(invalid.diagnostics()) == 3


@pytest.mark.parametrize("indices", [{"train": [0, 99]}, {"train": ["bad", 0]}, {"train": [0]}])
def test_invalid_split_positions_are_diagnostic(indices):
    table = TaintTable()
    source = np.arange(8).reshape(4, 2)
    table.register_split(train=source[:2], test=source[2:], source=source, row_indices=indices)
    assert any(d["code"] == "LP410" for d in table.diagnostics())
    assert not table.lineage_for(source[:2]).supported or table.diagnostics()


def test_transform_requires_explicit_positions_when_length_changes():
    table = TaintTable()
    source = np.arange(8).reshape(4, 2)
    output = source[[2, 0]].copy()
    assert not table.track_transform(source=source, transformed=output)
    assert table.diagnostics()[-1]["code"] == "LP412"
    assert not table.track_transform(source=source, transformed=output, row_indices=[0, 9])
    assert table.diagnostics()[-1]["code"] == "LP413"
    assert table.track_transform(source=source, transformed=output, row_indices=[2, 0], origin=7)
    assert table.lineage_for(output).transform_origins == frozenset({7})
    assert table.lineage_for(output).rows == tuple(
        table.lineage_for(source).rows[i] for i in [2, 0]
    )


def test_pandas_sort_reset_and_duplicate_index_policy():
    table = TaintTable()
    frame = pd.DataFrame({"a": [4, 2, 3]}, index=["c", "a", "b"])
    table.register_split(train=frame, test=frame.iloc[:0])
    sorted_frame = frame.sort_index()
    assert table.track_pandas_operation(frame, sorted_frame)
    assert table.lineage_for(sorted_frame).rows == tuple(
        table.lineage_for(frame).rows[i] for i in [1, 2, 0]
    )
    reset = sorted_frame.reset_index()
    assert table.track_pandas_operation(sorted_frame, reset, method="reset_index")
    assert table.lineage_for(reset).rows == table.lineage_for(sorted_frame).rows
    duplicate = pd.DataFrame({"a": [1, 2]}, index=[0, 0])
    table.register_split(train=duplicate, test=duplicate.iloc[:0])
    assert not table.track_pandas_operation(duplicate, duplicate.copy())
    assert table.diagnostics()[-1]["code"] == "LP414"


def test_row_and_column_assembly_preserve_provenance():
    table = TaintTable()
    train, test = np.arange(4).reshape(2, 2), np.arange(4, 8).reshape(2, 2)
    table.register_split(train=train, test=test)
    combined = np.concatenate([train, test])
    assert table.track_concat([train, test], combined)
    assert table.lineage_for(combined).mixed
    assert table.membership(table.lineage_for(combined).rows)["test"] == 2
    columns = np.hstack([train, train])
    assert table.track_column_assembly([train, train], columns)
    assert table.lineage_for(columns).rows == table.lineage_for(train).rows
    assert not table.track_column_assembly([train, test], np.hstack([train, test]))
    assert not table.track_concat([], combined)
    assert not table.track_column_assembly([np.zeros((2, 2))], columns)


def test_numpy_out_assignment_updates_destination_lineage():
    from sklearn.preprocessing import StandardScaler

    with watch(Config()) as session:
        from sklearn.model_selection import train_test_split

        source = np.arange(40, dtype=float).reshape(20, 2)
        train, test = train_test_split(source, test_size=0.5, random_state=0)
        prepared = train.copy()
        returned = np.add(test, 0, out=prepared)
        assert returned is prepared
        StandardScaler().fit(prepared)
    assert "S001" in {f.rule_id for f in session.findings}


def test_numpy_multiple_outputs_and_masked_output_are_tracked():
    from sklearn.preprocessing import StandardScaler

    with watch(Config()) as session:
        from sklearn.model_selection import train_test_split

        source = np.arange(40, dtype=float).reshape(20, 2) / 3
        train, test = train_test_split(source, test_size=0.5, random_state=0)
        fractions, integers = np.modf(test)
        assert session.taint.lineage_for(fractions).rows == session.taint.lineage_for(test).rows
        assert session.taint.lineage_for(integers).rows == session.taint.lineage_for(test).rows
        prepared = train.copy()
        mask = np.ones(prepared.shape, dtype=bool)
        mask[0] = False
        np.add(test, 0, out=prepared, where=mask)
        StandardScaler().fit(prepared)
    assert "P002" in {f.rule_id for f in session.findings}


def test_numpy_inplace_arithmetic_runs_once_on_tracking_failure(monkeypatch):
    with watch(Config()) as session:
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(np.arange(20, dtype=float).reshape(10, 2), random_state=0)
        prepared = train.copy()
        before = prepared.view(np.ndarray).copy()

        def fail(*args):
            raise RuntimeError("tracking failed")

        monkeypatch.setattr(session.taint, "track_numpy_operation", fail)
        returned = np.add(prepared, 1, out=prepared)
        assert returned is prepared
        np.testing.assert_array_equal(prepared.view(np.ndarray), before + 1)
        assert session.taint.lineage_for(prepared) is None


def test_tracked_arrays_remain_usable_after_session_with_out():
    with watch(Config()):
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(np.arange(20, dtype=float).reshape(10, 2), random_state=0)
    before = train.view(np.ndarray).copy()
    assert np.add(train, 1, out=train) is train
    np.testing.assert_array_equal(train.view(np.ndarray), before + 1)
    parts = np.modf(train, out=(train, None))
    assert parts[0] is train
    np.testing.assert_array_equal(parts[1], before + 1)
