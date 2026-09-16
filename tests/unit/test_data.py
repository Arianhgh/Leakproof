from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ml_leakproof
from ml_leakproof.core.config import Config
from ml_leakproof.data.engine import DataEngine
from ml_leakproof.data.input import DataAuditInput


def test_exact_overlap():
    rows = pd.DataFrame({"a": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    train = rows.iloc[[0, 1, 2]].reset_index(drop=True)
    test = rows.iloc[[2, 3]].reset_index(drop=True)
    findings = DataEngine(Config()).run(DataAuditInput(train=train, test=test, target="target"))
    assert "D001" in {f.rule_id for f in findings}


def test_group_overlap():
    train = pd.DataFrame({"x": [1, 2], "pid": ["p1", "p2"], "target": [0, 1]})
    test = pd.DataFrame({"x": [3], "pid": ["p2"], "target": [0]})
    findings = DataEngine(Config()).run(
        DataAuditInput(train=train, test=test, target="target", group="pid")
    )
    assert "D003" in {f.rule_id for f in findings}


def test_temporal_overlap():
    train = pd.DataFrame(
        {"x": [1, 2], "ts": pd.to_datetime(["2020-01-01", "2021-01-01"]), "target": [0, 1]}
    )
    test = pd.DataFrame({"x": [3], "ts": pd.to_datetime(["2020-06-01"]), "target": [0]})
    findings = DataEngine(Config()).run(
        DataAuditInput(train=train, test=test, target="target", time="ts")
    )
    assert "TM001" in {f.rule_id for f in findings}


def test_public_audit_data_api():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 60)
    train = pd.DataFrame({"noise": rng.rand(60), "leak": y, "target": y})
    test = train.iloc[:10].copy()
    findings = ml_leakproof.audit_data(train, test, target="target", config=Config())
    assert "D005" in {f.rule_id for f in findings}


def test_hash_exclude_avoids_identifier_overlap():
    train = pd.DataFrame({"id": [1, 2], "x": [10, 20], "target": [0, 1]})
    test = pd.DataFrame({"id": [1], "x": [999], "target": [0]})
    cfg = Config()
    cfg.data.hash_include = ["id"]
    assert "D001" in {
        f.rule_id
        for f in DataEngine(cfg).run(DataAuditInput(train=train, test=test, target="target"))
    }
    cfg.data.hash_exclude = ["id"]
    assert "D001" not in {
        f.rule_id
        for f in DataEngine(cfg).run(DataAuditInput(train=train, test=test, target="target"))
    }


def test_image_path_near_duplicate_detection(tmp_path):
    from PIL import Image

    train_img = tmp_path / "train.png"
    test_img = tmp_path / "test.png"
    Image.new("RGB", (8, 8), color="white").save(train_img)
    Image.new("RGB", (8, 8), color="white").save(test_img)
    train = pd.DataFrame({"path": [str(train_img)], "target": [0]})
    test = pd.DataFrame({"path": [str(test_img)], "target": [0]})
    findings = DataEngine(Config()).run(
        DataAuditInput(
            train=train,
            test=test,
            target="target",
            feature_types={"path": "image"},
        )
    )
    assert "D002" in {f.rule_id for f in findings}


def test_nonfinite_grouped_probe_is_reported_as_partial():
    train = pd.DataFrame(
        {
            "feature": [0, 1, 2, 3, 4, 5],
            "group": ["a", "a", "b", "b", "c", "c"],
            "target": [0, 0, 0, 0, 1, 1],
        }
    )
    test = pd.DataFrame({"feature": [100, 101], "group": ["d", "e"], "target": [0, 1]})
    result = DataEngine(Config()).run_result(
        DataAuditInput(train=train, test=test, target="target", group="group")
    )
    assert result.partial
    assert "D005" in result.coverage.unavailable_checks
    assert any(d.code == "LP330" and d.rule_id == "D005" for d in result.diagnostics)


@pytest.mark.parametrize(
    "changes",
    [
        {"train": [1, 2]},
        {"test": pd.DataFrame({"different": [1]})},
        {"target": "missing"},
        {"target": 1},
        {"group": "missing"},
        {"time": "missing"},
        {"feature_types": []},
        {"feature_types": {"missing": "numeric"}},
        {"feature_types": {"x": "unsupported"}},
        {"val": pd.DataFrame({"different": [1]})},
    ],
)
def test_public_invalid_data_returns_failed_result(changes):
    import pytest

    from ml_leakproof import AnalysisError, analyze_data, audit_data

    args = {"train": pd.DataFrame({"x": [1, 2]}), "test": pd.DataFrame({"x": [3, 4]}), **changes}
    result = analyze_data(**args, config=Config(select=["D001"]))
    assert result.failed and result.diagnostics[0].code == "LP300"
    with pytest.raises(AnalysisError):
        audit_data(**args, config=Config(select=["D001"]))
    assert audit_data(**args, config=Config(select=["D001"]), allow_partial=True) == []


def test_duplicate_columns_and_unknown_hash_policy_fail():
    from ml_leakproof import analyze_data
    from ml_leakproof.core.config import DataConfig

    frame = pd.DataFrame([[1, 2]], columns=["x", "x"])
    assert analyze_data(frame, frame, config=Config(select=["D001"])).failed
    frame = pd.DataFrame({"x": [1]})
    result = analyze_data(
        frame, frame, config=Config(select=["D001"], data=DataConfig(hash_include=["missing"]))
    )
    assert result.failed and result.diagnostics[0].code == "LP301"


def test_public_data_layer_disabled():
    from ml_leakproof import Layer, analyze_data

    assert analyze_data(None, None, config=Config(layers=[Layer.STATIC])).failed


def test_data_quality_notes_remain_visible():
    from ml_leakproof import analyze_data

    frame = pd.DataFrame({"x": [float("inf"), None]}, index=[0, 0])
    result = analyze_data(frame, frame.iloc[:0], config=Config(select=["D001"]))
    assert result.complete
    notes = " ".join(result.coverage.notes)
    assert all(word in notes for word in ["empty", "missing", "duplicate", "infinity"])


def test_temporal_range_validation():
    import pytest

    from ml_leakproof.data.temporal import time_ranges

    with pytest.raises(ValueError, match="missing"):
        time_ranges(pd.DataFrame({"x": [1]}), "time")
    with pytest.raises(ValueError, match="could not be parsed"):
        time_ranges(pd.DataFrame({"time": ["not a time"]}), "time")
    assert time_ranges(pd.DataFrame({"time": [None]}), "time") == (None, None)
    minimum, maximum = time_ranges(
        pd.DataFrame({"time": ["2024-01-01T00:00:00+00:00", "2024-01-01T01:00:00-05:00"]}), "time"
    )
    assert (maximum - minimum).total_seconds() == 6 * 3600
