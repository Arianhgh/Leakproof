from __future__ import annotations

import numpy as np
import pandas as pd

import leakproof
from leakproof.core.config import Config
from leakproof.data.engine import DataEngine
from leakproof.data.input import DataAuditInput


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
    train = pd.DataFrame({"x": [1, 2], "ts": pd.to_datetime(["2020-01-01", "2021-01-01"]), "target": [0, 1]})
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
    findings = leakproof.audit_data(train, test, target="target", config=Config())
    assert "D005" in {f.rule_id for f in findings}


def test_hash_exclude_avoids_identifier_overlap():
    train = pd.DataFrame({"id": [1, 2], "x": [10, 20], "target": [0, 1]})
    test = pd.DataFrame({"id": [1], "x": [999], "target": [0]})
    cfg = Config()
    cfg.data.hash_include = ["id"]
    assert "D001" in {
        f.rule_id for f in DataEngine(cfg).run(DataAuditInput(train=train, test=test, target="target"))
    }
    cfg.data.hash_exclude = ["id"]
    assert "D001" not in {
        f.rule_id for f in DataEngine(cfg).run(DataAuditInput(train=train, test=test, target="target"))
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
