from __future__ import annotations

from leakproof.core.config import Config
from leakproof.core.models import Layer, Severity


def test_defaults():
    cfg = Config()
    assert cfg.fail_on is Severity.HIGH
    assert cfg.select == ["ALL"]
    assert Layer.STATIC in cfg.layers


def test_merge_from_table():
    cfg = Config()
    cfg._merge(
        {
            "select": ["P001", "D*"],
            "ignore": ["R001"],
            "fail_on": "critical",
            "layers": ["static"],
            "severity": {"S002": "low"},
            "data": {"imbalance_ratio": 0.8},
            "llm": {"enabled": True, "provider": "openai"},
        }
    )
    assert cfg.select == ["P001", "D*"]
    assert cfg.ignore == ["R001"]
    assert cfg.fail_on is Severity.CRITICAL
    assert cfg.layers == [Layer.STATIC]
    assert cfg.severity_overrides["S002"] == "low"
    assert cfg.data.imbalance_ratio == 0.8
    assert cfg.llm.enabled and cfg.llm.provider == "openai"


def test_load_from_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.leakproof]\nfail_on = 'medium'\nignore = ['R002']\n"
    )
    cfg = Config.load(start=tmp_path)
    assert cfg.fail_on is Severity.MEDIUM
    assert "R002" in cfg.ignore


def test_apply_cli_overrides():
    cfg = Config()
    cfg.apply_cli(select=["C001"], ignore=["R001"], layers=["static", "data"], fail_on="low")
    assert cfg.select == ["C001"]
    assert "R001" in cfg.ignore
    assert cfg.layers == [Layer.STATIC, Layer.DATA]
    assert cfg.fail_on is Severity.LOW
