from __future__ import annotations

import pytest

from ml_leakproof.core.config import Config, ConfigError, LLMConfig
from ml_leakproof.core.models import Layer, Severity


def test_defaults():
    cfg = Config()
    assert cfg.fail_on is Severity.HIGH
    assert cfg.min_confidence == 0.60
    assert cfg.gate_confidence == 0.75
    assert cfg.profile == "ci"
    assert cfg.select == ["ALL"]
    assert Layer.STATIC in cfg.layers


def test_merge_from_table():
    cfg = Config()
    cfg._merge(
        {
            "select": ["P001", "D*"],
            "ignore": ["R001"],
            "fail_on": "critical",
            "min_confidence": 0.4,
            "gate_confidence": 0.9,
            "profile": "research",
            "layers": ["static"],
            "severity": {"S002": "low"},
            "data": {"imbalance_ratio": 0.8},
            "llm": {"enabled": True, "provider": "openai", "model": "gpt-test"},
        }
    )
    assert cfg.select == ["P001", "D*"]
    assert cfg.ignore == ["R001"]
    assert cfg.fail_on is Severity.CRITICAL
    assert cfg.min_confidence == 0.4
    assert cfg.gate_confidence == 0.9
    assert cfg.profile == "research"
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


def test_enabled_llm_requires_explicit_model():
    with pytest.raises(ConfigError, match="explicit llm.provider and llm.model"):
        Config(llm=LLMConfig(enabled=True))


def test_apply_cli_overrides():
    cfg = Config()
    cfg.apply_cli(
        select=["C001"],
        ignore=["R001"],
        layers=["static", "data"],
        fail_on="low",
        min_confidence=0.3,
        gate_confidence=0.8,
        profile="notebook",
    )
    assert cfg.select == ["C001"]
    assert "R001" in cfg.ignore
    assert cfg.layers == [Layer.STATIC, Layer.DATA]
    assert cfg.fail_on is Severity.LOW
    assert cfg.min_confidence == 0.3
    assert cfg.gate_confidence == 0.8
    assert cfg.profile == "notebook"
