"""Focused release-gate tests for validation, hashing, fixes, and hooks."""

from __future__ import annotations

import os
import stat
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import ml_leakproof.autofix as autofix
import ml_leakproof.core.config as config_module
import ml_leakproof.runtime.hooks as hooks
from ml_leakproof.core.config import Config, ConfigError, DataConfig, LLMConfig
from ml_leakproof.core.models import Category, Finding, Fix, Layer, Location, Severity
from ml_leakproof.data.hashing import (
    _normalize_cell,
    canonical_cell,
    canonical_row,
    overlap,
    overlap_details,
    row_hashes,
)
from ml_leakproof.static.engine import StaticEngine


@pytest.mark.parametrize(
    "kwargs",
    [
        {"select": "P001"},
        {"ignore": "P001"},
        {"exclude": "tests/"},
        {"layers": ["static"]},
        {"fail_on": object()},
        {"min_confidence": "0.5"},
        {"gate_confidence": float("nan")},
        {"allow_partial": "false"},
        {"severity_overrides": []},
        {"severity_overrides": {"": "low"}},
        {"severity_overrides": {"P001": 1}},
        {"data": object()},
        {"llm": object()},
    ],
)
def test_programmatic_config_rejects_invalid_types(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        Config(**kwargs)


def test_config_profiles_and_isolated_copy() -> None:
    cfg = Config(profile="RESEARCH")
    assert cfg.profile == "research"
    assert cfg.min_confidence == 0.0
    assert cfg.explicit_select_all is False
    cfg._merge({"select": ["ALL"]})
    assert cfg.explicit_select_all is True
    copied = cfg.copy()
    copied.ignore.append("P001")
    assert copied.ignore != cfg.ignore


def test_config_merge_and_cli_precedence() -> None:
    cfg = Config()
    cfg._merge(
        {
            "profile": "research",
            "select": ["P001"],
            "ignore": ["R001"],
            "fail_on": "critical",
            "min_confidence": 0.4,
            "gate_confidence": 0.9,
            "exclude": ["vendor/"],
            "layers": [],
            "severity": {"P001": "medium"},
            "data": {"sample_cap": 10, "hash_include": ["id"]},
            "llm": {"provider": "OPENAI", "model": "test-model"},
            "allow_partial": True,
        }
    )
    assert cfg.fail_on is Severity.CRITICAL
    assert cfg.layers == []
    assert cfg.data.sample_cap == 10
    assert cfg.llm.provider == "openai"
    assert cfg.allow_partial is True
    cfg.apply_cli(
        select=["D001"],
        ignore=["D002"],
        layers=["data"],
        fail_on="low",
        exclude=["generated/"],
        profile="ci",
        allow_partial=False,
    )
    assert cfg.select == ["D001"]
    assert cfg.ignore == ["R001", "D002"]
    assert cfg.layers == [Layer.DATA]
    assert cfg.fail_on is Severity.LOW
    assert cfg.allow_partial is False
    assert cfg.exclude[-1] == "generated/"


@pytest.mark.parametrize(
    "table",
    [
        {"unknown": True},
        {"select": "P001"},
        {"ignore": "P001"},
        {"exclude": "tests"},
        {"layers": ["unknown"]},
        {"severity": []},
        {"severity": {1: "low"}},
        {"severity": {"P001": 1}},
        {"allow_partial": 1},
    ],
)
def test_config_merge_rejects_bad_tables(table: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        Config()._merge(table)


def test_config_merge_rejects_non_table() -> None:
    with pytest.raises(ConfigError, match="configuration must be a table"):
        Config()._merge([])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "target,values",
    [
        (DataConfig(), {"sample_cap": True}),
        (DataConfig(), {"sample_cap": "10"}),
        (DataConfig(), {"neardup_distance": "0.1"}),
        (DataConfig(), {"hash_include": "id"}),
        (DataConfig(), {"unknown": 1}),
        (LLMConfig(), {"enabled": "false"}),
        (LLMConfig(), {"provider": 1}),
        (LLMConfig(), {"timeout_seconds": "20"}),
        (LLMConfig(), {"model": 1}),
    ],
)
def test_nested_config_merge_rejects_bad_values(target: Any, values: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        config_module._merge_dataclass(target, values, "nested")


def test_nested_config_merge_accepts_all_supported_shapes() -> None:
    data = DataConfig()
    config_module._merge_dataclass(
        data,
        {
            "neardup_distance": 0.1,
            "sample_cap": 10,
            "hash_include": ["id"],
            "deterministic_sampling": True,
        },
        "data",
    )
    llm = LLMConfig()
    config_module._merge_dataclass(
        llm,
        {"enabled": False, "provider": None, "model": None, "retries": 0},
        "llm",
    )
    assert data.hash_include == ["id"]
    assert llm.retries == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("neardup_text_threshold", "bad"),
        ("neardup_distance", float("inf")),
        ("imbalance_ratio", -0.1),
        ("target_leakage_score", 1.1),
        ("mi_floor", -1.0),
        ("mi_floor", float("nan")),
        ("mi_dominance_ratio", 0.5),
        ("mi_dominance_ratio", float("nan")),
        ("imagehash_distance", 0),
        ("sample_cap", False),
        ("hash_chunk_size", "bad"),
        ("max_examples", 0),
        ("max_neardup_candidates", -1),
        ("max_neardup_pairs", 0),
        ("probe_cv_folds", 0),
        ("hash_include", [1]),
        ("deterministic_sampling", 1),
    ],
)
def test_data_config_validation_edges(field: str, value: Any) -> None:
    data = DataConfig()
    setattr(data, field, value)
    with pytest.raises(ConfigError):
        config_module._validate_data(data)


@pytest.mark.parametrize(
    "llm",
    [
        LLMConfig(enabled=1),  # type: ignore[arg-type]
        LLMConfig(provider="unknown"),
        LLMConfig(provider=1),  # type: ignore[arg-type]
        LLMConfig(enabled=True),
        LLMConfig(model=1),  # type: ignore[arg-type]
        LLMConfig(timeout_seconds="20"),  # type: ignore[arg-type]
        LLMConfig(timeout_seconds=float("nan")),
        LLMConfig(timeout_seconds=0),
        LLMConfig(retries=True),  # type: ignore[arg-type]
        LLMConfig(retries=-1),
        LLMConfig(max_context_chars=0),
    ],
)
def test_llm_validation_edges(llm: LLMConfig) -> None:
    with pytest.raises(ConfigError):
        config_module._validate_llm(llm)


def test_toml_helpers_and_config_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "pyproject.toml"
    project.write_text("[tool.leakproof]\nprofile = 'notebook'\nignore = ['R001']\n")
    (tmp_path / "leakproof.toml").write_text("profile = 'research'\nmin_confidence = 0.2\n")
    explicit = tmp_path / "explicit.toml"
    explicit.write_text("[tool.leakproof]\nprofile = 'ci'\nmin_confidence = 0.1\n")
    cfg = Config.load(explicit, start=tmp_path)
    assert cfg.profile == "ci"
    assert cfg.min_confidence == 0.1
    assert "R001" in cfg.ignore

    bad = tmp_path / "bad.toml"
    bad.write_text("broken = [\n")
    with pytest.raises(ConfigError, match="invalid TOML"):
        config_module._read_toml(bad)
    with pytest.raises(ConfigError, match="could not read"):
        config_module._read_toml(tmp_path / "missing.toml")
    with pytest.raises(ConfigError, match="does not exist"):
        Config.load(tmp_path / "missing.toml", start=tmp_path)

    monkeypatch.setattr(config_module.tomllib, "load", lambda _fh: [])
    with pytest.raises(ConfigError, match="not a TOML table"):
        config_module._read_toml(project)


@pytest.mark.parametrize(
    "contents,pattern",
    [
        ("tool = 'not-a-table'\n", r"\[tool\] must be a table"),
        ("[tool]\nleakproof = 'bad'\n", r"\[tool\.leakproof\] must be a table"),
    ],
)
def test_project_toml_shape_validation(tmp_path: Path, contents: str, pattern: str) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text(contents)
    with pytest.raises(ConfigError, match=pattern):
        Config.load(start=tmp_path)


def test_explicit_toml_table_shape_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "config.toml"
    path.write_text("x = 1\n")
    monkeypatch.setattr(config_module, "_read_toml", lambda _path: {"tool": {}})
    with pytest.raises(ConfigError, match=r"no \[tool.leakproof\]"):
        config_module._read_toml_table(path)
    monkeypatch.setattr(config_module, "_read_toml", lambda _path: {"tool": {"leakproof": []}})
    with pytest.raises(ConfigError, match="table is invalid"):
        config_module._read_toml_table(path)


def test_find_root_file_and_git_marker(tmp_path: Path) -> None:
    nested = tmp_path / "src"
    nested.mkdir()
    file = nested / "m.py"
    file.write_text("x = 1\n")
    assert config_module._find_root(file) == nested
    (tmp_path / ".git").mkdir()
    assert config_module._find_root(nested) == tmp_path
    other = tmp_path / "other"
    other.mkdir()
    assert config_module._find_root(other) == tmp_path


def test_hash_canonicalization_is_type_aware() -> None:
    values = [
        None,
        pd.NA,
        True,
        1,
        1.0,
        -0.0,
        Decimal("1.00"),
        date(2024, 1, 2),
        datetime(2024, 1, 2, 3, 4),
        time(3, 4),
        b"a|b",
        np.int64(3),
        pd.Timestamp("2024-01-02T03:04:05Z"),
        pd.Categorical(["a"])[0],
        object(),
    ]
    encoded = [canonical_cell(value) for value in values]
    assert canonical_cell(None) == canonical_cell(pd.NA)
    assert len(set(encoded)) == len(encoded) - 1
    assert canonical_cell("1|2") != canonical_cell("1") + canonical_cell("2")
    assert canonical_cell(0.0) == canonical_cell(-0.0)
    assert _normalize_cell("hello") == canonical_cell("hello").hex()
    assert canonical_row(["a", 1]).startswith(b"R")


def test_hashing_handles_arrays_missing_values_and_sample_indices() -> None:
    frame = pd.DataFrame({"a": ["x", "y", None], "b": [1.0, 2.0, np.nan]})
    hashes = row_hashes(frame, columns=["a", "b"], sample_cap=2)
    assert len(hashes) == 2
    assert hashes.dtype.name == "string"
    assert hashes.attrs["sampled"] is True
    assert hashes.attrs["columns"] == ["a", "b"]
    assert hashes.attrs["original_indices"] == [0, 1]
    full = row_hashes(frame)
    assert full.attrs["sampled"] is False
    assert len(full.attrs["canonical_rows"]) == 3


def test_hash_overlap_verifies_records_and_preserves_positions() -> None:
    train = pd.DataFrame({"a": ["left|right", "same"], "b": [1, 2]})
    test = pd.DataFrame({"a": ["same", "different"], "b": [2, 3]})
    train_hashes = row_hashes(train)
    test_hashes = row_hashes(test)
    details = overlap_details(train_hashes, test_hashes)
    assert details["count"] == 1
    assert details["pairs"] == [(1, 0)]
    assert overlap(train_hashes, test_hashes)[1] == 1
    empty = pd.Series([], dtype="string")
    assert overlap_details(empty, empty)["count"] == 0


def _r001_findings(path: Path) -> list[Finding]:
    return [finding for finding in StaticEngine(Config()).run_file(path) if finding.rule_id == "R001"]


def test_autofix_preview_write_stale_digest_and_idempotence(tmp_path: Path) -> None:
    path = tmp_path / "seed.py"
    path.write_text(
        "from sklearn.model_selection import train_test_split\n"
        "a, b = train_test_split(X, y)\n"
    )
    findings = _r001_findings(path)
    assert findings
    previews = autofix.preview_fixes(findings)
    assert previews and "random_state=0" in previews[0]
    assert "random_state=0" not in path.read_text()
    assert autofix.apply_fixes(findings, write=False) == {path: 1}
    assert autofix.apply_fixes(findings) == {path: 1}
    assert "random_state=0" in path.read_text()
    assert autofix.apply_fixes(_r001_findings(path)) == {}

    path.write_text("from sklearn.model_selection import train_test_split\na, b = train_test_split(X, y)\n")
    stale = findings[0]
    path.write_text(path.read_text() + "# changed\n")
    assert autofix.apply_fixes([stale]) == {}


def test_autofix_skips_ambiguous_and_noneligible_fixes(tmp_path: Path) -> None:
    path = tmp_path / "seed.py"
    path.write_text("from sklearn.model_selection import train_test_split\na, b = train_test_split(X, y, **kwargs)\n")
    finding = Finding(
        rule_id="R001",
        category=Category.DETERMINISM,
        severity=Severity.LOW,
        layer=Layer.STATIC,
        message="seed",
        location=Location(file=path, line=2, col=6),
        fix=Fix(summary="seed", autofixable=True),
        evidence={"callable": "train_test_split", "analysis_source_sha256": autofix._source_digest(path)},
    )
    assert autofix.apply_fixes([finding]) == {}
    assert autofix.preview_fixes([finding]) == []
    assert autofix.apply_fixes([]) == {}
    assert autofix.apply_fixes(
        [
            Finding(
                rule_id="P001",
                category=Category.PREPROCESSING,
                severity=Severity.HIGH,
                layer=Layer.STATIC,
                message="x",
                location=Location(file=path, line=2, col=4),
                fix=Fix(summary="x", autofixable=True),
            )
        ]
    ) == {}

    path.write_text(
        "from sklearn.model_selection import train_test_split\n"
        "a, b = train_test_split(X, y, random_state=1)\n"
    )
    finding.evidence["analysis_source_sha256"] = autofix._source_digest(path)
    assert autofix.apply_fixes([finding]) == {}

    assert autofix._cst_callable_name(SimpleNamespace(value="plain")) == "plain"
    assert (
        autofix._cst_callable_name(
            SimpleNamespace(
                value=SimpleNamespace(value="root"),
                attr=SimpleNamespace(value="leaf"),
            )
        )
        == "root.leaf"
    )
    assert autofix._cst_callable_name(object()) == ""


def test_autofix_handles_unreadable_paths_and_write_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "missing.py"
    finding = Finding(
        rule_id="R001",
        category=Category.DETERMINISM,
        severity=Severity.LOW,
        layer=Layer.STATIC,
        message="seed",
        location=Location(file=missing, line=1, col=0),
        fix=Fix(summary="seed", autofixable=True),
        evidence={"callable": "train_test_split"},
    )
    assert autofix.apply_fixes([finding]) == {}

    path = tmp_path / "write.py"
    path.write_text("x = 1\n")
    monkeypatch.setattr(autofix.os, "replace", lambda _source, _target: (_ for _ in ()).throw(OSError("no replace")))
    with pytest.raises(OSError, match="no replace"):
        autofix._atomic_write(path, "x = 2\n", "utf-8")
    assert path.read_text() == "x = 1\n"

    monkeypatch.setattr(autofix, "_rewrite", lambda _path, _findings: ("x = 2\n", 1, "utf-8"))
    assert autofix.preview_fixes([finding]) == []


def test_atomic_write_preserves_permissions(tmp_path: Path) -> None:
    path = tmp_path / "write.py"
    path.write_text("x = 1\n")
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    autofix._atomic_write(path, "x = 2\n", "utf-8")
    assert path.read_text() == "x = 2\n"
    assert stat.S_IMODE(path.stat().st_mode) == stat.S_IRUSR | stat.S_IWUSR


class _FakeSession:
    def __init__(self) -> None:
        self._hook_depth = 0
        self._cv_depth = 0
        self.events: list[tuple[str, Any]] = []
        self.diagnostics: list[tuple[str, str]] = []
        self.taint = SimpleNamespace(
            track_concat=lambda sources, result: self.events.append(("concat", (sources, result))),
            track_column_assembly=lambda sources, result: self.events.append(("columns", (sources, result))),
        )

    def add_diagnostic(self, code: str, message: str, **_: Any) -> None:
        self.diagnostics.append((code, message))

    def on_split(self, *args: Any, **kwargs: Any) -> None:
        self.events.append(("split", (args, kwargs)))

    def on_cv(self, *args: Any, **kwargs: Any) -> None:
        self.events.append(("cv", (args, kwargs)))

    def on_fit(self, *args: Any, **kwargs: Any) -> None:
        self.events.append(("fit", (args, kwargs)))

    def on_apply(self, *args: Any, **kwargs: Any) -> None:
        self.events.append(("apply", (args, kwargs)))

    def on_score(self, *args: Any, **kwargs: Any) -> None:
        self.events.append(("score", (args, kwargs)))

    def track_transform(self, **kwargs: Any) -> None:
        self.events.append(("transform", kwargs))


def test_hook_helpers_and_direct_callbacks() -> None:
    session = _FakeSession()
    manager = hooks.HookManager(session)

    def original(data: Any) -> tuple[Any, Any]:
        return data[:1], data[1:]

    manager._on_split(session, original, (), {"X": [1, 2]}, ([1], [2]))
    manager._on_split(session, original, (), {"X": [1, 2]}, ([1], [2], [3]))
    manager._on_split(session, original, (), {"X": [1, 2]}, (object(), object()))
    manager._on_split(session, original, (), {}, [1])
    manager._on_split(session, original, (), {}, ([1], [2]))
    manager._on_cv(session, original, (), {"X": [1]})
    manager._on_concat(session, original, (), {}, [])
    manager._on_concat(session, original, ([1],), {"axis": 0}, [1])
    assert any(code == "LP420" for code, _ in session.diagnostics)
    assert hooks._cv_input(("estimator", [1]), {}) == [1]
    assert hooks._cv_input((), {"x": [2]}) == [2]
    assert hooks._cv_input((), {}) is None
    assert hooks._extract_eval_inputs({"eval_set": [([1], [0]), "raw"], "validation_data": [3]}) == [
        [1],
        "raw",
        3,
    ]
    assert hooks._extract_eval_inputs({}) == []
    assert hooks._extract_eval_inputs({"eval_data": "scalar"}) == ["scalar"]

    def concatenate(values: Any, axis: int = 0) -> Any:
        return values

    def hstack(values: Any) -> Any:
        return values

    manager._on_concat(session, concatenate, ([1], 1), {}, [1])
    manager._on_concat(session, hstack, ([1],), {}, [1])
    session.taint.track_column_assembly = lambda _sources, _result: (_ for _ in ()).throw(RuntimeError("bad lineage"))
    manager._on_concat(session, concatenate, ([1], 1), {}, [1])
    assert any(code == "LP414" for code, _ in session.diagnostics)


def test_hook_argument_binding_and_method_dispatch() -> None:
    class Estimator:
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

        def score(self, X, y=None):
            return 0.5

    estimator = Estimator()
    assert hooks._bound_input(Estimator.fit, estimator, (), {"X": [1]}) == [1]
    assert hooks._bound_input(Estimator.fit, estimator, ([2],), {}) == [2]
    assert hooks._bound_input(lambda self: self, estimator, (), {}) is None
    assert hooks._bound_input(lambda self, other: self, estimator, ("fallback",), {}) == "fallback"
    assert hooks._bound_input(lambda x: x, estimator, (), {"X": [3]}) == [3]
    session = _FakeSession()
    manager = hooks.HookManager(session)
    manager._on_method(session, Estimator.fit, hooks.EventKind.FIT, estimator, ([1],), {"eval_set": [([2], [0])] }, estimator)
    manager._on_method(session, Estimator.transform, hooks.EventKind.APPLY, estimator, ([1],), {}, [1])
    manager._on_method(session, Estimator.transform, hooks.EventKind.APPLY, estimator, ([1],), {}, None)
    manager._on_method(session, Estimator.score, hooks.EventKind.SCORE, estimator, ([1],), {}, 0.5)
    manager._on_method(session, Estimator.fit, hooks.EventKind.SPLIT, estimator, ([1],), {}, estimator)
    assert {"fit", "apply", "score", "transform"} <= {event[0] for event in session.events}

    class Operations:
        def fit_transform(self, X):
            return [*X]

        def fit_resample(self, X, y):
            return [*X], [*y]

        def predict(self, X):
            return X

    operations = Operations()
    manager._on_method(
        session,
        Operations.fit_transform,
        hooks.EventKind.FIT,
        operations,
        ([1],),
        {},
        [1],
    )
    manager._on_method(
        session,
        Operations.fit_resample,
        hooks.EventKind.FIT,
        operations,
        ([1], [0]),
        {},
        ([1], [0]),
    )
    manager._on_method(
        session,
        Operations.predict,
        hooks.EventKind.APPLY,
        operations,
        ([1],),
        {},
        [1],
    )


def test_patch_manager_is_transactional_and_nested(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession()
    manager = hooks.HookManager(session)
    monkeypatch.setattr(manager, "_install_patches", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        manager.install()
    assert hooks.current_session() is None

    calls: list[str] = []
    manager._install_patches = lambda: calls.append("installed")
    manager.install()
    assert hooks.current_session() is session
    nested = hooks.HookManager(_FakeSession())
    nested.install()
    nested.remove()
    assert hooks.current_session() is session
    manager.remove()
    assert hooks.current_session() is None
    assert calls == ["installed"]
    hooks.HookManager(_FakeSession()).remove()

    module = SimpleNamespace(plain=lambda value: value)
    manager._patch_function(module, "plain", hooks.EventKind.APPLY)
    existing = hooks.HookManager(_FakeSession())
    existing.install()
    existing.remove()
    assert hooks.current_session() is None


def test_hook_wrapper_exceptions_and_cleanup() -> None:
    session = _FakeSession()
    manager = hooks.HookManager(session)
    module = SimpleNamespace()

    def boom() -> None:
        raise ValueError("boom")

    module.boom = boom
    manager._patch_function(module, "boom", hooks.EventKind.APPLY)
    token = hooks._CURRENT_SESSION.set(session)
    try:
        with pytest.raises(ValueError, match="boom"):
            module.boom()
    finally:
        hooks._CURRENT_SESSION.reset(token)
        manager._remove_patches()

    class Target:
        def method(self, value):
            return value

    manager._patch_method(Target, "method", hooks.EventKind.APPLY)
    assert Target().method(4) == 4
    manager._remove_patches()

    class FailingMeta(type):
        def __setattr__(cls, _name, _value):
            raise RuntimeError("cannot restore")

    failing = FailingMeta("Failing", (), {})
    hooks._PATCHES.append((failing, "method", object()))
    hooks._PATCHED_KEYS.add((id(failing), "method"))
    manager._remove_patches()


def test_call_site_and_optional_integration_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_stack() -> Any:
        raise RuntimeError("no stack")

    monkeypatch.setattr(hooks.inspect, "stack", fail_stack)
    assert hooks._call_site() == "<unknown>"


def test_patch_function_and_method_preserve_behavior() -> None:
    session = _FakeSession()
    manager = hooks.HookManager(session)
    module = SimpleNamespace()

    def split(data):
        return (data[:1], data[1:])

    def concat(values, axis=0):
        return values

    def cv(estimator, X):
        return X

    module.split = split
    module.concat = concat
    module.cv = cv
    manager._patch_function(module, "split", hooks.EventKind.SPLIT)
    manager._patch_function(module, "concat", hooks.EventKind.APPLY)
    manager._patch_function(module, "cv", hooks.EventKind.CV)
    manager._patch_function(module, "cv", hooks.EventKind.CV)
    assert module.split([1, 2]) == ([1], [2])
    assert module.concat([[1], [2]], axis=1) == [[1], [2]]
    assert module.cv(None, [1]) == [1]
    token = hooks._CURRENT_SESSION.set(session)
    try:
        assert module.split([1, 2]) == ([1], [2])
        assert module.concat([[1], [2]], axis=0) == [[1], [2]]
        assert module.cv(None, [1]) == [1]
    finally:
        hooks._CURRENT_SESSION.reset(token)
        manager._remove_patches()


def test_optional_hook_discovery_and_misc_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    class Fake:
        def fit(self, X):
            return self

        def fit_resample(self, X, y=None):
            return X, y

        def predict(self, X):
            return X

    available = {
        "imblearn.pipeline": SimpleNamespace(Pipeline=Fake),
        "xgboost.sklearn": SimpleNamespace(XGBClassifier=Fake, XGBRegressor=Fake, XGBRanker=Fake),
        "lightgbm.sklearn": SimpleNamespace(LGBMClassifier=Fake, LGBMRegressor=Fake, LGBMRanker=Fake),
        "imblearn.over_sampling": SimpleNamespace(SMOTE=Fake),
        "imblearn.under_sampling": SimpleNamespace(),
    }

    def fake_import(name: str) -> Any:
        if name not in available:
            raise ImportError(name)
        return available[name]

    manager = hooks.HookManager(_FakeSession())
    recorded: list[tuple[type, str, hooks.EventKind]] = []
    monkeypatch.setattr(hooks.importlib, "import_module", fake_import)
    monkeypatch.setattr(manager, "_patch_method", lambda cls, name, kind: recorded.append((cls, name, kind)))
    manager._patch_optional_integrations()
    assert any(name == "fit_resample" for _, name, _ in recorded)
    assert hooks._bound_input(lambda x: x, object(), (), {"X": [1]}) == [1]
    assert hooks._bound_input(lambda x: x, object(), (), {"x": [2]}) == [2]
