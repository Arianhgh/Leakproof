"""Exercise real plugin discovery paths, including atomic failure."""

from types import SimpleNamespace

import pytest

from ml_leakproof.core import registry
from ml_leakproof.core.config import Config
from ml_leakproof.core.models import Category, Layer, Severity
from ml_leakproof.core.rule import StaticRule


class PluginRule(StaticRule):
    id = "vendor-X001"
    name = "example"
    category = Category.METRIC
    severity = Severity.HIGH
    layers = (Layer.STATIC,)

    def check(self, ctx):
        return []


@pytest.fixture
def isolated(monkeypatch):
    registry.all_rules()
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(registry, "_DIAGNOSTICS", [])
    monkeypatch.setattr(registry, "_ENTRYPOINTS_LOADED", False)
    monkeypatch.setattr(registry, "_BUILTIN_RULE_MODULES", [])
    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: [])


@pytest.mark.parametrize(
    "value", [PluginRule, PluginRule(), lambda: PluginRule(), lambda: PluginRule]
)
def test_factories_produce_fresh_rules(isolated, value):
    registry.register(value)
    a = registry.all_rules()["vendor-X001"]
    b = registry.all_rules()["vendor-X001"]
    assert a is not b
    a.severity = Severity.LOW
    assert b.severity is Severity.HIGH
    with pytest.raises(AssertionError):
        registry.register(PluginRule)


@pytest.mark.parametrize("value", [object(), lambda: object()])
def test_invalid_factory_rejected(isolated, value):
    with pytest.raises(TypeError):
        registry.register(value)


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("id", ""),
        ("name", ""),
        ("category", "metric"),
        ("severity", "high"),
        ("layers", []),
        ("layers", ("static",)),
        ("advisory_only", "true"),
    ],
)
def test_invalid_metadata(isolated, attribute, value):
    rule = PluginRule()
    setattr(rule, attribute, value)
    with pytest.raises(ValueError):
        registry._validate_rule(rule)


def test_plugin_load_is_atomic_and_errors_are_visible(isolated, monkeypatch):
    class InvalidRule(PluginRule):
        id = "UNPREFIXED"

    entries = [
        SimpleNamespace(name="bad", load=lambda: lambda: [PluginRule, InvalidRule]),
        SimpleNamespace(name="good", load=lambda: PluginRule),
    ]
    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: entries)
    assert set(registry.all_rules()) == {"vendor-X001"}
    diagnostics = registry.consume_diagnostics()
    assert len(diagnostics) == 1 and diagnostics[0].code == "LP101"
    assert "vendor-prefixed" in diagnostics[0].message
    assert not registry.consume_diagnostics()
    assert set(registry.all_rules()) == {"vendor-X001"}


def test_duplicate_plugin_and_non_iterable_errors(isolated, monkeypatch):
    entries = [
        SimpleNamespace(name="duplicate", load=lambda: lambda: [PluginRule, PluginRule]),
        SimpleNamespace(name="invalid", load=lambda: 7),
    ]
    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: entries)
    assert registry.all_rules() == {}
    assert len(registry.consume_diagnostics()) == 2


def test_import_failure_is_diagnostic(isolated, monkeypatch):
    monkeypatch.setattr(registry, "_BUILTIN_RULE_MODULES", ["nonexistent_leakproof_test_module"])
    assert registry.all_rules() == {}
    assert registry.consume_diagnostics()[0].code == "LP100"


def test_selection_overrides_and_default_disabled(isolated):
    class Disabled(PluginRule):
        enabled_by_default = False

    registry.register(Disabled)
    assert not registry.load_all(Config())
    cfg = Config()
    cfg.apply_cli(select=["ALL"])
    cfg.severity_overrides = {"vendor-X001": "low"}
    rules = registry.load_all(cfg)
    assert len(rules) == 1 and rules[0].severity is Severity.LOW
    assert not registry.load_all(cfg, layers=(Layer.DATA,))
    cfg.ignore = ["vendor-*"]
    assert not registry.load_all(cfg)
