"""Rule and adapter discovery with isolated instances and atomic plugins."""

from __future__ import annotations

import fnmatch
import importlib
from collections.abc import Callable, Iterable
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

from .models import Category, Diagnostic, DiagnosticLevel, Layer, Severity
from .rule import Rule

if TYPE_CHECKING:
    from .config import Config

RuleFactory = Callable[[], Rule]

_REGISTRY: dict[str, RuleFactory] = {}
_ENTRYPOINTS_LOADED = False
_DIAGNOSTICS: list[Diagnostic] = []
_BUILTIN_RULE_MODULES = [
    "ml_leakproof.static.rules",
    "ml_leakproof.data.rules",
    "ml_leakproof.runtime.rules",
]


def _factory_for(value: Any) -> RuleFactory:
    if isinstance(value, type) and issubclass(value, Rule):
        return value
    if isinstance(value, Rule):
        def fresh(value: Rule = value) -> Rule:
            return type(value)()

        return fresh
    if callable(value):
        def factory(value: Any = value) -> Rule:
            produced = value()
            if isinstance(produced, Rule):
                return produced
            if isinstance(produced, type) and issubclass(produced, Rule):
                return produced()
            raise TypeError("rule factory must return a Rule instance or Rule class")

        return factory
    raise TypeError("rule registration must be a Rule class, instance, or factory")


def _validate_rule(rule: Rule, *, plugin: str | None = None) -> None:
    if not isinstance(rule, Rule):
        raise TypeError("registered value is not a Rule")
    if not isinstance(rule.id, str) or not rule.id.strip():
        raise ValueError("rule id must be a non-empty string")
    if plugin and "-" not in rule.id:
        raise ValueError(f"plugin rule id must be vendor-prefixed: {rule.id}")
    if not isinstance(rule.name, str) or not rule.name:
        raise ValueError(f"rule {rule.id} has no name")
    if not isinstance(rule.category, Category):
        raise ValueError(f"rule {rule.id} has no category")
    if not isinstance(rule.severity, Severity):
        raise ValueError(f"rule {rule.id} has invalid severity")
    if not isinstance(rule.layers, tuple) or not rule.layers:
        raise ValueError(f"rule {rule.id} must declare at least one layer")
    if not all(isinstance(layer, Layer) for layer in rule.layers):
        raise ValueError(f"rule {rule.id} has invalid layers")
    if not isinstance(rule.advisory_only, bool):
        raise ValueError(f"rule {rule.id} has invalid advisory_only metadata")
    for layer in rule.layers:
        if layer is Layer.STATIC and not hasattr(rule, "check"):
            raise ValueError(f"rule {rule.id} declares static but has no check method")
        if layer is Layer.DATA and not hasattr(rule, "check"):
            raise ValueError(f"rule {rule.id} declares data but has no check method")
        if layer is Layer.RUNTIME and not hasattr(rule, "on_event"):
            raise ValueError(f"rule {rule.id} declares runtime but has no on_event method")


def register(value: Any) -> Any:
    """Register a built-in Rule class without sharing mutable instances."""

    factory = _factory_for(value)
    instance = factory()
    _validate_rule(instance)
    if instance.id in _REGISTRY:
        raise AssertionError(f"duplicate rule id {instance.id}")
    _REGISTRY[instance.id] = factory
    return value


def _diagnostic(code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
    _DIAGNOSTICS.append(
        Diagnostic(
            code=code,
            message=message,
            level=DiagnosticLevel.ERROR,
            details=details or {},
        )
    )


def consume_diagnostics() -> list[Diagnostic]:
    diagnostics = list(_DIAGNOSTICS)
    _DIAGNOSTICS.clear()
    return diagnostics


def _import_builtins() -> None:
    for module_name in _BUILTIN_RULE_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            _diagnostic(
                "LP100",
                f"could not import built-in rule module {module_name}: {exc}",
                details={"module": module_name, "error_type": type(exc).__name__},
            )


def _entry_points_for(group: str) -> list[Any]:
    try:
        return list(entry_points(group=group))
    except TypeError:  # pragma: no cover - Python 3.10 compatibility
        return list(entry_points().select(group=group))


def _load_entrypoints() -> None:
    global _ENTRYPOINTS_LOADED
    if _ENTRYPOINTS_LOADED:
        return
    _ENTRYPOINTS_LOADED = True
    for ep in _entry_points_for("ml_leakproof.rules"):
        try:
            loaded = ep.load()
            produced = loaded() if callable(loaded) else loaded
            if isinstance(produced, Rule) or isinstance(produced, type):
                produced = [produced]
            if not isinstance(produced, Iterable):
                raise TypeError("plugin entry point must return an iterable of rules")
            candidate: list[tuple[str, RuleFactory]] = []
            for value in produced:
                factory = _factory_for(value)
                rule = factory()
                _validate_rule(rule, plugin=ep.name)
                if rule.id in _REGISTRY or any(rule.id == rid for rid, _ in candidate):
                    raise ValueError(f"duplicate rule id {rule.id} from plugin {ep.name}")
                candidate.append((rule.id, factory))
            # Commit only after every registration in this entry point passes.
            _REGISTRY.update(dict(candidate))
        except Exception as exc:
            _diagnostic(
                "LP101",
                f"failed to load rule plugin {ep.name}: {exc}",
                details={"plugin": ep.name, "error_type": type(exc).__name__},
            )


def all_rules() -> dict[str, Rule]:
    """Return fresh rule instances for inspection or a command listing."""

    _import_builtins()
    _load_entrypoints()
    return {rule_id: factory() for rule_id, factory in _REGISTRY.items()}


def _matches_any(rule_id: str, patterns: list[str]) -> bool:
    return any(pattern == "ALL" or fnmatch.fnmatchcase(rule_id, pattern) for pattern in patterns)


def load_all(
    config: Config | None = None,
    *,
    layers: tuple[Layer, ...] | None = None,
) -> list[Rule]:
    """Resolve active rules and instantiate each one for this analysis only."""

    from .config import Config

    cfg = config or Config()
    cfg.validate()
    _import_builtins()
    _load_entrypoints()
    active_layers = tuple(cfg.layers) if layers is None else tuple(layers)
    selected: list[Rule] = []

    for rule_id, factory in _REGISTRY.items():
        selected_by_pattern = _matches_any(rule_id, cfg.select)
        if not selected_by_pattern:
            continue
        if _matches_any(rule_id, cfg.ignore):
            continue
        # A default ALL means "all enabled-by-default rules".  An explicit ALL
        # is an opt-in to disabled plugin rules too.
        if not rule_id or (not cfg.explicit_select_all and not factory().enabled_by_default):
            continue
        if not active_layers:
            continue
        rule = factory()
        if not any(layer in active_layers for layer in rule.layers):
            continue
        override = cfg.severity_overrides.get(rule.id)
        if override is not None:
            rule.severity = Severity.from_str(override)
        selected.append(rule)
    selected.sort(key=lambda rule: rule.id)
    return selected
