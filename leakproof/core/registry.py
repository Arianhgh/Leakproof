"""Rule and adapter discovery via decorators + entry points."""

from __future__ import annotations

import fnmatch
import importlib
from typing import TYPE_CHECKING

from .models import Layer, Severity
from .rule import Rule

if TYPE_CHECKING:
    from .config import Config

from importlib.metadata import entry_points  # type: ignore

_REGISTRY: dict[str, Rule] = {}
_ENTRYPOINTS_LOADED = False

# Modules that register built-in rules via the @register decorator on import.
_BUILTIN_RULE_MODULES = [
    "leakproof.static.rules",
    "leakproof.data.rules",
    "leakproof.runtime.rules",
]


def register(rule_cls: type[Rule]) -> type[Rule]:
    """Class decorator that instantiates and registers a rule."""
    inst = rule_cls()
    if inst.id in _REGISTRY:
        raise AssertionError(f"duplicate rule id {inst.id}")
    _REGISTRY[inst.id] = inst
    return rule_cls


def _import_builtins() -> None:
    for mod in _BUILTIN_RULE_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as exc:  # pragma: no cover - defensive
            # A missing optional dependency (e.g. pandas for data rules) must not
            # break the static layer. Surface a debug-friendly warning.
            import warnings

            warnings.warn(f"leakproof: could not import {mod}: {exc}", stacklevel=2)


def _load_entrypoints() -> None:
    global _ENTRYPOINTS_LOADED
    if _ENTRYPOINTS_LOADED:
        return
    _ENTRYPOINTS_LOADED = True
    try:
        eps = entry_points(group="leakproof.rules")
    except TypeError:  # pragma: no cover - py<3.10 compat path
        eps = entry_points().get("leakproof.rules", [])  # type: ignore
    for ep in eps:
        try:
            factory = ep.load()
            produced = factory()
            for rule in produced:
                if rule.id in _REGISTRY:
                    raise AssertionError(f"duplicate rule id {rule.id} from plugin {ep.name}")
                _REGISTRY[rule.id] = rule
        except Exception as exc:  # pragma: no cover - defensive
            import warnings

            warnings.warn(f"leakproof: failed to load rule plugin {ep.name}: {exc}", stacklevel=2)


def all_rules() -> dict[str, Rule]:
    """Return every known rule (built-in + plugins), importing on first call."""
    _import_builtins()
    _load_entrypoints()
    return dict(_REGISTRY)


def _matches_any(rule_id: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if pat == "ALL":
            return True
        if fnmatch.fnmatch(rule_id, pat):
            return True
    return False


def load_all(config: Config | None = None, *, layers: tuple[Layer, ...] | None = None) -> list[Rule]:
    """Resolve the active rule set given a config.

    Applies select/ignore globs, enabled_by_default, layer filtering, and
    per-rule severity overrides.
    """
    from .config import Config

    cfg = config or Config()
    rules = all_rules()
    selected: list[Rule] = []

    select = cfg.select or ["ALL"]
    ignore = cfg.ignore or []
    active_layers = layers if layers is not None else tuple(cfg.layers)

    for rule in rules.values():
        if not _matches_any(rule.id, select):
            continue
        if _matches_any(rule.id, ignore):
            continue
        if not rule.enabled_by_default and not _matches_any(rule.id, select):
            continue
        if active_layers and not any(layer in active_layers for layer in rule.layers):
            continue
        # severity override
        override = cfg.severity_overrides.get(rule.id)
        if override is not None:
            rule.severity = Severity.from_str(override)
        selected.append(rule)

    selected.sort(key=lambda r: r.id)
    return selected
