"""Validated configuration and configuration-file precedence."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from .models import Layer, Severity

DEFAULT_LAYERS = (Layer.STATIC, Layer.DATA, Layer.RUNTIME)
PROFILES = {"ci", "notebook", "research"}
PROFILE_DEFAULTS: dict[str, dict[str, float]] = {
    "ci": {"min_confidence": 0.60, "gate_confidence": 0.75},
    "notebook": {"min_confidence": 0.30, "gate_confidence": 0.75},
    "research": {"min_confidence": 0.00, "gate_confidence": 0.75},
}
VALID_PROVIDERS = {"anthropic", "openai", "ollama"}


class ConfigError(ValueError):
    """Raised for invalid configuration, including invalid programmatic values."""


@dataclass
class DataConfig:
    neardup_text_threshold: float = 0.8
    neardup_distance: float = 0.05
    imagehash_distance: int = 4
    imbalance_ratio: float = 0.9
    sample_cap: int = 200_000
    target_leakage_score: float = 0.98
    mi_floor: float = 0.2
    mi_dominance_ratio: float = 1.5
    hash_include: list[str] = field(default_factory=list)
    hash_exclude: list[str] = field(default_factory=list)
    hash_chunk_size: int = 50_000
    max_examples: int = 10
    max_neardup_candidates: int = 100_000
    max_neardup_pairs: int = 100_000
    deterministic_sampling: bool = False
    probe_cv_folds: int = 5


@dataclass
class LLMConfig:
    enabled: bool = False
    provider: str | None = None
    model: str | None = None
    timeout_seconds: float = 20.0
    retries: int = 2
    max_context_chars: int = 12_000


@dataclass
class Config:
    # ["ALL"] is the public default, while _select_explicit distinguishes it
    # from an explicit ALL in a project file.  This lets disabled-by-default
    # plugin rules remain opt-in unless the user explicitly selects ALL.
    select: list[str] = field(default_factory=lambda: ["ALL"])
    ignore: list[str] = field(default_factory=list)
    fail_on: Severity = Severity.HIGH
    min_confidence: float = 0.60
    gate_confidence: float = 0.75
    profile: str = "ci"
    exclude: list[str] = field(
        default_factory=lambda: ["tests/", "examples/", ".venv/", "build/", "dist/"]
    )
    layers: list[Layer] = field(default_factory=lambda: list(DEFAULT_LAYERS))
    severity_overrides: dict[str, str] = field(default_factory=dict)
    data: DataConfig = field(default_factory=DataConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    allow_partial: bool = False
    root: Path | None = None
    _select_explicit: bool = field(default=False, init=False, repr=False)
    _min_confidence_explicit: bool = field(default=False, init=False, repr=False)
    _gate_confidence_explicit: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        # A caller may construct Config(profile="research") directly.
        self.profile = _profile(self.profile)
        if self.profile != "ci" and self.min_confidence == PROFILE_DEFAULTS["ci"]["min_confidence"]:
            self.min_confidence = PROFILE_DEFAULTS[self.profile]["min_confidence"]
        self.validate()

    @classmethod
    def load(
        cls,
        path: Path | None = None,
        *,
        start: Path | None = None,
        root: Path | None = None,
    ) -> Config:
        """Load built-ins, profile defaults, project config, then explicit config.

        ``leakproof.toml`` and ``[tool.leakproof]`` remain the supported project
        configuration names.  An explicit file is merged after both project
        sources, which makes command-line/config-file precedence predictable.
        """

        start = (root or start or Path.cwd()).resolve()
        project_root = _find_root(start)
        cfg = cls(root=project_root)

        # Project-level configuration sources are intentionally deterministic:
        # pyproject first, then the dedicated file.
        pyproject = project_root / "pyproject.toml"
        if pyproject.exists():
            data = _read_toml(pyproject)
            tool = data.get("tool", {})
            if not isinstance(tool, dict):
                raise ConfigError("[tool] must be a table in pyproject.toml")
            table = tool.get("leakproof", {})
            if table and not isinstance(table, dict):
                raise ConfigError("[tool.leakproof] must be a table")
            cfg._merge(table, source=pyproject)

        project_file = project_root / "leakproof.toml"
        if project_file.exists():
            cfg._merge(_read_toml_table(project_file), source=project_file)

        if path is not None:
            explicit = path.expanduser().resolve()
            if not explicit.is_file():
                raise ConfigError(f"configuration file does not exist: {path}")
            cfg._merge(_read_toml_table(explicit), source=explicit)
        cfg.validate()
        return cfg

    def copy(self) -> Config:
        """Return an isolated configuration suitable for a new analysis."""

        import copy

        return copy.deepcopy(self)

    def validate(self) -> None:
        self.profile = _profile(self.profile)
        if not isinstance(self.select, list) or not all(isinstance(v, str) for v in self.select):
            raise ConfigError("select must be a list of rule-id glob strings")
        if not isinstance(self.ignore, list) or not all(isinstance(v, str) for v in self.ignore):
            raise ConfigError("ignore must be a list of rule-id glob strings")
        if not isinstance(self.exclude, list) or not all(isinstance(v, str) for v in self.exclude):
            raise ConfigError("exclude must be a list of path strings")
        if not isinstance(self.layers, list) or not all(isinstance(v, Layer) for v in self.layers):
            raise ConfigError("layers must be a list containing static, data, or runtime")
        if not isinstance(self.fail_on, Severity):
            if not isinstance(self.fail_on, str):
                raise ConfigError("fail_on must be a severity string")
            self.fail_on = Severity.from_str(self.fail_on)
        _confidence(self.min_confidence, "min_confidence")
        _confidence(self.gate_confidence, "gate_confidence")
        if not isinstance(self.allow_partial, bool):
            raise ConfigError("allow_partial must be a boolean")
        if not isinstance(self.severity_overrides, dict):
            raise ConfigError("severity must be a mapping of rule ID to severity")
        for rule_id, severity in self.severity_overrides.items():
            if not isinstance(rule_id, str) or not rule_id:
                raise ConfigError("severity override IDs must be non-empty strings")
            if not isinstance(severity, str):
                raise ConfigError("severity override values must be severity strings")
            Severity.from_str(severity)
        if not isinstance(self.data, DataConfig):
            raise ConfigError("data must be a DataConfig instance")
        if not isinstance(self.llm, LLMConfig):
            raise ConfigError("llm must be an LLMConfig instance")
        _validate_data(self.data)
        _validate_llm(self.llm)

    def _merge(self, table: dict[str, Any], *, source: Path | None = None) -> None:
        if not isinstance(table, dict):
            raise ConfigError(f"configuration must be a table: {source or '<programmatic>'}")
        allowed = {
            "select",
            "ignore",
            "fail_on",
            "min_confidence",
            "gate_confidence",
            "profile",
            "exclude",
            "layers",
            "severity",
            "data",
            "llm",
            "allow_partial",
        }
        unknown = sorted(set(table) - allowed)
        if unknown:
            where = f" in {source}" if source else ""
            raise ConfigError(f"unknown configuration key(s){where}: {', '.join(unknown)}")

        if "profile" in table:
            self.profile = _profile(table["profile"])
            if not self._min_confidence_explicit:
                self.min_confidence = PROFILE_DEFAULTS[self.profile]["min_confidence"]
            if not self._gate_confidence_explicit:
                self.gate_confidence = PROFILE_DEFAULTS[self.profile]["gate_confidence"]
        if "select" in table:
            self.select = _string_list(table["select"], "select")
            self._select_explicit = True
        if "ignore" in table:
            self.ignore = _string_list(table["ignore"], "ignore")
        if "fail_on" in table:
            self.fail_on = Severity.from_str(table["fail_on"])
        if "min_confidence" in table:
            self.min_confidence = _confidence(table["min_confidence"], "min_confidence")
            self._min_confidence_explicit = True
        if "gate_confidence" in table:
            self.gate_confidence = _confidence(table["gate_confidence"], "gate_confidence")
            self._gate_confidence_explicit = True
        if "exclude" in table:
            self.exclude = _string_list(table["exclude"], "exclude")
        if "layers" in table:
            raw_layers = _string_list(table["layers"], "layers")
            try:
                self.layers = [Layer(value) for value in raw_layers]
            except ValueError as exc:
                raise ConfigError("layers must contain only static, data, or runtime") from exc
        if "severity" in table:
            if not isinstance(table["severity"], dict):
                raise ConfigError("severity must be a table of rule ID to severity")
            if any(not isinstance(key, str) or not key for key in table["severity"]):
                raise ConfigError("severity override IDs must be non-empty strings")
            if any(not isinstance(value, str) for value in table["severity"].values()):
                raise ConfigError("severity override values must be severity strings")
            self.severity_overrides.update(
                {key: Severity.from_str(value).value for key, value in table["severity"].items()}
            )
        if "data" in table:
            _merge_dataclass(self.data, table["data"], "data")
        if "llm" in table:
            _merge_dataclass(self.llm, table["llm"], "llm")
        if "allow_partial" in table:
            if not isinstance(table["allow_partial"], bool):
                raise ConfigError("allow_partial must be a boolean")
            self.allow_partial = table["allow_partial"]
        self.validate()

    def apply_cli(
        self,
        *,
        select: list[str] | None = None,
        ignore: list[str] | None = None,
        layers: list[str] | None = None,
        fail_on: str | None = None,
        exclude: list[str] | None = None,
        min_confidence: float | None = None,
        gate_confidence: float | None = None,
        profile: str | None = None,
        allow_partial: bool | None = None,
    ) -> None:
        if select is not None:
            self.select = _string_list(select, "select")
            self._select_explicit = True
        if ignore is not None:
            self.ignore = self.ignore + _string_list(ignore, "ignore")
        if layers is not None:
            values = _string_list(layers, "layers")
            try:
                self.layers = [Layer(value) for value in values]
            except ValueError as exc:
                raise ConfigError("layers must contain only static, data, or runtime") from exc
        if fail_on is not None:
            self.fail_on = Severity.from_str(fail_on)
        if min_confidence is not None:
            self.min_confidence = _confidence(min_confidence, "min_confidence")
            self._min_confidence_explicit = True
        if gate_confidence is not None:
            self.gate_confidence = _confidence(gate_confidence, "gate_confidence")
            self._gate_confidence_explicit = True
        if profile is not None:
            self.profile = _profile(profile)
            if min_confidence is None and not self._min_confidence_explicit:
                self.min_confidence = PROFILE_DEFAULTS[self.profile]["min_confidence"]
            if gate_confidence is None and not self._gate_confidence_explicit:
                self.gate_confidence = PROFILE_DEFAULTS[self.profile]["gate_confidence"]
        if exclude is not None:
            self.exclude = self.exclude + _string_list(exclude, "exclude")
        if allow_partial is not None:
            if not isinstance(allow_partial, bool):
                raise ConfigError("allow_partial must be a boolean")
            self.allow_partial = allow_partial
        self.validate()

    @property
    def explicit_select_all(self) -> bool:
        return self._select_explicit and "ALL" in self.select


def _merge_dataclass(target: Any, values: Any, name: str) -> None:
    if not isinstance(values, dict):
        raise ConfigError(f"{name} must be a table")
    allowed = set(vars(target))
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ConfigError(f"unknown {name} key(s): {', '.join(unknown)}")
    for key, value in values.items():
        current = getattr(target, key)
        expected = type(current)
        if expected is bool:
            if not isinstance(value, bool):
                raise ConfigError(f"{name}.{key} must be a boolean")
        elif expected is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"{name}.{key} must be an integer")
        elif expected is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"{name}.{key} must be a finite number")
            value = float(value)
        elif expected is list:
            value = _string_list(value, f"{name}.{key}")
        elif current is None:
            if value is not None and not isinstance(value, str):
                raise ConfigError(f"{name}.{key} has invalid type")
        elif value is not None and not isinstance(value, expected):
            raise ConfigError(f"{name}.{key} has invalid type")
        setattr(target, key, value)


def _validate_data(data: DataConfig) -> None:
    for name in ("neardup_text_threshold", "neardup_distance", "imbalance_ratio", "target_leakage_score"):
        value = getattr(data, name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ConfigError(f"data.{name} must be finite")
    for name in ("neardup_text_threshold", "neardup_distance", "imbalance_ratio", "target_leakage_score"):
        value = getattr(data, name)
        if not 0 <= value <= 1:
            raise ConfigError(f"data.{name} must be between 0 and 1")
    if data.mi_floor < 0 or not math.isfinite(data.mi_floor):
        raise ConfigError("data.mi_floor must be finite and non-negative")
    if data.mi_dominance_ratio < 1 or not math.isfinite(data.mi_dominance_ratio):
        raise ConfigError("data.mi_dominance_ratio must be finite and at least 1")
    for name in (
        "imagehash_distance",
        "sample_cap",
        "hash_chunk_size",
        "max_examples",
        "max_neardup_candidates",
        "max_neardup_pairs",
        "probe_cv_folds",
    ):
        value = getattr(data, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(f"data.{name} must be a positive integer")
    for name in ("hash_include", "hash_exclude"):
        _string_list(getattr(data, name), f"data.{name}")
    if not isinstance(data.deterministic_sampling, bool):
        raise ConfigError("data.deterministic_sampling must be a boolean")


def _validate_llm(llm: LLMConfig) -> None:
    if not isinstance(llm.enabled, bool):
        raise ConfigError("llm.enabled must be a boolean")
    if llm.provider is not None:
        if not isinstance(llm.provider, str) or llm.provider.lower() not in VALID_PROVIDERS:
            raise ConfigError("llm.provider must be one of: anthropic, openai, ollama")
        llm.provider = llm.provider.lower()
    if llm.enabled and (not llm.provider or not llm.model):
        raise ConfigError("llm.enabled requires explicit llm.provider and llm.model")
    if llm.model is not None and not isinstance(llm.model, str):
        raise ConfigError("llm.model must be a string or null")
    if not isinstance(llm.timeout_seconds, (int, float)) or isinstance(llm.timeout_seconds, bool):
        raise ConfigError("llm.timeout_seconds must be a finite positive number")
    if not math.isfinite(float(llm.timeout_seconds)) or llm.timeout_seconds <= 0:
        raise ConfigError("llm.timeout_seconds must be a finite positive number")
    if isinstance(llm.retries, bool) or not isinstance(llm.retries, int) or llm.retries < 0:
        raise ConfigError("llm.retries must be a non-negative integer")
    if isinstance(llm.max_context_chars, bool) or not isinstance(llm.max_context_chars, int) or llm.max_context_chars <= 0:
        raise ConfigError("llm.max_context_chars must be a positive integer")


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            value = tomllib.load(fh)
    except OSError as exc:
        raise ConfigError(f"could not read configuration {path}: {exc}") from exc
    except Exception as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"configuration is not a TOML table: {path}")
    return value


def _read_toml_table(path: Path) -> dict[str, Any]:
    data = _read_toml(path)
    if "tool" in data:
        tool = data.get("tool")
        if not isinstance(tool, dict) or "leakproof" not in tool:
            raise ConfigError(f"explicit config {path} has no [tool.leakproof] table")
        table = tool["leakproof"]
    else:
        table = data
    if not isinstance(table, dict):
        raise ConfigError(f"configuration table is invalid: {path}")
    return table


def _find_root(start: Path) -> Path:
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    for parent in (cur, *cur.parents):
        if (parent / "pyproject.toml").exists() or (parent / "leakproof.toml").exists():
            return parent
        if (parent / ".git").exists():
            return parent
    return cur


def _confidence(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be a finite number between 0 and 1")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ConfigError(f"{field} must be between 0 and 1")
    return normalized


def _profile(value: Any) -> str:
    if not isinstance(value, str):
        raise ConfigError("profile must be a string")
    profile = value.strip().lower()
    if profile not in PROFILES:
        raise ConfigError(f"profile must be one of: {', '.join(sorted(PROFILES))}")
    return profile


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{field_name} must be a list of strings")
    return list(value)
