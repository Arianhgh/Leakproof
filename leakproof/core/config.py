"""Configuration loading and merging.

Resolution order (later overrides earlier):
  built-in defaults -> pyproject.toml [tool.leakproof] -> leakproof.toml -> CLI flags
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from .models import Layer, Severity

DEFAULT_LAYERS = (Layer.STATIC, Layer.DATA, Layer.RUNTIME)


@dataclass
class DataConfig:
    neardup_text_threshold: float = 0.8
    neardup_distance: float = 0.05
    imbalance_ratio: float = 0.9
    sample_cap: int = 200_000
    target_leakage_score: float = 0.98  # univariate score above this is suspicious
    mi_zscore: float = 3.0  # MI z-score above the cohort flagged


@dataclass
class LLMConfig:
    enabled: bool = False
    provider: str = "anthropic"
    model: str | None = None


@dataclass
class Config:
    select: list[str] = field(default_factory=lambda: ["ALL"])
    ignore: list[str] = field(default_factory=list)
    fail_on: Severity = Severity.HIGH
    exclude: list[str] = field(
        default_factory=lambda: ["tests/", "examples/", ".venv/", "build/", "dist/"]
    )
    layers: list[Layer] = field(default_factory=lambda: list(DEFAULT_LAYERS))
    severity_overrides: dict[str, str] = field(default_factory=dict)
    data: DataConfig = field(default_factory=DataConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path | None = None, *, start: Path | None = None) -> Config:
        """Discover and merge config from disk.

        If ``path`` is given it is loaded as a leakproof.toml-style file.
        Otherwise we look for pyproject.toml then leakproof.toml walking up
        from ``start`` (default: cwd).
        """
        cfg = cls()
        start = start or Path.cwd()

        if path is not None:
            cfg._merge(_read_toml_table(path, explicit=True))
            return cfg

        root = _find_root(start)
        pyproject = root / "pyproject.toml"
        if pyproject.exists():
            data = _read_toml(pyproject)
            table = data.get("tool", {}).get("leakproof", {})
            cfg._merge(table)
        leakproof_toml = root / "leakproof.toml"
        if leakproof_toml.exists():
            cfg._merge(_read_toml_table(leakproof_toml, explicit=True))
        return cfg

    def _merge(self, table: dict) -> None:
        if not table:
            return
        if "select" in table:
            self.select = list(table["select"])
        if "ignore" in table:
            self.ignore = list(table["ignore"])
        if "fail_on" in table:
            self.fail_on = Severity.from_str(str(table["fail_on"]))
        if "exclude" in table:
            self.exclude = list(table["exclude"])
        if "layers" in table:
            self.layers = [Layer(layer) for layer in table["layers"]]
        if "severity" in table and isinstance(table["severity"], dict):
            self.severity_overrides.update({k: str(v) for k, v in table["severity"].items()})
        if "data" in table and isinstance(table["data"], dict):
            d = table["data"]
            for key in vars(self.data):
                if key in d:
                    setattr(self.data, key, type(getattr(self.data, key))(d[key]))
        if "llm" in table and isinstance(table["llm"], dict):
            ll = table["llm"]
            if "enabled" in ll:
                self.llm.enabled = bool(ll["enabled"])
            if "provider" in ll:
                self.llm.provider = str(ll["provider"])
            if "model" in ll:
                self.llm.model = str(ll["model"])

    # CLI overrides ----------------------------------------------------
    def apply_cli(
        self,
        *,
        select: list[str] | None = None,
        ignore: list[str] | None = None,
        layers: list[str] | None = None,
        fail_on: str | None = None,
        exclude: list[str] | None = None,
    ) -> None:
        if select:
            self.select = list(select)
        if ignore:
            self.ignore = list(self.ignore) + list(ignore)
        if layers:
            self.layers = [Layer(layer) for layer in layers]
        if fail_on:
            self.fail_on = Severity.from_str(fail_on)
        if exclude:
            self.exclude = list(self.exclude) + list(exclude)


def _read_toml(path: Path) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def _read_toml_table(path: Path, *, explicit: bool) -> dict:
    """Read a leakproof.toml.

    Accept either a top-level table or a [tool.leakproof] table (so the same
    file shape works in either location).
    """
    data = _read_toml(path)
    if "tool" in data and isinstance(data["tool"], dict) and "leakproof" in data["tool"]:
        return data["tool"]["leakproof"]
    return data


def _find_root(start: Path) -> Path:
    cur = start.resolve()
    for parent in [cur, *cur.parents]:
        if (parent / "pyproject.toml").exists() or (parent / "leakproof.toml").exists():
            return parent
        if (parent / ".git").exists():
            return parent
    return cur
