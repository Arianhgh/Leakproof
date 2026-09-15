"""Framework metadata used by static analysis and plugin adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any

from ...core.models import Diagnostic, DiagnosticLevel


@dataclass
class FrameworkAdapter:
    name: str
    split_functions: set[str] = field(default_factory=set)
    cv_splitters: set[str] = field(default_factory=set)
    group_cv_splitters: set[str] = field(default_factory=set)
    temporal_cv_splitters: set[str] = field(default_factory=set)
    transformers: set[str] = field(default_factory=set)
    stateless_transformers: set[str] = field(default_factory=set)
    estimators: set[str] = field(default_factory=set)
    resamplers: set[str] = field(default_factory=set)
    feature_selectors: set[str] = field(default_factory=set)
    learn_methods: set[str] = field(
        default_factory=lambda: {"fit", "fit_transform", "fit_resample", "partial_fit"}
    )
    apply_methods: set[str] = field(
        default_factory=lambda: {
            "transform",
            "predict",
            "predict_proba",
            "decision_function",
            "score",
        }
    )
    pipeline_constructors: set[str] = field(default_factory=set)
    search_constructors: set[str] = field(default_factory=set)
    metrics: dict[str, tuple[int, int]] = field(default_factory=dict)
    imbalance_sensitive_metrics: set[str] = field(default_factory=set)
    seeded_callables: set[str] = field(default_factory=set)


class AdapterRegistry:
    """Merged view with conservative name resolution and cached lookups."""

    def __init__(self, adapters: list[FrameworkAdapter]):
        self.adapters = list(adapters)
        self._cache: dict[tuple[str, str], bool] = {}
        self._sets: dict[str, set[str]] = {
            attr: self._union(attr)
            for attr in (
                "split_functions",
                "cv_splitters",
                "group_cv_splitters",
                "temporal_cv_splitters",
                "transformers",
                "stateless_transformers",
                "estimators",
                "resamplers",
                "feature_selectors",
                "pipeline_constructors",
                "search_constructors",
                "learn_methods",
                "apply_methods",
                "imbalance_sensitive_metrics",
                "seeded_callables",
            )
        }
        self._framework_roots = {adapter.name.split(".", 1)[0] for adapter in self.adapters}

    def _union(self, attr: str) -> set[str]:
        out: set[str] = set()
        for adapter in self.adapters:
            out |= set(getattr(adapter, attr))
        return out

    @staticmethod
    def _tail(name: str) -> str:
        return name.rsplit(".", 1)[-1]

    def _matches(self, attr: str, name: str) -> bool:
        key = (attr, name)
        if key in self._cache:
            return self._cache[key]
        values = self._sets[attr]
        if name in values:
            answer = True
        elif "." not in name:
            # Bare names are only accepted for plugin-provided exact symbols.
            answer = False
        else:
            root = name.split(".", 1)[0]
            tail = self._tail(name)
            answer = root in self._framework_roots and tail in {
                self._tail(value) for value in values
            }
        self._cache[key] = answer
        return answer

    def is_split_function(self, name: str) -> bool:
        return self._matches("split_functions", name)

    def is_cv_splitter(self, name: str) -> bool:
        return self._matches("cv_splitters", name)

    def is_group_cv_splitter(self, name: str) -> bool:
        return self._matches("group_cv_splitters", name)

    def is_temporal_cv_splitter(self, name: str) -> bool:
        return self._matches("temporal_cv_splitters", name)

    def is_transformer(self, name: str) -> bool:
        return self._matches("transformers", name)

    def is_stateless_transformer(self, name: str) -> bool:
        return self._matches("stateless_transformers", name)

    def is_estimator(self, name: str) -> bool:
        return self._matches("estimators", name)

    def is_resampler(self, name: str) -> bool:
        return self._matches("resamplers", name)

    def is_feature_selector(self, name: str) -> bool:
        return self._matches("feature_selectors", name)

    def is_pipeline_constructor(self, name: str) -> bool:
        return self._matches("pipeline_constructors", name)

    def is_search_constructor(self, name: str) -> bool:
        return self._matches("search_constructors", name)

    def is_learn_method(self, method: str) -> bool:
        return method in self._sets["learn_methods"]

    def is_apply_method(self, method: str) -> bool:
        return method in self._sets["apply_methods"]

    def is_seeded(self, name: str) -> bool:
        return self._matches("seeded_callables", name)

    @property
    def metrics(self) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        for adapter in self.adapters:
            for key, value in adapter.metrics.items():
                out[self._tail(key)] = value
        return out

    def is_imbalance_sensitive_metric(self, name: str) -> bool:
        return self._matches("imbalance_sensitive_metrics", name)


def _builtin_adapters() -> list[FrameworkAdapter]:
    from .imblearn import adapter as imblearn_adapter
    from .keras import adapter as keras_adapter
    from .lightgbm import adapter as lgbm_adapter
    from .numpy import adapter as numpy_adapter
    from .pandas import adapter as pandas_adapter
    from .sklearn import adapter as sklearn_adapter
    from .torch import adapter as torch_adapter
    from .xgboost import adapter as xgboost_adapter

    return [
        sklearn_adapter(),
        pandas_adapter(),
        numpy_adapter(),
        imblearn_adapter(),
        xgboost_adapter(),
        lgbm_adapter(),
        torch_adapter(),
        keras_adapter(),
    ]


_LAST_DIAGNOSTICS: list[Diagnostic] = []


def consume_adapter_diagnostics() -> list[Diagnostic]:
    diagnostics = list(_LAST_DIAGNOSTICS)
    _LAST_DIAGNOSTICS.clear()
    return diagnostics


def load_adapters() -> AdapterRegistry:
    adapters = _builtin_adapters()
    for ep in _entry_points_for("ml_leakproof.adapters"):
        try:
            loaded = ep.load()
            produced = loaded() if callable(loaded) else loaded
            if not isinstance(produced, FrameworkAdapter):
                raise TypeError("adapter entry point must return FrameworkAdapter")
            if not produced.name or not isinstance(produced.name, str):
                raise ValueError("adapter name must be a non-empty string")
            if any(existing.name == produced.name for existing in adapters):
                raise ValueError(f"duplicate adapter name {produced.name}")
            adapters.append(produced)
        except Exception as exc:
            _LAST_DIAGNOSTICS.append(
                Diagnostic(
                    code="LP102",
                    message=f"failed to load adapter plugin {ep.name}: {exc}",
                    level=DiagnosticLevel.ERROR,
                    details={"plugin": ep.name, "error_type": type(exc).__name__},
                )
            )
    return AdapterRegistry(adapters)


def _entry_points_for(group: str) -> list[Any]:
    try:
        return list(entry_points(group=group))
    except TypeError:  # pragma: no cover
        return list(entry_points().select(group=group))


def register_adapter(factory: Any) -> Any:
    """Decorator/helper retained for plugin authors."""

    return factory
