"""Framework knowledge used by the static engine.

Adapters keep framework-specific facts out of the rules: which callables split data,
which classes learn from data, which wrappers make a transformation safe, and how metric
functions order their arguments. Third-party packages can add adapters through the
``leakproof.adapters`` entry-point group.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from importlib.metadata import entry_points


@dataclass
class FrameworkAdapter:
    """Symbol metadata contributed by one library or plugin."""

    name: str
    split_functions: set[str] = field(default_factory=set)
    cv_splitters: set[str] = field(default_factory=set)
    group_cv_splitters: set[str] = field(default_factory=set)
    temporal_cv_splitters: set[str] = field(default_factory=set)
    transformers: set[str] = field(default_factory=set)
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
    """Merged view over all loaded adapters with fast membership helpers."""

    def __init__(self, adapters: list[FrameworkAdapter]):
        self.adapters = adapters

    def _union(self, attr: str) -> set[str]:
        out: set[str] = set()
        for a in self.adapters:
            out |= getattr(a, attr)
        return out

    def _union_dict(self, attr: str) -> dict:
        out: dict = {}
        for a in self.adapters:
            out.update(getattr(a, attr))
        return out

    @staticmethod
    def _tail(name: str) -> str:
        return name.rsplit(".", 1)[-1]

    def is_split_function(self, name: str) -> bool:
        names = self._union("split_functions")
        return name in names or self._tail(name) in {self._tail(n) for n in names}

    def is_cv_splitter(self, name: str) -> bool:
        return self._tail(name) in {self._tail(n) for n in self._union("cv_splitters")}

    def is_group_cv_splitter(self, name: str) -> bool:
        return self._tail(name) in {self._tail(n) for n in self._union("group_cv_splitters")}

    def is_temporal_cv_splitter(self, name: str) -> bool:
        return self._tail(name) in {self._tail(n) for n in self._union("temporal_cv_splitters")}

    def is_transformer(self, name: str) -> bool:
        return self._tail(name) in {self._tail(n) for n in self._union("transformers")}

    def is_estimator(self, name: str) -> bool:
        return self._tail(name) in {self._tail(n) for n in self._union("estimators")}

    def is_resampler(self, name: str) -> bool:
        return self._tail(name) in {self._tail(n) for n in self._union("resamplers")}

    def is_feature_selector(self, name: str) -> bool:
        return self._tail(name) in {self._tail(n) for n in self._union("feature_selectors")}

    def is_pipeline_constructor(self, name: str) -> bool:
        return self._tail(name) in {self._tail(n) for n in self._union("pipeline_constructors")}

    def is_search_constructor(self, name: str) -> bool:
        return self._tail(name) in {self._tail(n) for n in self._union("search_constructors")}

    def is_learn_method(self, method: str) -> bool:
        return method in self._union("learn_methods")

    def is_apply_method(self, method: str) -> bool:
        return method in self._union("apply_methods")

    def is_seeded(self, name: str) -> bool:
        return self._tail(name) in {self._tail(n) for n in self._union("seeded_callables")}

    @property
    def metrics(self) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        for a in self.adapters:
            for k, v in a.metrics.items():
                out[self._tail(k)] = v
        return out

    def is_imbalance_sensitive_metric(self, name: str) -> bool:
        return self._tail(name) in {
            self._tail(n) for n in self._union("imbalance_sensitive_metrics")
        }


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


def load_adapters() -> AdapterRegistry:
    adapters = _builtin_adapters()
    try:
        eps = entry_points(group="leakproof.adapters")
    except TypeError:  # pragma: no cover
        eps = entry_points().get("leakproof.adapters", [])  # type: ignore
    for ep in eps:
        try:
            factory = ep.load()
            adapters.append(factory())
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"leakproof: failed to load adapter {ep.name}: {exc}", stacklevel=2)
    return AdapterRegistry(adapters)


def register_adapter(factory):
    """Decorator/helper for plugin authors (used with the entry point)."""
    return factory
