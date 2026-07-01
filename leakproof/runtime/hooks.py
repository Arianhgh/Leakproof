"""Runtime hooks for split/fit/score row tracking."""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable


def _call_site() -> str:
    try:
        for frame in inspect.stack()[2:]:
            fn = frame.filename
            if "leakproof" + "/" not in fn.replace("\\", "/") and "/wrapt/" not in fn:
                return f"{fn}:{frame.lineno}"
    except Exception:
        pass
    return "<unknown>"


class HookManager:
    def __init__(self, session: RuntimeSession):  # type: ignore[name-defined]  # noqa: F821
        self.session = session
        self._patches: list[tuple[object, str, object]] = []

    def install(self) -> None:
        try:
            import wrapt
        except Exception:
            warnings.warn("leakproof: wrapt not installed; runtime layer disabled", stacklevel=2)
            return
        try:
            import sklearn.base  # noqa: F401
            import sklearn.model_selection as ms
        except Exception:
            warnings.warn("leakproof: scikit-learn not installed; runtime layer disabled", stacklevel=2)
            return

        self._wrap(wrapt, ms, "train_test_split", self._on_split)
        self._wrap_method(wrapt, "sklearn.base", "BaseEstimator", "fit", self._on_fit)
        for cls_path, method, handler in [
            ("sklearn.base.TransformerMixin", "fit_transform", self._on_fit),
            ("sklearn.base.ClassifierMixin", "score", self._on_score),
            ("sklearn.base.RegressorMixin", "score", self._on_score),
            ("sklearn.pipeline.Pipeline", "fit", self._on_fit),
            ("sklearn.pipeline.Pipeline", "fit_transform", self._on_fit),
            ("sklearn.pipeline.Pipeline", "score", self._on_score),
            ("imblearn.pipeline.Pipeline", "fit", self._on_fit),
            ("imblearn.pipeline.Pipeline", "fit_resample", self._on_fit),
            ("xgboost.sklearn.XGBClassifier", "fit", self._on_fit),
            ("xgboost.sklearn.XGBClassifier", "score", self._on_score),
            ("xgboost.sklearn.XGBRegressor", "fit", self._on_fit),
            ("xgboost.sklearn.XGBRegressor", "score", self._on_score),
            ("lightgbm.sklearn.LGBMClassifier", "fit", self._on_fit),
            ("lightgbm.sklearn.LGBMClassifier", "score", self._on_score),
            ("lightgbm.sklearn.LGBMRegressor", "fit", self._on_fit),
            ("lightgbm.sklearn.LGBMRegressor", "score", self._on_score),
        ]:
            mod, cls = cls_path.rsplit(".", 1)
            self._wrap_method(wrapt, mod, cls, method, handler)

    def _wrap(self, wrapt, module, name: str, handler: Callable) -> None:
        try:
            original = getattr(module, name)

            def wrapper(wrapped, instance, args, kwargs):
                result = wrapped(*args, **kwargs)
                try:
                    handler(args, kwargs, result, None)
                except Exception:
                    pass
                return result

            wrapt.wrap_function_wrapper(module, name, wrapper)
            self._patches.append((module, name, original))
        except Exception:
            pass

    def _wrap_method(self, wrapt, module_path: str, cls_name: str, method: str, handler: Callable) -> None:
        try:
            import importlib

            module = importlib.import_module(module_path)
            cls = getattr(module, cls_name, None)
            if cls is None or method not in cls.__dict__:
                return
            original = cls.__dict__[method]

            def wrapper(wrapped, instance, args, kwargs):
                result = wrapped(*args, **kwargs)
                try:
                    handler(args, kwargs, result, instance)
                except Exception:
                    pass
                return result

            wrapt.wrap_function_wrapper(cls, method, wrapper)
            self._patches.append((cls, method, original))
        except Exception:
            pass

    def remove(self) -> None:
        for target, name, original in reversed(self._patches):
            try:
                setattr(target, name, original)
            except Exception:
                pass
        self._patches.clear()

    def _on_split(self, args, kwargs, result, instance) -> None:
        from .taint import row_hashes

        if not args:
            return
        args[0]
        if not isinstance(result, (list, tuple)) or len(result) < 2:
            return
        x_train, x_test = result[0], result[1]
        train_h = row_hashes(x_train)
        test_h = row_hashes(x_test)
        self.session.on_split(_call_site(), train_h, test_h)

    def _on_fit(self, args, kwargs, result, instance) -> None:
        from .taint import row_hashes

        X = args[0] if args else None
        if X is None:
            return
        seen = row_hashes(X)
        cls = type(instance).__name__ if instance is not None else "estimator"
        self.session.on_fit(_call_site(), cls, seen)

    def _on_score(self, args, kwargs, result, instance) -> None:
        from .taint import row_hashes

        X = args[0] if args else None
        if X is None:
            return
        seen = row_hashes(X)
        cls = type(instance).__name__ if instance is not None else "estimator"
        self.session.on_score(_call_site(), cls, seen)
