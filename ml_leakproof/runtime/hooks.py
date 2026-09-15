"""Transactional runtime hooks for supported sklearn-family operations."""

from __future__ import annotations

import contextvars
import copy
import functools
import importlib
import inspect
import threading
from collections.abc import Callable
from typing import Any

from .taint import EventKind

_CURRENT_SESSION: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "ml_leakproof_runtime_session", default=None
)
_IN_HOOK_CALLBACK: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "ml_leakproof_runtime_hook_callback", default=False
)
_LOCK = threading.RLock()
_PATCHES: list[tuple[Any, str, Any]] = []
_PATCHED_KEYS: set[tuple[int, str]] = set()


def current_session() -> Any | None:
    return _CURRENT_SESSION.get()


def _call_site() -> str:
    try:
        for frame in inspect.stack()[2:]:
            filename = frame.filename.replace("\\", "/")
            if "/ml_leakproof/" not in filename and "/wrapt/" not in filename:
                return f"{frame.filename}:{frame.lineno}"
    except Exception:
        pass
    return "<unknown>"


class HookManager:
    """Shared patch manager; nested contexts only change the active session."""

    def __init__(self, session: Any) -> None:
        self.session = session
        self._token: contextvars.Token[Any | None] | None = None
        self._installed_by_me = False

    def install(self) -> None:
        with _LOCK:
            if not _PATCHES:
                try:
                    self._install_patches()
                except Exception:
                    self._remove_patches()
                    raise
                self._installed_by_me = True
            self._token = _CURRENT_SESSION.set(self.session)

    def remove(self) -> None:
        with _LOCK:
            if self._token is not None:
                _CURRENT_SESSION.reset(self._token)
                self._token = None
            # An outer context may still be active.  Its contextvar value is
            # restored by reset, so patches remain until the final release.
            if _CURRENT_SESSION.get() is None and _PATCHES:
                self._remove_patches()

    def _install_patches(self) -> None:
        ms = importlib.import_module("sklearn.model_selection")
        validation = importlib.import_module("sklearn.model_selection._validation")
        for module in (ms, validation):
            if hasattr(module, "train_test_split"):
                self._patch_function(module, "train_test_split", EventKind.SPLIT)
            for cv_name in ("cross_val_score", "cross_validate", "cross_val_predict"):
                if hasattr(module, cv_name):
                    self._patch_function(module, cv_name, EventKind.CV)
        try:
            pandas = importlib.import_module("pandas")
        except Exception:
            pandas = None
        if pandas is not None:
            self._patch_function(pandas, "concat", EventKind.APPLY)
            for cls in (pandas.DataFrame, pandas.Series):
                # These operations preserve row identity (or preserve the
                # pandas index, which lets the lineage table map a subset).
                # ``__getitem__`` covers boolean/index selection and column
                # extraction without attempting to patch pandas' indexer
                # implementation itself.
                for method in (
                    "__getitem__",
                    "copy",
                    "reset_index",
                    "sort_values",
                    "sort_index",
                    "astype",
                    "fillna",
                    "dropna",
                ):
                    if hasattr(cls, method):
                        self._patch_method(cls, method, EventKind.APPLY)
        try:
            numpy = importlib.import_module("numpy")
        except Exception:
            numpy = None
        if numpy is not None:
            self._patch_function(numpy, "concatenate", EventKind.APPLY)
            self._patch_function(numpy, "vstack", EventKind.APPLY)
            self._patch_function(numpy, "hstack", EventKind.APPLY)
            self._patch_function(numpy, "column_stack", EventKind.APPLY)
            # These functions can deliberately strip an ndarray subclass.
            # Observe them while a session is active so ancestry is propagated
            # by the operation rather than guessed from equal values.
            for name in ("array", "asarray", "asanyarray", "copy"):
                if hasattr(numpy, name):
                    self._patch_function(numpy, name, EventKind.APPLY)
            for name in ("zeros_like", "ones_like", "empty_like", "full_like"):
                if hasattr(numpy, name):
                    self._patch_function(numpy, name, EventKind.APPLY)

        base = importlib.import_module("sklearn.base")
        self._patch_method(base.TransformerMixin, "fit_transform", EventKind.FIT)
        self._patch_method(base.ClassifierMixin, "score", EventKind.SCORE)
        self._patch_method(base.RegressorMixin, "score", EventKind.SCORE)

        # all_estimators imports concrete sklearn classes and lets ordinary
        # estimator .fit/.partial_fit methods be observed, including keyword X.
        try:
            from sklearn.utils import all_estimators

            candidates = [cls for _, cls in all_estimators()]
        except Exception:
            candidates = []
        candidates.extend(
            [
                importlib.import_module("sklearn.pipeline").Pipeline,
                importlib.import_module("sklearn.compose").ColumnTransformer,
            ]
        )
        for class_name in (
            "GridSearchCV",
            "RandomizedSearchCV",
            "HalvingGridSearchCV",
            "HalvingRandomSearchCV",
        ):
            try:
                if class_name.startswith("Halving"):
                    importlib.import_module("sklearn.experimental.enable_halving_search_cv")
                candidates.append(getattr(ms, class_name))
            except AttributeError:
                pass
        seen_classes: set[type] = set()
        for cls in candidates:
            if not isinstance(cls, type) or cls in seen_classes:
                continue
            seen_classes.add(cls)
            for method, kind in (
                ("fit", EventKind.FIT),
                ("partial_fit", EventKind.FIT),
                ("fit_transform", EventKind.FIT),
                ("fit_resample", EventKind.FIT),
                ("transform", EventKind.APPLY),
                ("predict", EventKind.APPLY),
                ("predict_proba", EventKind.APPLY),
                ("decision_function", EventKind.APPLY),
                ("score", EventKind.SCORE),
            ):
                owner = next(
                    (candidate for candidate in cls.__mro__ if method in candidate.__dict__),
                    None,
                )
                if owner is not None:
                    self._patch_method(owner, method, kind)

        self._patch_optional_integrations()

    def _patch_optional_integrations(self) -> None:
        candidates = [
            ("imblearn.pipeline", "Pipeline"),
            ("xgboost.sklearn", "XGBClassifier"),
            ("xgboost.sklearn", "XGBRegressor"),
            ("xgboost.sklearn", "XGBRanker"),
            ("lightgbm.sklearn", "LGBMClassifier"),
            ("lightgbm.sklearn", "LGBMRegressor"),
            ("lightgbm.sklearn", "LGBMRanker"),
        ]
        for module_name, class_name in candidates:
            try:
                cls = getattr(importlib.import_module(module_name), class_name)
            except Exception:
                continue
            for method, kind in (
                ("fit", EventKind.FIT),
                ("predict", EventKind.APPLY),
                ("predict_proba", EventKind.APPLY),
                ("score", EventKind.SCORE),
            ):
                owner = next(
                    (candidate for candidate in cls.__mro__ if method in candidate.__dict__),
                    None,
                )
                if owner is not None:
                    self._patch_method(owner, method, kind)
        for module_name, class_names in (
            ("imblearn.over_sampling", ("SMOTE", "SMOTENC", "SMOTEN", "ADASYN", "RandomOverSampler")),
            ("imblearn.under_sampling", ("RandomUnderSampler", "NearMiss", "TomekLinks")),
        ):
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            for class_name in class_names:
                cls = getattr(module, class_name, None)
                if cls is not None:
                    self._patch_method(cls, "fit_resample", EventKind.FIT)

    def _patch_function(self, module: Any, name: str, kind: EventKind) -> None:
        original = getattr(module, name)
        key = (id(module), name)
        if key in _PATCHED_KEYS:
            return

        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            session = current_session()
            if session is None or _IN_HOOK_CALLBACK.get():
                return original(*args, **kwargs)
            split_random_state = (
                _clone_random_state(kwargs.get("random_state"))
                if kind is EventKind.SPLIT
                else None
            )
            if kind is EventKind.CV:
                # ``cross_val_score`` delegates to ``cross_validate``.  Both
                # public names are patched so imports made before and after
                # ``watch`` are observable, but one user call must still
                # produce one CV event.
                if session._cv_depth == 0:
                    token = _IN_HOOK_CALLBACK.set(True)
                    try:
                        self._on_cv(session, original, args, kwargs)
                    finally:
                        _IN_HOOK_CALLBACK.reset(token)
                session._cv_depth += 1
                try:
                    return original(*args, **kwargs)
                finally:
                    session._cv_depth -= 1
            try:
                result = original(*args, **kwargs)
            except Exception:
                raise
            token = _IN_HOOK_CALLBACK.set(True)
            try:
                if kind is EventKind.SPLIT:
                    result = self._on_split(
                        session,
                        original,
                        args,
                        kwargs,
                        result,
                        random_state=split_random_state,
                    )
                elif kind is EventKind.APPLY:
                    operation = getattr(original, "__name__", "")
                    if operation in {
                        "concat",
                        "concatenate",
                        "vstack",
                        "hstack",
                        "column_stack",
                    }:
                        result = self._on_concat(session, original, args, kwargs, result)
                    elif operation in {"array", "asarray", "asanyarray", "copy"}:
                        result = self._on_numpy_copy(session, original, args, kwargs, result)
                    elif operation in {"zeros_like", "ones_like", "empty_like", "full_like"}:
                        result = self._on_numpy_like(session, args, kwargs, result)
            finally:
                _IN_HOOK_CALLBACK.reset(token)
            return result

        setattr(module, name, wrapper)
        _PATCHES.append((module, name, original))
        _PATCHED_KEYS.add(key)

    def _patch_method(self, cls: type, name: str, kind: EventKind) -> None:
        original = cls.__dict__.get(name)
        if original is None or not callable(original):
            return
        key = (id(cls), name)
        if key in _PATCHED_KEYS:
            return

        @functools.wraps(original)
        def wrapper(instance: Any, *args: Any, **kwargs: Any) -> Any:
            session = current_session()
            if session is None:
                return original(instance, *args, **kwargs)
            outer = session._hook_depth
            session._hook_depth += 1
            try:
                result = original(instance, *args, **kwargs)
            finally:
                session._hook_depth -= 1
            # Nested delegated operations are represented by their outer public
            # operation, preventing Pipeline.fit from becoming many fits/scores.
            if outer == 0:
                token = _IN_HOOK_CALLBACK.set(True)
                try:
                    result = self._on_method(
                        session, original, kind, instance, args, kwargs, result
                    )
                finally:
                    _IN_HOOK_CALLBACK.reset(token)
            return result

        setattr(cls, name, wrapper)
        _PATCHES.append((cls, name, original))
        _PATCHED_KEYS.add(key)

    def _remove_patches(self) -> None:
        for target, name, original in reversed(_PATCHES):
            try:
                setattr(target, name, original)
            except Exception:
                pass
        _PATCHES.clear()
        _PATCHED_KEYS.clear()

    def _on_split(
        self,
        session: Any,
        original: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any,
        *,
        random_state: Any = None,
    ) -> Any:
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            session.add_diagnostic("LP420", "split operation returned no trackable train/test arrays")
            return result
        sources = list(args)
        if not sources:
            source = next(
                (kwargs[name] for name in ("X", "x", "data") if name in kwargs),
                None,
            )
            if source is not None:
                sources = [source]
        if not sources:
            session.add_diagnostic("LP420", "split operation has no trackable input")
            return result
        if len(result) != 2 * len(sources):
            session.add_diagnostic(
                "LP420",
                f"split operation returned {len(result)} outputs for {len(sources)} inputs",
            )
            return result
        train_len = _row_count(result[0])
        test_len = _row_count(result[1])
        if train_len is None or test_len is None:
            session.add_diagnostic("LP420", "split output row counts are unavailable")
            return result
        indices = _split_indices(
            sources[0],
            kwargs,
            train_len=train_len,
            test_len=test_len,
            random_state=random_state,
        )
        function = getattr(original, "__name__", "split")
        for position, source in enumerate(sources):
            train, test = result[2 * position], result[2 * position + 1]
            session.on_split(
                _call_site(),
                source=source,
                train=train,
                test=test,
                row_indices=indices if source is not None else None,
                function=function,
            )
        adapted = [_wrap_numpy_result(session, value) for value in result]
        return tuple(adapted) if isinstance(result, tuple) else adapted

    def _on_cv(
        self, session: Any, original: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        session.on_cv(
            _call_site(),
            function=getattr(original, "__name__", "cross_val_score"),
            data=_cv_input(args, kwargs),
        )

    def _on_concat(
        self,
        session: Any,
        original: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any,
    ) -> Any:
        if not args:
            sources_value = kwargs.get("objs", kwargs.get("arrays"))
            if sources_value is None:
                return result
            sources = sources_value if isinstance(sources_value, (list, tuple)) else (sources_value,)
        else:
            sources = args[0] if isinstance(args[0], (list, tuple)) else args
        operation = getattr(original, "__name__", "concat")
        if operation == "concatenate":
            if len(args) > 1:
                axis = args[1]
            elif "axis" in kwargs:
                axis = kwargs["axis"]
            else:
                # numpy.concatenate defaults to row-wise concatenation.  A
                # missing axis is not the same as an explicit axis=None,
                # which requests flattening.
                axis = 0
        else:
            axis = kwargs.get("axis", 0)
        if operation in {"hstack", "vstack"}:
            if operation == "vstack":
                axis = 0
            else:
                # np.hstack concatenates rows for 1-D inputs and columns for
                # 2-D inputs. Inspect the result so a 1-D train/test join is
                # not mistaken for incompatible column assembly.
                axis = 1 if getattr(result, "ndim", 1) > 1 else 0
        if operation == "column_stack":
            axis = 1
        try:
            if axis is None and getattr(result, "ndim", 1) > 1:
                session.add_diagnostic("LP414", "flattened concat has unsupported row lineage")
                return result
            tracked = None
            if axis in (1, -1):
                tracked = session.taint.track_column_assembly(sources, result)
            else:
                tracked = session.taint.track_concat(sources, result)
            # Libraries use concatenate internally for bookkeeping and index
            # generation.  An untracked internal operation is not an
            # unsupported user-data lineage event and must not make a clean
            # runtime session partial.  Emit LP414 only when the operation
            # touched data that the session already knows how to track.
            lineage_for = getattr(session.taint, "lineage_for", None)
            source_lineages = (
                [lineage_for(source) for source in sources]
                if callable(lineage_for)
                else []
            )
            if tracked is False and any(lineage is not None for lineage in source_lineages):
                session.add_diagnostic("LP414", f"{operation} lineage could not be propagated")
        except Exception:
            source_lineages = []
            try:
                lineage_for = getattr(session.taint, "lineage_for", None)
                if callable(lineage_for):
                    source_lineages = [lineage_for(source) for source in sources]
            except Exception:
                pass
            if source_lineages and any(lineage is not None for lineage in source_lineages):
                session.add_diagnostic("LP414", "concat lineage could not be propagated")
            elif not callable(getattr(session.taint, "lineage_for", None)):
                session.add_diagnostic("LP414", "concat lineage could not be propagated")
        if tracked is True:
            return _wrap_numpy_result(session, result)
        return result

    def _on_numpy_copy(
        self,
        session: Any,
        original: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any,
    ) -> Any:
        """Propagate ancestry through NumPy conversions by source identity."""

        source = args[0] if args else None
        if source is None:
            for name in ("a", "object", "x"):
                if name in kwargs:
                    source = kwargs[name]
                    break
        lineage_for = getattr(session.taint, "lineage_for", None)
        if source is None or not callable(lineage_for):
            return result
        try:
            source_lineage = lineage_for(source)
        except Exception:
            source_lineage = None
        if source_lineage is None:
            return result
        operation = getattr(original, "__name__", "NumPy copy")
        tracker = getattr(session, "track_transform", None)
        tracked = False
        if callable(tracker):
            try:
                tracked = bool(tracker(source=source, transformed=result))
            except Exception:
                tracked = False
        if tracked:
            return _wrap_numpy_result(session, result)
        marker = getattr(session.taint, "mark_unsupported", None)
        if callable(marker):
            marker(result, f"NumPy {operation} could not propagate array provenance")
        return result

    def _on_numpy_like(
        self,
        session: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any,
    ) -> Any:
        """Keep shape-only NumPy constructors independent from their prototype."""

        source = args[0] if args else kwargs.get("prototype")
        lineage_for = getattr(session.taint, "lineage_for", None)
        if source is None or not callable(lineage_for):
            return result
        try:
            has_lineage = lineage_for(source) is not None
        except Exception:
            has_lineage = False
        if not has_lineage:
            return result
        try:
            import numpy as np

            if isinstance(result, np.ndarray) and type(result) is not np.ndarray:
                return result.view(np.ndarray)
        except Exception:
            pass
        return result

    def _on_method(
        self,
        session: Any,
        original: Any,
        kind: EventKind,
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any,
    ) -> Any:
        method = getattr(original, "__name__", "operation")
        data = _bound_input(original, instance, args, kwargs)
        call_site = _call_site()
        cls = type(instance).__name__
        if kind is EventKind.FIT:
            eval_inputs = _extract_eval_inputs(kwargs)
            session.on_fit(
                call_site,
                cls,
                method=method,
                data=data,
                result=result,
                eval_inputs=eval_inputs,
                kwargs=kwargs,
                instance=instance,
            )
            if method in {"fit_transform", "fit_resample"}:
                output = result[0] if method == "fit_resample" and isinstance(result, (tuple, list)) else result
                if output is not instance and output is not None:
                    tracked = session.track_transform(
                        source=data,
                        transformed=output,
                        origin=_transform_origin(session, instance),
                    )
                    if tracked:
                        wrapped = _wrap_numpy_result(session, output)
                        if method == "fit_resample" and isinstance(result, (tuple, list)):
                            values = list(result)
                            values[0] = wrapped
                            result = tuple(values) if isinstance(result, tuple) else values
                        else:
                            result = wrapped
        elif kind is EventKind.APPLY:
            if method == "transform" and result is not None:
                tracked = session.track_transform(
                    source=data,
                    transformed=result,
                    origin=_transform_origin(session, instance),
                )
                if tracked:
                    result = _wrap_numpy_result(session, result)
            elif method in {
                "__getitem__",
                "copy",
                "reset_index",
                "sort_values",
                "sort_index",
                "astype",
                "fillna",
                "dropna",
            } and result is not None:
                # Pandas operations expose a stable index even when rows are
                # filtered or reordered.  Use it when available; a scalar
                # extraction simply remains untracked.
                tracker = getattr(session.taint, "track_pandas_operation", None)
                if callable(tracker):
                    tracked = tracker(
                        instance,
                        result,
                        method=method,
                        args=args,
                        kwargs=kwargs,
                    )
                    if tracked is False:
                        try:
                            has_source_lineage = session.taint.lineage_for(instance) is not None
                        except Exception:
                            has_source_lineage = False
                        if has_source_lineage:
                            session.add_diagnostic(
                                "LP414",
                                f"pandas {method} lineage could not be propagated",
                            )
            session.on_apply(
                call_site,
                cls,
                method=method,
                data=data,
                result=result,
                instance=instance,
            )
        elif kind is EventKind.SCORE:
            session.on_score(call_site, cls, method=method, data=data, kwargs=kwargs)
        return result


def _bound_input(
    original: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any | None:
    try:
        signature = inspect.signature(original)
        bound = signature.bind_partial(instance, *args, **kwargs)
        arguments = list(bound.arguments.items())
        for name, value in arguments[1:]:
            if name in {"X", "x", "data", "features", "X_train"}:
                return value
        return arguments[1][1] if len(arguments) > 1 else None
    except Exception:
        if args:
            return args[0]
        if "X" in kwargs:
            return kwargs["X"]
        return kwargs.get("x")


def _transform_origin(session: Any, instance: Any) -> int | None:
    resolver = getattr(session, "transform_origin", None)
    return resolver(instance) if callable(resolver) else None


def _cv_input(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any | None:
    # sklearn CV helpers take estimator, X, y; keyword calls are supported too.
    if "X" in kwargs:
        return kwargs["X"]
    if "x" in kwargs:
        return kwargs["x"]
    return args[1] if len(args) > 1 else None


def _wrap_numpy_result(session: Any, value: Any) -> Any:
    wrapper = getattr(getattr(session, "taint", None), "wrap_numpy_result", None)
    if callable(wrapper):
        try:
            return wrapper(value)
        except Exception:
            pass
    return value


def _row_count(value: Any) -> int | None:
    try:
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) > 0:
            return int(shape[0])
    except Exception:
        pass
    try:
        return len(value)
    except Exception:
        return None


def _clone_random_state(value: Any) -> Any:
    """Snapshot sklearn's split RNG before the wrapped call consumes it."""

    try:
        import numpy as np

        if value is None:
            state = np.random.RandomState()
            state.set_state(np.random.get_state())
            return state
        if isinstance(value, np.random.RandomState):
            state = np.random.RandomState()
            state.set_state(value.get_state())
            return state
        if isinstance(value, np.random.Generator):
            generator = np.random.default_rng()
            generator.bit_generator.state = copy.deepcopy(value.bit_generator.state)
            return generator
    except Exception:
        return value
    return value


def _split_indices(
    source: Any,
    kwargs: dict[str, Any],
    *,
    train_len: int,
    test_len: int,
    random_state: Any,
) -> dict[str, list[int]] | None:
    """Recreate train_test_split's exact positions from its pre-call RNG."""

    try:
        import numpy as np
        from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

        if not kwargs.get("shuffle", True):
            train = np.arange(train_len, dtype=int)
            test = np.arange(train_len, train_len + test_len, dtype=int)
        else:
            stratify = kwargs.get("stratify")
            splitter_class = StratifiedShuffleSplit if stratify is not None else ShuffleSplit
            splitter = splitter_class(
                n_splits=1,
                train_size=train_len,
                test_size=test_len,
                random_state=random_state,
            )
            train, test = next(splitter.split(X=source, y=stratify))
        if len(train) != train_len or len(test) != test_len:
            return None
        return {
            "train": [int(index) for index in train],
            "test": [int(index) for index in test],
        }
    except Exception:
        return None


def _extract_eval_inputs(kwargs: dict[str, Any]) -> list[Any]:
    values: list[Any] = []
    for name in ("eval_set", "validation_data", "eval_data"):
        if name not in kwargs:
            continue
        value = kwargs[name]
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, (list, tuple)) and item:
                    values.append(item[0])
                else:
                    values.append(item)
        else:
            values.append(value)
    return values
