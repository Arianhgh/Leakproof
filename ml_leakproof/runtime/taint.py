"""Runtime provenance tracking with per-dataset split namespaces."""

from __future__ import annotations

import hashlib
import itertools
import operator
import weakref
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, SupportsIndex, cast


class EventKind(str, Enum):
    SPLIT = "split"
    FIT = "fit"
    APPLY = "apply"
    SCORE = "score"
    CV = "cross_validation"


@dataclass
class RuntimeEvent:
    kind: EventKind
    call_site: str
    run_ordinal: int
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RowRef:
    dataset_id: str
    key: str

    def token(self) -> str:
        return f"{self.dataset_id}:{self.key}"


@dataclass
class Lineage:
    rows: tuple[RowRef, ...]
    supported: bool = True
    mixed: bool = False
    # IDs of learned preprocessing fits whose output contributed to this
    # object.  Row identity alone is not enough to detect CV reuse: an unused
    # transformed array can contain the same rows as the array passed to CV.
    transform_origins: frozenset[int] = frozenset()


@dataclass
class _TrackedObject:
    ref: weakref.ReferenceType[Any] | None
    strong: Any | None
    lineage: Lineage

    def matches(self, obj: Any) -> bool:
        if self.strong is not None:
            return self.strong is obj
        return self.ref is not None and self.ref() is obj


@dataclass
class _UnsupportedObject:
    ref: weakref.ReferenceType[Any] | None
    strong: Any | None
    message: str

    def matches(self, obj: Any) -> bool:
        if self.strong is not None:
            return self.strong is obj
        return self.ref is not None and self.ref() is obj


_LINEAGE_ATTRIBUTE = "_ml_leakproof_lineage"
_UNSUPPORTED_ATTRIBUTE = "_ml_leakproof_unsupported"
_OWNER_ATTRIBUTE = "_ml_leakproof_taint_owner"
_TRACKED_ARRAY_TYPE: type[Any] | None = None


def _restore_serialized_array(data: Any) -> Any:
    """Restore tracked arrays as plain arrays, without importing NumPy eagerly."""

    import numpy as np

    return np.asarray(data)


def _tracked_array_type() -> type[Any] | None:
    """Create the NumPy subclass used to carry operation ancestry."""

    global _TRACKED_ARRAY_TYPE
    if _TRACKED_ARRAY_TYPE is not None:
        return _TRACKED_ARRAY_TYPE
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        return None

    class TrackedArray(np.ndarray):
        def __reduce_ex__(
            self, protocol: SupportsIndex
        ) -> tuple[Any, tuple[Any, ...]]:
            """Serialize data only; lineage belongs to the originating session."""

            del protocol
            return _restore_serialized_array, (self.view(np.ndarray),)

        def __array_finalize__(self, source: Any) -> None:
            if source is not None:
                owner = getattr(source, _OWNER_ATTRIBUTE, None)
                if owner is not None:
                    setattr(self, _OWNER_ATTRIBUTE, owner)
                unsupported = getattr(source, _UNSUPPORTED_ATTRIBUTE, None)
                if isinstance(unsupported, str):
                    setattr(self, _UNSUPPORTED_ATTRIBUTE, unsupported)
                    return
                lineage = getattr(source, _LINEAGE_ATTRIBUTE, None)
                if not isinstance(lineage, Lineage):
                    return
                # Native copies, dtype casts, and row-preserving ufuncs keep
                # the same shape and memory layout.  A different layout may
                # be a row selection, transpose, or reduction and must be
                # resolved by that operation rather than inheriting all rows.
                same_shape = getattr(self, "shape", None) == getattr(source, "shape", None)
                same_layout = getattr(self, "strides", None) == getattr(
                    source, "strides", None
                )
                dtype_changed = getattr(self, "dtype", None) != getattr(
                    source, "dtype", None
                )
                if same_shape and (same_layout or dtype_changed):
                    setattr(self, _LINEAGE_ATTRIBUTE, lineage)
                else:
                    setattr(
                        self,
                        _UNSUPPORTED_ATTRIBUTE,
                        "NumPy operation could not propagate array provenance",
                    )

        def __setitem__(self, key: Any, value: Any) -> None:
            super().__setitem__(key, value)
            try:
                from .hooks import current_session

                session = current_session()
                table = getattr(session, "taint", None)
                if table is None or getattr(self, _OWNER_ATTRIBUTE, None) is not getattr(
                    table, "_owner_token", None
                ):
                    return
                tracker = getattr(table, "track_array_mutation", None)
                if callable(tracker):
                    tracker(self, key, value)
            except Exception:
                # The assignment itself has already succeeded.  A failed
                # provenance update is handled as unknown coverage when the
                # object is consumed by the active session.
                try:
                    from .hooks import current_session

                    session = current_session()
                    table = getattr(session, "taint", None)
                    if table is not None:
                        marker = getattr(table, "mark_unsupported", None)
                        if callable(marker):
                            marker(self, "in-place NumPy mutation could not propagate array provenance")
                except Exception:
                    pass

        def __array_ufunc__(
            self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any
        ) -> Any:
            if method != "__call__":
                base_inputs = tuple(
                    value.view(np.ndarray) if isinstance(value, np.ndarray) else value
                    for value in inputs
                )
                call_kwargs = dict(kwargs)
                out_values = call_kwargs.get("out")
                if out_values is not None:
                    outputs = out_values if isinstance(out_values, tuple) else (out_values,)
                    call_kwargs["out"] = tuple(
                        value.view(np.ndarray) if isinstance(value, np.ndarray) else value
                        for value in outputs
                    )
                return getattr(ufunc, method)(*base_inputs, **call_kwargs)
            try:
                from .hooks import current_session

                session = current_session()
                table = getattr(session, "taint", None)
                if table is None:
                    base_inputs = tuple(
                        value.view(np.ndarray) if isinstance(value, np.ndarray) else value
                        for value in inputs
                    )
                    return getattr(ufunc, method)(*base_inputs, **kwargs)
                arrays = [
                    value
                    for value in inputs
                    if isinstance(value, np.ndarray)
                    and getattr(value, "ndim", 0) > 0
                ]
                lineages = [table.lineage_for(value) for value in arrays]
                if not any(lineage is not None for lineage in lineages):
                    base_inputs = tuple(
                        value.view(np.ndarray) if isinstance(value, np.ndarray) else value
                        for value in inputs
                    )
                    return getattr(ufunc, method)(*base_inputs, **kwargs)

                # Run the operation on base arrays so ndarray's default
                # ``__array_finalize__`` cannot silently choose the first
                # operand's ancestry.  The table then combines every input
                # lineage or records an unsupported propagation.
                base_inputs = tuple(
                    value.view(np.ndarray) if isinstance(value, np.ndarray) else value
                    for value in inputs
                )
                call_kwargs = dict(kwargs)
                out_values = call_kwargs.get("out")
                if out_values is not None:
                    outputs = out_values if isinstance(out_values, tuple) else (out_values,)
                    call_kwargs["out"] = tuple(
                        value.view(np.ndarray) if isinstance(value, np.ndarray) else value
                        for value in outputs
                    )
                result = getattr(ufunc, method)(*base_inputs, **call_kwargs)
                tracker = getattr(table, "track_numpy_operation", None)
                if callable(tracker):
                    tracked_result = tracker(arrays, result)
                    if out_values is not None:
                        # NumPy returns the supplied out object, not its base
                        # view.  The table has already updated its lineage.
                        return out_values[0] if isinstance(out_values, tuple) else out_values
                    return tracked_result
                return result
            except Exception:
                base_inputs = tuple(
                    value.view(np.ndarray) if isinstance(value, np.ndarray) else value
                    for value in inputs
                )
                return getattr(ufunc, method)(*base_inputs, **kwargs)

        def __getitem__(self, key: Any) -> Any:
            result = super().__getitem__(key)
            if not isinstance(result, np.ndarray) or result.ndim == 0:
                return result
            lineage = getattr(self, _LINEAGE_ATTRIBUTE, None)
            if not isinstance(lineage, Lineage):
                return result
            if lineage.mixed and len(lineage.rows) != self.shape[0]:
                if hasattr(result, _LINEAGE_ATTRIBUTE):
                    delattr(result, _LINEAGE_ATTRIBUTE)
                setattr(
                    result,
                    _UNSUPPORTED_ATTRIBUTE,
                    "NumPy indexing could not resolve mixed array provenance",
                )
                return result
            positions = _numpy_row_positions(self, result, key)
            if positions is None:
                if hasattr(result, _LINEAGE_ATTRIBUTE):
                    delattr(result, _LINEAGE_ATTRIBUTE)
                setattr(
                    result,
                    _UNSUPPORTED_ATTRIBUTE,
                    "NumPy indexing could not propagate array provenance",
                )
                return result
            setattr(
                result,
                _LINEAGE_ATTRIBUTE,
                Lineage(
                    tuple(lineage.rows[index] for index in positions),
                    supported=lineage.supported,
                    mixed=lineage.mixed,
                    transform_origins=lineage.transform_origins,
                ),
            )
            if hasattr(result, _UNSUPPORTED_ATTRIBUTE):
                delattr(result, _UNSUPPORTED_ATTRIBUTE)
            return result

    _TRACKED_ARRAY_TYPE = TrackedArray
    return _TRACKED_ARRAY_TYPE


def _numpy_row_positions(source: Any, result: Any, key: Any) -> list[int] | None:
    """Return exact row positions for a row-preserving ndarray selection."""

    try:
        positions = _numpy_row_positions_for_key(source, key)
        if positions is None:
            return None
        if getattr(result, "shape", (None,))[0] != len(positions):
            return None
        return positions
    except Exception:
        return None


def _numpy_row_positions_for_key(source: Any, key: Any) -> list[int] | None:
    """Return source row positions selected by an ndarray indexing key."""

    try:
        import numpy as np

        row_key = key[0] if isinstance(key, tuple) and key else key
        if row_key is Ellipsis:
            row_key = slice(None)
        if row_key is None:
            return None
        if isinstance(row_key, slice):
            return list(range(*row_key.indices(source.shape[0])))
        if np.isscalar(row_key):
            position = operator.index(cast(Any, row_key))
            if position < 0:
                position += source.shape[0]
            if position < 0 or position >= source.shape[0]:
                return None
            return [position]
        index_array = np.asarray(row_key)
        if index_array.ndim != 1:
            return None
        if index_array.dtype.kind == "b":
            if len(index_array) != source.shape[0]:
                return None
            return [int(position) for position in np.flatnonzero(index_array).tolist()]
        positions: list[int] = []
        for value in index_array.tolist():
            position = operator.index(value)
            if position < 0:
                position += source.shape[0]
            if position < 0 or position >= source.shape[0]:
                return None
            positions.append(position)
        return positions
    except Exception:
        return None


def _numpy_key_covers_all_columns(source: Any, key: Any) -> bool:
    """Return whether an ndarray assignment replaces complete selected rows."""

    if not isinstance(key, tuple) or len(key) <= 1:
        return True
    for column_key in key[1:]:
        if column_key is Ellipsis:
            continue
        if not isinstance(column_key, slice) or column_key != slice(None):
            return False
    return True


def _should_report_unsupported() -> bool:
    """Defer internal estimator bookkeeping until an object is user-consumed."""

    try:
        from .hooks import current_session

        session = current_session()
    except Exception:
        return True
    return session is None or getattr(session, "_hook_depth", 0) == 0


def row_hashes(data: Any, cap: int = 200_000) -> list[str]:
    """Best-effort readable row digests for diagnostics, never row identity."""

    try:
        import numpy as np
    except Exception:  # pragma: no cover
        return []
    try:
        import pandas as pd

        if isinstance(data, (pd.DataFrame, pd.Series)):
            array = data.to_numpy()
        else:
            array = np.asarray(data)
    except Exception:
        try:
            array = np.asarray(data)
        except Exception:
            return []
    if array.ndim == 0:
        return []
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    array = array[:cap]
    out: list[str] = []
    for row in array:
        try:
            value = repr(row.tolist()).encode("utf-8")
        except Exception:
            value = repr(row).encode("utf-8")
        out.append(hashlib.blake2b(value, digest_size=16).hexdigest())
    return out


class TaintTable:
    """Track row lineage and split membership inside isolated namespaces."""

    def __init__(self) -> None:
        self.dataset_counter = itertools.count(1)
        self._owner_token = object()
        self._objects: list[_TrackedObject] = []
        self._unsupported_objects: list[_UnsupportedObject] = []
        self._split_sets: dict[str, dict[str, set[RowRef]]] = {}
        self._full_sets: dict[str, set[RowRef]] = {}
        self._diagnostics: list[dict[str, Any]] = []
        self._last_match_ambiguous = False
        self.score_counts: dict[str, int] = {}

    @property
    def train_hashes(self) -> set[str]:
        return {row.token() for splits in self._split_sets.values() for row in splits.get("train", set())}

    @property
    def test_hashes(self) -> set[str]:
        return {row.token() for splits in self._split_sets.values() for row in splits.get("test", set())}

    @property
    def val_hashes(self) -> set[str]:
        return {row.token() for splits in self._split_sets.values() for row in splits.get("val", set())}

    def eval_side(self) -> set[str]:
        return self.test_hashes | self.val_hashes

    def diagnostics(self) -> list[dict[str, Any]]:
        return list(self._diagnostics)

    def _new_dataset(self) -> str:
        return f"dataset-{next(self.dataset_counter)}"

    def _track_object(self, obj: Any, lineage: Lineage) -> None:
        if obj is None:
            return
        try:
            ref = weakref.ref(obj)
            strong = None
        except TypeError:
            ref = None
            strong = obj
        # Replace only the same object; never use an unvalidated id as identity.
        for tracked in self._objects:
            if tracked.matches(obj):
                tracked.lineage = lineage
                self._set_array_metadata(obj, lineage)
                self._clear_unsupported(obj)
                return
        self._objects.append(_TrackedObject(ref=ref, strong=strong, lineage=lineage))
        self._set_array_metadata(obj, lineage)
        self._clear_unsupported(obj)
        if len(self._objects) > 100_000:
            self._objects = [tracked for tracked in self._objects if tracked.matches(tracked.ref() if tracked.ref else tracked.strong)]

    def _clear_unsupported(self, obj: Any) -> None:
        self._unsupported_objects = [
            item for item in self._unsupported_objects if not item.matches(obj)
        ]

    def _forget_object(self, obj: Any) -> None:
        self._objects = [item for item in self._objects if not item.matches(obj)]
        for attribute in (_OWNER_ATTRIBUTE, _LINEAGE_ATTRIBUTE, _UNSUPPORTED_ATTRIBUTE):
            try:
                delattr(obj, attribute)
            except Exception:
                pass

    def _set_array_metadata(self, obj: Any, lineage: Lineage) -> None:
        try:
            array_type = _tracked_array_type()
            if array_type is not None and isinstance(obj, array_type):
                setattr(obj, _OWNER_ATTRIBUTE, self._owner_token)
                setattr(obj, _LINEAGE_ATTRIBUTE, lineage)
                if hasattr(obj, _UNSUPPORTED_ATTRIBUTE):
                    delattr(obj, _UNSUPPORTED_ATTRIBUTE)
        except Exception:
            pass

    def mark_unsupported(self, obj: Any, message: str) -> None:
        """Remember an untracked derived object until it is consumed."""

        if obj is None:
            return
        self._forget_object(obj)
        for item in self._unsupported_objects:
            if item.matches(obj):
                return
        try:
            ref = weakref.ref(obj)
            strong = None
        except TypeError:
            ref = None
            strong = obj
        self._unsupported_objects.append(
            _UnsupportedObject(ref=ref, strong=strong, message=message)
        )

    def track_array_mutation(self, target: Any, key: Any, value: Any) -> bool:
        """Update or invalidate lineage after an in-place ndarray assignment."""

        lineage = self.lineage_for(target)
        if lineage is None:
            return False
        try:
            import numpy as np

            if not isinstance(target, np.ndarray) or target.ndim == 0:
                self.mark_unsupported(target, "in-place NumPy mutation has no row axis")
                return False
            positions = _numpy_row_positions_for_key(target, key)
            if positions is None or len(lineage.rows) != target.shape[0]:
                self.mark_unsupported(
                    target,
                    "in-place NumPy mutation could not propagate array provenance",
                )
                return False
            if not isinstance(value, np.ndarray) or value.ndim == 0:
                # Assigning a scalar leaves each row's source identity intact.
                return True
            source_lineage = self.lineage_for(value)
            if source_lineage is None or len(source_lineage.rows) != len(positions):
                self.mark_unsupported(
                    target,
                    "in-place NumPy mutation used an untracked or incompatible array",
                )
                return False
            origins = lineage.transform_origins | source_lineage.transform_origins
            if _numpy_key_covers_all_columns(target, key):
                rows = list(lineage.rows)
                for position, row in zip(positions, source_lineage.rows):
                    rows[position] = row
                updated = Lineage(
                    tuple(rows),
                    supported=lineage.supported and source_lineage.supported,
                    mixed=lineage.mixed or source_lineage.mixed,
                    transform_origins=origins,
                )
            else:
                rows = list(dict.fromkeys((*lineage.rows, *source_lineage.rows)))
                updated = Lineage(
                    tuple(rows),
                    supported=False,
                    mixed=True,
                    transform_origins=origins,
                )
            self._track_object(target, updated)
            return True
        except Exception:
            self.mark_unsupported(
                target,
                "in-place NumPy mutation could not propagate array provenance",
            )
            return False

    def track_numpy_operation(self, sources: Iterable[Any], transformed: Any) -> Any:
        """Propagate ancestry through an ndarray operation using input identity."""

        try:
            import numpy as np
        except Exception:  # pragma: no cover
            return transformed
        if not isinstance(transformed, np.ndarray) or transformed.ndim == 0:
            return transformed
        arrays = [
            source
            for source in sources
            if isinstance(source, np.ndarray) and getattr(source, "ndim", 0) > 0
        ]
        lineages = [self.lineage_for(source) for source in arrays]
        known = [lineage for lineage in lineages if lineage is not None]
        if not known:
            return transformed
        if len(known) != len(lineages):
            self.mark_unsupported(
                transformed,
                "NumPy operation combined tracked and untracked arrays",
            )
            return transformed
        origins = frozenset().union(*(lineage.transform_origins for lineage in known))
        if all(lineage.rows == known[0].rows for lineage in known[1:]):
            source_len = len(self._values_for(arrays[0])) if arrays else 0
            if len(known[0].rows) != transformed.shape[0] and not (
                known[0].mixed and source_len == transformed.shape[0]
            ):
                self.mark_unsupported(
                    transformed,
                    "NumPy operation changed the row axis during provenance propagation",
                )
                return transformed
            lineage = Lineage(
                known[0].rows,
                supported=all(item.supported for item in known),
                mixed=any(item.mixed for item in known),
                transform_origins=origins,
            )
        elif all(len(item.rows) == transformed.shape[0] for item in known):
            rows = list(dict.fromkeys(row for item in known for row in item.rows))
            lineage = Lineage(
                tuple(rows),
                supported=False,
                mixed=True,
                transform_origins=origins,
            )
        else:
            self.mark_unsupported(
                transformed,
                "NumPy operation changed the row axis during provenance propagation",
            )
            return transformed
        self._track_object(transformed, lineage)
        return self.wrap_numpy_result(transformed)

    def lineage_for(self, obj: Any) -> Lineage | None:
        for tracked in reversed(self._objects):
            if tracked.matches(obj):
                return tracked.lineage
        metadata_owner = getattr(obj, _OWNER_ATTRIBUTE, None)
        metadata = getattr(obj, _LINEAGE_ATTRIBUTE, None)
        if metadata_owner is self._owner_token and isinstance(metadata, Lineage):
            self._track_object(obj, metadata)
            return metadata
        unsupported_metadata = getattr(obj, _UNSUPPORTED_ATTRIBUTE, None)
        if metadata_owner is self._owner_token and isinstance(unsupported_metadata, str):
            if _should_report_unsupported():
                self._record_unsupported_diagnostic(unsupported_metadata)
            return None
        # NumPy row/column slices retain a reference to their base array. Use
        # that object identity to recover lineage, then verify the row values
        # before attaching it to the view. Searching by equal content alone
        # would incorrectly merge duplicate-valued rows from unrelated data
        # namespaces.
        base = getattr(obj, "base", None)
        visited: set[int] = set()
        while base is not None and id(base) not in visited:
            visited.add(id(base))
            for tracked in reversed(self._objects):
                if not tracked.matches(base):
                    continue
                rows = self._shared_numpy_rows_if_possible(base, obj, tracked.lineage.rows)
                if rows is None and len(tracked.lineage.rows) == len(self._values_for(base)):
                    rows = self._match_output_rows(base, obj, tracked.lineage.rows)
                if rows is not None:
                    lineage = Lineage(
                        rows,
                        supported=tracked.lineage.supported,
                        mixed=tracked.lineage.mixed,
                        transform_origins=tracked.lineage.transform_origins,
                    )
                    self._track_object(obj, lineage)
                    return lineage
            base = getattr(base, "base", None)
        # NumPy may normalize a view's ``base`` to an underlying 1-D buffer,
        # so the direct base walk above can miss ordinary slices of a tracked
        # reshaped array.  Memory sharing is an identity relation (unlike
        # equal-content matching), and row pointers let us preserve slicing
        # and column-selection lineage without merging unrelated datasets.
        try:
            import numpy as np

            if isinstance(obj, np.ndarray):
                for tracked in reversed(self._objects):
                    candidate = tracked.ref() if tracked.ref is not None else tracked.strong
                    if not isinstance(candidate, np.ndarray) or not np.shares_memory(candidate, obj):
                        continue
                    rows = self._shared_numpy_rows(candidate, obj, tracked.lineage.rows)
                    if rows is not None:
                        lineage = Lineage(
                            rows,
                            supported=tracked.lineage.supported,
                            mixed=tracked.lineage.mixed,
                            transform_origins=tracked.lineage.transform_origins,
                        )
                        self._track_object(obj, lineage)
                        return lineage
        except Exception:
            pass
        for unsupported in reversed(self._unsupported_objects):
            if unsupported.matches(obj):
                message = unsupported.message
                if _should_report_unsupported():
                    self._record_unsupported_diagnostic(message)
                return None
        return None

    def _record_unsupported_diagnostic(self, message: str) -> None:
        if not any(
            item.get("code") == "LP417" and item.get("message") == message
            for item in self._diagnostics
        ):
            self._diagnostics.append({"code": "LP417", "message": message})

    def wrap_numpy_result(self, obj: Any) -> Any:
        """Attach operation ancestry to a NumPy result without inspecting values."""

        try:
            import numpy as np

            array_type = _tracked_array_type()
            if array_type is None or not isinstance(obj, np.ndarray):
                return obj
            lineage = self.lineage_for(obj)
            if lineage is None:
                return obj
            if isinstance(obj, array_type):
                setattr(obj, _LINEAGE_ATTRIBUTE, lineage)
                return obj
            wrapped = obj.view(array_type)
            setattr(wrapped, _LINEAGE_ATTRIBUTE, lineage)
            self._track_object(wrapped, lineage)
            return wrapped
        except Exception:
            return obj

    @staticmethod
    def _shared_numpy_rows(
        source: Any, output: Any, source_rows: tuple[RowRef, ...]
    ) -> tuple[RowRef, ...] | None:
        """Map rows for a NumPy view by their memory addresses."""

        if source.ndim == 0 or output.ndim == 0 or len(source_rows) != source.shape[0]:
            return None
        if output.shape[0] != len(output) or source.strides[0] == 0 or output.strides[0] == 0:
            return None
        source_ptr = int(source.__array_interface__["data"][0])
        output_ptr = int(output.__array_interface__["data"][0])
        positions = {
            source_ptr + index * int(source.strides[0]): index
            for index in range(source.shape[0])
        }
        mapped: list[RowRef] = []
        for index in range(output.shape[0]):
            source_index = positions.get(output_ptr + index * int(output.strides[0]))
            if source_index is None:
                # A column slice starts at an offset inside each source row,
                # so its row pointers do not equal the source row starts even
                # though the row stride is unchanged.  Map that fixed offset
                # without using equal-content matching.
                if (
                    source.ndim < 2
                    or output.ndim < 2
                    or int(output.strides[0]) != int(source.strides[0])
                    or int(source.strides[0]) <= 0
                ):
                    return None
                delta = output_ptr - source_ptr
                offset = delta % int(source.strides[0])
                start = (delta - offset) // int(source.strides[0])
                pointer = output_ptr + index * int(output.strides[0])
                row_delta = pointer - source_ptr - offset
                if row_delta % int(source.strides[0]) != 0:
                    return None
                source_index = row_delta // int(source.strides[0])
                if not (0 <= start < source.shape[0]) or not (
                    0 <= source_index < source.shape[0]
                ):
                    return None
            mapped.append(source_rows[source_index])
        return tuple(mapped)

    def _values_for(self, obj: Any) -> list[Any]:
        # Keep pandas/NumPy support independent from scipy. scipy is useful
        # for sparse matrices but is not a dependency of the runtime extra;
        # importing it in the same try block used to make a trackable
        # DataFrame fall through to ``list(frame)`` (column names).
        numpy_module: Any = None
        pandas_module: Any = None
        sparse_module: Any = None
        try:
            import numpy as np

            numpy_module = np
        except Exception:
            pass
        try:
            import pandas as pd

            pandas_module = pd
        except Exception:
            pass
        try:
            from scipy import sparse

            sparse_module = sparse
        except Exception:
            pass

        try:
            if pandas_module is not None and isinstance(obj, pandas_module.DataFrame):
                return [row for row in obj.to_numpy()]
            if pandas_module is not None and isinstance(obj, pandas_module.Series):
                return list(obj.to_numpy())
            if sparse_module is not None and sparse_module.issparse(obj):
                matrix = obj.tocsr()
                return [matrix.getrow(index) for index in range(matrix.shape[0])]
            if numpy_module is not None:
                array = numpy_module.asarray(obj)
                if array.ndim == 0:
                    return []
                return list(array if array.ndim > 1 else array.reshape(-1, 1))
        except Exception:
            pass
        try:
            return list(obj)
        except Exception:
            return []

    def _match_output_rows(
        self, source: Any, output: Any, source_rows: tuple[RowRef, ...]
    ) -> tuple[RowRef, ...] | None:
        self._last_match_ambiguous = False
        source_values = row_hashes(source, cap=max(len(source_rows), 1))
        output_values = row_hashes(output, cap=max(len(source_rows), 1))
        if not output_values:
            return None
        source_counts: dict[str, int] = {}
        output_counts: dict[str, int] = {}
        for digest in source_values:
            source_counts[digest] = source_counts.get(digest, 0) + 1
        for digest in output_values:
            output_counts[digest] = output_counts.get(digest, 0) + 1
        if any(count > 1 for count in source_counts.values()) or any(
            count > 1 for count in output_counts.values()
        ):
            self._last_match_ambiguous = True
            message = "row provenance is ambiguous because split values are duplicated"
            if not any(
                item.get("code") == "LP415" and item.get("message") == message
                for item in self._diagnostics
            ):
                self._diagnostics.append({"code": "LP415", "message": message})
            return None
        positions: dict[str, list[int]] = {}
        for position, digest in enumerate(source_values):
            positions.setdefault(digest, []).append(position)
        used: dict[str, int] = {}
        matched: list[RowRef] = []
        for digest in output_values:
            available = positions.get(digest, [])
            offset = used.get(digest, 0)
            if offset >= len(available):
                return None
            matched.append(source_rows[available[offset]])
            used[digest] = offset + 1
        return tuple(matched)

    def register_split(
        self,
        *,
        train: Any,
        test: Any,
        val: Any | None = None,
        name: str = "holdout",
        row_ids: dict[str, Iterable[Any]] | None = None,
        row_indices: dict[str, Iterable[int]] | None = None,
        source: Any | None = None,
    ) -> str:
        """Register one dataset namespace and attach lineage to each split object."""

        dataset_id = f"{name}-{next(self.dataset_counter)}"
        row_ids = row_ids or {}
        row_indices = row_indices or {}
        unknown_row_id_splits = set(row_ids) - {"train", "test", "val"}
        if unknown_row_id_splits:
            self._diagnostics.append(
                {
                    "code": "LP410",
                    "message": f"row_ids contains unknown split(s): {sorted(unknown_row_id_splits)}",
                }
            )
        unknown_row_index_splits = set(row_indices) - {"train", "test", "val"}
        if unknown_row_index_splits:
            self._diagnostics.append(
                {
                    "code": "LP410",
                    "message": f"row_indices contains unknown split(s): {sorted(unknown_row_index_splits)}",
                }
            )
        split_objects = {"train": train, "test": test}
        if val is not None:
            split_objects["val"] = val
        split_sets: dict[str, set[RowRef]] = {}
        source_lineage = self.lineage_for(source) if source is not None else None
        if source is not None:
            source_rows = tuple(
                RowRef(dataset_id, f"row-{index}")
                for index in range(len(self._values_for(source)))
            )
            previous_rows = source_lineage.rows if source_lineage is not None else ()
            source_lineage = Lineage(
                source_rows,
                supported=source_lineage.supported if source_lineage else True,
                mixed=source_lineage.mixed if source_lineage else False,
                transform_origins=(
                    source_lineage.transform_origins if source_lineage else frozenset()
                ),
            )
            self._track_object(source, source_lineage)
            self._full_sets[dataset_id] = set(source_rows)
            if previous_rows:
                self._rebind_descendants(previous_rows, source_rows)
        for split_name, obj in split_objects.items():
            output_len = len(self._values_for(obj))
            split_supported = source_lineage.supported if source_lineage else True
            provided = row_ids.get(split_name)
            if provided is not None:
                values = list(provided)
                if len(values) != output_len:
                    self._diagnostics.append({"code": "LP410", "message": f"row_ids length does not match {split_name}"})
                    rows = tuple(RowRef(dataset_id, f"{split_name}-row-{index}") for index in range(output_len))
                else:
                    rows = tuple(RowRef(dataset_id, f"id-{_stable_token(value)}") for value in values)
            elif split_name in row_indices:
                try:
                    positions = [operator.index(value) for value in row_indices[split_name]]
                except (TypeError, ValueError):
                    positions = []
                    split_supported = False
                    self._diagnostics.append(
                        {"code": "LP410", "message": f"row_indices are invalid for {split_name}"}
                    )
                if (
                    source_lineage is None
                    or len(positions) != output_len
                    or any(index < 0 or index >= len(source_lineage.rows) for index in positions)
                ):
                    if split_supported:
                        self._diagnostics.append(
                            {"code": "LP410", "message": f"row_indices do not map {split_name} to the source"}
                        )
                    split_supported = False
                    rows = tuple(RowRef(dataset_id, f"{split_name}-row-{index}") for index in range(output_len))
                else:
                    rows = tuple(source_lineage.rows[index] for index in positions)
            elif source_lineage is not None:
                matched = self._shared_numpy_rows_if_possible(source, obj, source_lineage.rows)
                if matched is None:
                    matched = self._indexed_pandas_rows(source, obj, source_lineage.rows)
                if matched is None:
                    diagnostic_count = len(self._diagnostics)
                    matched = self._match_output_rows(source, obj, source_lineage.rows)
                    if matched is None:
                        split_supported = False
                        if len(self._diagnostics) == diagnostic_count and not self._last_match_ambiguous:
                            self._diagnostics.append(
                                {
                                    "code": "LP414",
                                    "message": f"{split_name} row provenance could not be mapped to the source",
                                }
                            )
                if matched is not None:
                    rows = matched
                else:
                    rows = tuple(RowRef(dataset_id, f"{split_name}-row-{index}") for index in range(output_len))
            else:
                rows = tuple(RowRef(dataset_id, f"{split_name}-row-{index}") for index in range(output_len))
            split_sets[split_name] = set(rows)
            self._track_object(
                obj,
                Lineage(
                    rows,
                    supported=split_supported,
                    mixed=source_lineage.mixed if source_lineage else False,
                    transform_origins=(
                        source_lineage.transform_origins if source_lineage else frozenset()
                    ),
                ),
            )
        self._split_sets[dataset_id] = split_sets
        return dataset_id

    def register_automatic_split(
        self,
        source: Any,
        train: Any,
        test: Any,
        *,
        name: str = "holdout",
        row_indices: dict[str, Iterable[int]] | None = None,
    ) -> str:
        return self.register_split(
            train=train,
            test=test,
            name=name,
            row_indices=row_indices,
            source=source,
        )

    def track_transform(
        self,
        *,
        source: Any,
        transformed: Any,
        row_indices: Iterable[int] | None = None,
        origin: int | None = None,
    ) -> bool:
        """Propagate row lineage through a supported row-preserving operation."""

        source_lineage = self.lineage_for(source)
        if source_lineage is None:
            source_lineage = self._ensure_lineage(source)
            if source_lineage is None:
                return False
        output_len = len(self._values_for(transformed))
        if row_indices is None:
            source_len = len(self._values_for(source))
            if source_lineage.mixed and output_len == source_len:
                self._track_object(
                    transformed,
                    Lineage(
                        source_lineage.rows,
                        supported=False,
                        mixed=True,
                        transform_origins=source_lineage.transform_origins
                        | ({origin} if origin is not None else set()),
                    ),
                )
                return True
            if output_len != len(source_lineage.rows):
                self._diagnostics.append({"code": "LP412", "message": "length-changing transform requires row_indices"})
                return False
            rows = source_lineage.rows
        else:
            indices = list(row_indices)
            if len(indices) != output_len or any(index < 0 or index >= len(source_lineage.rows) for index in indices):
                self._diagnostics.append({"code": "LP413", "message": "row_indices do not map the transformed output"})
                return False
            rows = tuple(source_lineage.rows[index] for index in indices)
        self._track_object(
            transformed,
            Lineage(
                tuple(rows),
                source_lineage.supported,
                source_lineage.mixed,
                source_lineage.transform_origins | ({origin} if origin is not None else set()),
            ),
        )
        return True

    def track_pandas_operation(
        self,
        source: Any,
        transformed: Any,
        *,
        method: str | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> bool:
        """Track a pandas result using its row index when it is available.

        Index-based mapping supports boolean/index selection, column
        extraction, sorting, copies, and reset/NA operations while avoiding a
        content-only join that could merge equal-valued rows from unrelated
        datasets. ``reset_index`` is handled explicitly because it can discard
        or replace the source index. Duplicate source indexes are intentionally
        left unsupported; callers can use :meth:`track_transform` with explicit
        positions.
        """

        try:
            kwargs = kwargs or {}
            source_lineage = self.lineage_for(source)
            if source_lineage is None:
                return False
            source_index = source.index
            transformed_index = transformed.index
            reset_drop = method == "reset_index" and (
                kwargs.get("drop") is True or (kwargs.get("drop") is None and len(args) > 1 and args[1] is True)
            )
            if reset_drop and len(source_lineage.rows) == len(transformed):
                self._track_object(
                    transformed,
                    Lineage(
                        source_lineage.rows,
                        source_lineage.supported,
                        source_lineage.mixed,
                        source_lineage.transform_origins,
                    ),
                )
                return True
            if not source_index.is_unique:
                self._diagnostics.append(
                    {
                        "code": "LP414",
                        "message": "pandas lineage requires a unique source index",
                    }
                )
                return False

            if method == "reset_index":
                drop = kwargs.get("drop")
                if drop is None and len(args) > 1:
                    drop = args[1]
                if drop is True:
                    matched = self._match_output_rows(
                        source, transformed, source_lineage.rows
                    )
                    if matched is None:
                        self._diagnostics.append(
                            {
                                "code": "LP414",
                                "message": "reset_index output could not be mapped to source rows",
                            }
                        )
                        return False
                    self._track_object(
                        transformed,
                        Lineage(
                            matched,
                            source_lineage.supported,
                            source_lineage.mixed,
                            source_lineage.transform_origins,
                        ),
                    )
                    return True

                # ``reset_index(drop=False)`` creates a new positional index
                # and inserts the original index as a column. Locate that
                # column and map it back to the unique source index rather
                # than interpreting the new RangeIndex as source identity.
                source_columns = set(getattr(source, "columns", []))
                candidates = [
                    column
                    for column in getattr(transformed, "columns", [])
                    if column not in source_columns
                ]
                for column in candidates:
                    try:
                        positions = source_index.get_indexer(transformed[column])
                    except Exception:
                        continue
                    if len(positions) == len(transformed) and all(
                        int(position) >= 0 for position in positions
                    ):
                        rows = tuple(source_lineage.rows[int(position)] for position in positions)
                        self._track_object(
                            transformed,
                            Lineage(
                                rows,
                                source_lineage.supported,
                                source_lineage.mixed,
                                source_lineage.transform_origins,
                            ),
                        )
                        return True

            positions = source_index.get_indexer(transformed_index)
            if any(int(position) < 0 for position in positions):
                return False
            return self.track_transform(
                source=source,
                transformed=transformed,
                row_indices=[int(position) for position in positions],
            )
        except Exception:
            return False

    @staticmethod
    def _shared_numpy_rows_if_possible(
        source: Any, output: Any, source_rows: tuple[RowRef, ...]
    ) -> tuple[RowRef, ...] | None:
        try:
            import numpy as np

            if isinstance(source, np.ndarray) and isinstance(output, np.ndarray):
                if np.shares_memory(source, output):
                    return TaintTable._shared_numpy_rows(source, output, source_rows)
        except Exception:
            return None
        return None

    @staticmethod
    def _indexed_pandas_rows(
        source: Any, output: Any, source_rows: tuple[RowRef, ...]
    ) -> tuple[RowRef, ...] | None:
        """Map pandas outputs by their stable index before using row values."""

        try:
            import pandas as pd

            pandas_types = (pd.DataFrame, pd.Series)
            if not isinstance(source, pandas_types) or not isinstance(output, pandas_types):
                return None
            source_index = source.index
            if not source_index.is_unique:
                return None
            positions = source_index.get_indexer(output.index)
            if len(positions) != len(output):
                return None
            if any(int(position) < 0 for position in positions):
                return None
            return tuple(source_rows[int(position)] for position in positions)
        except Exception:
            return None

    def _ensure_lineage(self, obj: Any) -> Lineage | None:
        values = self._values_for(obj)
        if not values:
            return None
        dataset_id = self._new_dataset()
        lineage = Lineage(tuple(RowRef(dataset_id, f"row-{index}") for index in range(len(values))))
        self._full_sets[dataset_id] = set(lineage.rows)
        self._track_object(obj, lineage)
        return lineage

    def track_concat(self, sources: Iterable[Any], transformed: Any) -> bool:
        lineages = [self.lineage_for(source) for source in sources]
        if not lineages or any(lineage is None for lineage in lineages):
            return False
        known_lineages = [lineage for lineage in lineages if lineage is not None]
        rows: list[RowRef] = []
        for lineage in known_lineages:
            rows.extend(lineage.rows)
        sides = {
            split_name
            for row in rows
            for split_sets in self._split_sets.values()
            for split_name, values in split_sets.items()
            if row in values
        }
        mixed = len({row.dataset_id for row in rows}) > 1 or len(sides) > 1
        origins = frozenset().union(*(lineage.transform_origins for lineage in known_lineages))
        self._track_object(
            transformed,
            Lineage(tuple(rows), mixed=mixed, transform_origins=frozenset(origins)),
        )
        return True

    def track_column_assembly(self, sources: Iterable[Any], transformed: Any) -> bool:
        """Propagate rows through a same-row column assembly, not row mixing."""

        lineages = [self.lineage_for(source) for source in sources]
        if not lineages or any(lineage is None for lineage in lineages):
            return False
        known_lineages = [lineage for lineage in lineages if lineage is not None]
        first = known_lineages[0]
        if any(lineage.rows != first.rows for lineage in known_lineages[1:]):
            self._diagnostics.append(
                {
                    "code": "LP414",
                    "message": "column assembly inputs have incompatible row lineage",
                }
            )
            return False
        origins = frozenset().union(*(lineage.transform_origins for lineage in known_lineages))
        self._track_object(
            transformed,
            Lineage(
                first.rows,
                supported=first.supported,
                mixed=any(lineage.mixed for lineage in known_lineages),
                transform_origins=frozenset(origins),
            ),
        )
        return True

    def _rebind_descendants(
        self,
        previous_rows: tuple[RowRef, ...],
        current_rows: tuple[RowRef, ...],
    ) -> None:
        """Move descendants of a source into the namespace created by a split."""

        if len(previous_rows) != len(current_rows):
            return
        mapping = dict(zip(previous_rows, current_rows))
        for tracked in self._objects:
            rows = tracked.lineage.rows
            if not rows or not all(row in mapping for row in rows):
                continue
            lineage = Lineage(
                tuple(mapping[row] for row in rows),
                supported=tracked.lineage.supported,
                mixed=tracked.lineage.mixed,
                transform_origins=tracked.lineage.transform_origins,
            )
            tracked.lineage = lineage
            candidate = tracked.ref() if tracked.ref is not None else tracked.strong
            self._set_array_metadata(candidate, lineage)

    def capture(self, obj: Any) -> dict[str, Any]:
        lineage = self.lineage_for(obj)
        return {
            "object": obj,
            "rows": lineage.rows if lineage else (),
            "tracked": lineage is not None,
            "dynamic": lineage is None,
            "mixed": lineage.mixed if lineage else False,
            "transform_origins": sorted(lineage.transform_origins) if lineage else [],
        }

    def resolve_capture(self, capture: dict[str, Any]) -> tuple[RowRef, ...]:
        if capture.get("dynamic"):
            lineage = self.lineage_for(capture.get("object"))
            if lineage is not None:
                return lineage.rows
        return tuple(capture.get("rows", ()))

    def membership(self, rows: Iterable[RowRef]) -> dict[str, int]:
        seen = set(rows)
        train = eval_rows = val = 0
        for split_sets in self._split_sets.values():
            train += len(seen & split_sets.get("train", set()))
            eval_rows += len(seen & split_sets.get("test", set()))
            val += len(seen & split_sets.get("val", set()))
        return {"train": train, "test": eval_rows, "val": val, "eval": eval_rows + val}

    @staticmethod
    def signature(rows: Iterable[Any]) -> str:
        h = hashlib.blake2b(digest_size=16)
        for value in sorted(str(row) for row in rows):
            h.update(value.encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()


def _stable_token(value: Any) -> str:
    try:
        return hashlib.blake2b(repr(value).encode("utf-8"), digest_size=12).hexdigest()
    except Exception:
        return str(id(value))
