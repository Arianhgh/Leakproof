"""Runtime context manager, script execution, and pytest integration."""

from __future__ import annotations

import runpy
import sys
import warnings
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from ..core.aggregate import annotate_gateability, dedupe, sort_findings, summarize
from ..core.config import Config, ConfigError
from ..core.context import RuntimeContext
from ..core.models import (
    AnalysisResult,
    CompletionStatus,
    Diagnostic,
    DiagnosticLevel,
    Finding,
    Layer,
)
from ..core.registry import consume_diagnostics, load_all
from ..core.rule import RuntimeRule
from .hooks import HookManager
from .taint import EventKind, RuntimeEvent, TaintTable


class RuntimeSession:
    """A context-local runtime provenance session."""

    def __init__(self, config: Config):
        self.config = config
        self.config.validate()
        if Layer.RUNTIME not in self.config.layers:
            raise ConfigError("watch requires the runtime layer")
        self.taint = TaintTable()
        self.ctx = RuntimeContext(config=config, taint=self.taint)
        self.rules: list[RuntimeRule] = [
            rule
            for rule in load_all(config, layers=(Layer.RUNTIME,))
            if isinstance(rule, RuntimeRule) and Layer.RUNTIME in rule.layers
        ]
        self.findings: list[Finding] = []
        self.diagnostics: list[Diagnostic] = consume_diagnostics()
        self._ordinal = 0
        self._hook_depth = 0
        self._cv_depth = 0
        self._fit_ordinals: dict[int, int] = {}
        self._hooks = HookManager(self)
        self._finalized = False
        self._entered = False
        self.result = AnalysisResult()

    def __enter__(self) -> RuntimeSession:
        if self._entered:
            return self
        self._entered = True
        try:
            self._hooks.install()
        except Exception as exc:
            self.add_diagnostic("LP400", f"runtime hook installation failed: {exc}", error=True)
        return self

    def __exit__(
        self,
        exc_type: Any = None,
        exc_value: BaseException | None = None,
        traceback: Any = None,
    ) -> None:
        if not self._entered:
            return
        try:
            self._hooks.remove()
        finally:
            self._finalize()
            self._entered = False

    def add_diagnostic(self, code: str, message: str, *, error: bool = False) -> None:
        self.diagnostics.append(
            Diagnostic(
                code=code,
                message=message,
                level=DiagnosticLevel.ERROR if error else DiagnosticLevel.WARNING,
                layer=Layer.RUNTIME,
            )
        )

    def _record(self, event: RuntimeEvent) -> None:
        self.ctx.events.append(event)

    def _finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        for event in self.ctx.events:
            if "input" in event.payload:
                rows = self.taint.resolve_capture(event.payload["input"])
                event.payload["seen"] = [row.token() for row in rows]
                membership = self.taint.membership(rows)
                event.payload["membership"] = membership
                event.payload["eval_rows_seen"] = membership["eval"]
                event.payload["train_rows_seen"] = membership["train"]
                event.payload["concat_sources"] = bool(event.payload["input"].get("mixed"))
            if "eval_inputs" in event.payload:
                eval_rows: list[dict[str, Any]] = []
                for capture in event.payload["eval_inputs"]:
                    rows = self.taint.resolve_capture(capture)
                    eval_rows.append({"rows": [row.token() for row in rows], "membership": self.taint.membership(rows)})
                event.payload["eval_membership"] = eval_rows
                event.payload["eval_rows_seen"] = sum(
                    int(item["membership"]["eval"]) for item in eval_rows
                )
            for rule in self.rules:
                try:
                    self.findings.extend(rule.on_event(event, self.ctx))
                except Exception as exc:
                    self.add_diagnostic(
                        "LP420",
                        f"runtime rule {rule.id} failed: {exc}",
                        error=True,
                    )
        for item in self.taint.diagnostics():
            self.add_diagnostic(item.get("code", "LP421"), item.get("message", "runtime lineage unsupported"))
        findings = annotate_gateability(
            sort_findings(dedupe(self.findings)),
            gate=self.config.fail_on,
            gate_confidence=self.config.gate_confidence,
            profile=self.config.profile,
        )
        self.findings = findings
        coverage = self.result.coverage
        coverage.requested_inputs = ["runtime session"]
        coverage.analyzed_inputs = ["instrumented operations"]
        coverage.executed_checks = sorted({rule.id for rule in self.rules})
        if self.diagnostics:
            self.result.completion = CompletionStatus.PARTIAL
        self.result.findings = findings
        self.result.diagnostics = list(self.diagnostics)
        self.result.summary = summarize(
            findings,
            self.config.fail_on,
            self.config.gate_confidence,
            diagnostics_count=len(self.diagnostics),
        )
        self.result.metadata.update({"hook_scope": "context-local", "subprocesses_tracked": False})

    def register_split(
        self,
        *,
        train: Any,
        test: Any,
        val: Any | None = None,
        name: str = "holdout",
        row_ids: dict[str, Iterable[Any]] | None = None,
        row_indices: dict[str, Iterable[int]] | None = None,
    ) -> str:
        dataset_id = self.taint.register_split(
            train=train,
            test=test,
            val=val,
            name=name,
            row_ids=row_ids,
            row_indices=row_indices,
        )
        self._ordinal += 1
        self._record(
            RuntimeEvent(
                kind=EventKind.SPLIT,
                call_site="explicit registration",
                run_ordinal=self._ordinal,
                payload={"dataset_id": dataset_id},
            )
        )
        return dataset_id

    def track_transform(
        self,
        *,
        source: Any,
        transformed: Any,
        row_indices: list[int] | None = None,
        origin: int | None = None,
    ) -> bool:
        return self.taint.track_transform(
            source=source,
            transformed=transformed,
            row_indices=row_indices,
            origin=origin,
        )

    def on_split(
        self,
        call_site: str,
        train_h: list[str] | None = None,
        test_h: list[str] | None = None,
        *,
        source: Any = None,
        train: Any = None,
        test: Any = None,
        val: Any = None,
        name: str = "holdout",
        function: str = "split",
        row_indices: dict[str, Iterable[int]] | None = None,
    ) -> None:
        if train is not None and test is not None:
            dataset_id = self.taint.register_automatic_split(
                source,
                train,
                test,
                name=name,
                row_indices=row_indices,
            )
        else:
            # Compatibility path for integrations that already provide hashes.
            dataset_id = self.taint.register_split(
                train=train_h or [], test=test_h or [], name=name
            )
        self._ordinal += 1
        self._record(
            RuntimeEvent(
                kind=EventKind.SPLIT,
                call_site=call_site,
                run_ordinal=self._ordinal,
                payload={"dataset_id": dataset_id, "function": function},
            )
        )

    def on_fit(
        self,
        call_site: str,
        cls: str,
        *,
        method: str = "fit",
        data: Any = None,
        result: Any = None,
        eval_inputs: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        instance: Any = None,
    ) -> int:
        self._ordinal += 1
        ordinal = self._ordinal
        if instance is not None:
            self._fit_ordinals[id(instance)] = ordinal
        payload = {
            "class": cls,
            "method": method,
            "input": self.taint.capture(data),
            "output": self.taint.capture(result),
            "eval_inputs": [self.taint.capture(value) for value in (eval_inputs or [])],
            "kwargs": _safe_kwargs(kwargs or {}),
            "components": _pipeline_components(instance),
            "in_cv": bool(self._cv_depth),
            "instance_id": id(instance) if instance is not None else None,
        }
        self._record(
            RuntimeEvent(
                kind=EventKind.FIT,
                call_site=call_site,
                run_ordinal=self._ordinal,
                payload=payload,
            )
        )
        return ordinal

    def transform_origin(self, instance: Any) -> int | None:
        if instance is None:
            return None
        return self._fit_ordinals.get(id(instance))

    def on_apply(
        self,
        call_site: str,
        cls: str,
        *,
        method: str,
        data: Any = None,
        result: Any = None,
        instance: Any = None,
    ) -> None:
        self._ordinal += 1
        self._record(
            RuntimeEvent(
                kind=EventKind.APPLY,
                call_site=call_site,
                run_ordinal=self._ordinal,
            payload={
                "class": cls,
                "method": method,
                "input": self.taint.capture(data),
                "output": self.taint.capture(result),
                "in_cv": bool(self._cv_depth),
                "instance_id": id(instance) if instance is not None else None,
            },
            )
        )

    def on_score(
        self,
        call_site: str,
        cls: str,
        *,
        method: str = "score",
        data: Any = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._ordinal += 1
        self._record(
            RuntimeEvent(
                kind=EventKind.SCORE,
                call_site=call_site,
                run_ordinal=self._ordinal,
                payload={
                    "class": cls,
                    "method": method,
                    "input": self.taint.capture(data),
                    "kwargs": _safe_kwargs(kwargs or {}),
                    "in_cv": bool(self._cv_depth),
                },
            )
        )

    def on_cv(self, call_site: str, *, function: str, data: Any = None) -> None:
        """Record a CV consumer and associate learned transforms with it.

        Shared row identity is deliberately not enough evidence here.  A
        transformed array may be computed and abandoned while CV runs on the
        original rows.  Transform lineage carries the fit ordinal instead, so
        only preprocessing that actually contributed to the CV input is marked.
        """

        cv_capture = self.taint.capture(data)
        cv_lineage = self.taint.lineage_for(data)
        cv_origins = cv_lineage.transform_origins if cv_lineage is not None else frozenset()
        reused: list[int] = []
        reused_classes: list[str] = []
        for event in self.ctx.events:
            if event.kind is not EventKind.FIT or event.payload.get("in_cv"):
                continue
            if event.run_ordinal in cv_origins:
                event.payload["reused_by_cv"] = True
                reused.append(event.run_ordinal)
                cls = event.payload.get("class")
                if isinstance(cls, str) and cls:
                    reused_classes.append(cls)
        self._ordinal += 1
        self._record(
            RuntimeEvent(
                kind=EventKind.CV,
                call_site=call_site,
                run_ordinal=self._ordinal,
                payload={
                    "function": function,
                    "input": cv_capture,
                    "reused_fit_ordinals": reused,
                    "reused_fit_classes": reused_classes,
                },
            )
        )

    def captured_frames(self) -> list[Any]:
        return cast(list[Any], self.ctx.scratch.get("captured_frames", []))


def _safe_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in {"eval_set", "validation_data", "eval_data"}:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
    return safe


def _pipeline_components(instance: Any) -> list[str]:
    if instance is None:
        return []
    components: list[str] = []
    for attr in ("steps", "transformers", "transformer_list"):
        value = getattr(instance, attr, None)
        if not value:
            continue
        try:
            items = value.values() if isinstance(value, dict) else value
            for item in items:
                component = item[1] if isinstance(item, tuple) and len(item) > 1 else item
                components.append(type(component).__name__)
        except Exception:
            continue
    return components


@contextmanager
def watch(config: Config | None = None) -> Iterator[RuntimeSession]:
    cfg = (config or Config.load()).copy()
    session = RuntimeSession(cfg)
    session.__enter__()
    try:
        yield session
    except BaseException:
        session.__exit__(*sys.exc_info())
        raise
    else:
        session.__exit__(None, None, None)


def run_script_result(path: Path, argv: list[str], config: Config | None = None) -> AnalysisResult:
    cfg = (config or Config.load(start=path)).copy()
    result: AnalysisResult
    saved_argv = sys.argv[:]
    saved_path = sys.path[:]
    script = path.expanduser().resolve()
    sys.argv = [str(script), *argv]
    script_parent = str(script.parent)
    if script_parent not in sys.path:
        sys.path.insert(0, script_parent)
    status = 0
    try:
        with watch(cfg) as session:
            try:
                runpy.run_path(str(script), run_name="__main__")
            except SystemExit as exc:
                if exc.code is None:
                    status = 0
                elif isinstance(exc.code, bool):
                    # bool is an int subclass, but SystemExit(True) is the
                    # conventional status 1 and should be serialized as an
                    # actual integer rather than a JSON boolean.
                    status = int(exc.code)
                elif isinstance(exc.code, int):
                    status = exc.code
                else:
                    status = 1
                if status:
                    session.add_diagnostic(
                        "LP430", f"script exited with status {status}", error=True
                    )
            except KeyboardInterrupt:
                # Let the context manager remove hooks, then preserve the
                # user's interrupt instead of converting Ctrl-C into a normal
                # failed-script report.
                raise
            except BaseException as exc:
                status = 1
                session.add_diagnostic(
                    "LP431",
                    f"script raised {type(exc).__name__}: {exc}",
                    error=True,
                )
            result = session.result
    finally:
        sys.argv = saved_argv
        sys.path[:] = saved_path
    result.script_exit_status = status
    if status:
        result.completion = CompletionStatus.FAILED
        result.summary = summarize(
            result.findings,
            cfg.fail_on,
            cfg.gate_confidence,
            diagnostics_count=len(result.diagnostics),
        )
    return result


def run_script(path: Path, argv: list[str], config: Config | None = None) -> tuple[list[Finding], int]:
    """Compatibility API returning findings and the original script status."""

    result = run_script_result(path, argv, config)
    return result.findings, int(result.script_exit_status or 0)


class _PytestPlugin:
    def __init__(self, config: Config, fail: bool):
        self.config = config
        self.fail = fail
        self.session: RuntimeSession | None = None

    def pytest_runtest_setup(self, item: Any) -> None:  # pragma: no cover
        self.session = RuntimeSession(self.config.copy())
        self.session.__enter__()

    def pytest_runtest_teardown(self, item: Any) -> None:  # pragma: no cover
        session = self.session
        if session is None:
            return
        try:
            session.__exit__(None, None, None)
            messages = [f"[ml-leakproof {finding.rule_id}] {finding.message}" for finding in session.findings]
            if self.fail and messages:
                raise AssertionError("\n".join(messages))
            for message in messages:
                warnings.warn(message, stacklevel=2)
        finally:
            self.session = None


def pytest_plugin(config: Config | None = None, *, fail: bool = False) -> _PytestPlugin:
    return _PytestPlugin(config or Config.load(), fail)
