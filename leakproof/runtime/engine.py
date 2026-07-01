"""Runtime engine: watch() context manager, script runner, pytest plugin."""

from __future__ import annotations

import runpy
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

from ..core.config import Config
from ..core.context import RuntimeContext
from ..core.models import Finding, Layer
from ..core.registry import load_all
from ..core.rule import RuntimeRule
from .hooks import HookManager
from .taint import EventKind, RuntimeEvent, TaintTable


class RuntimeSession:
    """Holds taint state, dispatches events to runtime rules, collects findings."""

    def __init__(self, config: Config):
        self.config = config
        self.taint = TaintTable()
        self.ctx = RuntimeContext(config=config, taint=self.taint)
        self.rules: list[RuntimeRule] = [
            r for r in load_all(config, layers=(Layer.RUNTIME,)) if isinstance(r, RuntimeRule)
        ]
        self.findings: list[Finding] = []
        self._ordinal = 0
        self._hooks = HookManager(self)

    def __enter__(self) -> RuntimeSession:
        self._hooks.install()
        return self

    def __exit__(self, *exc) -> None:
        self._hooks.remove()
        self._finalize()

    def _record(self, event: RuntimeEvent) -> None:
        # We mark split membership live so the taint table is complete, but defer
        # rule dispatch to _finalize(): a transformer fit *before* the split can
        # only be judged once we know which rows end up in the held-out set.
        self.ctx.events.append(event)

    def _finalize(self) -> None:
        for event in self.ctx.events:
            for rule in self.rules:
                try:
                    for f in rule.on_event(event, self.ctx):
                        self.findings.append(f)
                except Exception as exc:  # pragma: no cover
                    warnings.warn(f"leakproof: runtime rule {rule.id} raised: {exc}", stacklevel=2)

    def on_split(self, call_site: str, train_h: list[str], test_h: list[str]) -> None:
        self.taint.mark_split(train_h, test_h)
        self._ordinal += 1
        self._record(
            RuntimeEvent(
                kind=EventKind.SPLIT,
                call_site=call_site,
                run_ordinal=self._ordinal,
                payload={"train": train_h, "test": test_h},
            )
        )

    def on_fit(self, call_site: str, cls: str, seen: list[str]) -> None:
        self._ordinal += 1
        self._record(
            RuntimeEvent(
                kind=EventKind.FIT,
                call_site=call_site,
                run_ordinal=self._ordinal,
                payload={"class": cls, "seen": seen},
            )
        )

    def on_score(self, call_site: str, cls: str, seen: list[str]) -> None:
        self._ordinal += 1
        self._record(
            RuntimeEvent(
                kind=EventKind.SCORE,
                call_site=call_site,
                run_ordinal=self._ordinal,
                payload={"class": cls, "seen": seen},
            )
        )

    def captured_frames(self) -> list:
        """Frames handed to splits (used by --from-script data discovery)."""
        return self.ctx.scratch.get("captured_frames", [])


@contextmanager
def watch(config: Config | None = None):
    """Instrument sklearn within the block; collect runtime findings.

    Usage::

        with leakproof.watch() as session:
            ... run training/eval ...
        print(session.findings)
    """
    cfg = config or Config.load()
    session = RuntimeSession(cfg)
    session.__enter__()
    try:
        yield session
    finally:
        session.__exit__(None, None, None)


def run_script(
    path: Path, argv: list[str], config: Config | None = None
) -> tuple[list[Finding], int]:
    """Execute a script under instrumentation; return (findings, exit_code)."""
    cfg = config or Config.load()
    saved_argv = sys.argv[:]
    sys.argv = [str(path), *argv]
    exit_code = 0
    with watch(cfg) as session:
        try:
            runpy.run_path(str(path), run_name="__main__")
        except SystemExit as exc:
            exit_code = int(exc.code) if isinstance(exc.code, int) else 0
        except Exception as exc:  # pragma: no cover - user script errors
            warnings.warn(f"leakproof: script raised {type(exc).__name__}: {exc}", stacklevel=2)
            exit_code = 1
        finally:
            sys.argv = saved_argv
    return session.findings, exit_code


class _PytestPlugin:
    def __init__(self, config: Config, fail: bool):
        self.config = config
        self.fail = fail
        self.session: RuntimeSession | None = None

    def pytest_runtest_setup(self, item):  # pragma: no cover - requires pytest
        self.session = RuntimeSession(self.config)
        self.session.__enter__()

    def pytest_runtest_teardown(self, item):  # pragma: no cover
        if self.session is None:
            return
        self.session.__exit__(None, None, None)
        for f in self.session.findings:
            msg = f"[leakproof {f.rule_id}] {f.message}"
            if self.fail:
                raise AssertionError(msg)
            warnings.warn(msg, stacklevel=2)
        self.session = None


def pytest_plugin(config: Config | None = None, *, fail: bool = False) -> _PytestPlugin:
    return _PytestPlugin(config or Config.load(), fail)
