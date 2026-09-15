"""Leakproof: static, dataset, and runtime leakage analysis for ML projects."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ._version import __version__
from .core.aggregate import annotate_gateability, dedupe, sort_findings, summarize
from .core.config import Config, ConfigError
from .core.models import (
    AnalysisError,
    AnalysisResult,
    Category,
    CompletionStatus,
    Coverage,
    Diagnostic,
    DiagnosticLevel,
    Finding,
    Fix,
    Layer,
    Location,
    Severity,
)

__all__ = [
    "analyze",
    "check",
    "analyze_data",
    "audit_data",
    "watch",
    "Config",
    "ConfigError",
    "AnalysisError",
    "AnalysisResult",
    "CompletionStatus",
    "Coverage",
    "Diagnostic",
    "DiagnosticLevel",
    "Finding",
    "Severity",
    "Category",
    "Layer",
    "Location",
    "Fix",
    "__version__",
]


def _finalize_result(result: AnalysisResult, config: Config) -> AnalysisResult:
    findings = sort_findings(dedupe(result.findings))
    findings = annotate_gateability(
        findings,
        gate=config.fail_on,
        gate_confidence=config.gate_confidence,
        profile=config.profile,
    )
    result.findings = findings
    result.summary = summarize(
        findings,
        config.fail_on,
        config.gate_confidence,
        diagnostics_count=len(result.diagnostics),
    )
    result.metadata.setdefault("tool", "ml-leakproof")
    result.metadata.setdefault("version", __version__)
    return result


def _config_for(config: Config | None, *, start: Path | None = None) -> Config:
    if config is not None:
        config.validate()
        return config.copy()
    return Config.load(start=start)


def analyze(
    paths: str | Path | Iterable[str | Path],
    config: Config | None = None,
    *,
    root: Path | None = None,
) -> AnalysisResult:
    """Run static analysis and return findings plus truthful coverage."""

    from .static.engine import StaticEngine

    if isinstance(paths, (str, Path)):
        paths = [paths]
    path_list = [Path(path) for path in paths]
    cfg = _config_for(config, start=path_list[0] if path_list else root)
    result = AnalysisResult()
    if Layer.STATIC not in cfg.layers:
        from .core.models import Diagnostic, DiagnosticLevel

        result.completion = CompletionStatus.FAILED
        result.diagnostics.append(
            Diagnostic(
                code="LP010",
                message="check/analyze is static-only; configuration must include the static layer",
                level=DiagnosticLevel.ERROR,
                layer=Layer.STATIC,
            )
        )
        return _finalize_result(result, cfg)
    result = StaticEngine(cfg).run_result(path_list, root=root)
    return _finalize_result(result, cfg)


def check(
    paths: str | Path | Iterable[str | Path],
    config: Config | None = None,
    *,
    root: Path | None = None,
    allow_partial: bool | None = None,
) -> list[Finding]:
    """Compatibility list API for static analysis.

    Incomplete analysis raises :class:`AnalysisError` unless partial results are
    explicitly allowed in the configuration or call.
    """

    cfg = _config_for(config, start=Path(paths[0]) if isinstance(paths, list) and paths else (Path(paths) if isinstance(paths, (str, Path)) else root))
    if allow_partial is not None:
        cfg.allow_partial = allow_partial
    result = analyze(paths, config=cfg, root=root)
    if result.completion is not CompletionStatus.COMPLETE and not cfg.allow_partial:
        raise AnalysisError("static analysis did not complete", result)
    return result.findings


def analyze_data(
    train: Any,
    test: Any,
    *,
    val: Any = None,
    target: str | None = None,
    group: str | None = None,
    time: str | None = None,
    feature_types: dict[str, str] | None = None,
    config: Config | None = None,
) -> AnalysisResult:
    """Audit explicit pandas splits and return findings plus diagnostics."""

    from .data.engine import DataEngine
    from .data.input import DataAuditInput

    cfg = _config_for(config)
    if Layer.DATA not in cfg.layers:
        result = AnalysisResult(
            completion=CompletionStatus.FAILED,
            diagnostics=[
                Diagnostic(
                    code="LP310",
                    message="data auditing was requested but the data layer is disabled",
                    level=DiagnosticLevel.ERROR,
                    layer=Layer.DATA,
                )
            ],
        )
        return _finalize_result(result, cfg)
    result = DataEngine(cfg).run_result(
        DataAuditInput(
            train=train,
            test=test,
            val=val,
            target=target,
            group=group,
            time=time,
            feature_types=feature_types,
        )
    )
    return _finalize_result(result, cfg)


def audit_data(
    train: Any,
    test: Any,
    *,
    val: Any = None,
    target: str | None = None,
    group: str | None = None,
    time: str | None = None,
    feature_types: dict[str, str] | None = None,
    config: Config | None = None,
    allow_partial: bool | None = None,
) -> list[Finding]:
    """Compatibility list API for dataset auditing."""

    cfg = _config_for(config)
    if allow_partial is not None:
        cfg.allow_partial = allow_partial
    result = analyze_data(
        train,
        test,
        val=val,
        target=target,
        group=group,
        time=time,
        feature_types=feature_types,
        config=cfg,
    )
    if result.completion is not CompletionStatus.COMPLETE and not cfg.allow_partial:
        raise AnalysisError("data analysis did not complete", result)
    return result.findings


def watch(config: Config | None = None) -> Any:
    """Return a runtime instrumentation context manager."""

    from .runtime.engine import watch as _watch

    return _watch(_config_for(config))
