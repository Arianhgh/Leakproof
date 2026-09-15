"""Data-layer orchestration with validation and truthful coverage."""

from __future__ import annotations

from ..core.config import Config
from ..core.context import DataContext
from ..core.models import (
    AnalysisResult,
    CompletionStatus,
    Diagnostic,
    DiagnosticLevel,
    Finding,
    Layer,
)
from ..core.registry import consume_diagnostics, load_all
from ..core.rule import DataRule
from .input import DataAuditInput, DataInputError


class DataEngine:
    def __init__(self, config: Config):
        self.config = config
        self.config.validate()
        self.rules: list[DataRule] = [
            rule
            for rule in load_all(config, layers=(Layer.DATA,))
            if isinstance(rule, DataRule) and Layer.DATA in rule.layers
        ]
        self._initial_diagnostics = consume_diagnostics()

    def run_result(self, audit_input: DataAuditInput) -> AnalysisResult:
        result = AnalysisResult()
        result.coverage.requested_inputs = [name for name, _ in audit_input.splits()]
        result.diagnostics.extend(self._initial_diagnostics)
        self._initial_diagnostics = []
        try:
            split_names = audit_input.validate()
        except DataInputError as exc:
            result.completion = CompletionStatus.FAILED
            result.diagnostics.append(
                Diagnostic(
                    code="LP300",
                    message=str(exc),
                    level=DiagnosticLevel.ERROR,
                    layer=Layer.DATA,
                )
            )
            result.coverage.failed_inputs.extend(result.coverage.requested_inputs)
            return result

        configured = set(audit_input.train.columns)
        unknown_hash_columns = sorted(
            (set(self.config.data.hash_include) | set(self.config.data.hash_exclude)) - configured,
            key=str,
        )
        if unknown_hash_columns:
            result.completion = CompletionStatus.FAILED
            result.diagnostics.append(
                Diagnostic(
                    code="LP301",
                    message=f"configured hash column(s) do not exist: {unknown_hash_columns}",
                    level=DiagnosticLevel.ERROR,
                    layer=Layer.DATA,
                )
            )
            result.coverage.failed_inputs.extend(result.coverage.requested_inputs)
            return result

        result.coverage.analyzed_inputs.extend(split_names)
        result.coverage.notes.extend(audit_input.quality_notes())
        result.coverage.notes.extend(
            [
                "overlap is reported as an observation under the declared split policy",
                "target predictivity and similarity-only checks are advisory",
            ]
        )
        ctx = DataContext(
            audit_input=audit_input,
            config=self.config,
            diagnostics=result.diagnostics,
            coverage=result.coverage,
        )
        for rule in self.rules:
            result.coverage.executed_checks.append(rule.id)
            try:
                result.findings.extend(rule.check(ctx))
            except Exception as exc:
                result.completion = CompletionStatus.PARTIAL
                result.diagnostics.append(
                    Diagnostic(
                        code="LP320",
                        message=f"data rule {rule.id} failed: {exc}",
                        level=DiagnosticLevel.ERROR,
                        layer=Layer.DATA,
                        rule_id=rule.id,
                        details={"error_type": type(exc).__name__},
                    )
                )
        if result.diagnostics and result.completion is CompletionStatus.COMPLETE:
            result.completion = CompletionStatus.PARTIAL
        result.coverage.executed_checks = sorted(set(result.coverage.executed_checks))
        result.coverage.unavailable_checks = sorted(set(result.coverage.unavailable_checks))
        result.coverage.sampled_checks = sorted(set(result.coverage.sampled_checks))
        return result

    def run(self, audit_input: DataAuditInput) -> list[Finding]:
        """Compatibility list API; use ``run_result`` for diagnostics."""

        return self.run_result(audit_input).findings
