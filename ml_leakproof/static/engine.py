"""Static analysis engine: discovery, parsing, rule isolation, and coverage."""

from __future__ import annotations

import ast
import hashlib
import os
import tokenize
import warnings
from pathlib import Path
from typing import Any, cast

from ..core.config import Config, ConfigError
from ..core.context import StaticContext
from ..core.models import (
    AnalysisResult,
    CompletionStatus,
    Diagnostic,
    DiagnosticLevel,
    Finding,
    Layer,
    Location,
)
from ..core.registry import all_rules, consume_diagnostics, load_all
from ..core.rule import StaticRule
from ..core.suppression import FileSuppressions, path_excluded
from .adapters import AdapterRegistry, consume_adapter_diagnostics, load_adapters
from .dataflow import DataFlow

SUPPORTED_SUFFIXES = {".py", ".ipynb"}


def discover_files(paths: list[Path], excludes: list[str], root: Path) -> list[Path]:
    """Discover supported files while pruning excluded directories early."""

    out: list[Path] = []
    for raw_path in paths:
        path = raw_path.expanduser()
        if not path.exists() or path_excluded(path, excludes, root=root):
            continue
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_SUFFIXES:
                out.append(path)
            continue
        if not path.is_dir():
            continue
        for directory, dir_names, file_names in os.walk(path):
            directory_path = Path(directory)
            dir_names[:] = sorted(
                name
                for name in dir_names
                if not path_excluded(directory_path / name, excludes, root=root)
            )
            for name in sorted(file_names):
                candidate = directory_path / name
                if candidate.suffix.lower() in SUPPORTED_SUFFIXES and not path_excluded(
                    candidate, excludes, root=root
                ):
                    out.append(candidate)
    seen: set[Path] = set()
    unique: list[Path] = []
    for file_path in out:
        resolved = file_path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(file_path)
    return unique


def _load_source(
    path: Path,
) -> tuple[str, dict[int, tuple[int, int, str | None]] | None, dict[str, Any]]:
    """Load source using its declared Python encoding and notebook metadata."""

    if path.suffix.lower() == ".ipynb":
        from .notebook import notebook_to_source

        source, line_map, metadata = notebook_to_source(path)
        return source, line_map, metadata
    with tokenize.open(str(path)) as handle:
        return handle.read(), None, {}


class StaticEngine:
    def __init__(self, config: Config, adapters: AdapterRegistry | None = None):
        self.config = config
        self.config.validate()
        self.adapters = adapters or load_adapters()
        self._initial_diagnostics = consume_diagnostics() + consume_adapter_diagnostics()
        self.rules: list[StaticRule] = [
            rule
            for rule in load_all(config, layers=(Layer.STATIC,))
            if isinstance(rule, StaticRule) and Layer.STATIC in rule.layers
        ]
        self._initial_diagnostics.extend(consume_diagnostics())
        self._known_rule_ids = set(all_rules())

    def run_result(self, paths: list[Path], *, root: Path | None = None) -> AnalysisResult:
        normalized = [Path(path).expanduser() for path in paths]
        root = self._resolve_root(normalized, root)
        result = AnalysisResult()
        result.coverage.requested_inputs = [str(path) for path in normalized]
        result.diagnostics.extend(self._initial_diagnostics)
        self._initial_diagnostics = []

        if not normalized:
            result.completion = CompletionStatus.FAILED
            result.diagnostics.append(
                Diagnostic(
                    code="LP002",
                    message="no scan paths were provided",
                    level=DiagnosticLevel.ERROR,
                    layer=Layer.STATIC,
                )
            )
            return result

        for path in normalized:
            if not path.exists():
                result.coverage.failed_inputs.append(str(path))
                result.diagnostics.append(
                    Diagnostic(
                        code="LP003",
                        message=f"scan path does not exist: {path}",
                        level=DiagnosticLevel.ERROR,
                        layer=Layer.STATIC,
                        location=Location(file=path),
                    )
                )
            elif path.is_file() and path.suffix.lower() not in SUPPORTED_SUFFIXES:
                result.coverage.skipped_inputs.append(str(path))
                result.diagnostics.append(
                    Diagnostic(
                        code="LP004",
                        message=f"unsupported scan input (expected .py or .ipynb): {path}",
                        level=DiagnosticLevel.WARNING,
                        layer=Layer.STATIC,
                        location=Location(file=path),
                    )
                )

        if result.coverage.failed_inputs:
            result.completion = CompletionStatus.FAILED

        files = discover_files(normalized, self.config.exclude, root)
        if not files:
            result.completion = (
                CompletionStatus.FAILED
                if result.coverage.failed_inputs
                else CompletionStatus.PARTIAL
            )
            result.diagnostics.append(
                Diagnostic(
                    code="LP005",
                    message="no eligible Python or notebook files were found",
                    level=DiagnosticLevel.WARNING,
                    layer=Layer.STATIC,
                    details={"requested_paths": [str(path) for path in normalized]},
                )
            )
            return result

        for file_path in files:
            file_result = self.run_file_result(file_path)
            result.findings.extend(file_result.findings)
            result.diagnostics.extend(file_result.diagnostics)
            result.coverage.analyzed_inputs.extend(file_result.coverage.analyzed_inputs)
            result.coverage.failed_inputs.extend(file_result.coverage.failed_inputs)
            result.coverage.skipped_inputs.extend(file_result.coverage.skipped_inputs)
            result.coverage.executed_checks.extend(file_result.coverage.executed_checks)
            result.coverage.unavailable_checks.extend(file_result.coverage.unavailable_checks)
            result.coverage.notes.extend(file_result.coverage.notes)
            if file_result.completion is not CompletionStatus.COMPLETE:
                result.completion = CompletionStatus.PARTIAL

        if result.coverage.failed_inputs and not result.coverage.analyzed_inputs:
            result.completion = CompletionStatus.FAILED
        elif result.diagnostics and result.completion is CompletionStatus.COMPLETE:
            # Diagnostics from a rule/parser/import failure are incomplete
            # coverage, even if some files were successfully checked.
            result.completion = CompletionStatus.PARTIAL
        result.coverage.executed_checks = sorted(set(result.coverage.executed_checks))
        result.coverage.unavailable_checks = sorted(set(result.coverage.unavailable_checks))
        return result

    def run(self, paths: list[Path], *, root: Path | None = None) -> list[Finding]:
        """Compatibility list API; use ``run_result`` to retain diagnostics."""

        return self.run_result(paths, root=root).findings

    def run_file_result(self, path: Path) -> AnalysisResult:
        result = AnalysisResult()
        result.coverage.requested_inputs = [str(path)]
        try:
            source, line_map, notebook_metadata = _load_source(path)
        except Exception as exc:
            result.completion = CompletionStatus.FAILED
            result.coverage.failed_inputs.append(str(path))
            result.diagnostics.append(
                Diagnostic(
                    code="LP006",
                    message=f"could not read {path}: {exc}",
                    level=DiagnosticLevel.ERROR,
                    layer=Layer.STATIC,
                    location=Location(file=path),
                    details={"error_type": type(exc).__name__},
                )
            )
            return result

        result.coverage.analyzed_inputs.append(str(path))
        if notebook_metadata.get("stored_cell_order"):
            result.coverage.notes.append(
                "notebook stored cell order analyzed; execution history was not reconstructed"
            )
            for cell_index in notebook_metadata.get("non_python_cells", []):
                result.completion = CompletionStatus.PARTIAL
                result.diagnostics.append(Diagnostic(
                    code="LP008",
                    message=f"notebook cell {cell_index} contains magic or shell code that was not analyzed",
                    level=DiagnosticLevel.WARNING,
                    layer=Layer.STATIC,
                    location=Location(file=path, cell_index=cell_index),
                ))
            for malformed in notebook_metadata.get("malformed_cells", []):
                result.completion = CompletionStatus.PARTIAL
                result.diagnostics.append(
                    Diagnostic(
                        code="LP007",
                        message=(
                            f"notebook cell {malformed['cell_index']} is malformed; "
                            "that cell was skipped while later cells were analyzed"
                        ),
                        level=DiagnosticLevel.WARNING,
                        layer=Layer.STATIC,
                        location=Location(
                            file=path,
                            line=malformed.get("line"),
                            context_label=f"notebook cell {malformed['cell_index']}",
                            cell_id=malformed.get("cell_id"),
                            cell_index=malformed.get("cell_index"),
                            cell_line=malformed.get("line"),
                        ),
                        details={"cell": malformed},
                    )
                )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            result.completion = CompletionStatus.PARTIAL
            result.diagnostics.append(
                Diagnostic(
                    code="LP000",
                    message=f"could not parse file; static analysis skipped: {exc.msg}",
                    level=DiagnosticLevel.ERROR,
                    layer=Layer.STATIC,
                    location=Location(file=path, line=exc.lineno, col=exc.offset),
                    details={"error_type": "SyntaxError"},
                )
            )
            result.coverage.notes.append("parse failure; no execution history was reconstructed")
            return result

        source_lines = source.splitlines()
        suppressions = FileSuppressions.parse(
            source,
            known_rule_ids=self._known_rule_ids,
        )
        result.diagnostics.extend(suppressions.diagnostics(path))
        if result.diagnostics:
            result.completion = CompletionStatus.PARTIAL
        if suppressions.ignore_file:
            result.coverage.notes.append("file suppressed by an ignore-file directive")
            return result

        try:
            dataflow = DataFlow(tree, self.adapters, source_lines)
        except Exception as exc:
            result.completion = CompletionStatus.PARTIAL
            result.diagnostics.append(
                Diagnostic(
                    code="LP110",
                    message=f"static data-flow analysis failed for {path}: {exc}",
                    level=DiagnosticLevel.ERROR,
                    layer=Layer.STATIC,
                    location=Location(file=path),
                    details={"error_type": type(exc).__name__},
                )
            )
            return result

        context = StaticContext(
            module_path=path,
            tree=tree,
            source=source,
            source_lines=source_lines,
            dataflow=dataflow,
            adapters=self.adapters,
            config=self.config,
            line_map=cast(
                dict[int, tuple[int, int] | tuple[int, int, str | None]] | None,
                line_map,
            ),
            notebook_metadata=notebook_metadata,
        )
        for rule in self.rules:
            result.coverage.executed_checks.append(rule.id)
            try:
                for finding in rule.check(context):
                    finding = _attach_snippet(finding, context)
                    finding = _attach_analysis_metadata(finding, path, notebook=bool(line_map))
                    if not suppressions.suppresses(finding):
                        result.findings.append(finding)
            except Exception as exc:
                result.completion = CompletionStatus.PARTIAL
                result.coverage.failed_inputs.append(str(path))
                result.diagnostics.append(
                    Diagnostic(
                        code="LP120",
                        message=f"rule {rule.id} failed on {path}: {exc}",
                        level=DiagnosticLevel.ERROR,
                        layer=Layer.STATIC,
                        rule_id=rule.id,
                        location=Location(file=path),
                        details={"error_type": type(exc).__name__},
                    )
                )
        return result

    def run_file(self, path: Path) -> list[Finding]:
        return self.run_file_result(path).findings

    def _resolve_root(self, paths: list[Path], root: Path | None) -> Path:
        if root is not None:
            return root.expanduser().resolve()
        if self.config.root is not None:
            return self.config.root.expanduser().resolve()
        existing = [path.resolve() for path in paths if path.exists()]
        if len(existing) <= 1:
            return (existing[0].parent if existing and existing[0].is_file() else Path.cwd()).resolve()
        parents = {(path if path.is_dir() else path.parent) for path in existing}
        common = Path(os.path.commonpath([str(parent) for parent in parents]))
        # Multiple unrelated roots make automatic project-config discovery
        # ambiguous.  Require an explicit root so one config governs the scan.
        if len(parents) > 1 and not (common / "pyproject.toml").exists() and not (
            common / "leakproof.toml"
        ).exists():
            raise ConfigError("multiple scan roots require an explicit root/configuration")
        return common


def _attach_snippet(finding: Finding, context: StaticContext) -> Finding:
    from dataclasses import replace

    location = finding.location
    if location.snippet is None and location.line is not None:
        snippet = context.snippet(location.line)
        if snippet is not None:
            location = replace(location, snippet=snippet.strip())
    if context.is_notebook and location.line is not None and context.line_map:
        mapped = context.line_map.get(location.line)
        if mapped:
            if len(mapped) == 2:
                cell_index, cell_line = mapped
                cell_id = None
            else:
                cell_index, cell_line, cell_id = mapped
            location = replace(
                location,
                cell_id=location.cell_id or cell_id,
                cell_index=location.cell_index if location.cell_index is not None else cell_index,
                cell_line=location.cell_line if location.cell_line is not None else cell_line,
            )
    return replace(finding, location=location)


def _attach_analysis_metadata(finding: Finding, path: Path, *, notebook: bool) -> Finding:
    """Attach the source snapshot used for safe autofix verification."""

    from dataclasses import replace

    evidence = dict(finding.evidence)
    try:
        evidence.setdefault("analysis_source_sha256", hashlib.sha256(path.read_bytes()).hexdigest())
    except OSError:
        pass
    fix = finding.fix
    if notebook and fix is not None and fix.autofixable:
        # Notebook JSON rewriting is intentionally out of scope for this release line.
        from ..core.models import Fix

        fix = Fix(
            summary=fix.summary,
            suggested_diff=fix.suggested_diff,
            autofixable=False,
        )
    return replace(finding, evidence=evidence, fix=fix)
