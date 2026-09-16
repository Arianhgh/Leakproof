"""The ``ml-leakproof`` command-line interface."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Any

import typer

from .core.aggregate import (
    annotate_gateability,
    dedupe,
    filter_min_confidence,
    gate_failed,
    sort_findings,
    summarize,
)
from .core.config import Config, ConfigError
from .core.models import (
    AnalysisResult,
    CompletionStatus,
    Diagnostic,
    DiagnosticLevel,
    Layer,
    Location,
)

app = typer.Typer(
    add_completion=False,
    help="A hybrid static, data, and runtime leakage/evaluation-rigor checker for ML code.",
    no_args_is_help=True,
)

EXIT_CLEAN = 0
EXIT_FINDINGS = 1
EXIT_ERROR = 2
_FORMATS = {"terminal", "json", "sarif", "markdown"}


def _load_config(path: Path | None, *, start: Path | None = None) -> Config:
    return Config.load(path, start=start)


def _normalize_result(result: AnalysisResult, cfg: Config) -> AnalysisResult:
    """Apply deterministic aggregation without changing completion semantics."""

    result.findings = annotate_gateability(
        sort_findings(dedupe(result.findings)),
        gate=cfg.fail_on,
        gate_confidence=cfg.gate_confidence,
        profile=cfg.profile,
    )
    if result.diagnostics and result.completion is CompletionStatus.COMPLETE:
        result.completion = CompletionStatus.PARTIAL
    result.summary = summarize(
        result.findings,
        cfg.fail_on,
        cfg.gate_confidence,
        diagnostics_count=len(result.diagnostics),
    )
    result.metadata.setdefault("tool", "ml-leakproof")
    try:
        from . import __version__

        result.metadata.setdefault("version", __version__)
    except Exception:  # pragma: no cover - defensive serialization fallback
        pass
    return result


def _presentation_result(result: AnalysisResult, cfg: Config) -> AnalysisResult:
    """Return a report view while retaining gate decisions over all findings."""

    all_findings = result.findings
    displayed = filter_min_confidence(all_findings, cfg.min_confidence)
    summary = summarize(
        displayed,
        cfg.fail_on,
        cfg.gate_confidence,
        total_findings=len(all_findings),
        hidden_count=len(all_findings) - len(displayed),
        diagnostics_count=len(result.diagnostics),
        gate_findings=all_findings,
    )
    return replace(result, findings=displayed, summary=summary)


def _validate_report_request(formats: list[str] | None, output: Path | None) -> list[str]:
    selected = list(formats or ["terminal"])
    if not selected:
        selected = ["terminal"]
    invalid = sorted(set(selected) - _FORMATS)
    if invalid:
        raise ConfigError(f"unknown report format(s): {', '.join(invalid)}")
    if len(selected) > 1 and output is None:
        raise ConfigError("multiple report formats require --output")
    if output is not None:
        if output.exists() and output.is_dir():
            raise ConfigError(f"report output is a directory: {output}")
        if not output.parent.exists():
            raise ConfigError(f"report output directory does not exist: {output.parent}")
    return selected


def _report_path(output: Path, fmt: str, count: int) -> Path:
    if count == 1:
        return output
    suffix = output.suffix
    stem = output.name[: -len(suffix)] if suffix else output.name
    return output.with_name(f"{stem}.{fmt}{suffix}")


def _emit_reports(
    result: AnalysisResult,
    cfg: Config,
    formats: list[str] | None,
    output: Path | None,
    *,
    root: Path | None = None,
    color: bool = True,
) -> None:
    from . import report

    selected = _validate_report_request(formats, output)
    presented = _presentation_result(result, cfg)
    for fmt in selected:
        rendered = report.render(
            fmt,
            presented.findings,
            presented.summary,
            color=color,
            result=presented,
            root=root,
        )
        if output is not None:
            destination = _report_path(output, fmt, len(selected))
            destination.write_text(rendered, encoding="utf-8")
            typer.echo(f"wrote {fmt} report to {destination}", err=True)
        elif fmt == "terminal":
            from .report.terminal import print_report

            print_report(
                presented.findings,
                presented.summary,
                color=color,
                result=presented,
            )
        else:
            typer.echo(rendered)


def _finish(
    result: AnalysisResult,
    cfg: Config,
    *,
    formats: list[str] | None,
    output: Path | None,
    root: Path | None,
    color: bool,
) -> None:
    result = _normalize_result(result, cfg)
    try:
        _emit_reports(result, cfg, formats, output, root=root, color=color)
    except (OSError, ValueError) as exc:
        typer.secho(f"ml-leakproof report error: {exc}", fg="red", err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    if result.completion is CompletionStatus.FAILED:
        raise typer.Exit(EXIT_ERROR)
    if result.completion is not CompletionStatus.COMPLETE and not cfg.allow_partial:
        raise typer.Exit(EXIT_ERROR)
    if result.script_exit_status not in (None, 0):
        raise typer.Exit(EXIT_ERROR)
    raise typer.Exit(
        EXIT_FINDINGS
        if gate_failed(result.findings, cfg.fail_on, cfg.gate_confidence)
        else EXIT_CLEAN
    )


def _config_or_exit(path: Path | None, *, start: Path | None = None) -> Config:
    try:
        return _load_config(path, start=start)
    except (ConfigError, OSError, ValueError) as exc:
        typer.secho(f"ml-leakproof configuration error: {exc}", fg="red", err=True)
        raise typer.Exit(EXIT_ERROR) from exc


@app.command()
def check(
    paths: list[Path] = typer.Argument(None, help="Files or directories to scan."),
    select: list[str] = typer.Option(None, "--select", help="Rule globs to run."),
    ignore: list[str] = typer.Option(None, "--ignore", help="Rule globs to skip."),
    layers: str | None = typer.Option(None, "--layers", help="Static only for check."),
    exclude: list[str] = typer.Option(None, "--exclude", help="Additional path exclusions."),
    fail_on: str | None = typer.Option(None, "--fail-on", help="Severity gate."),
    min_confidence: float | None = typer.Option(
        None, "--min-confidence", help="Hide findings below this presentation threshold."
    ),
    gate_confidence: float | None = typer.Option(
        None, "--gate-confidence", help="Minimum confidence required to affect the exit code."
    ),
    profile: str | None = typer.Option(None, "--profile", help="ci|notebook|research."),
    fmt: list[str] = typer.Option(None, "--format", help="terminal|json|sarif|markdown."),
    output: Path | None = typer.Option(None, "--output", help="Write report(s) to file(s)."),
    explain: bool = typer.Option(False, "--explain", help="Attach optional LLM explanations."),
    fix: bool = typer.Option(False, "--fix", help="Apply verified autofixes and rescan."),
    fix_preview: bool = typer.Option(False, "--fix-preview", help="Show a fix diff without writing."),
    llm_triage: bool = typer.Option(False, "--llm-triage", help="Triage only parse-failed files."),
    allow_partial: bool = typer.Option(False, "--allow-partial", help="Permit a partial exit status."),
    config: Path | None = typer.Option(None, "--config", help="Path to a leakproof.toml."),
    no_color: bool = typer.Option(False, "--no-color", help="Disable colored output."),
) -> None:
    """Run the static layer without executing analyzed code."""

    from .static.engine import StaticEngine, discover_files

    scan_paths = [Path(p) for p in (paths or [Path(".")])]
    cfg = _config_or_exit(config, start=scan_paths[0] if scan_paths else Path.cwd())
    try:
        if layers is not None and any(
            item.strip() != Layer.STATIC.value for item in layers.split(",")
        ):
            raise ConfigError("check is static-only; --layers may contain only static")
        cfg.apply_cli(
            select=list(select) if select else None,
            ignore=list(ignore) if ignore else None,
            layers=layers.split(",") if layers else None,
            exclude=list(exclude) if exclude else None,
            fail_on=fail_on,
            min_confidence=min_confidence,
            gate_confidence=gate_confidence,
            profile=profile,
            allow_partial=True if allow_partial else None,
        )
        if Layer.STATIC not in cfg.layers:
            raise ConfigError("check is static-only and requires the static layer")
        _validate_report_request(fmt, output)
        if (explain or llm_triage) and not cfg.llm.enabled:
            raise ConfigError("--explain/--llm-triage require [llm].enabled = true")
    except (ConfigError, ValueError) as exc:
        typer.secho(f"ml-leakproof configuration error: {exc}", fg="red", err=True)
        raise typer.Exit(EXIT_ERROR) from exc

    try:
        engine = StaticEngine(cfg)
        result = engine.run_result(scan_paths, root=cfg.root)
        if llm_triage:
            from .llm.triage import triage_file

            failed_files = {
                str(diagnostic.location.file)
                for diagnostic in result.diagnostics
                if diagnostic.code == "LP000" and diagnostic.location and diagnostic.location.file
            }
            scan_root = engine._resolve_root(scan_paths, cfg.root)
            for file_path in discover_files(scan_paths, cfg.exclude, scan_root):
                if str(file_path) not in failed_files:
                    continue
                try:
                    triaged, diagnostics = triage_file(file_path, cfg)
                except Exception as exc:
                    result.diagnostics.append(
                        Diagnostic(
                            code="LP250",
                            message=f"LLM triage failed for {file_path}: {exc}",
                            level=DiagnosticLevel.WARNING,
                            layer=Layer.STATIC,
                            location=Location(file=file_path),
                        )
                    )
                    continue
                result.findings.extend(triaged)
                result.diagnostics.extend(diagnostics)
        if explain:
            from .llm.explain import explain_all

            gated = [
                finding
                for finding in result.findings
                if not finding.advisory_only
                and finding.severity.gate_rank >= cfg.fail_on.gate_rank
            ]
            explained, diagnostics = explain_all(gated, cfg)
            replacements = {old.key: new for old, new in zip(gated, explained)}
            result.findings = [
                replacements.get(finding.key, finding) for finding in result.findings
            ]
            result.diagnostics.extend(diagnostics)
        if fix or fix_preview:
            from .autofix import AutofixUnavailable, apply_fixes, preview_fixes

            try:
                if fix_preview and not fix:
                    for diff in preview_fixes(result.findings):
                        # Keep machine-readable report formats clean on
                        # stdout; previews are human-facing diagnostics.
                        typer.echo(diff, err=True)
                if fix:
                    applied = apply_fixes(result.findings, write=True)
                    for path, count in applied.items():
                        typer.echo(f"autofixed {count} issue(s) in {path}", err=True)
                    if applied:
                        result = StaticEngine(cfg).run_result(scan_paths, root=cfg.root)
            except AutofixUnavailable as exc:
                result.completion = CompletionStatus.PARTIAL
                result.diagnostics.append(
                    Diagnostic(
                        code="LP600",
                        message=str(exc),
                        level=DiagnosticLevel.ERROR,
                        layer=Layer.STATIC,
                    )
                )
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # pragma: no cover - command boundary
        typer.secho(f"ml-leakproof error: {exc}", fg="red", err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    _finish(result, cfg, formats=fmt, output=output, root=cfg.root, color=not no_color)


@app.command()
def run(
    script: Path = typer.Argument(..., help="Script to execute under runtime instrumentation."),
    args: list[str] = typer.Argument(None, help="Arguments passed to the script."),
    fmt: list[str] = typer.Option(None, "--format"),
    output: Path | None = typer.Option(None, "--output"),
    fail_on: str | None = typer.Option(None, "--fail-on"),
    min_confidence: float | None = typer.Option(None, "--min-confidence"),
    gate_confidence: float | None = typer.Option(None, "--gate-confidence"),
    profile: str | None = typer.Option(None, "--profile"),
    allow_partial: bool = typer.Option(False, "--allow-partial"),
    config: Path | None = typer.Option(None, "--config"),
    no_color: bool = typer.Option(False, "--no-color"),
) -> None:
    """Execute a script under runtime instrumentation and report its status."""

    from .runtime.engine import run_script_result

    cfg = _config_or_exit(config, start=script)
    try:
        cfg.apply_cli(
            fail_on=fail_on,
            min_confidence=min_confidence,
            gate_confidence=gate_confidence,
            profile=profile,
            allow_partial=True if allow_partial else None,
        )
        if Layer.RUNTIME not in cfg.layers:
            raise ConfigError("run requires the runtime layer")
        _validate_report_request(fmt, output)
        if not script.exists() or not script.is_file():
            raise ConfigError(f"script does not exist: {script}")
        script_stdout = io.StringIO()
        with redirect_stdout(script_stdout):
            result = run_script_result(script, list(args or []), cfg)
        # Keep stdout machine-readable for JSON/SARIF/Markdown consumers.  A
        # user's print() output remains visible, but is routed to stderr so it
        # cannot corrupt the selected report format.
        if output_text := script_stdout.getvalue():
            typer.echo(output_text, err=True, nl=False)
    except KeyboardInterrupt:
        raise
    except (ConfigError, ValueError) as exc:
        typer.secho(f"ml-leakproof error: {exc}", fg="red", err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    except Exception as exc:  # pragma: no cover
        typer.secho(f"ml-leakproof error: {exc}", fg="red", err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    _finish(result, cfg, formats=fmt, output=output, root=script.parent, color=not no_color)


@app.command(name="audit-data")
def audit_data_cmd(
    train: Path = typer.Option(..., "--train", help="CSV/Parquet training split."),
    test: Path = typer.Option(..., "--test", help="CSV/Parquet test split."),
    val: Path | None = typer.Option(None, "--val", help="Optional CSV/Parquet validation split."),
    target: str | None = typer.Option(None, "--target"),
    group: str | None = typer.Option(None, "--group"),
    time: str | None = typer.Option(None, "--time"),
    fmt: list[str] = typer.Option(None, "--format"),
    output: Path | None = typer.Option(None, "--output"),
    fail_on: str | None = typer.Option(None, "--fail-on"),
    min_confidence: float | None = typer.Option(None, "--min-confidence"),
    gate_confidence: float | None = typer.Option(None, "--gate-confidence"),
    profile: str | None = typer.Option(None, "--profile"),
    allow_partial: bool = typer.Option(False, "--allow-partial"),
    config: Path | None = typer.Option(None, "--config"),
    no_color: bool = typer.Option(False, "--no-color"),
) -> None:
    """Audit explicit train, validation, and test data files."""

    cfg = _config_or_exit(config, start=train)
    try:
        cfg.apply_cli(
            fail_on=fail_on,
            min_confidence=min_confidence,
            gate_confidence=gate_confidence,
            profile=profile,
            allow_partial=True if allow_partial else None,
        )
        if Layer.DATA not in cfg.layers:
            raise ConfigError("audit-data requires the data layer")
        _validate_report_request(fmt, output)
        import pandas as pd

        from .data.engine import DataEngine
        from .data.input import DataAuditInput

        def _read(path: Path) -> Any:
            if not path.exists() or not path.is_file():
                raise ConfigError(f"data input does not exist: {path}")
            if path.suffix.lower() in {".parquet", ".pq"}:
                return pd.read_parquet(path)
            if path.suffix.lower() == ".csv":
                return pd.read_csv(path)
            raise ConfigError(f"unsupported data input (expected CSV or Parquet): {path}")

        audit_input = DataAuditInput(
            train=_read(train),
            test=_read(test),
            val=_read(val) if val else None,
            target=target,
            group=group,
            time=time,
        )
        result = DataEngine(cfg).run_result(audit_input)
    except KeyboardInterrupt:
        raise
    except (ConfigError, ValueError, OSError, ImportError) as exc:
        typer.secho(f"ml-leakproof data error: {exc}", fg="red", err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    except Exception as exc:  # pragma: no cover
        typer.secho(f"ml-leakproof data error: {exc}", fg="red", err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    _finish(result, cfg, formats=fmt, output=output, root=train.parent, color=not no_color)


@app.command()
def rules(
    layer: str | None = typer.Option(None, "--layer", help="Filter by layer."),
    fmt: str = typer.Option("table", "--format", help="table|json"),
) -> None:
    """List rule IDs, categories, severities, and supported layers."""

    from .core.registry import all_rules

    if fmt not in {"table", "json"}:
        typer.secho("format must be table or json", fg="red", err=True)
        raise typer.Exit(EXIT_ERROR)
    try:
        selected_layer = Layer(layer) if layer else None
    except ValueError as exc:
        typer.secho(f"unknown layer: {layer}", fg="red", err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    rule_values = sorted(all_rules().values(), key=lambda rule: rule.id)
    if selected_layer:
        rule_values = [rule for rule in rule_values if selected_layer in rule.layers]
    records: list[dict[str, Any]] = [
        {
            "id": rule.id,
            "name": rule.name,
            "category": rule.category.value,
            "severity": rule.severity.value,
            "layers": [item.value for item in rule.layers],
            "advisory_only": rule.advisory_only,
        }
        for rule in rule_values
    ]
    if fmt == "json":
        typer.echo(json.dumps(records, indent=2, ensure_ascii=False))
        return
    from rich.console import Console
    from rich.table import Table

    table = Table(show_header=True, header_style="bold")
    for column in ("ID", "Name", "Category", "Severity", "Layers", "Advisory"):
        table.add_column(column)
    for record in records:
        table.add_row(
            str(record["id"]),
            str(record["name"]),
            str(record["category"]),
            str(record["severity"]),
            ",".join(str(value) for value in record["layers"]),
            "yes" if record["advisory_only"] else "no",
        )
    Console().print(table)


@app.command()
def explain(rule_id: str = typer.Argument(..., help="Rule ID, e.g. P001.")) -> None:
    """Print a rule rationale and packaged leaky/clean examples."""

    from .core.registry import all_rules

    rule = all_rules().get(rule_id)
    if rule is None:
        typer.secho(f"unknown rule: {rule_id}", fg="red", err=True)
        raise typer.Exit(EXIT_ERROR)
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print(Panel.fit(f"[bold]{rule.id}[/bold] — {rule.name}", style="cyan"))
    console.print(f"[bold]Category:[/bold] {rule.category.value}")
    console.print(f"[bold]Severity:[/bold] {rule.severity.value}")
    console.print(f"[bold]Layers:[/bold] {', '.join(item.value for item in rule.layers)}")
    console.print()
    console.print(rule.rationale or "(no rationale provided)")
    if rule.references:
        console.print("\n[bold]References:[/bold]")
        for reference in rule.references:
            console.print(f"  - {reference}")
    _print_fixture(console, rule_id, "leaky")
    _print_fixture(console, rule_id, "clean")


def _print_fixture(console: Any, rule_id: str, kind: str) -> None:
    from importlib import resources

    base = resources.files("ml_leakproof").joinpath("examples").joinpath(kind)
    matches = sorted(
        (item for item in base.iterdir() if item.name.startswith(f"{rule_id}__")),
        key=lambda item: item.name,
    )
    if not matches:
        return
    from rich.syntax import Syntax

    resource = matches[0]
    console.print(f"\n[bold]{kind} example[/bold] ({resource.name}):")
    console.print(Syntax(resource.read_text(encoding="utf-8"), "python", line_numbers=True))


@app.command()
def corpus(
    manifest: Path = typer.Option(..., "--manifest", help="JSON manifest of pinned repositories."),
    workdir: Path = typer.Option(Path(".leakproof-corpus"), "--workdir"),
    output: Path = typer.Option(Path("corpus-report.json"), "--output"),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    """Run the bounded, source-only corpus benchmark."""

    from .corpus import run_corpus

    cfg = _config_or_exit(config, start=manifest)
    try:
        if output.exists() and output.is_dir():
            raise ConfigError(f"corpus output is a directory: {output}")
        if not output.parent.exists():
            raise ConfigError(f"corpus output directory does not exist: {output.parent}")
        report = run_corpus(manifest, workdir, cfg)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # pragma: no cover
        typer.secho(f"ml-leakproof corpus error: {exc}", fg="red", err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    typer.echo(f"wrote corpus report to {output}", err=True)
    if report["n_failed"] or report["n_partial"]:
        raise typer.Exit(EXIT_ERROR)


@app.command()
def version() -> None:
    """Print the installed Leakproof version."""

    from . import __version__

    typer.echo(__version__)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
