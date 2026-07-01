"""leakproof command-line interface (typer)."""

from __future__ import annotations

from pathlib import Path

import typer

from .core.aggregate import dedupe, gate_failed, sort_findings, summarize
from .core.config import Config

app = typer.Typer(
    add_completion=False,
    help="A hybrid (static + data + runtime) leakage and evaluation-rigor checker for ML code.",
    no_args_is_help=True,
)

EXIT_CLEAN = 0
EXIT_FINDINGS = 1
EXIT_ERROR = 2


def _load_config(config: Path | None) -> Config:
    return Config.load(config) if config else Config.load()


def _emit_reports(findings, summary, formats: list[str], output: Path | None, color: bool) -> None:
    from . import report

    formats = formats or ["terminal"]
    for fmt in formats:
        if fmt == "terminal":
            from .report.terminal import print_report

            print_report(findings, summary, color=color)
            continue
        text = report.render(fmt, findings, summary, color=color)
        if output:
            out_path = output if len(formats) == 1 else output.with_suffix(f".{fmt}")
            out_path.write_text(text, encoding="utf-8")
            typer.echo(f"wrote {fmt} report to {out_path}")
        else:
            typer.echo(text)


@app.command()
def check(
    paths: list[Path] = typer.Argument(None, help="Files or directories to scan."),
    select: list[str] = typer.Option(None, "--select", help="Rule globs to run (e.g. P001 C00* D*)."),
    ignore: list[str] = typer.Option(None, "--ignore", help="Rule globs to skip."),
    layers: str | None = typer.Option(None, "--layers", help="Comma list: static,data,runtime."),
    fail_on: str | None = typer.Option(None, "--fail-on", help="Severity gate (default: high)."),
    fmt: list[str] = typer.Option(None, "--format", help="terminal|json|sarif|markdown (repeatable)."),
    output: Path | None = typer.Option(None, "--output", help="Write report to file."),
    explain: bool = typer.Option(False, "--explain", help="Attach LLM explanations (requires llm.enabled)."),
    fix: bool = typer.Option(False, "--fix", help="Apply autofixable fixes only."),
    llm_triage: bool = typer.Option(False, "--llm-triage", help="LLM-propose sites in unparseable files."),
    config: Path | None = typer.Option(None, "--config", help="Path to a leakproof.toml."),
    no_color: bool = typer.Option(False, "--no-color", help="Disable colored output."),
) -> None:
    """Run the static layer (the default zero-setup analysis)."""
    from .static.engine import StaticEngine

    cfg = _load_config(config)
    cfg.apply_cli(
        select=list(select) if select else None,
        ignore=list(ignore) if ignore else None,
        layers=layers.split(",") if layers else None,
        fail_on=fail_on,
    )
    scan_paths = [Path(p) for p in (paths or [Path(".")])]
    root = Path.cwd()

    try:
        engine = StaticEngine(cfg)
        findings = engine.run(scan_paths, root=root)
    except Exception as exc:  # pragma: no cover
        typer.secho(f"leakproof error: {exc}", fg="red", err=True)
        raise typer.Exit(EXIT_ERROR) from exc

    if llm_triage and cfg.llm.enabled:
        from .llm.triage import triage_file
        from .static.engine import discover_files

        for f in discover_files(scan_paths, cfg.exclude, root):
            findings.extend(triage_file(f, cfg))

    findings = sort_findings(dedupe(findings))

    if fix:
        from .autofix import apply_fixes

        applied = apply_fixes(findings)
        for path, n in applied.items():
            typer.echo(f"autofixed {n} issue(s) in {path}")

    if explain and cfg.llm.enabled:
        from .llm.explain import explain_all

        gated = [f for f in findings if f.severity.gate_rank >= cfg.fail_on.gate_rank]
        explained = {id(f): e for f, e in zip(gated, explain_all(gated, cfg))}
        findings = [explained.get(id(f), f) for f in findings]

    summary = summarize(findings, cfg.fail_on)
    _emit_reports(findings, summary, list(fmt or []), output, color=not no_color)

    raise typer.Exit(EXIT_FINDINGS if gate_failed(findings, cfg.fail_on) else EXIT_CLEAN)


@app.command()
def run(
    script: Path = typer.Argument(..., help="Script to execute under runtime instrumentation."),
    args: list[str] = typer.Argument(None, help="Arguments passed to the script."),
    fmt: list[str] = typer.Option(None, "--format"),
    output: Path | None = typer.Option(None, "--output"),
    fail_on: str | None = typer.Option(None, "--fail-on"),
    config: Path | None = typer.Option(None, "--config"),
    no_color: bool = typer.Option(False, "--no-color"),
) -> None:
    """Execute a script under runtime instrumentation and report runtime findings."""
    from .runtime.engine import run_script

    cfg = _load_config(config)
    cfg.apply_cli(fail_on=fail_on)
    try:
        findings, _ = run_script(script, list(args or []), cfg)
    except Exception as exc:  # pragma: no cover
        typer.secho(f"leakproof error: {exc}", fg="red", err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    findings = sort_findings(dedupe(findings))
    summary = summarize(findings, cfg.fail_on)
    _emit_reports(findings, summary, list(fmt or []), output, color=not no_color)
    raise typer.Exit(EXIT_FINDINGS if gate_failed(findings, cfg.fail_on) else EXIT_CLEAN)


@app.command(name="audit-data")
def audit_data_cmd(
    train: Path = typer.Option(..., "--train", help="CSV/Parquet path for the training split."),
    test: Path = typer.Option(..., "--test", help="CSV/Parquet path for the test split."),
    val: Path | None = typer.Option(None, "--val"),
    target: str | None = typer.Option(None, "--target"),
    group: str | None = typer.Option(None, "--group"),
    time: str | None = typer.Option(None, "--time"),
    fmt: list[str] = typer.Option(None, "--format"),
    output: Path | None = typer.Option(None, "--output"),
    fail_on: str | None = typer.Option(None, "--fail-on"),
    config: Path | None = typer.Option(None, "--config"),
    no_color: bool = typer.Option(False, "--no-color"),
) -> None:
    """Run the data layer on explicit train/test (and optional val) files."""
    cfg = _load_config(config)
    cfg.apply_cli(fail_on=fail_on)
    try:
        import pandas as pd

        from .data.engine import DataEngine
        from .data.input import DataAuditInput

        def _read(p: Path):
            return pd.read_parquet(p) if p.suffix in (".parquet", ".pq") else pd.read_csv(p)

        ai = DataAuditInput(
            train=_read(train),
            test=_read(test),
            val=_read(val) if val else None,
            target=target,
            group=group,
            time=time,
        )
        findings = DataEngine(cfg).run(ai)
    except Exception as exc:
        typer.secho(f"leakproof error: {exc}", fg="red", err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    findings = sort_findings(dedupe(findings))
    summary = summarize(findings, cfg.fail_on)
    _emit_reports(findings, summary, list(fmt or []), output, color=not no_color)
    raise typer.Exit(EXIT_FINDINGS if gate_failed(findings, cfg.fail_on) else EXIT_CLEAN)


@app.command()
def rules(
    layer: str | None = typer.Option(None, "--layer", help="Filter by layer."),
    fmt: str = typer.Option("table", "--format", help="table|json"),
) -> None:
    """List rules: id, name, category, severity, layers."""
    from .core.registry import all_rules

    rs = sorted(all_rules().values(), key=lambda r: r.id)
    if layer:
        rs = [r for r in rs if any(layer == lay.value for lay in r.layers)]
    if fmt == "json":
        import json

        typer.echo(
            json.dumps(
                [
                    {
                        "id": r.id,
                        "name": r.name,
                        "category": r.category.value,
                        "severity": r.severity.value,
                        "layers": [lay.value for lay in r.layers],
                    }
                    for r in rs
                ],
                indent=2,
            )
        )
        return
    from rich.console import Console
    from rich.table import Table

    table = Table(show_header=True, header_style="bold")
    for col in ("ID", "Name", "Category", "Severity", "Layers"):
        table.add_column(col)
    for r in rs:
        table.add_row(
            r.id, r.name, r.category.value, r.severity.value, ",".join(lay.value for lay in r.layers)
        )
    Console().print(table)


@app.command()
def explain(rule_id: str = typer.Argument(..., help="Rule id, e.g. P001.")) -> None:
    """Print a rule's rationale plus its leaky/clean fixture examples."""
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
    console.print(f"[bold]Layers:[/bold] {', '.join(lay.value for lay in rule.layers)}")
    console.print()
    console.print(rule.rationale or "(no rationale provided)")
    if rule.references:
        console.print("\n[bold]References:[/bold]")
        for ref in rule.references:
            console.print(f"  - {ref}")
    # surface fixtures if present
    _print_fixture(console, rule_id, "leaky")
    _print_fixture(console, rule_id, "clean")


def _print_fixture(console, rule_id: str, kind: str) -> None:
    base = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / kind
    if not base.exists():
        return
    matches = sorted(base.glob(f"{rule_id}__*.py"))
    if not matches:
        return
    from rich.syntax import Syntax

    console.print(f"\n[bold]{kind} example[/bold] ({matches[0].name}):")
    console.print(Syntax(matches[0].read_text(encoding="utf-8"), "python", line_numbers=True))


@app.command()
def corpus(
    manifest: Path = typer.Option(..., "--manifest", help="JSON manifest of repos to audit."),
    workdir: Path = typer.Option(Path(".leakproof-corpus"), "--workdir"),
    output: Path = typer.Option(Path("corpus-report.json"), "--output"),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    """Run the corpus-audit benchmark over a manifest of public repos."""
    from .corpus import run_corpus

    cfg = _load_config(config)
    try:
        report = run_corpus(manifest, workdir, cfg)
    except Exception as exc:
        typer.secho(f"leakproof error: {exc}", fg="red", err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    import json

    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    typer.echo(f"wrote corpus report to {output}")


@app.command()
def version() -> None:
    """Print the leakproof version."""
    from . import __version__

    typer.echo(__version__)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
