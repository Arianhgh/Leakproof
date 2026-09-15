"""Rich terminal output."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..core.aggregate import Summary
from ..core.models import AnalysisResult, Finding, Severity

_STYLE = {
    Severity.CRITICAL: "bold white on red",
    Severity.HIGH: "bold red",
    Severity.MEDIUM: "yellow",
    Severity.LOW: "cyan",
    Severity.INFO: "blue",
}
_COLOR = {
    Severity.CRITICAL: "red",
    Severity.HIGH: "red",
    Severity.MEDIUM: "yellow",
    Severity.LOW: "cyan",
    Severity.INFO: "blue",
}


def _emit(
    console: Any,
    findings: list[Finding],
    summary: Summary,
    *,
    result: AnalysisResult | None = None,
) -> None:
    from rich.table import Table
    from rich.text import Text

    if not findings:
        console.print(Text("OK No leakage or evaluation-rigor issues found.", style="green"))
        if summary.hidden:
            console.print(Text(f"{summary.hidden} finding(s) hidden by presentation filters.", style="yellow"))
        _emit_diagnostics(console, result)
        return

    by_file: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        key = str(f.location.file) if f.location.file is not None else "<data / runtime>"
        by_file[key].append(f)

    for file_key in sorted(by_file):
        console.print()
        console.print(Text(file_key, style="bold underline"))
        group = sorted(
            by_file[file_key], key=lambda f: (-f.severity.gate_rank, f.location.line or 0)
        )
        for f in group:
            sev = f.severity
            head = Text()
            head.append(f"  {sev.value.upper():8}", style=_STYLE.get(sev, ""))
            head.append(f" {f.rule_id} ", style="bold")
            head.append(f.location.short(), style="dim")
            console.print(head)
            console.print(Text(f"    {f.message}", style=_COLOR.get(sev, "")))
            if f.location.snippet:
                console.print(Text(f"    | {f.location.snippet}", style="dim"))
            if f.fix:
                console.print(Text(f"    -> fix: {f.fix.summary}", style="green"))
            console.print(
                Text(
                    f"    confidence={f.confidence:.2f} advisory={f.advisory_only} "
                    f"gateable={bool(f.gateable)}",
                    style="dim",
                )
            )

    console.print()
    table = Table(title="Summary", show_header=True, header_style="bold")
    table.add_column("Severity")
    table.add_column("Count", justify="right")
    for sev in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]:
        c = summary.by_severity.get(sev.value, 0)
        if c:
            table.add_row(Text(sev.value, style=_COLOR.get(sev, "")), str(c))
    table.add_row("displayed", str(summary.displayed if summary.displayed is not None else summary.total))
    if summary.hidden:
        table.add_row("hidden", str(summary.hidden))
    table.add_row("total", str(summary.total))
    console.print(table)

    if summary.by_category:
        cat_table = Table(show_header=True, header_style="bold")
        cat_table.add_column("Category")
        cat_table.add_column("Count", justify="right")
        for cat, c in summary.by_category.items():
            cat_table.add_row(cat, str(c))
        console.print(cat_table)

    console.print(Text(f"\nGate: {summary.gate} - {summary.gated_count} gateable finding(s)."))
    _emit_diagnostics(console, result)


def render(
    findings: list[Finding],
    summary: Summary,
    *,
    color: bool = True,
    result: AnalysisResult | None = None,
) -> str:
    import io

    from rich.console import Console

    console = Console(
        record=True, no_color=not color, force_terminal=color, width=100, file=io.StringIO()
    )
    _emit(console, findings, summary, result=result)
    return console.export_text(styles=color)


def print_report(
    findings: list[Finding],
    summary: Summary,
    *,
    color: bool = True,
    result: AnalysisResult | None = None,
) -> None:
    from rich.console import Console

    console = Console(no_color=not color)
    _emit(console, findings, summary, result=result)


def _emit_diagnostics(console: Any, result: AnalysisResult | None) -> None:
    if result is None:
        return
    from rich.text import Text

    if result.completion.value != "complete":
        console.print(
            Text(
                f"\nAnalysis {result.completion.value}: requested coverage is incomplete.",
                style="bold yellow" if result.completion.value == "partial" else "bold red",
            )
        )
    if result.script_exit_status not in (None, 0):
        console.print(Text(f"Script exit status: {result.script_exit_status}", style="bold red"))
    for diagnostic in result.diagnostics:
        location = f" @ {diagnostic.location.short()}" if diagnostic.location else ""
        style = "red" if diagnostic.level.value == "error" else "yellow"
        console.print(Text(f"{diagnostic.level.value.upper()} {diagnostic.code}{location}: {diagnostic.message}", style=style))
