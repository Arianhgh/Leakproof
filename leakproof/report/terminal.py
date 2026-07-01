"""Rich terminal output."""

from __future__ import annotations

from collections import defaultdict

from ..core.aggregate import Summary
from ..core.models import Finding, Severity

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


def _emit(console, findings: list[Finding], summary: Summary) -> None:
    from rich.table import Table
    from rich.text import Text

    if not findings:
        console.print("[green]OK No leakage or evaluation-rigor issues found.[/green]")
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
            console.print(f"    {f.message}", style=_COLOR.get(sev, ""))
            if f.location.snippet:
                console.print(f"    | {f.location.snippet}", style="dim")
            if f.fix:
                console.print(f"    -> fix: {f.fix.summary}", style="green")

    console.print()
    table = Table(title="Summary", show_header=True, header_style="bold")
    table.add_column("Severity")
    table.add_column("Count", justify="right")
    for sev in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]:
        c = summary.by_severity.get(sev.value, 0)
        if c:
            table.add_row(Text(sev.value, style=_COLOR.get(sev, "")), str(c))
    table.add_row("total", str(summary.total))
    console.print(table)

    if summary.by_category:
        cat_table = Table(show_header=True, header_style="bold")
        cat_table.add_column("Category")
        cat_table.add_column("Count", justify="right")
        for cat, c in summary.by_category.items():
            cat_table.add_row(cat, str(c))
        console.print(cat_table)

    console.print(
        f"\nGate: [bold]{summary.gate}[/bold] - {summary.gated_count} gateable finding(s)."
    )


def render(findings: list[Finding], summary: Summary, *, color: bool = True) -> str:
    import io

    from rich.console import Console

    console = Console(
        record=True, no_color=not color, force_terminal=color, width=100, file=io.StringIO()
    )
    _emit(console, findings, summary)
    return console.export_text(styles=color)


def print_report(findings: list[Finding], summary: Summary, *, color: bool = True) -> None:
    from rich.console import Console

    console = Console(no_color=not color)
    _emit(console, findings, summary)
