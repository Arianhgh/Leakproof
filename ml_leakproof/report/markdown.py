"""Markdown report."""

from __future__ import annotations

from ..core.aggregate import Summary
from ..core.models import AnalysisResult, Finding, Severity

_LABEL = {
    Severity.CRITICAL: "CRITICAL",
    Severity.HIGH: "HIGH",
    Severity.MEDIUM: "MEDIUM",
    Severity.LOW: "LOW",
    Severity.INFO: "INFO",
}


def render(
    findings: list[Finding],
    summary: Summary,
    *,
    top: int = 25,
    result: AnalysisResult | None = None,
) -> str:
    lines: list[str] = []
    lines.append("# Leakproof report")
    lines.append("")
    if summary.total == 0:
        lines.append("No leakage or evaluation-rigor issues found.")
    else:
        lines.append(
            f"**{summary.total} finding(s)** ({summary.displayed} displayed, {summary.hidden} hidden) "
            f"- gate: `{summary.gate}` ({summary.gated_count} gateable, "
            f"confidence >= {summary.gate_confidence:.2f})"
        )
        lines.append("")
        lines.append("| Severity | Count |")
        lines.append("|----------|-------|")
        for sev, count in summary.by_severity.items():
            lines.append(f"| {sev} | {count} |")
        lines.append("")

        if summary.by_category:
            lines.append("| Category | Count |")
            lines.append("|----------|-------|")
            for cat, count in summary.by_category.items():
                lines.append(f"| {cat} | {count} |")
            lines.append("")

        lines.append("## Findings")
        lines.append("")
        for f in findings[:top]:
            label = _LABEL.get(f.severity, f.severity.value.upper())
            loc = f.location.short()
            gateable = "yes" if f.gateable else "no"
            lines.append(f"### `{f.rule_id}` [{label}] - {_escape(f.message.splitlines()[0])}")
            lines.append("")
            lines.append(
                f"- **severity:** {f.severity.value} | **layer:** {f.layer.value} "
                f"| **confidence:** {f.confidence:.2f} | **advisory-only:** {f.advisory_only} "
                f"| **gateable:** {gateable}"
            )
            lines.append(f"- **location:** `{_escape_code(loc)}`")
            if f.location.snippet:
                lines.append(f"- **code:** `{_escape_code(f.location.snippet)}`")
            if f.fix:
                lines.append(f"- **fix:** {_escape(f.fix.summary)}")
            if f.references:
                refs = ", ".join(f"[ref]({r})" for r in f.references)
                lines.append(f"- **references:** {refs}")
            lines.append("")
        if summary.displayed is not None and summary.displayed > top:
            lines.append(f"_...and {summary.displayed - top} more displayed finding(s)._\n")

    if result is not None:
        lines.append("## Analysis status")
        lines.append("")
        lines.append(f"- **completion:** `{result.completion.value}`")
        if result.script_exit_status is not None:
            lines.append(f"- **script exit status:** `{result.script_exit_status}`")
        if result.diagnostics:
            lines.append("- **diagnostics:**")
            for diagnostic in result.diagnostics:
                lines.append(f"  - `{diagnostic.code}` ({diagnostic.level.value}): {_escape(diagnostic.message)}")
        if result.coverage.notes:
            lines.append("- **coverage notes:**")
            for note in result.coverage.notes:
                lines.append(f"  - {_escape(note)}")
    return "\n".join(lines)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("`", "\\`").replace("|", "\\|")


def _escape_code(value: str) -> str:
    return value.replace("`", "\\`").replace("\n", " ")
