"""Markdown report."""

from __future__ import annotations

from ..core.aggregate import Summary
from ..core.models import Finding, Severity

_LABEL = {
    Severity.CRITICAL: "CRITICAL",
    Severity.HIGH: "HIGH",
    Severity.MEDIUM: "MEDIUM",
    Severity.LOW: "LOW",
    Severity.INFO: "INFO",
}


def render(findings: list[Finding], summary: Summary, *, top: int = 25) -> str:
    lines: list[str] = []
    lines.append("# leakproof report")
    lines.append("")
    if summary.total == 0:
        lines.append("No leakage or evaluation-rigor issues found.")
        return "\n".join(lines)

    lines.append(
        f"**{summary.total} finding(s)** - gate: `{summary.gate}` "
        f"({summary.gated_count} gateable, confidence >= {summary.gate_confidence:.2f})"
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
        lines.append(f"### `{f.rule_id}` [{label}] - {f.message.splitlines()[0]}")
        lines.append("")
        lines.append(
            f"- **severity:** {f.severity.value} | **layer:** {f.layer.value} "
            f"| **confidence:** {f.confidence:.2f} | **gateable:** {gateable}"
        )
        lines.append(f"- **location:** `{loc}`")
        if f.location.snippet:
            lines.append(f"- **code:** `{f.location.snippet}`")
        if f.fix:
            lines.append(f"- **fix:** {f.fix.summary}")
        if f.references:
            refs = ", ".join(f"[ref]({r})" for r in f.references)
            lines.append(f"- **references:** {refs}")
        lines.append("")
    if summary.total > top:
        lines.append(f"_...and {summary.total - top} more._")
    return "\n".join(lines)
