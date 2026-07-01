"""Markdown report for PR comments / portfolio screenshots."""

from __future__ import annotations

from ..core.aggregate import Summary
from ..core.models import Finding, Severity

_EMOJI = {
    Severity.CRITICAL: "🛑",
    Severity.HIGH: "🔴",
    Severity.MEDIUM: "🟠",
    Severity.LOW: "🟡",
    Severity.INFO: "🔵",
}


def render(findings: list[Finding], summary: Summary, *, top: int = 25) -> str:
    lines: list[str] = []
    lines.append("# leakproof report")
    lines.append("")
    if summary.total == 0:
        lines.append("✅ No leakage or evaluation-rigor issues found.")
        return "\n".join(lines)

    lines.append(f"**{summary.total} finding(s)** — gate: `{summary.gate}` "
                 f"({summary.gated_count} at/above gate)")
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
        emoji = _EMOJI.get(f.severity, "")
        loc = f.location.short()
        lines.append(f"### {emoji} `{f.rule_id}` — {f.message.splitlines()[0]}")
        lines.append("")
        lines.append(f"- **severity:** {f.severity.value} · **layer:** {f.layer.value} "
                     f"· **confidence:** {f.confidence:.2f}")
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
        lines.append(f"_…and {summary.total - top} more._")
    return "\n".join(lines)
