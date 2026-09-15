"""Report formatters."""

from __future__ import annotations

from pathlib import Path

from ..core.aggregate import Summary
from ..core.models import AnalysisResult, Finding

VALID_FORMATS = ("terminal", "json", "sarif", "markdown")


def render(
    fmt: str,
    findings: list[Finding],
    summary: Summary,
    *,
    color: bool = True,
    result: AnalysisResult | None = None,
    root: Path | None = None,
) -> str:
    if fmt == "terminal":
        from . import terminal

        return terminal.render(findings, summary, color=color, result=result)
    if fmt == "json":
        from . import json_report

        return json_report.render(findings, summary, result=result)
    if fmt == "sarif":
        from . import sarif

        return sarif.render(findings, result=result, root=root)
    if fmt == "markdown":
        from . import markdown

        return markdown.render(findings, summary, result=result)
    raise ValueError(f"unknown format: {fmt}")
