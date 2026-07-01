"""Report formatters."""

from __future__ import annotations

from ..core.aggregate import Summary
from ..core.models import Finding

VALID_FORMATS = ("terminal", "json", "sarif", "markdown")


def render(fmt: str, findings: list[Finding], summary: Summary, *, color: bool = True) -> str:
    if fmt == "terminal":
        from . import terminal

        return terminal.render(findings, summary, color=color)
    if fmt == "json":
        from . import json_report

        return json_report.render(findings, summary)
    if fmt == "sarif":
        from . import sarif

        return sarif.render(findings)
    if fmt == "markdown":
        from . import markdown

        return markdown.render(findings, summary)
    raise ValueError(f"unknown format: {fmt}")
