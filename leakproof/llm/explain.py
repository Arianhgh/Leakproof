"""LLM explanation of confirmed findings.

The LLM never *creates* a finding; it explains one the deterministic engine
already produced and proposes a unified-diff fix.
"""

from __future__ import annotations

from dataclasses import replace

from ..core.config import Config
from ..core.models import Finding, Fix

_SYSTEM = (
    "You are a meticulous ML methodology reviewer. You are given a leakage/eval finding that "
    "a deterministic analyzer has already confirmed. Explain in plain English (1) why it is a "
    "problem, (2) the concrete failure it causes (e.g. optimistic metrics), and (3) a minimal "
    "fix. If you can, end with a unified diff fenced as ```diff. Never dispute the finding; it "
    "is the analyzer output this explanation is based on."
)


def _prompt(finding: Finding, code_context: str) -> str:
    return (
        f"Rule: {finding.rule_id} — {finding.message}\n"
        f"Category: {finding.category.value}\n"
        f"Severity: {finding.severity.value}\n"
        f"Location: {finding.location.short()}\n"
        f"Evidence: {finding.evidence}\n\n"
        f"Code context:\n```python\n{code_context}\n```\n"
    )


def code_context(finding: Finding, *, window: int = 8) -> str:
    loc = finding.location
    if loc.file is None or loc.line is None:
        return loc.snippet or ""
    try:
        lines = loc.file.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return loc.snippet or ""
    lo = max(0, loc.line - window)
    hi = min(len(lines), loc.line + window)
    return "\n".join(lines[lo:hi])


def explain_finding(finding: Finding, config: Config) -> Finding:
    """Return a copy of the finding with an LLM-authored explanation attached."""
    from .provider import get_provider

    provider = get_provider(config.llm.provider, config.llm.model)
    ctx = code_context(finding)
    text = provider.complete(_SYSTEM, _prompt(finding, ctx))
    diff = _extract_diff(text)
    evidence = dict(finding.evidence)
    evidence["llm_explanation"] = text
    new_fix = finding.fix
    if diff:
        new_fix = Fix(
            summary=finding.fix.summary if finding.fix else "LLM-proposed fix",
            suggested_diff=diff,
            autofixable=finding.fix.autofixable if finding.fix else False,
        )
    return replace(finding, evidence=evidence, fix=new_fix)


def explain_all(findings: list[Finding], config: Config) -> list[Finding]:
    return [explain_finding(f, config) for f in findings]


def _extract_diff(text: str) -> str | None:
    if "```diff" in text:
        body = text.split("```diff", 1)[1]
        return body.split("```", 1)[0].strip()
    return None
