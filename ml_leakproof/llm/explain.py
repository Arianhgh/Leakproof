"""Optional explanations for deterministic findings.

LLM text enriches a finding but never creates, removes, or gates one.  Model
patches stay in evidence and are never converted into automatic fixes.
"""

from __future__ import annotations

from dataclasses import replace

from ..core.config import Config
from ..core.models import Diagnostic, DiagnosticLevel, Finding, Layer

_SYSTEM = (
    "You are a careful ML methodology reviewer. Explain the deterministic finding below in "
    "plain English. Separate observed evidence from uncertainty, state what the analyzer did "
    "not prove, and suggest a minimal human-reviewed remediation. If useful, provide a diff "
    "as a suggestion only; it is never an automatic fix. Keep the response concise."
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


def code_context(finding: Finding, *, window: int = 8, max_chars: int | None = None) -> str:
    loc = finding.location
    if loc.file is None or loc.line is None:
        text = loc.snippet or ""
    else:
        try:
            lines = loc.file.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            text = loc.snippet or ""
        else:
            lo = max(0, loc.line - window)
            hi = min(len(lines), loc.line + window)
            text = "\n".join(lines[lo:hi])
    return text if max_chars is None else text[:max_chars]


def explain_finding(finding: Finding, config: Config) -> Finding:
    """Return a copy enriched with an LLM explanation."""

    from .provider import get_provider

    provider = get_provider(
        config.llm.provider,
        config.llm.model,
        timeout_seconds=config.llm.timeout_seconds,
        retries=config.llm.retries,
        max_context_chars=config.llm.max_context_chars,
    )
    context = code_context(
        finding,
        max_chars=min(config.llm.max_context_chars, 8_000),
    )
    text = provider.complete(_SYSTEM, _prompt(finding, context))
    evidence = dict(finding.evidence)
    evidence["llm_explanation"] = text
    diff = _extract_diff(text)
    if diff:
        evidence["llm_suggested_diff"] = diff
    # The existing deterministic Fix, if any, is intentionally preserved.
    return replace(finding, evidence=evidence)


def explain_all(
    findings: list[Finding], config: Config
) -> tuple[list[Finding], list[Diagnostic]]:
    """Enrich findings best-effort and preserve them if a provider fails."""

    out: list[Finding] = []
    diagnostics: list[Diagnostic] = []
    for finding in findings:
        try:
            out.append(explain_finding(finding, config))
        except Exception as exc:
            out.append(finding)
            diagnostics.append(
                Diagnostic(
                    code="LP251",
                    message=f"LLM explanation failed for {finding.rule_id}: {exc}",
                    level=DiagnosticLevel.WARNING,
                    layer=Layer.STATIC,
                    rule_id=finding.rule_id,
                    location=finding.location,
                )
            )
    return out, diagnostics


def _extract_diff(text: str) -> str | None:
    marker = "```diff"
    if marker not in text:
        return None
    body = text.split(marker, 1)[1]
    return body.split("```", 1)[0].strip() or None
