"""Advisory LLM triage for files with deterministic parse failures."""

from __future__ import annotations

import json
from pathlib import Path

from ..core.config import Config
from ..core.models import (
    Category,
    Diagnostic,
    DiagnosticLevel,
    Finding,
    Layer,
    Location,
    Severity,
)

_SYSTEM = (
    "You are triaging one Python source file after a deterministic parser failure. Identify "
    "at most a few plausible leakage or evaluation-rigor sites. Respond ONLY with a JSON array "
    "of objects containing integer line and string reason. These are advisory hints, not proof."
)


def triage_file(path: Path, config: Config) -> tuple[list[Finding], list[Diagnostic]]:
    from .provider import get_provider

    diagnostics: list[Diagnostic] = []
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return [], [
            Diagnostic(
                code="LP252",
                message=f"could not read source for LLM triage: {exc}",
                level=DiagnosticLevel.WARNING,
                layer=Layer.STATIC,
                location=Location(file=path),
            )
        ]
    try:
        provider = get_provider(
            config.llm.provider,
            config.llm.model,
            timeout_seconds=config.llm.timeout_seconds,
            retries=config.llm.retries,
            max_context_chars=config.llm.max_context_chars,
        )
        raw = provider.complete(
            _SYSTEM,
            f"File: {path.name}\n```python\n{source[:config.llm.max_context_chars]}\n```",
        )
        items = json.loads(_strip_fence(raw))
    except Exception as exc:
        return [], [
            Diagnostic(
                code="LP250",
                message=f"LLM triage failed for {path}: {exc}",
                level=DiagnosticLevel.WARNING,
                layer=Layer.STATIC,
                location=Location(file=path),
            )
        ]
    if not isinstance(items, list):
        diagnostics.append(
            Diagnostic(
                code="LP253",
                message="LLM triage response was not a JSON array",
                level=DiagnosticLevel.WARNING,
                layer=Layer.STATIC,
                location=Location(file=path),
            )
        )
        return [], diagnostics

    line_count = max(1, len(source.splitlines()))
    out: list[Finding] = []
    for item in items[:10]:
        if not isinstance(item, dict):
            diagnostics.append(_response_diagnostic(path, "triage item was not an object"))
            continue
        try:
            line = int(item["line"])
            reason = str(item["reason"]).strip()
        except (KeyError, TypeError, ValueError):
            diagnostics.append(_response_diagnostic(path, "triage item requires line and reason"))
            continue
        if not 1 <= line <= line_count or not reason:
            diagnostics.append(_response_diagnostic(path, "triage location or reason was invalid"))
            continue
        out.append(
            Finding(
                rule_id="LP-TRIAGE",
                category=Category.PREPROCESSING,
                severity=Severity.INFO,
                layer=Layer.STATIC,
                message=f"[llm-triage] {reason}",
                location=Location(file=path, line=line),
                confidence=0.30,
                advisory_only=True,
                evidence={
                    "source": "llm-triage",
                    "reason_codes": ["llm-triage", "advisory-only"],
                },
            )
        )
    return out, diagnostics


def _response_diagnostic(path: Path, message: str) -> Diagnostic:
    return Diagnostic(
        code="LP253",
        message=message,
        level=DiagnosticLevel.WARNING,
        layer=Layer.STATIC,
        location=Location(file=path),
    )


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    return text.strip()
