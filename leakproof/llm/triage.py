"""Optional LLM triage for code the static layer could not parse or resolve.

Candidates are returned at reduced confidence and marked `llm-triage`; they never gate CI
by default and are re-verified by the deterministic layers where possible.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..core.config import Config
from ..core.models import Category, Finding, Layer, Location, Severity

_SYSTEM = (
    "You are scanning ML code that a static analyzer could not fully parse. Identify likely "
    "data-leakage or evaluation-rigor sites. Respond ONLY with a JSON array of objects: "
    '[{"line": int, "rule_id": "Pxxx|Sxxx|...", "reason": str}]. Be conservative.'
)


def triage_file(path: Path, config: Config) -> list[Finding]:
    from .provider import get_provider

    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    provider = get_provider(config.llm.provider, config.llm.model)
    raw = provider.complete(_SYSTEM, f"File: {path}\n```python\n{source}\n```")
    try:
        items = json.loads(_strip_fence(raw))
    except Exception:
        return []
    out: list[Finding] = []
    for item in items if isinstance(items, list) else []:
        rid = str(item.get("rule_id", "LP-TRIAGE"))
        out.append(
            Finding(
                rule_id=rid,
                category=Category.PREPROCESSING,
                severity=Severity.INFO,
                layer=Layer.STATIC,
                message=f"[llm-triage] {item.get('reason', 'possible leakage site')}",
                location=Location(file=path, line=int(item.get("line", 1) or 1)),
                confidence=0.3,
                evidence={"source": "llm-triage", "proposed_rule": rid},
            )
        )
    return out


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    return text.strip()
