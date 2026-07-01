"""Optional LLM layer (explanation-only; never the sole basis for a finding)."""

from __future__ import annotations

__all__ = ["explain_all", "explain_finding", "triage_file", "get_provider"]


def __getattr__(name):  # lazy to avoid importing optional SDKs eagerly
    if name in ("explain_all", "explain_finding"):
        from .explain import explain_all, explain_finding

        return {"explain_all": explain_all, "explain_finding": explain_finding}[name]
    if name == "triage_file":
        from .triage import triage_file

        return triage_file
    if name == "get_provider":
        from .provider import get_provider

        return get_provider
    raise AttributeError(name)
