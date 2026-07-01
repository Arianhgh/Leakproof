"""Data-layer engine: orchestrate DataRules over a DataAuditInput."""

from __future__ import annotations

import warnings

from ..core.config import Config
from ..core.context import DataContext
from ..core.models import Finding, Layer
from ..core.registry import load_all
from ..core.rule import DataRule
from .input import DataAuditInput


class DataEngine:
    def __init__(self, config: Config):
        self.config = config
        self.rules: list[DataRule] = [
            r for r in load_all(config, layers=(Layer.DATA,)) if isinstance(r, DataRule)
        ]

    def run(self, audit_input: DataAuditInput) -> list[Finding]:
        ctx = DataContext(audit_input=audit_input, config=self.config)
        out: list[Finding] = []
        for rule in self.rules:
            try:
                out.extend(rule.check(ctx))
            except Exception as exc:  # pragma: no cover - a buggy rule must not crash
                warnings.warn(f"leakproof: data rule {rule.id} raised: {exc}", stacklevel=2)
        return out
