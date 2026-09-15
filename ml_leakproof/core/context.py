"""Per-layer rule contexts passed to ``Rule.check`` / ``Rule.on_event``."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .models import Coverage, Diagnostic

if TYPE_CHECKING:
    from ..static.adapters import AdapterRegistry
    from ..static.dataflow import DataFlow
    from .config import Config


@dataclass
class StaticContext:
    module_path: Path
    tree: ast.AST
    source: str
    source_lines: list[str]
    dataflow: DataFlow
    adapters: AdapterRegistry
    config: Config
    # For notebooks: maps concatenated line -> (cell_index, cell_line, cell_id).
    # The two-item form remains accepted for callers built against 0.1.
    line_map: dict[int, tuple[int, int] | tuple[int, int, str | None]] | None = None
    notebook_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_notebook(self) -> bool:
        return self.line_map is not None

    def snippet(self, line: int | None) -> str | None:
        if line is None or line < 1 or line > len(self.source_lines):
            return None
        return self.source_lines[line - 1].rstrip("\n")


@dataclass
class DataContext:
    audit_input: Any  # DataAuditInput (avoid hard pandas import at module load)
    config: Config
    diagnostics: list[Diagnostic] = field(default_factory=list)
    coverage: Coverage = field(default_factory=Coverage)
    cache: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeContext:
    config: Config
    taint: Any  # runtime.taint.TaintTable
    events: list[Any] = field(default_factory=list)
    scratch: dict[str, Any] = field(default_factory=dict)  # per-session rule state
