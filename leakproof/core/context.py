"""Per-layer rule contexts passed to ``Rule.check`` / ``Rule.on_event``."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    # for notebooks: maps concatenated line -> (cell_index, cell_line)
    line_map: dict[int, tuple[int, int]] | None = None

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


@dataclass
class RuntimeContext:
    config: Config
    taint: Any  # runtime.taint.TaintTable
    events: list = field(default_factory=list)
    scratch: dict = field(default_factory=dict)  # per-session rule state
