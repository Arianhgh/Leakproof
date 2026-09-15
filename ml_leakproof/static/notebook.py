"""Notebook source normalization with cell-aware locations and coverage notes."""

from __future__ import annotations

import ast
import warnings
from pathlib import Path
from typing import Any


def notebook_to_source(
    path: Path,
) -> tuple[str, dict[int, tuple[int, int, str | None]], dict[str, Any]]:
    """Return source-order Python, a line map, and notebook coverage metadata.

    Stored cell order is analyzed; execution history is not reconstructed.  Magic
    cells are replaced line-for-line with ``pass`` so later valid cells remain
    analyzable.  A syntactically malformed code cell is similarly isolated and
    recorded as partial coverage.
    """

    import nbformat

    notebook: Any = nbformat.read(str(path), as_version=4)  # type: ignore[no-untyped-call]
    lines: list[str] = []
    line_map: dict[int, tuple[int, int, str | None]] = {}
    metadata: dict[str, Any] = {
        "stored_cell_order": True,
        "execution_history_reconstructed": False,
        "malformed_cells": [],
        "non_python_cells": [],
    }
    out_line = 0

    for cell_index, cell in enumerate(notebook.cells):
        if cell.get("cell_type") != "code":
            continue
        cell_id = cell.get("id")
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        raw_lines = str(source).splitlines()
        normalized = _normalize_cell(raw_lines)
        if normalized[1]:
            metadata["non_python_cells"].append(cell_index)
        cell_lines = normalized[0]
        if cell_lines and not normalized[1]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    ast.parse("\n".join(cell_lines), filename=f"{path}:cell-{cell_index}")
            except SyntaxError as exc:
                metadata["malformed_cells"].append(
                    {"cell_index": cell_index, "cell_id": cell_id, "message": exc.msg, "line": exc.lineno}
                )
                cell_lines = [_pass_line(line) for line in cell_lines]
        for cell_line_no, text in enumerate(cell_lines, start=1):
            out_line += 1
            line_map[out_line] = (cell_index, cell_line_no, cell_id)
            lines.append(text)
        out_line += 1
        line_map[out_line] = (cell_index, len(cell_lines) + 1, cell_id)
        lines.append("")

    return "\n".join(lines), line_map, metadata


def _normalize_cell(lines: list[str]) -> tuple[list[str], bool]:
    if not lines:
        return [], False
    first_code = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first_code is not None and lines[first_code].lstrip().startswith("%%"):
        return [_pass_line(line) for line in lines], True
    normalized: list[str] = []
    saw_magic = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(("%", "!", "?")):
            saw_magic = True
            normalized.append(_pass_line(line))
        else:
            normalized.append(line)
    return normalized, saw_magic


def _pass_line(line: str) -> str:
    prefix = line[: len(line) - len(line.lstrip())]
    return f"{prefix}pass  # leakproof: notebook-magic"
