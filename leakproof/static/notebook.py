"""Notebook -> concatenated module with a line map.

v1 limitation: cells are concatenated in source order. Out-of-order execution
is NOT modeled; findings on notebooks should note this.
"""

from __future__ import annotations

from pathlib import Path


def notebook_to_source(path: Path) -> tuple[str, dict[int, tuple[int, int]]]:
    import nbformat

    nb = nbformat.read(str(path), as_version=4)
    lines: list[str] = []
    line_map: dict[int, tuple[int, int]] = {}
    out_line = 0
    for cell_index, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        # strip IPython magics / shell escapes that aren't valid Python
        cell_lines = src.splitlines()
        for cell_line_no, text in enumerate(cell_lines, start=1):
            stripped = text.lstrip()
            if stripped.startswith(("%", "!", "?")):
                text = "pass  # leakproof: notebook-magic"
            out_line += 1
            line_map[out_line] = (cell_index, cell_line_no)
            lines.append(text)
        # cell separator (blank line) to avoid bleed between cells
        out_line += 1
        line_map[out_line] = (cell_index, len(cell_lines) + 1)
        lines.append("")
    return "\n".join(lines), line_map
