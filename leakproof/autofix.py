"""Opt-in autofix: limited to a safe, mechanical subset.

v1 autofix only applies fixes the rule marks ``autofixable`` and that we can
perform with a confident, localized rewrite. Currently:

  - R001: add ``random_state=0`` to a splitter/estimator call missing a seed.

Everything else is suggestion-only.
"""

from __future__ import annotations

import re
from pathlib import Path

from .core.models import Finding


def apply_fixes(findings: list[Finding]) -> dict[Path, int]:
    """Apply autofixable fixes in place. Return {path: num_fixes_applied}."""
    applied: dict[Path, int] = {}
    # group by file, apply bottom-up so earlier edits don't shift later lines
    by_file: dict[Path, list[Finding]] = {}
    for f in findings:
        if not (f.fix and f.fix.autofixable and f.location.file and f.location.line):
            continue
        by_file.setdefault(f.location.file, []).append(f)

    for path, group in by_file.items():
        try:
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        except Exception:
            continue
        count = 0
        for f in sorted(group, key=lambda x: -(x.location.line or 0)):
            idx = (f.location.line or 1) - 1
            if idx < 0 or idx >= len(lines):
                continue
            new = _fix_line(f, lines[idx])
            if new is not None and new != lines[idx]:
                lines[idx] = new
                count += 1
        if count:
            path.write_text("".join(lines), encoding="utf-8")
            applied[path] = count
    return applied


_CALL_RE = re.compile(r"(\b\w+\s*\()([^)]*)(\))")


def _fix_line(finding: Finding, line: str) -> str | None:
    if finding.rule_id == "R001":
        return _add_random_state(line)
    return None


def _add_random_state(line: str) -> str | None:
    if "random_state" in line:
        return None
    m = _CALL_RE.search(line)
    if not m:
        return None
    head, args, tail = m.group(1), m.group(2).strip(), m.group(3)
    insert = "random_state=0" if not args else f"{args}, random_state=0"
    return line[: m.start()] + head + insert + tail + line[m.end() :]
