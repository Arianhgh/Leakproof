"""Inline + path-based suppression of findings.

Supported forms:
  - ``# leakproof: ignore[P001]``  (end of the offending line; rule id required)
  - ``# leakproof: ignore[P001,C002]``  (multiple ids)
  - ``# leakproof: ignore-file``  (suppress the whole file)
  - path excludes from config

Bare ``# leakproof: ignore`` with no rule id is rejected (a no-op that emits a
warning) so suppressions stay auditable.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

from .models import Finding

_INLINE_RE = re.compile(r"#\s*leakproof:\s*ignore\[([^\]]*)\]")
_BARE_RE = re.compile(r"#\s*leakproof:\s*ignore(?!\[|-file)")
_FILE_RE = re.compile(r"#\s*leakproof:\s*ignore-file")


class FileSuppressions:
    """Parsed suppression directives for one source file."""

    def __init__(self, ignore_file: bool, per_line: dict[int, set[str]]):
        self.ignore_file = ignore_file
        self.per_line = per_line

    @classmethod
    def parse(cls, source: str) -> FileSuppressions:
        ignore_file = False
        per_line: dict[int, set[str]] = {}
        for lineno, line in enumerate(source.splitlines(), start=1):
            if _FILE_RE.search(line):
                ignore_file = True
            m = _INLINE_RE.search(line)
            if m:
                ids = {tok.strip() for tok in m.group(1).split(",") if tok.strip()}
                if ids:
                    per_line.setdefault(lineno, set()).update(ids)
                else:
                    warnings.warn(
                        f"leakproof: empty ignore[] at line {lineno}; rule id required",
                        stacklevel=2,
                    )
            elif _BARE_RE.search(line):
                warnings.warn(
                    f"leakproof: bare `# leakproof: ignore` at line {lineno} is rejected; "
                    "use ignore[RULE_ID]",
                    stacklevel=2,
                )
        return cls(ignore_file, per_line)

    def suppresses(self, finding: Finding) -> bool:
        if self.ignore_file:
            return True
        line = finding.location.line
        if line is None:
            return False
        ids = self.per_line.get(line)
        return bool(ids and finding.rule_id in ids)


def path_excluded(path: Path, excludes: list[str], *, root: Path | None = None) -> bool:
    """Whether ``path`` matches any exclude prefix/glob."""
    root = root or Path.cwd()
    try:
        rel = path.resolve().relative_to(root.resolve())
        rel_str = str(rel)
    except ValueError:
        rel_str = str(path)
    norm = rel_str.replace("\\", "/")
    for pattern in excludes:
        pat = pattern.rstrip("/")
        if norm == pat or norm.startswith(pat + "/"):
            return True
        # glob support
        from fnmatch import fnmatch

        if fnmatch(norm, pattern) or fnmatch(norm, pattern.rstrip("/") + "/*"):
            return True
        if f"/{pat}/" in f"/{norm}/":
            return True
    return False
