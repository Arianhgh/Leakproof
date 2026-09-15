"""Token-aware inline and file suppressions."""

from __future__ import annotations

import io
import re
import tokenize
import warnings
from pathlib import Path

from .models import Diagnostic, DiagnosticLevel, Finding, Layer, Location

_DIRECTIVE_RE = re.compile(r"^leakproof\s*:\s*(?P<body>.+?)\s*$", re.IGNORECASE)
_IGNORE_RE = re.compile(r"^ignore\s*\[(?P<ids>[^\]]*)\]\s*$", re.IGNORECASE)
_FILE_RE = re.compile(r"^ignore-file\s*$", re.IGNORECASE)


class FileSuppressions:
    """Parsed comments for one source file."""

    def __init__(
        self,
        ignore_file: bool,
        per_line: dict[int, set[str]],
        issues: list[tuple[int, str]],
    ):
        self.ignore_file = ignore_file
        self.per_line = per_line
        self._issues = issues

    @classmethod
    def parse(
        cls,
        source: str,
        *,
        known_rule_ids: set[str] | None = None,
    ) -> FileSuppressions:
        ignore_file = False
        per_line: dict[int, set[str]] = {}
        issues: list[tuple[int, str]] = []
        try:
            tokens = tokenize.generate_tokens(io.StringIO(source).readline)
            comments = [token for token in tokens if token.type == tokenize.COMMENT]
        except (tokenize.TokenError, IndentationError) as exc:
            # The parser will report the source error separately.  A tokenization
            # issue must not cause a broad regex-based file suppression.
            issues.append((1, f"could not parse suppression comments: {exc}"))
            comments = []
        for token in comments:
            comment = token.string.lstrip()[1:].strip()
            match = _DIRECTIVE_RE.match(comment)
            if not match:
                continue
            body = match.group("body")
            # Notebook normalization uses a Leakproof-prefixed marker for a
            # replaced magic line.  It is metadata, not a suppression.
            if body.lower().startswith("notebook-magic"):
                continue
            if _FILE_RE.match(body):
                ignore_file = True
                continue
            ignore_match = _IGNORE_RE.match(body)
            if not ignore_match:
                issues.append((token.start[0], "malformed bare suppression; use ignore[RULE_ID]"))
                continue
            raw_ids = [value.strip() for value in ignore_match.group("ids").split(",")]
            ids = {value for value in raw_ids if value}
            if not ids:
                issues.append((token.start[0], "empty suppression; rule ID is required"))
                continue
            invalid = sorted(value for value in ids if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value))
            if invalid:
                issues.append((token.start[0], f"invalid suppression rule ID(s): {', '.join(invalid)}"))
                ids -= set(invalid)
            if known_rule_ids is not None:
                unknown = sorted(value for value in ids if value not in known_rule_ids)
                if unknown:
                    issues.append((token.start[0], f"unknown suppression rule ID(s): {', '.join(unknown)}"))
                    ids -= set(unknown)
            if ids:
                per_line.setdefault(token.start[0], set()).update(ids)
        for line, message in issues:
            warnings.warn(f"leakproof: {message} at line {line}", stacklevel=2)
        return cls(ignore_file, per_line, issues)

    def diagnostics(self, path: Path | None = None) -> list[Diagnostic]:
        return [
            Diagnostic(
                code="LP201",
                message=message,
                level=DiagnosticLevel.WARNING,
                layer=Layer.STATIC,
                location=Location(file=path, line=line),
            )
            for line, message in self._issues
        ]

    def suppresses(self, finding: Finding) -> bool:
        if self.ignore_file:
            return True
        line = finding.location.line
        if line is None:
            return False
        ids = self.per_line.get(line)
        return bool(ids and finding.rule_id in ids)


def path_excluded(path: Path, excludes: list[str], *, root: Path | None = None) -> bool:
    """Return whether a path matches an exclude prefix or glob."""

    root = root or Path.cwd()
    try:
        relative = path.resolve().relative_to(root.resolve())
        normalized = str(relative).replace("\\", "/")
    except ValueError:
        normalized = str(path).replace("\\", "/")
    from fnmatch import fnmatch

    for pattern in excludes:
        normalized_pattern = pattern.replace("\\", "/").rstrip("/")
        if normalized == normalized_pattern or normalized.startswith(normalized_pattern + "/"):
            return True
        if fnmatch(normalized, pattern) or fnmatch(normalized, normalized_pattern + "/*"):
            return True
        if f"/{normalized_pattern}/" in f"/{normalized}/":
            return True
    return False
