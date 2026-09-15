"""Conservative LibCST-backed autofixes.

Only a verified seed argument is currently writable.  The transformer targets
the exact call location recorded by static analysis, so multiline and nested
calls are handled without regex rewriting.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from .core.models import Finding


class AutofixUnavailable(RuntimeError):
    """Raised when the optional LibCST dependency is not installed."""


_SEED_NAMES = {"random_state", "seed", "random_seed"}


def _libcst() -> tuple[Any, Any, Any]:
    try:
        import libcst as cst
        from libcst.metadata import MetadataWrapper, PositionProvider
    except ImportError as exc:  # pragma: no cover - exercised in base installs
        raise AutofixUnavailable(
            "autofix requires the optional dependency; install ml-leakproof[fix]"
        ) from exc
    return cst, MetadataWrapper, PositionProvider


def _source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _eligible(findings: list[Finding]) -> dict[Path, list[Finding]]:
    grouped: dict[Path, list[Finding]] = {}
    for finding in findings:
        fix = finding.fix
        location = finding.location
        if (
            fix is None
            or not fix.autofixable
            or location.file is None
            or location.line is None
            or finding.rule_id != "R001"
            or location.cell_id is not None
        ):
            continue
        grouped.setdefault(location.file, []).append(finding)
    return grouped


class _SeedTransformer:
    def __init__(self, cst: Any, MetadataWrapper: Any, PositionProvider: Any, findings: list[Finding]):
        self.cst = cst
        self.MetadataWrapper = MetadataWrapper
        self.PositionProvider = PositionProvider
        self.targets = {
            (
                finding.location.line,
                finding.location.col,
                str(finding.evidence.get("callable", "")),
            )
            for finding in findings
        }
        self.changed = 0

    def transform(self, source: str) -> str:
        module = self.cst.parse_module(source)

        transformer = self

        class Visitor(self.cst.CSTTransformer):  # type: ignore[name-defined,misc]
            METADATA_DEPENDENCIES = (transformer.PositionProvider,)

            def leave_Call(self, original_node: Any, updated_node: Any) -> Any:
                position = self.get_metadata(transformer.PositionProvider, original_node)
                line = position.start.line
                col = position.start.column
                callable_name = _cst_callable_name(original_node.func)
                exact = (line, col, callable_name) in transformer.targets
                line_only = any(
                    target_line == line
                    and target_col is None
                    and target_name == callable_name
                    for target_line, target_col, target_name in transformer.targets
                )
                if not (exact or line_only):
                    return updated_node
                if any(
                    getattr(argument.keyword, "value", None) in _SEED_NAMES
                    for argument in original_node.args
                    if argument.keyword is not None
                ):
                    return updated_node
                if any(argument.star for argument in original_node.args):
                    return updated_node
                argument = transformer.cst.Arg(
                    keyword=transformer.cst.Name("random_state"),
                    value=transformer.cst.Integer("0"),
                    equal=transformer.cst.AssignEqual(
                        whitespace_before=transformer.cst.SimpleWhitespace(""),
                        whitespace_after=transformer.cst.SimpleWhitespace(""),
                    ),
                )
                transformer.changed += 1
                return updated_node.with_changes(args=(*updated_node.args, argument))

        return str(self.MetadataWrapper(module).visit(Visitor()).code)


def _cst_callable_name(node: Any) -> str:
    names: list[str] = []
    while hasattr(node, "value") and hasattr(node, "attr"):
        names.append(node.attr.value)
        node = node.value
    if hasattr(node, "value"):
        names.append(node.value)
    return ".".join(reversed(names))


def _rewrite(path: Path, findings: list[Finding]) -> tuple[str, int, str] | None:
    cst, MetadataWrapper, PositionProvider = _libcst()
    import tokenize

    try:
        with tokenize.open(str(path)) as handle:
            source = handle.read()
        with path.open("rb") as raw:
            encoding, _ = tokenize.detect_encoding(raw.readline)
    except Exception:
        return None
    expected = {
        value
        for value in (finding.evidence.get("analysis_source_sha256") for finding in findings)
        if value
    }
    if expected and _source_digest(path) not in expected:
        return None
    transformer = _SeedTransformer(cst, MetadataWrapper, PositionProvider, findings)
    updated = transformer.transform(source)
    if not transformer.changed or updated == source:
        return None
    # LibCST guarantees a well-formed concrete tree; compile as the final
    # syntax guard before any file replacement.
    compile(updated, str(path), "exec")
    return updated, transformer.changed, encoding


def preview_fixes(findings: list[Finding]) -> list[str]:
    """Return unified diffs without writing files."""

    previews: list[str] = []
    for path, group in _eligible(findings).items():
        rewritten = _rewrite(path, group)
        if rewritten is None:
            continue
        updated, _, _ = rewritten
        try:
            import tokenize

            with tokenize.open(str(path)) as handle:
                original = handle.read()
        except Exception:
            continue
        previews.append(
            "".join(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    updated.splitlines(keepends=True),
                    fromfile=str(path),
                    tofile=str(path),
                )
            )
        )
    return previews


def _atomic_write(path: Path, text: str, encoding: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding=encoding,
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, mode)
    try:
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def apply_fixes(findings: list[Finding], *, write: bool = True) -> dict[Path, int]:
    """Apply verified fixes atomically, returning ``{path: count}``.

    ``write=False`` is a compatibility-friendly dry-run; callers wanting the
    actual diff should use :func:`preview_fixes`.
    """

    applied: dict[Path, int] = {}
    for path, group in _eligible(findings).items():
        rewritten = _rewrite(path, group)
        if rewritten is None:
            continue
        updated, count, encoding = rewritten
        if write:
            _atomic_write(path, updated, encoding)
        applied[path] = count
    return applied
