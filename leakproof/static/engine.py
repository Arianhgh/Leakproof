"""Static analysis engine: discover files, parse, run StaticRules."""

from __future__ import annotations

import ast
from pathlib import Path

from ..core.config import Config
from ..core.context import StaticContext
from ..core.models import Finding, Layer
from ..core.registry import load_all
from ..core.rule import StaticRule
from ..core.suppression import FileSuppressions, path_excluded
from .adapters import AdapterRegistry, load_adapters
from .dataflow import DataFlow


def discover_files(paths: list[Path], excludes: list[str], root: Path) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            for ext in ("*.py", "*.ipynb"):
                for f in sorted(p.rglob(ext)):
                    if not path_excluded(f, excludes, root=root):
                        out.append(f)
        elif p.is_file():
            if p.suffix in (".py", ".ipynb") and not path_excluded(p, excludes, root=root):
                out.append(p)
    # de-dupe preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for f in out:
        rp = f.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(f)
    return uniq


def _load_source(path: Path) -> tuple[str, dict[int, tuple[int, int]] | None]:
    """Return source text and an optional notebook line map."""
    if path.suffix == ".ipynb":
        from .notebook import notebook_to_source

        return notebook_to_source(path)
    return path.read_text(encoding="utf-8", errors="replace"), None


class StaticEngine:
    def __init__(self, config: Config, adapters: AdapterRegistry | None = None):
        self.config = config
        self.adapters = adapters or load_adapters()
        self.rules: list[StaticRule] = [
            r
            for r in load_all(config, layers=(Layer.STATIC,))
            if isinstance(r, StaticRule)
        ]

    def run(self, paths: list[Path], *, root: Path | None = None) -> list[Finding]:
        root = root or Path.cwd()
        files = discover_files(paths, self.config.exclude, root)
        findings: list[Finding] = []
        for f in files:
            findings.extend(self.run_file(f))
        return findings

    def run_file(self, path: Path) -> list[Finding]:
        try:
            source, line_map = _load_source(path)
        except Exception:
            return []
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            # unparseable -> a single advisory; --llm-triage can pick it up
            from ..core.models import Category, Location, Severity

            return [
                Finding(
                    rule_id="LP000",
                    category=Category.DETERMINISM,
                    severity=Severity.INFO,
                    layer=Layer.STATIC,
                    message=f"Could not parse file ({exc.msg}); static analysis skipped.",
                    location=Location(file=path, line=exc.lineno),
                    confidence=1.0,
                )
            ]

        source_lines = source.splitlines()
        suppressions = FileSuppressions.parse(source)
        if suppressions.ignore_file:
            return []

        dataflow = DataFlow(tree, self.adapters, source_lines)
        ctx = StaticContext(
            module_path=path,
            tree=tree,
            source=source,
            source_lines=source_lines,
            dataflow=dataflow,
            adapters=self.adapters,
            config=self.config,
            line_map=line_map,
        )

        out: list[Finding] = []
        for rule in self.rules:
            try:
                for finding in rule.check(ctx):
                    # attach snippet if missing
                    finding = _attach_snippet(finding, ctx)
                    if not suppressions.suppresses(finding):
                        out.append(finding)
            except Exception:  # pragma: no cover - a buggy rule must not crash the run
                import warnings

                warnings.warn(f"leakproof: rule {rule.id} raised on {path}", stacklevel=2)
        return out


def _attach_snippet(finding: Finding, ctx: StaticContext) -> Finding:
    if finding.location.snippet is None and finding.location.line is not None:
        snippet = ctx.snippet(finding.location.line)
        if snippet is not None:
            from dataclasses import replace

            loc = replace(finding.location, snippet=snippet.strip())
            return replace(finding, location=loc)
    return finding
