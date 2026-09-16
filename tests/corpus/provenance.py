"""Content identity for release inputs, independent of Git worktree dirtiness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def source_snapshot(root: Path = ROOT) -> dict[str, object]:
    files = []
    for directory in ("ml_leakproof", "tests", "scripts", "docs", "examples", ".github"):
        files.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file()
            and not any(
                part in {"__pycache__", "build", "dist", ".pytest_cache"}
                or part.endswith(".egg-info")
                for part in path.parts
            )
            and path.suffix not in {".pyc", ".pyo"}
            and path.name != ".DS_Store"
        )
    files.extend(
        root / name
        for name in (
            "pyproject.toml",
            "uv.lock",
            "README.md",
            "CHANGELOG.md",
            "LICENSE",
            "corpus-manifest.json",
            "corpus-labeled.json",
            "leakproof.toml",
            "action.yml",
            ".pre-commit-hooks.yaml",
        )
        if (root / name).is_file()
    )
    hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }
    digest = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"algorithm": "sha256", "digest": digest, "files": hashes}


def verify_snapshot(report: dict, root: Path = ROOT) -> None:
    actual = source_snapshot(root)
    if report.get("source_snapshot") != actual:
        raise ValueError("release inputs changed; regenerate and review the benchmark")
