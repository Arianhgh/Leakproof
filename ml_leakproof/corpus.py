"""Bounded, source-only corpus auditing for reproducible benchmarks."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import __version__
from .core.aggregate import gate_failed
from .core.config import Config
from .static.engine import StaticEngine

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _git(
    args: list[str],
    *,
    timeout: int = 300,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _validate_manifest(manifest: Any, manifest_path: Path) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict) or not isinstance(manifest.get("repos"), list):
        raise ValueError(f"{manifest_path} must contain a repos array")
    repos: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, repo in enumerate(manifest["repos"]):
        if not isinstance(repo, dict):
            raise ValueError(f"manifest repo {index} must be an object")
        name = repo.get("name")
        url = repo.get("url")
        commit = repo.get("commit")
        if (
            not isinstance(name, str)
            or name in {".", ".."}
            or not _NAME_RE.fullmatch(name)
            or ".." in name
        ):
            raise ValueError(f"manifest repo {index} has an unsafe name")
        if name in names:
            raise ValueError(f"manifest contains duplicate repo name: {name}")
        if not isinstance(url, str) or not url.startswith(("https://", "http://")):
            raise ValueError(f"manifest repo {name} must use an HTTP(S) clone URL")
        if not isinstance(commit, str) or not _SHA_RE.fullmatch(commit):
            raise ValueError(f"manifest repo {name} must pin a full 40-character commit SHA")
        for field_name in ("expected_rule_ids", "expected_false_positives"):
            if field_name in repo and (
                not isinstance(repo[field_name], list)
                or not all(isinstance(value, str) for value in repo[field_name])
            ):
                raise ValueError(f"manifest repo {name} field {field_name} must be a string array")
        if "domain" in repo and not isinstance(repo["domain"], str):
            raise ValueError(f"manifest repo {name} domain must be a string")
        if "rationale" in repo and not isinstance(repo["rationale"], str):
            raise ValueError(f"manifest repo {name} rationale must be a string")
        names.add(name)
        repos.append({**repo, "name": name, "url": url, "commit": commit.lower()})
    return repos


def _under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _remote_matches(dest: Path, url: str) -> bool:
    remote = _git(["-C", str(dest), "remote", "get-url", "origin"], timeout=30)
    if remote.returncode != 0:
        return False
    actual = remote.stdout.strip().rstrip("/")
    expected = url.rstrip("/")
    return actual == expected or actual.removesuffix(".git") == expected.removesuffix(".git")


def _checkout_pinned(url: str, dest: Path, commit: str) -> dict[str, Any]:
    """Clone/fetch only inside the managed destination; never reset an existing tree."""

    try:
        if dest.exists():
            if not dest.is_dir() or not (dest / ".git").exists():
                return {"status": "clone_failed", "error": "destination is not a git checkout"}
            if not _remote_matches(dest, url):
                return {"status": "clone_failed", "error": "existing checkout remote does not match"}
            dirty = _git(["-C", str(dest), "status", "--porcelain"], timeout=30)
            if dirty.stdout.strip():
                return {"status": "clone_failed", "error": "existing checkout has uncommitted changes"}
        else:
            clone = _git(["clone", "--filter=blob:none", "--no-checkout", url, str(dest)])
            if clone.returncode != 0:
                return {"status": "clone_failed", "error": clone.stderr.strip()[-1000:]}
        fetched = _git(["-C", str(dest), "fetch", "--depth", "1", "origin", commit])
        if fetched.returncode != 0:
            return {"status": "clone_failed", "error": fetched.stderr.strip()[-1000:]}
        checkout = _git(["-C", str(dest), "checkout", "--detach", "FETCH_HEAD"])
        if checkout.returncode != 0:
            return {"status": "clone_failed", "error": checkout.stderr.strip()[-1000:]}
        resolved = _git(["-C", str(dest), "rev-parse", "HEAD"], timeout=30)
        if resolved.returncode != 0 or resolved.stdout.strip().lower() != commit:
            return {"status": "clone_failed", "error": "resolved commit did not match manifest"}
        return {
            "status": "ok",
            "resolved_commit": resolved.stdout.strip().lower(),
            "remote_url": _git(["-C", str(dest), "remote", "get-url", "origin"], timeout=30).stdout.strip(),
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "clone_failed", "error": str(exc)}


def _environment() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for module_name in ("pandas", "numpy", "sklearn", "scipy", "nbformat"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            continue
    return versions


def run_corpus(manifest_path: Path, workdir: Path, config: Config) -> dict[str, Any]:
    """Run a manifest without executing repository code."""

    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repos = _validate_manifest(manifest, manifest_path)
    managed = workdir.expanduser().resolve()
    if managed == managed.parent:
        raise ValueError("corpus workdir cannot be the filesystem root")
    if managed.exists() and not managed.is_dir():
        raise ValueError(f"corpus workdir is not a directory: {managed}")
    managed.mkdir(parents=True, exist_ok=True)
    if not _under(managed, managed.parent):
        raise ValueError("invalid corpus workdir")

    cfg = config.copy()
    engine = StaticEngine(cfg)
    per_rule: Counter[str] = Counter()
    per_domain_rule: dict[str, Counter[str]] = defaultdict(Counter)
    repo_results: list[dict[str, Any]] = []
    flagged = 0
    analyzed = 0
    labeled = 0
    failures = 0
    clone_failures = 0
    analysis_failures = 0
    partial_scans = 0

    for repo in repos:
        name = repo["name"]
        dest = managed / name
        if not _under(dest, managed):
            repo_results.append(
                {
                    "name": name,
                    "status": "clone_failed",
                    "completion": "failed",
                    "rate_eligible": False,
                    "label_eligible": False,
                    "flagged": None,
                    "error": "unsafe destination",
                }
            )
            failures += 1
            clone_failures += 1
            continue
        checkout = _checkout_pinned(repo["url"], dest, repo["commit"])
        if checkout["status"] != "ok":
            repo_results.append(
                {
                    "name": name,
                    **checkout,
                    "completion": "failed",
                    "rate_eligible": False,
                    "label_eligible": False,
                    "flagged": None,
                }
            )
            failures += 1
            clone_failures += 1
            continue
        result = engine.run_result([dest], root=dest)
        completion = result.completion.value
        complete = completion == "complete"
        if complete:
            analyzed += 1
        elif completion == "partial":
            partial_scans += 1
        else:
            analysis_failures += 1
            failures += 1
        rule_counts = Counter(f.rule_id for f in result.findings)
        per_rule.update(rule_counts)
        domain = str(repo.get("domain", "unknown"))
        per_domain_rule[domain].update(rule_counts)
        expected = set(repo.get("expected_rule_ids", []))
        expected_fp = set(repo.get("expected_false_positives", []))
        is_labeled = bool(expected or expected_fp or "expected_rule_ids" in repo)
        is_flagged = gate_failed(result.findings, cfg.fail_on, cfg.gate_confidence) if complete else None
        if is_flagged is not None:
            flagged += int(is_flagged)
        if is_labeled and complete:
            labeled += 1
        observed = set(rule_counts)
        repo_results.append(
            {
                "name": name,
                "domain": domain,
                "status": "ok" if complete else completion,
                "resolved_commit": checkout["resolved_commit"],
                "remote_url": checkout["remote_url"],
                "completion": completion,
                "rate_eligible": complete,
                "label_eligible": is_labeled and complete,
                "n_findings": len(result.findings),
                "by_rule": dict(rule_counts),
                "flagged": is_flagged,
                "expected_rule_ids": sorted(expected),
                "expected_false_positives": sorted(expected_fp),
                "missing_expected": sorted(expected - observed),
                "unexpected_rule_ids": sorted(observed - expected - expected_fp),
                "analysis_failures": [diagnostic.to_dict() for diagnostic in result.diagnostics],
                "rationale": repo.get("rationale"),
            }
        )

    return {
        "schema_version": "2.0",
        "tool": {"name": "ml-leakproof", "version": __version__},
        "manifest": {"path": str(manifest_path), "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()},
        "environment": _environment(),
        "configuration": {
            "profile": cfg.profile,
            "fail_on": cfg.fail_on.value,
            "gate_confidence": cfg.gate_confidence,
        },
        "n_repos": len(repos),
        "n_analyzed": analyzed,
        "n_failed": failures,
        "n_clone_failed": clone_failures,
        "n_analysis_failed": analysis_failures,
        "n_partial": partial_scans,
        "n_labeled": labeled,
        "flagged_repository_rate": (flagged / analyzed) if analyzed else None,
        "findings_by_rule": dict(per_rule.most_common()),
        "findings_by_domain": {domain: dict(counts.most_common()) for domain, counts in per_domain_rule.items()},
        "repos": repo_results,
        "notes": [
            "Repositories were scanned as source; repository code was not executed.",
            "flagged_repository_rate is not a labeled detection metric.",
            "Clone failures, failed analyses, and partial scans are reported and excluded from the analyzed denominator and repository rates.",
        ],
    }
