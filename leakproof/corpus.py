"""Corpus audit: clone a manifest of public ML repos and aggregate leakage rates.

Manifest format (JSON)::

    {
      "repos": [
        {"name": "example", "url": "https://github.com/owner/repo", "domain": "tabular"},
        ...
      ]
    }

Produces an aggregate report: findings per rule and per domain, plus a leakage
rate (fraction of repos with at least one >= MEDIUM finding).
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from .core.config import Config
from .core.models import Severity
from .static.engine import StaticEngine


def _clone(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True,
            capture_output=True,
            timeout=300,
        )
        return True
    except Exception:
        return False


def run_corpus(manifest_path: Path, workdir: Path, config: Config) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repos = manifest.get("repos", [])
    workdir.mkdir(parents=True, exist_ok=True)
    engine = StaticEngine(config)

    per_rule: Counter = Counter()
    per_domain_rule: dict[str, Counter] = defaultdict(Counter)
    repo_results: list[dict] = []
    repos_with_leakage = 0

    for repo in repos:
        name = repo["name"]
        dest = workdir / name
        ok = _clone(repo["url"], dest)
        if not ok:
            repo_results.append({"name": name, "status": "clone_failed"})
            continue
        findings = engine.run([dest], root=dest)
        rule_counts = Counter(f.rule_id for f in findings)
        per_rule.update(rule_counts)
        domain = repo.get("domain", "unknown")
        per_domain_rule[domain].update(rule_counts)
        has_leak = any(f.severity.gate_rank >= Severity.MEDIUM.gate_rank for f in findings)
        repos_with_leakage += int(has_leak)
        repo_results.append(
            {
                "name": name,
                "domain": domain,
                "status": "ok",
                "n_findings": len(findings),
                "by_rule": dict(rule_counts),
                "has_leakage": has_leak,
            }
        )

    n_ok = sum(1 for r in repo_results if r.get("status") == "ok")
    return {
        "n_repos": len(repos),
        "n_analyzed": n_ok,
        "leakage_rate": (repos_with_leakage / n_ok) if n_ok else 0.0,
        "findings_by_rule": dict(per_rule.most_common()),
        "findings_by_domain": {d: dict(c.most_common()) for d, c in per_domain_rule.items()},
        "repos": repo_results,
    }
