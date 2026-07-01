"""Run leakproof across a small repo corpus."""

from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from .core.aggregate import gate_failed
from .core.config import Config
from .static.engine import StaticEngine


def _clone(url: str, dest: Path, ref: str | None = None) -> bool:
    try:
        if not dest.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(dest)],
                check=True,
                capture_output=True,
                timeout=300,
            )
        if ref:
            fetched = subprocess.run(
                ["git", "-C", str(dest), "fetch", "--depth", "1", "origin", ref],
                capture_output=True,
                timeout=300,
            )
            checkout_target = "FETCH_HEAD" if fetched.returncode == 0 else ref
            subprocess.run(
                ["git", "-C", str(dest), "checkout", "--detach", checkout_target],
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
        ref = repo.get("ref") or repo.get("commit")
        ok = _clone(repo["url"], dest, ref)
        if not ok:
            repo_results.append({"name": name, "status": "clone_failed"})
            continue
        findings = engine.run([dest], root=dest)
        rule_counts = Counter(f.rule_id for f in findings)
        per_rule.update(rule_counts)
        domain = repo.get("domain", "unknown")
        per_domain_rule[domain].update(rule_counts)
        has_leak = gate_failed(findings, config.fail_on, config.gate_confidence)
        repos_with_leakage += int(has_leak)
        expected = set(repo.get("expected_rule_ids", []))
        expected_fp = set(repo.get("expected_false_positives", []))
        observed = set(rule_counts)
        repo_results.append(
            {
                "name": name,
                "domain": domain,
                "ref": ref,
                "status": "ok",
                "n_findings": len(findings),
                "by_rule": dict(rule_counts),
                "has_leakage": has_leak,
                "expected_rule_ids": sorted(expected),
                "expected_false_positives": sorted(expected_fp),
                "missing_expected": sorted(expected - observed),
                "unexpected_rule_ids": sorted(observed - expected - expected_fp),
                "rationale": repo.get("rationale"),
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
