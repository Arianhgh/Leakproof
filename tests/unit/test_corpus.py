"""Source-only corpus audit against temporary local Git repositories."""

import json
import subprocess

import pytest
from typer.testing import CliRunner

from ml_leakproof.cli import app
from ml_leakproof.core.config import Config
from ml_leakproof.corpus import _checkout_pinned, _validate_manifest, run_corpus


def git(path, *args):
    return subprocess.run(
        ["git", "-C", str(path), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def corpus_repo(tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init")
    git(origin, "config", "user.email", "test@example.invalid")
    git(origin, "config", "user.name", "Test")
    (origin / "example.py").write_text(
        "from sklearn.model_selection import train_test_split\na,b=train_test_split(X)\n"
    )
    git(origin, "add", ".")
    git(origin, "commit", "-m", "fixture")
    repo = dict(
        name="example",
        url="https://corpus.example.invalid/example",
        commit=git(origin, "rev-parse", "HEAD"),
        domain="tabular",
        expected_rule_ids=["R001"],
        expected_false_positives=[],
        rationale="Explicit unseeded splitter.",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"repos": [repo]}))
    import ml_leakproof.corpus as module

    def local_transport(document, path):
        validated = _validate_manifest(document, path)
        return [{**item, "url": origin.as_uri()} for item in validated]

    monkeypatch.setattr(module, "_validate_manifest", local_transport)
    return origin, repo, manifest


def test_pinned_source_scan_and_labels(tmp_path, corpus_repo):
    origin, repo, manifest = corpus_repo
    workdir = tmp_path / "work"
    report = run_corpus(manifest, workdir, Config(profile="research"))
    assert report["n_analyzed"] == report["n_labeled"] == 1
    assert report["n_failed"] == 0
    row = report["repos"][0]
    assert row["resolved_commit"] == repo["commit"]
    assert row["missing_expected"] == [] and row["by_rule"]["R001"] == 1
    assert report["flagged_repository_rate"] == 0
    # A clean cached checkout can be reused without executing its code.
    assert run_corpus(manifest, workdir, Config())["n_analyzed"] == 1
    (workdir / "example" / "example.py").write_text("dirty")
    failed = run_corpus(manifest, workdir, Config())
    assert failed["n_clone_failed"] == 1 and failed["flagged_repository_rate"] is None
    assert "uncommitted" in failed["repos"][0]["error"]


def test_partial_scan_excluded_from_denominator(tmp_path, corpus_repo):
    origin, repo, manifest = corpus_repo
    (origin / "broken.py").write_text("if broken")
    git(origin, "add", ".")
    git(origin, "commit", "-m", "broken file")
    repo["commit"] = git(origin, "rev-parse", "HEAD")
    manifest.write_text(json.dumps({"repos": [repo]}))
    report = run_corpus(manifest, tmp_path / "work", Config())
    assert report["n_partial"] == 1 and report["n_analyzed"] == 0
    assert report["repos"][0]["label_eligible"] is False
    assert report["flagged_repository_rate"] is None


def test_checkout_refuses_wrong_destination(tmp_path, corpus_repo):
    origin, repo, manifest = corpus_repo
    bad = tmp_path / "bad"
    bad.mkdir()
    assert "not a git checkout" in _checkout_pinned(repo["url"], bad, repo["commit"])["error"]
    assert "remote does not match" in _checkout_pinned(repo["url"], origin, repo["commit"])["error"]
    assert _checkout_pinned(origin.as_uri(), tmp_path / "new", "0" * 40)["status"] == "clone_failed"


@pytest.mark.parametrize(
    "change",
    [
        {"name": "../escape"},
        {"name": "/root"},
        {"commit": "main"},
        {"url": ""},
        {"expected_rule_ids": "R001"},
        {"expected_false_positives": [1]},
        {"domain": 1},
        {"rationale": 1},
    ],
)
def test_invalid_manifest_fields(tmp_path, corpus_repo, change):
    _, repo, manifest = corpus_repo
    with pytest.raises(ValueError):
        _validate_manifest({"repos": [{**repo, **change}]}, manifest)


@pytest.mark.parametrize("document", [[], {}, {"repos": "bad"}, {"repos": [7]}])
def test_invalid_manifest_shape(tmp_path, document):
    with pytest.raises(ValueError):
        _validate_manifest(document, tmp_path / "manifest.json")


def test_manifest_duplicate_and_workdir_errors(tmp_path, corpus_repo):
    _, repo, manifest = corpus_repo
    with pytest.raises(ValueError):
        _validate_manifest({"repos": [repo, repo]}, manifest)
    with pytest.raises(ValueError):
        run_corpus(tmp_path / "missing", tmp_path / "work", Config())
    with pytest.raises(ValueError):
        run_corpus(manifest, manifest, Config())


def test_corpus_cli(tmp_path, corpus_repo):
    _, repo, manifest = corpus_repo
    out = tmp_path / "report.json"
    response = CliRunner().invoke(
        app,
        [
            "corpus",
            "--manifest",
            str(manifest),
            "--workdir",
            str(tmp_path / "work"),
            "--output",
            str(out),
        ],
    )
    assert response.exit_code == 0, response.output
    assert json.loads(out.read_text())["n_repos"] == 1
