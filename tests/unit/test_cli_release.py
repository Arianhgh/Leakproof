"""Command contracts through Typer, including failures and file reports."""

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from ml_leakproof.cli import app
from ml_leakproof.core.registry import all_rules
from tests._harness import LEAKY, run_fixture_result

runner = CliRunner()


@pytest.mark.parametrize(
    "args",
    [
        ["rules", "--format", "xml"],
        ["rules", "--layer", "unknown"],
        ["explain", "BAD"],
        ["check", "--format", "xml"],
        ["check", "--format", "json", "--format", "sarif"],
        ["check", "--profile", "missing"],
        ["check", "--fail-on", "missing"],
        ["check", "--layer", "runtime"],
    ],
)
def test_invalid_options_return_usage_error(args):
    result = runner.invoke(app, args)
    assert result.exit_code == 2, result.output


def test_catalog_advisory_metadata_matches_actual_findings():
    response = runner.invoke(app, ["rules", "--format", "json"])
    records = {r["id"]: r for r in json.loads(response.stdout)}
    advisory = {
        "C002",
        "C003",
        "C005",
        "D002",
        "D005",
        "D006",
        "M001",
        "M003",
        "R001",
        "R002",
        "S002",
        "T003",
        "TM002",
    }
    assert {rid for rid, r in records.items() if r["advisory_only"]} == advisory
    for path in LEAKY.glob("*.py"):
        for finding in run_fixture_result(path).findings:
            assert finding.advisory_only == records[finding.rule_id]["advisory_only"]
    assert set(records) == set(all_rules())
    assert runner.invoke(app, ["rules"]).exit_code == 0
    filtered = runner.invoke(app, ["rules", "--layer", "runtime", "--format", "json"])
    assert all("runtime" in r["layers"] for r in json.loads(filtered.stdout))


def test_multiple_reports_and_display_independent_gate(tmp_path):
    from tests.unit.test_cli import LEAKY as source

    path = tmp_path / "leak.py"
    path.write_text(source)
    output = tmp_path / "report.txt"
    result = runner.invoke(
        app,
        [
            "check",
            str(path),
            "--format",
            "json",
            "--format",
            "sarif",
            "--format",
            "markdown",
            "--output",
            str(output),
            "--min-confidence",
            "1",
        ],
    )
    assert result.exit_code == 1, result.output
    data = json.loads((tmp_path / "report.json.txt").read_text())
    assert data["findings"] == []
    assert (tmp_path / "report.sarif.txt").is_file()
    assert "Leakproof" in (tmp_path / "report.markdown.txt").read_text()


def test_output_errors_and_partial_exit(tmp_path):
    path = tmp_path / "bad.py"
    path.write_text("if broken")
    assert runner.invoke(app, ["check", str(path), "--format", "json"]).exit_code == 2
    assert runner.invoke(app, ["check", str(path), "--allow-partial"]).exit_code == 0
    for output in [tmp_path, tmp_path / "missing" / "report.json"]:
        assert runner.invoke(app, ["check", str(path), "--output", str(output)]).exit_code == 2


def test_runtime_errors_and_user_exit(tmp_path):
    for script, status in [('raise RuntimeError("oops")', 1), ("raise SystemExit(7)", 7)]:
        path = tmp_path / "script.py"
        path.write_text(script)
        response = runner.invoke(app, ["run", str(path), "--format", "json"])
        assert response.exit_code == 2
        data = json.loads(response.stdout)
        assert data["script_exit_status"] == status
    assert runner.invoke(app, ["run", str(tmp_path / "missing.py")]).exit_code == 2


def test_csv_audit_reports_overlap_and_invalid_inputs(tmp_path):
    train, test = tmp_path / "train.csv", tmp_path / "test.csv"
    pd.DataFrame({"id": [1, 2, 3], "label": [0, 1, 0]}).to_csv(train, index=False)
    pd.DataFrame({"id": [1, 4], "label": [0, 1]}).to_csv(test, index=False)
    config = tmp_path / "config.toml"
    config.write_text('select = ["D001"]\n')
    args = [
        "audit-data",
        "--train",
        str(train),
        "--test",
        str(test),
        "--target",
        "label",
        "--config",
        str(config),
        "--format",
        "json",
    ]
    response = runner.invoke(app, args)
    assert response.exit_code == 1, response.output
    assert json.loads(response.stdout)["findings"][0]["rule_id"] == "D001"
    test.unlink()
    assert runner.invoke(app, args).exit_code == 2


def test_version_and_fix_preview(tmp_path):
    assert "0.2.0rc2" in runner.invoke(app, ["version"]).stdout
    path = tmp_path / "seed.py"
    source = "from sklearn.model_selection import train_test_split\na,b = train_test_split(X)\n"
    path.write_text(source)
    response = runner.invoke(app, ["check", str(path), "--select", "R001", "--fix-preview"])
    assert response.exit_code == 0
    assert "random_state" in response.output
    assert path.read_text() == source


def test_parquet_audit_includes_backend_in_data_extra(tmp_path):
    train, test = tmp_path / "train.parquet", tmp_path / "test.parquet"
    pd.DataFrame({"x": [1, 2]}).to_parquet(train)
    pd.DataFrame({"x": [1, 3]}).to_parquet(test)
    config = tmp_path / "config.toml"
    config.write_text('select = ["D001"]\n')
    result = runner.invoke(
        app,
        [
            "audit-data",
            "--train",
            str(train),
            "--test",
            str(test),
            "--config",
            str(config),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)["completion"] == "complete"


def test_report_write_failure_has_stable_exit_code(tmp_path, monkeypatch):
    from pathlib import Path

    path = tmp_path / "clean.py"
    path.write_text("x = 1")
    original = Path.write_text

    def write(self, *args, **kwargs):
        if self.name == "output.json":
            raise PermissionError("read only")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write)
    result = runner.invoke(
        app, ["check", str(path), "--format", "json", "--output", str(tmp_path / "output.json")]
    )
    assert result.exit_code == 2
    assert "report error" in result.stderr
