from __future__ import annotations

import json

from typer.testing import CliRunner

from leakproof.cli import app

runner = CliRunner()

LEAKY = """
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X = pd.read_csv("d.csv"); y = X.pop("target")
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)
"""


def test_check_exit_code_on_findings(tmp_path):
    f = tmp_path / "m.py"
    f.write_text(LEAKY)
    res = runner.invoke(app, ["check", str(f), "--no-color"])
    assert res.exit_code == 1
    assert "P001" in res.stdout


def test_check_clean_exit_zero(tmp_path):
    f = tmp_path / "clean.py"
    f.write_text("x = 1\n")
    res = runner.invoke(app, ["check", str(f), "--no-color"])
    assert res.exit_code == 0


def test_check_json_format(tmp_path):
    f = tmp_path / "m.py"
    f.write_text(LEAKY)
    res = runner.invoke(app, ["check", str(f), "--format", "json"])
    obj = json.loads(res.stdout)
    assert any(x["rule_id"] == "P001" for x in obj["findings"])


def test_select_ignore(tmp_path):
    f = tmp_path / "m.py"
    f.write_text(LEAKY)
    res = runner.invoke(app, ["check", str(f), "--select", "R001", "--format", "json"])
    obj = json.loads(res.stdout)
    ids = {x["rule_id"] for x in obj["findings"]}
    assert ids <= {"R001"}


def test_fail_on_gate(tmp_path):
    f = tmp_path / "m.py"
    f.write_text(LEAKY)
    # only R001 (low) selected; gate high -> exit 0
    res = runner.invoke(app, ["check", str(f), "--select", "R001", "--no-color"])
    assert res.exit_code == 0


def test_rules_command():
    res = runner.invoke(app, ["rules", "--format", "json"])
    obj = json.loads(res.stdout)
    assert any(r["id"] == "P001" for r in obj)


def test_explain_command():
    res = runner.invoke(app, ["explain", "P001"])
    assert res.exit_code == 0
    assert "P001" in res.stdout


def test_autofix_adds_random_state(tmp_path):
    f = tmp_path / "m.py"
    f.write_text(
        "from sklearn.model_selection import train_test_split\n"
        "a, b, c, d = train_test_split(X, y)\n"
    )
    res = runner.invoke(app, ["check", str(f), "--select", "R001", "--fix", "--no-color"])
    assert res.exit_code == 0
    assert "random_state=0" in f.read_text()
