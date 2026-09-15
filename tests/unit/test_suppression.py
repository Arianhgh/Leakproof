from __future__ import annotations

from pathlib import Path

from ml_leakproof.core.config import Config
from ml_leakproof.core.suppression import FileSuppressions, path_excluded
from ml_leakproof.static.engine import StaticEngine

LEAKY = """
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X = pd.read_csv("d.csv"); y = X.pop("target")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # leakproof: ignore[P001]
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)
"""


def test_inline_suppression(tmp_path):
    f = tmp_path / "m.py"
    f.write_text(LEAKY)
    findings = StaticEngine(Config()).run_file(f)
    assert "P001" not in {x.rule_id for x in findings}


def test_ignore_file(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("# leakproof: ignore-file\n" + LEAKY)
    findings = StaticEngine(Config()).run_file(f)
    assert findings == []


def test_bare_ignore_rejected(recwarn):
    sup = FileSuppressions.parse("x = 1  # leakproof: ignore\n")
    assert not sup.per_line
    assert any("bare" in str(w.message) for w in recwarn.list)


def test_path_excluded():
    root = Path("/proj")
    assert path_excluded(Path("/proj/tests/a.py"), ["tests/"], root=root)
    assert not path_excluded(Path("/proj/src/a.py"), ["tests/"], root=root)
