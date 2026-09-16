"""Install an unpublished artifact in clean environments and test public entry points.

Run with the locked dev stack: python scripts/smoke_distribution.py dist/<artifact>.
No commands in this script publish, tag, or upload distributions.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import venv
from pathlib import Path

BASE_SMOKE = """
import importlib.util
import json
from pathlib import Path
import ml_leakproof as lp
assert lp.__version__ == "0.2.0rc1"
for name in ("numpy", "pandas", "sklearn", "anthropic", "openai", "libcst"):
    assert importlib.util.find_spec(name) is None, name
source = Path("leaky.py")
source.write_text("from sklearn.preprocessing import StandardScaler\\nfrom sklearn.model_selection import train_test_split\\nX = load_data()\\nz = StandardScaler().fit_transform(X)\\na,b = train_test_split(z, random_state=0)\\n")
result = lp.analyze(source, lp.Config())
assert result.complete
assert "P001" in {f.rule_id for f in result.findings}
assert lp.check(source, lp.Config())
assert not importlib.util.find_spec("tests")
"""

EXTRAS_SMOKE = """
import numpy as np
import pandas as pd
from pathlib import Path
from ml_leakproof import Config, analyze_data, watch
from ml_leakproof.autofix import apply_fixes
from ml_leakproof import analyze
train = pd.DataFrame({"feature": list(range(12)), "label": [i % 2 for i in range(12)]})
test = pd.DataFrame({"feature": [0,13], "label": [0,1]})
result = analyze_data(train, test, target="label", config=Config(select=["D001"]))
assert result.complete and {f.rule_id for f in result.findings} == {"D001"}
train.to_parquet("train.parquet")
test.to_parquet("test.parquet")
with watch(Config()) as session:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X = np.arange(160, dtype=float).reshape(80,2)
    transformed = StandardScaler().fit_transform(X)
    train_test_split(transformed, random_state=0)
assert session.result.complete
assert "P001" in {f.rule_id for f in session.findings}
path = Path("seed.py")
path.write_text("from sklearn.model_selection import train_test_split\\na,b = train_test_split(X)\\n")
result = analyze(path, Config())
assert apply_fixes(result.findings, write=True)
assert "random_state=0" in path.read_text()
"""


def run(command: list[str], cwd: Path, *, expected: int = 0) -> str:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    process = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    if process.returncode != expected:
        raise RuntimeError(
            f"{command} returned {process.returncode}, expected {expected}\n{process.stdout}\n{process.stderr}"
        )
    return process.stdout


def main(artifact: Path) -> None:
    if not artifact.is_file():
        raise ValueError(f"artifact not found: {artifact}")
    with tempfile.TemporaryDirectory(prefix="leakproof-artifact-") as temporary:
        root = Path(temporary)
        for extras in (False, True):
            environment = root / ("extras" if extras else "base")
            venv.EnvBuilder(with_pip=True).create(environment)
            bin_dir = environment / ("Scripts" if os.name == "nt" else "bin")
            python = str(bin_dir / ("python.exe" if os.name == "nt" else "python"))
            cli = str(bin_dir / ("ml-leakproof.exe" if os.name == "nt" else "ml-leakproof"))
            requirement = str(artifact.resolve()) + ("[data,runtime,fix]" if extras else "")
            run([python, "-m", "pip", "install", requirement], root)
            run([python, "-m", "pip", "check"], root)
            run([python, "-c", EXTRAS_SMOKE if extras else BASE_SMOKE], root)
            assert "0.2.0rc1" in run([python, "-m", "ml_leakproof", "version"], root)
            records = json.loads(run([cli, "rules", "--format", "json"], root))
            assert len(records) == 30
            assert next(r for r in records if r["id"] == "M003")["advisory_only"]
            assert "P001" in run([cli, "explain", "P001"], root)
            if extras:
                run(
                    [
                        cli,
                        "audit-data",
                        "--train",
                        "train.parquet",
                        "--test",
                        "test.parquet",
                        "--target",
                        "label",
                        "--format",
                        "json",
                    ],
                    root,
                    expected=1,
                )
            else:
                run(
                    [cli, "check", "leaky.py", "--format", "sarif", "--output", "report.sarif"],
                    root,
                    expected=1,
                )
                assert json.loads((root / "report.sarif").read_text())["version"] == "2.1.0"
                plugin = Path(__file__).resolve().parents[1] / "tests/integration/example_plugin"
                plugin_copy = root / "plugin-source"
                shutil.copytree(
                    plugin,
                    plugin_copy,
                    ignore=shutil.ignore_patterns("build", "*.egg-info", "__pycache__"),
                )
                run([python, "-m", "pip", "install", str(plugin_copy)], root)
                (root / "plugin_sample.py").write_text("value = eval('1 + 1')\n")
                response = json.loads(
                    run(
                        [
                            cli,
                            "check",
                            "plugin_sample.py",
                            "--select",
                            "myorg-X001",
                            "--fail-on",
                            "medium",
                            "--format",
                            "json",
                        ],
                        root,
                        expected=1,
                    )
                )
                assert response["completion"] == "complete"
                assert response["findings"][0]["rule_id"] == "myorg-X001"
                run(
                    [
                        python,
                        "-c",
                        "from ml_leakproof.static.adapters import load_adapters; assert load_adapters().is_transformer('MyCustomScaler')",
                    ],
                    root,
                )

            print(
                f"{artifact.name}: {'extras' if extras else 'base'} installation passed", flush=True
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    main(parser.parse_args().artifact)
