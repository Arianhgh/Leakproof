"""Notebook isolation and the public incomplete-analysis contract."""

import nbformat
import pytest

from ml_leakproof import AnalysisError, Config, Layer, analyze, check
from ml_leakproof.static.engine import StaticEngine, discover_files
from ml_leakproof.static.notebook import notebook_to_source


def notebook(tmp_path, cells):
    path = tmp_path / "review.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(s) for s in cells]), path
    )
    return path


@pytest.mark.parametrize(
    "skipped",
    [
        "if broken\n    nested = 1",
        "%%bash\nif true; then\n    echo hello\nfi",
        "%matplotlib inline\nif broken\n    nested = 1",
    ],
)
def test_skipped_cell_does_not_hide_later_findings(tmp_path, skipped):
    path = notebook(
        tmp_path,
        [
            skipped,
            "from sklearn.model_selection import train_test_split\na, b = train_test_split(X)",
        ],
    )
    result = analyze(path, Config(profile="research"))
    assert result.completion.value == "partial"
    finding = next(f for f in result.findings if f.rule_id == "R001")
    assert finding.location.cell_index == 1
    assert finding.location.cell_line == 2
    assert finding.location.cell_id
    assert not any(d.code == "LP000" for d in result.diagnostics)
    with pytest.raises(AnalysisError):
        check(path, Config())
    assert check(path, Config(), allow_partial=True)


def test_line_magic_preserves_python_and_reports_gap(tmp_path):
    path = notebook(tmp_path, ["", "%matplotlib inline\nx = 1", "y = 2"])
    source, mapping, metadata = notebook_to_source(path)
    assert "x = 1" in source and "y = 2" in source
    assert metadata["non_python_cells"] == [1]
    assert mapping
    result = StaticEngine(Config()).run_file_result(path)
    assert result.completion.value == "partial"
    assert [d.code for d in result.diagnostics] == ["LP008"]


def test_valid_notebook_is_complete(tmp_path):
    path = notebook(tmp_path, ["x = 1", "y = x + 1"])
    result = analyze(path, Config())
    assert result.complete
    assert result.coverage.notes


def test_selected_rules_do_not_invalidate_other_known_suppressions(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("x = 1  # leakproof: ignore[R001]\n")
    assert analyze(path, Config(select=["P001"])).complete
    path.write_text("x = 1  # leakproof: ignore[NONEXISTENT]\n")
    with pytest.warns(UserWarning):
        result = StaticEngine(Config(select=["P001"])).run_file_result(path)
    assert not result.complete
    assert result.diagnostics[0].code == "LP201"


def test_discovery_deduplication_and_pruning(tmp_path):
    source = tmp_path / "ok.py"
    source.write_text("x = 1")
    excluded = tmp_path / "vendor"
    excluded.mkdir()
    (excluded / "bad.py").write_text("x = 1")
    assert discover_files([tmp_path, source], ["vendor/"], tmp_path) == [source]


def test_static_failure_and_disabled_layer(tmp_path):
    assert analyze([], Config()).failed
    assert analyze(tmp_path / "missing.py", Config()).failed
    text = tmp_path / "readme.txt"
    text.write_text("hello")
    result = analyze(text, Config())
    assert not result.complete
    assert str(text) in result.coverage.skipped_inputs
    assert analyze(text, Config(layers=[Layer.DATA])).failed
    invalid = tmp_path / "bad.ipynb"
    invalid.write_text("not a notebook")
    result = analyze(invalid, Config())
    assert result.failed
    assert str(invalid) in result.coverage.failed_inputs
    assert not result.coverage.analyzed_inputs
