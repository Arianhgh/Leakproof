# leakproof

> A leakage and evaluation-rigor checker for ML and data-science projects.

ML pipelines can run cleanly and still be wrong. `leakproof` checks for common
leakage and evaluation mistakes in three ways:

- **static** - AST and def-use/taint analysis; no execution or dataset required.
- **data** - exact and near-duplicate overlap, group/temporal leakage, target-leakage probes.
- **runtime** - sklearn/pandas instrumentation for fits that touch held-out rows.

Reports are available for terminals, JSON, Markdown, and SARIF. The optional LLM layer
only explains existing findings.

## Install

```bash
pip install leakproof            # static layer, zero native deps
pip install "leakproof[data]"    # + data layer (pandas/sklearn/datasketch/imagehash)
pip install "leakproof[runtime]" # + runtime instrumentation (wrapt/sklearn)
pip install "leakproof[llm]"     # + optional LLM explanations
```

## Quickstart

```bash
# Static analysis of a repo (zero setup):
leakproof check path/to/project

# JSON / SARIF for CI:
leakproof check . --format sarif --output leakproof.sarif --fail-on high

# Fail CI only on confident findings:
leakproof check . --fail-on high --gate-confidence 0.75 --profile ci

# Hide low-confidence findings:
leakproof check . --min-confidence 0.6

# List or explain rules:
leakproof rules
leakproof explain P001

# Runtime instrumentation of a training script:
leakproof run train.py

# Data-layer audit of explicit splits:
leakproof audit-data --train train.csv --test test.csv --target y --group patient_id --time date
```

As a library:

```python
import leakproof

findings = leakproof.check("project/")            # static
findings = leakproof.audit_data(train, test, target="y", group="patient_id")  # data

with leakproof.watch() as session:                # runtime
    train_and_evaluate()
print(session.findings)
```

## Confidence

Reports show every finding by default. The exit code only fails when a finding meets
both `--fail-on` and `--gate-confidence` (default `0.75`). Use `--min-confidence`
to keep noisy findings out of a report. Profiles are `ci`, `notebook`, and `research`.

## What it detects

The built-in catalog covers split hygiene, cross-validation, preprocessing leakage,
data overlap, target leakage, temporal leakage, adaptivity/test-set reuse, metric misuse,
and determinism. Run `leakproof rules` for the full list, or see [docs/rules.md](docs/rules.md).

| Family | Examples |
|--------|----------|
| Split `S` | fit on full data before split, resampling before split, feature selection on full X |
| CV `C` | preprocessing fit outside the CV loop, plain KFold on grouped/time-series data, tuning on test |
| Preprocessing `P` | `fit_transform` on full X, fit on `concat([train, test])`, imputation across the split |
| Data overlap `D` | exact / near-duplicate rows across splits, group in both splits, duplicate test rows |
| Target leakage `D` | single feature near-perfectly predicts the target, suspiciously high MI |
| Temporal `TM` | train/test time overlap, look-ahead features |
| Adaptivity `T` | test scored multiple times, early stopping on test, best-of-N seeds reported |
| Metric `M` | accuracy on imbalanced target, threshold tuned on test, metric on training data |
| Determinism `R` | missing `random_state`, nondeterministic framework ops |

## Benchmark transparency

Every rule has at least one *leaky* fixture that should be flagged and one *clean* fixture
that should stay quiet. CI runs the fixture suite to catch both missed detections and
false positives:

```bash
pytest tests/ -q
python -m tests.corpus.metrics   # prints the per-rule precision/recall table
```

See [docs/benchmark.md](docs/benchmark.md) for the published table and the corpus-audit
methodology.

## CI integration

- **pre-commit**: see `.pre-commit-hooks.yaml`.
- **GitHub Action**: see `action.yml` and [docs/ci.md](docs/ci.md). Uploads SARIF to GitHub
  code scanning.

## Extending

Third parties add rules and framework adapters via entry points — no fork required. See
[docs/plugins.md](docs/plugins.md) and `tests/integration/example_plugin/`.

## License

MIT.
