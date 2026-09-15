# Leakproof

Leakproof is a static, dataset, and runtime checker for data leakage and
evaluation-rigor mistakes in machine-learning workflows. The release candidate
is `0.2.0rc1`; the distribution is `ml-leakproof`, the Python package is
`ml_leakproof`, and the executable is `ml-leakproof`.

## Install

```bash
pip install ml-leakproof                 # static analysis
pip install "ml-leakproof[data]"         # pandas / NumPy / scikit-learn data checks
pip install "ml-leakproof[runtime]"      # runtime provenance instrumentation
pip install "ml-leakproof[llm]"          # optional explanation providers
pip install "ml-leakproof[fix]"          # LibCST autofixes
```

The base install does not import or require pandas, scikit-learn, runtime
integrations, or LLM SDKs. `python -m ml_leakproof` is equivalent to the
`ml-leakproof` command.

## Quickstart

```bash
ml-leakproof check .
ml-leakproof check . --format sarif --output leakproof.sarif --fail-on high
ml-leakproof run train.py --format json --output runtime.json
ml-leakproof audit-data --train train.csv --test test.csv --target label
ml-leakproof rules
ml-leakproof explain P001
ml-leakproof version
```

`check` is static-only. `run` executes one script inside runtime instrumentation;
the script's own exit status is preserved in the result. `audit-data` validates
explicit pandas train/validation/test frames and never executes model code.

## Result contract

Result-oriented APIs return `AnalysisResult`, not an unqualified list:

```python
from ml_leakproof import AnalysisError, analyze, analyze_data, check

result = analyze("src")
print(result.findings)
print(result.diagnostics)
print(result.coverage, result.completion)  # complete | partial | failed

try:
    findings = check("src")
except AnalysisError as error:
    # The partial result is available for inspection and serialization.
    print(error.result.coverage, error.result.diagnostics)

data_result = analyze_data(train, test, val=validation, target="label")
```

The list wrappers `check` and `audit_data` raise `AnalysisError` when requested
analysis is incomplete unless `allow_partial=True` is supplied in the call or
configuration. Partial results are never silently presented as complete.

## Exit codes and profiles

| Code | Meaning |
|---|---|
| `0` | No gateable finding at or above the configured severity gate |
| `1` | At least one gateable finding meets the severity and confidence gate |
| `2` | Usage/configuration error, failed analysis, incomplete analysis, or non-zero script exit |

`--min-confidence` controls display only. It does not change the gate decision.
`--gate-confidence` controls the minimum confidence required to fail a gate.
The built-in profiles are:

| Profile | Display threshold | Gate threshold |
|---|---:|---:|
| `ci` | `0.60` | `0.75` |
| `notebook` | `0.30` | `0.75` |
| `research` | `0.00` | `0.75` |

Findings marked `advisory_only`—for example similarity and target-predictivity
probes, grouped/time hints, and reproducibility/reporting suggestions—are
visible evidence but never gate a run. A clean scan means that no checked rule
produced a gateable finding; it is not proof that a methodology is valid.

## Configuration and suppressions

Keep the dedicated `leakproof.toml` filename, or use `[tool.leakproof]` in
`pyproject.toml`. CLI flags override project configuration. Unknown keys and
invalid ranges fail loudly. LLM use is disabled by default and requires an
explicit provider and model.

```toml
[tool.leakproof]
profile = "ci"
select = ["ALL"]
fail_on = "high"
min_confidence = 0.60
gate_confidence = 0.75
layers = ["static", "data", "runtime"]
```

Suppress only a reviewed, confirmed false positive:

```python
# leakproof: ignore[P001]
```

File-wide suppression is available with `# leakproof: ignore-file`; malformed
or unknown directives are reported as diagnostics. Suppression markers inside
strings are not interpreted as directives.

## What is checked

The built-in catalog has 30 rules spanning split hygiene, cross-validation,
preprocessing, overlap, target leakage, temporal boundaries, test-set
adaptivity, metrics, and determinism. See the [rule catalog](docs/rules.md) or
run `ml-leakproof rules --format json`.

Static analysis is order- and scope-aware, understands ordinary and keyword
arguments, reassignment, aliases, pipelines, and common sklearn adapters. It
does not execute source code. Notebook cells are analyzed in stored order;
execution history and out-of-order state are not reconstructed, and malformed
or magic cells are isolated with coverage diagnostics.

The data layer checks every declared split pair, including validation, with
type-aware exact row hashes. Near-duplicate, target-predictivity, mutual
information, and imbalance checks are bounded and advisory; capped or
unavailable work is recorded in `coverage` and `diagnostics`. Input data stays
in the current process unless the explicitly enabled LLM explanation layer is
used; datasets are not sent to an LLM by the data audit itself.

Runtime tracking supports explicit registration as well as automatic hooks for
common sklearn splitters, estimators, pipelines, CV helpers, pandas/NumPy
assembly, and supported integrations. Hooks are transactional and nested
sessions are isolated. Runtime instrumentation covers the current process and
does not reconstruct child-process execution. Import split helpers inside the
`watch()` scope when relying on automatic hooks; explicit `register_split()` is
available for integrations that cannot be patched automatically.

NumPy outputs from learned preprocessing carry session-owned, operation-based
ancestry through supported copies, casts, and row selections. Equal-valued
independent arrays are not linked; unsupported propagation is reported as
partial coverage instead of being treated as clean.

Tracked arrays serialize as ordinary NumPy arrays for pickle/joblib
compatibility. Lineage remains local to the originating watch session and is
intentionally not restored when a saved array or estimator is loaded.

## Autofix and LLM explanations

`ml-leakproof check --fix-preview` shows a unified diff. `--fix` applies only
verified LibCST edits (currently deterministic seed arguments), checks the
source digest, writes atomically, and rescans. It never rewrites notebooks or
calls that use `**kwargs`.

`--explain` may add bounded, uncertainty-aware explanations to deterministic
findings. `--llm-triage` is limited to files that failed static parsing. LLM
output cannot create, delete, or replace a deterministic finding, and is never
required for a normal scan.

## CI, plugins, and corpus benchmarks

- [GitHub Action and CI usage](docs/ci.md)
- [Rule and adapter plugins](docs/plugins.md)
- [0.2 migration guide](docs/migration-0.2.md)
- [Benchmark methodology](docs/benchmark.md)
- Immutable-commit [corpus manifest](corpus-manifest.json)

Third-party plugins use the `ml_leakproof.rules` and
`ml_leakproof.adapters` entry-point groups and vendor-prefixed rule IDs. See
the plugin guide for the migration from the pre-0.2 import and executable
names.

## License

MIT; see [LICENSE](LICENSE).
