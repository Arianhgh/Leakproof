# Migrating to 0.2

The 0.2 release candidate intentionally uses a new distribution and import
name. It does not install compatibility aliases for the unrelated PyPI
project named `leakproof`.

## Names and entry points

Replace the old package and command names:

| Before 0.2 | 0.2 release candidate |
|---|---|
| `pip install leakproof` | `pip install ml-leakproof` |
| `import leakproof` | `import ml_leakproof` |
| `leakproof check ...` | `ml-leakproof check ...` |
| `leakproof.rules` plugin group | `ml_leakproof.rules` |
| `leakproof.adapters` plugin group | `ml_leakproof.adapters` |

Configuration and source suppressions keep their existing spellings:
`leakproof.toml`, `[tool.leakproof]`, and `# leakproof: ...`.

## Results and incomplete analysis

`analyze()` and `analyze_data()` return an `AnalysisResult` containing findings,
diagnostics, coverage, and a `complete`, `partial`, or `failed` status. The
list-returning `check()` and `audit_data()` helpers raise `AnalysisError` for
incomplete work unless `allow_partial=True` is passed or configured. Update
callers that previously treated an empty list as proof of a clean scan.

CLI exit codes are now stable: `0` means no gateable finding, `1` means a
gateable finding, and `2` means invalid usage, incomplete/failed analysis, or a
non-zero script exit. `--min-confidence` only controls displayed findings;
`--gate-confidence` controls the gate.

## Configuration changes

The default display profile is `ci` (`min_confidence = 0.60`), while
`notebook` displays from `0.30` and `research` from `0.00`. All profiles keep a
`0.75` default gate confidence. Invalid types, unknown keys, non-positive
resource limits, and enabled LLM configuration without an explicit provider
and model now fail validation.

The removed `data.mi_zscore` setting has no 0.2 equivalent: mutual-information
results are advisory and the implemented detector uses `mi_floor` and
`mi_dominance_ratio`.

## Runtime and autofix changes

Runtime tracking supports explicit split registration and automatic hooks for
documented sklearn-family operations. Automatic tracking of a split helper
aliased before entering `watch()` remains outside the support boundary; call
`session.register_split(...)` for that case.

Learned NumPy transform outputs now carry session-owned, operation-based
ancestry through supported copies, dtype casts, and row selections. Equal
values in an independently created array do not establish ancestry. If a
derived operation cannot be propagated, the runtime reports an `LP417`
diagnostic and marks the result partial when it is consumed.

Tracked arrays use a module-level serialization reducer, so split arrays and
fitted estimators remain compatible with pickle and joblib after `watch()`
exits. Saved values contain data only: session-owned lineage is not restored
when they are loaded.

Install `ml-leakproof[fix]` to use the LibCST-backed seed fixer. The historical
`[cst]` extra remains as a dependency alias. Notebook autofix is unavailable
in this release candidate, and LLM explanations/triage remain opt-in and
advisory.
