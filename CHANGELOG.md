# Changelog

## 0.2.0rc1 — release candidate

Status: unpublished and untagged; local release validation is documented in
`docs/releasing.md`.

### Release hardening

- Pin development tools and add a cross-version dependency lock; validate strict
  typing and full-package branch coverage on Python 3.10–3.14.
- Correct advisory metadata for all 13 advisory-only rules.
- Isolate malformed notebook cells, report skipped magics as partial analysis,
  preserve file-read failure coverage, and accept suppressions for unselected
  rules that exist in the catalog.
- Fix Unicode source columns and portable SARIF fingerprints and artifact paths.
- Handle report write errors with exit code 2; corpus failures also return 2
  after preserving their report.
- Fix optional image/LLM backend typing, report unreadable image inputs as partial,
  and include the Parquet backend with the data extra.
- Ignore estimator-internal array bookkeeping in runtime lineage hooks, including
  scikit-learn 1.7 score calculations. Preserve destination provenance for NumPy
  `out=` writes and masked/multiple outputs; bookkeeping failures never repeat
  a successful in-place numeric operation.
- Expand regression tests and independently reviewed source scopes across five
  pinned repositories; replace the obsolete M002 label with a reasoned negative.
- Fingerprint benchmark source inputs and smoke-test base and extra installations
  from both wheel and source archive without publishing.

### Library features

- Renamed the distribution and executable to `ml-leakproof` and the import
  package to `ml_leakproof`; retained `leakproof.toml` and `[tool.leakproof]`.
- Added the `AnalysisResult` / `Diagnostic` / `Coverage` contract with explicit
  `complete`, `partial`, and `failed` states.
- Added result-aware `check`, `run`, and `audit-data` CLI commands with stable
  exit codes, profiles, report schema 2.0, JSON, Markdown, terminal, and SARIF
  output.
- Expanded static analysis with source order, scope, reassignment, keyword
  argument, pipeline, notebook, and local-symbol handling.
- Added type-aware exact overlap checks for train/validation/test pairs and
  bounded advisory similarity and target-leakage probes.
- Added explicit and automatic runtime lineage tracking, nested sessions,
  evaluation-set checks, CV reuse checks, and script status preservation.
- Automatic holdout tracking now preserves exact split positions for every
  returned array, refuses ambiguous duplicate-value matches, and validates the
  runtime layer before installing hooks. Runtime preprocessing ancestry is
  session-owned and operation-based: supported NumPy copies and casts retain
  it, equal-content arrays do not, shape-only constructors stay independent,
  and unsupported propagation reports partial coverage.
- Tracked NumPy arrays serialize as ordinary arrays through a module-level
  reducer for pickle/joblib compatibility; session-owned lineage is not
  persisted, and importing the taint module remains NumPy-lazy.
- Added safe LibCST seed autofixes, opt-in bounded LLM explanations, plugin
  entry points, pinned source-only corpus audits, CI/action/pre-commit support,
  and release validation artifacts.

## 0.1 compatibility note

The 0.2 release candidate does not install `leakproof` compatibility aliases.
Migrate imports to `ml_leakproof`, commands to `ml-leakproof`, and plugin entry
point groups to `ml_leakproof.rules` / `ml_leakproof.adapters`. Configuration
and suppression spellings remain `leakproof.toml`, `[tool.leakproof]`, and
`# leakproof: ...`.
