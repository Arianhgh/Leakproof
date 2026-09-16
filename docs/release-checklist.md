# 0.2.0rc2 release checklist

This checklist records preparation of the documentation-corrected release
candidate that supersedes `0.2.0rc1`.

## Preflight

- [x] `ml_leakproof` is the only Python package in the wheel.
- [x] `ml-leakproof` is the only console entry point.
- [x] Version is `0.2.0rc2`.
- [x] `leakproof.toml` and `[tool.leakproof]` remain supported.
- [x] `ruff check ml_leakproof` passes.
- [x] `mypy --strict ml_leakproof` passes.
- [x] Unit and fixture tests pass.
- [x] Focused branch coverage for configuration, gating, hashing, autofix, and hooks is at least 90%.
- [x] Wheel and sdist build successfully.
- [x] Wheel metadata contains no legacy distribution/package name.
- [x] Corpus repositories are pinned to full commit SHAs.
- [x] Migration guide and release notes are present.
- [x] Machine-readable benchmark results are committed.

## Human review before publishing

- [ ] Review the public API and result/partial-result behavior.
- [ ] Review data handling and any explicit LLM provider configuration.
- [ ] Review the generated corpus report and any expected labels.
- [ ] Run the supported Python versions in CI.
- [ ] Confirm the PyPI project and trusted-publishing environment are correct.
- [ ] Decide whether to publish the release candidate.
- [ ] Create the release tag only after the publish decision.

The manual release workflow defaults to validation-only. It requires an explicit
publish input and a protected PyPI environment before it can upload artifacts.
