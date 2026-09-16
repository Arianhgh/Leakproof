# CI integration

Leakproof is designed to make incomplete analysis visible instead of silently
turning missing coverage into a green build.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | No gateable finding at or above `fail_on` |
| `1` | At least one gateable finding meets `fail_on` and `gate_confidence` |
| `2` | Usage/configuration error, incomplete/failed analysis, or non-zero script exit |

Display filtering (`--min-confidence`) does not affect the gate. Advisory-only
findings do not affect the gate. Use `--allow-partial` only when the workflow
explicitly accepts incomplete coverage.

## Direct CLI workflow

```yaml
name: leakproof

on:
  pull_request:
  push:

permissions:
  contents: read
  security-events: write

jobs:
  static:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      # Until publication, install an explicitly reviewed source revision.
      - run: python -m pip install "git+https://github.com/Arianhgh/Leakproof.git@<reviewed-commit-sha>"
      - run: >-
          ml-leakproof check . --format sarif --output leakproof.sarif
          --profile ci --fail-on high
      - uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: leakproof.sarif
```

The command intentionally runs before SARIF upload; a finding may produce exit
code `1`, while `if: always()` still uploads the report.

## Repository action

When this repository is checked out, the bundled composite action can be used
as a local action:

```yaml
      - uses: ./
        with:
          paths: |
            src
            notebooks
          select: "P* C* S*"
          fail-on: high
          layers: static
          output: leakproof.sarif
```

`action.yml` parses newline-separated `paths` and space-separated `select`
values as argument arrays. It does not evaluate input text as shell code, and
it preserves the CLI exit code. The action installs the checked-out package, so
it is suitable for development branches; no release tag or marketplace
publication is implied by this repository.

## pre-commit

Use the repository's hook definition or copy the equivalent entry into a local
configuration:

```yaml
repos:
  - repo: https://github.com/Arianhgh/Leakproof
    rev: main
    hooks:
      - id: ml-leakproof
```

The hook checks staged Python files with the static layer and a high-severity
gate. Add a project `leakproof.toml` when the default excludes or rule selection
need to change.


## Testing this library

The repository CI tests Python 3.10 through 3.14 using `uv.lock`. The lock records
compatible dependency versions for each Python version; runtime dependencies in
the published metadata remain flexible. Mypy and Ruff are pinned in the dev
extra. Mypy uses the active interpreter version so newer NumPy stubs are not
incorrectly parsed as Python 3.10 syntax.

```bash
uv sync --locked --extra dev --extra data --extra runtime --extra llm
uv run --no-sync ruff check ml_leakproof tests/corpus scripts
uv run --no-sync mypy --strict ml_leakproof
uv run --no-sync coverage run -m pytest -q
uv run --no-sync coverage report
```

Coverage includes every library module, even unimported modules, with branches
enabled and an 85% minimum. Only packaged example programs are excluded. CI also
builds and smoke-tests both distribution formats, regenerates the synthetic
benchmark for the tested source, and validates the reviewed corpus scopes at
immutable upstream commits. Reports and distributions are saved as workflow
artifacts. The CI workflow never publishes the package. Publishing is available
only through the separate manual release workflow with its explicit `publish`
input and protected PyPI environment.
