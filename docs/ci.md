# CI integration

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | clean (or only findings below `--fail-on`) |
| 1 | findings at or above `--fail-on` (default `high`) |
| 2 | tool error |

## pre-commit

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/leakproof/leakproof
    rev: v0.1.0
    hooks:
      - id: leakproof
```

The hook runs the fast static layer on staged Python files only.

## GitHub Actions

Minimal workflow that uploads SARIF to GitHub code scanning:

```yaml
name: leakproof
on: [push, pull_request]
permissions:
  contents: read
  security-events: write
jobs:
  leakproof:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: leakproof/leakproof-action@v1
        with:
          paths: "."
          fail-on: high
          layers: static
      - uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: leakproof.sarif
```

Or call the CLI directly:

```yaml
      - run: pip install leakproof
      - run: leakproof check . --format sarif --output leakproof.sarif --fail-on high
```

The bundled `action.yml` wraps exactly this. Inputs: `paths`, `select`, `fail-on`,
`layers`, `output`.
