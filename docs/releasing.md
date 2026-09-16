# Local release validation

This is the unpublished `0.2.0rc1` candidate. The repository never publishes on
push: `.github/workflows/release.yml` is manual, defaults to validation-only,
and uploads to PyPI only when its protected `publish` input is explicitly enabled.
The commands below build and validate local artifacts; they do not create a tag,
release, or PyPI upload.

The latest local results are recorded in [`release-validation.json`](../release-validation.json),
including interpreter versions, coverage, source fingerprints, and artifact hashes.

## Reproduce validation

Use Python 3.10–3.14 and uv 0.12.15. The committed lock resolves dependencies for
each supported interpreter and platform.

```bash
uv sync --locked --extra dev --extra data --extra runtime --extra llm
uv run --no-sync ruff check ml_leakproof tests/corpus scripts
uv run --no-sync mypy --strict ml_leakproof
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --no-sync coverage run -m pytest -q
uv run --no-sync coverage report
uv run --no-sync python -m tests.corpus.labeled
uv run --no-sync python -m tests.corpus.metrics --json > benchmark-results.json
uv run --no-sync python -m tests.corpus.metrics --verify benchmark-results.json
```

CI runs the checks for every supported Python version. Coverage includes the
whole library with branch measurement enabled, including optional modules that
are not imported by a particular test. The minimum is 85%; the packaged teaching
examples are the only excluded source directory.

The reviewed corpus runner requires Git/network access on the first run.
`--offline` reuses clean checkouts at their pinned commits. The broader corpus
scanner can legitimately return partial results for legacy Python or notebook
magics. Those gaps are reported and excluded from clean-analysis rates.

## Build and test the actual distributions

```bash
uv run --no-sync python -m build
uv run --no-sync twine check --strict dist/*.whl dist/*.tar.gz
uv run --no-sync python scripts/smoke_distribution.py dist/ml_leakproof-0.2.0rc1-py3-none-any.whl
uv run --no-sync python scripts/smoke_distribution.py dist/ml_leakproof-0.2.0rc1.tar.gz
```

Each smoke check creates clean temporary environments outside the repository,
installs the artifact, runs `pip check`, and checks the public API and CLI.
The base installation verifies that optional scientific/LLM libraries are absent,
then installs the example plugin and checks actual rule/adapter entry-point discovery.
The extras installation exercises runtime detection, CSV/Parquet data auditing,
and an actual autofix. Packaged rule examples must be available via `explain`.

Do not reuse an old benchmark after editing release inputs. Regenerate it and
verify the SHA-256 content manifest. Git revision and dirty status are also
recorded; the content fingerprint is the authoritative identity for uncommitted
release preparation. CI generates fresh evidence rather than relying on a
previously committed benchmark.

## Scope of the evidence

Passing these checks establishes tested functionality and installation behavior.
The synthetic benchmark is a regression suite; the ten labeled real-world
source scopes are still a small sample. Static analysis does not reconstruct
arbitrary execution, notebook execution history, or external modules. Runtime
hooks apply to supported operations in the current process. Unavailable or
unsupported analysis remains explicit in diagnostics and completion status.
Optional LLM providers are tested with injected clients without making network
requests or requiring credentials.

Public GitHub CI must run on the updated revision after it is pushed. Local
validation cannot change the status of an older public run. Publishing remains
a separate, explicit action.
