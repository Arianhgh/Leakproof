# Benchmark and validation

The repository contains paired, labeled fixtures for the built-in catalog. They
are a regression suite for rule behavior, not a claim that a finite corpus proves
absence of leakage in arbitrary projects.

Run the fixture checks and metrics locally:

```bash
pytest -q
python -m tests.corpus.metrics
python -m tests.corpus.metrics --json > benchmark-results.json
python -m tests.corpus.metrics --verify benchmark-results.json
```

Each leaky fixture is evaluated against its named rule for recall. Each clean
fixture is evaluated against every catalog rule, so an unrelated false positive
is counted. Precision and recall are reported as `—` when their denominator is
zero. Gateable false positives are counted separately from advisory findings.
The committed `benchmark-results.json` records the package version, source
revision, configuration, dependency versions, fixture counts, and pinned corpus
SHAs alongside the rule metrics. It also includes a SHA-256 manifest of library
code, tests, configuration, dependency lock, and release documentation. The
`--verify` command rejects changed, added, or removed release inputs. Generated
reports are excluded to avoid self-referential hashes. `source_dirty` remains
truthful when validating uncommitted edits; the content manifest identifies the
exact tested files independently of the Git commit. CI generates a fresh report
for every tested revision instead of trusting a previously committed result.

## Corpus benchmark

`corpus-manifest.json` contains public repositories pinned to full 40-character
commit SHAs. The corpus command performs a source-only static scan; it never
executes repository code:

```bash
ml-leakproof corpus \
  --manifest corpus-manifest.json \
  --workdir .leakproof-corpus \
  --output corpus-report.json
```

The report schema is `2.0` and records the manifest digest, resolved commit,
remote URL, environment, configuration, per-repository completion and failure
diagnostics, findings by rule/domain, and `flagged_repository_rate`. A flagged
repository rate is an operational corpus statistic, not a detection accuracy
metric. Clone failures, failed analyses, and partial scans are excluded from
the analyzed denominator and remain visible in the report. Notebook magic/shell code and legacy Python parse failures make some repositories
partial. Such repositories remain visible but do not count as clean negative
evidence. The corpus CLI writes the report and returns exit code `2` whenever
any repository failed or was only partially analyzed.

The benchmark worktree is managed under `.leakproof-corpus` and is excluded from
package artifacts. Existing checkouts must have the expected remote and no dirty
changes before they are fetched or checked out.

## Interpreting results

Exact overlap findings are observations under the supplied split policy. Near
duplicates, target predictivity, mutual information, and imbalance are advisory
hypotheses. Similarity and probe work is bounded; sampling, unavailable optional
backends, and parser failures are reported in coverage/diagnostics rather than
being presented as complete evidence.


## Reviewed real-world regression cases

`corpus-labeled.json` contains ten reviewed source scopes across all five pinned
repositories. Each case records its path, zero-based notebook cells (or Python
file), selected rules, positive/negative expectation, and reasoning. The runner
extracts original source text in stored order and hashes the exact excerpt; it
never executes repository code. Labels apply only to the selected rules and
scope. They do not label every finding in a repository or establish general
precision/recall.

```bash
python -m tests.corpus.labeled --output corpus-labeled-results.json
# Reuse clean cached checkouts at their pinned commits without network access:
python -m tests.corpus.labeled --offline
```

The cases include global feature selection before CV, preprocessing before a
holdout split, pipeline-based clean CV, training-only text vocabularies, mean-only
CV reporting, search-score reuse, and fixed-threshold plots. Missing expected
findings, unexpected selected findings, checkout failures, and incomplete case
analysis all fail this regression gate.

The old `M002` label for `09_classification_metrics.ipynb` was incorrect under the
current rule contract: the notebook plots an ROC curve and compares fixed
thresholds 0.5 and 0.3. It does not choose a threshold by optimizing test-curve
metrics. The reviewed scope is now an explicit `M002` negative. This correction
is not a claim that repeated inspection of test metrics is sound methodology.
