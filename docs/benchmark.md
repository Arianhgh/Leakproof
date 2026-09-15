# Benchmark and validation

The repository contains paired, labeled fixtures for the built-in catalog. They
are a regression suite for rule behavior, not a claim that a finite corpus proves
absence of leakage in arbitrary projects.

Run the fixture checks and metrics locally:

```bash
pytest -q
python -m tests.corpus.metrics
python -m tests.corpus.metrics --json > benchmark-results.json
```

Each leaky fixture is evaluated against its named rule for recall. Each clean
fixture is evaluated against every catalog rule, so an unrelated false positive
is counted. Precision and recall are reported as `—` when their denominator is
zero. Gateable false positives are counted separately from advisory findings.
The committed `benchmark-results.json` records the package version, source
revision, configuration, dependency versions, fixture counts, and pinned corpus
SHAs alongside the rule metrics.

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
the analyzed denominator and remain visible in the report. The current pinned
run has three complete repositories and two partial repositories, so its
rate is `1/3`, not `2/5`.

The benchmark worktree is managed under `.leakproof-corpus` and is excluded from
package artifacts. Existing checkouts must have the expected remote and no dirty
changes before they are fetched or checked out.

## Interpreting results

Exact overlap findings are observations under the supplied split policy. Near
duplicates, target predictivity, mutual information, and imbalance are advisory
hypotheses. Similarity and probe work is bounded; sampling, unavailable optional
backends, and parser failures are reported in coverage/diagnostics rather than
being presented as complete evidence.
