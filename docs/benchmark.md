# Benchmark & validation

The benchmark suite is intentionally small and inspectable. It is meant to answer two
questions during development: did a rule still catch the canonical leak, and did it start
firing on the matching clean example?

## Fixture suite

Every catalog rule has at least one **leaky** fixture that must be flagged and one
**clean** fixture that must stay quiet. They live in
`tests/fixtures/{leaky,clean}/<RULE_ID>__<slug>.py` and double as the worked examples
surfaced by `leakproof explain <RULE_ID>`.

Regenerate them from a single source of truth:

```bash
python tests/fixtures/_gen.py
```

## Per-rule precision / recall

Computed in CI on the fixture suite:

```bash
python -m tests.corpus.metrics
```

Because this suite is made from paired examples, per-rule precision/recall is expected to
be 1.00. Its real job is regression protection: CI fails (`tests/unit/test_benchmark.py`)
if a rule stops flagging its leaky fixture or starts firing on clean fixtures. The
clean-set false-positive gate is intentionally strict.

| metric | value |
|--------|-------|
| rules with paired fixtures | 30 / 30 |
| macro recall (leaky set) | 1.00 |
| clean-set false-positive rate | 0.00 |

## Corpus audit

The fixture suite proves the rules fire on controlled cases. The **corpus audit** measures
how often public ML repositories trip the same rules. Provide a manifest of repositories:

```json
{
  "repos": [
    {"name": "example-tabular", "url": "https://github.com/owner/repo", "domain": "tabular"},
    {"name": "example-nlp", "url": "https://github.com/owner/repo2", "domain": "nlp"}
  ]
}
```

Run:

```bash
leakproof corpus --manifest manifest.json --output corpus-report.json
```

The report aggregates findings by rule and domain and includes a **leakage rate**: the
fraction of analyzed repositories with at least one medium-or-higher finding.

> Methodology notes: the static layer runs with zero configuration; the data layer runs
> only where splits are discoverable. Repos that fail to clone are reported as
> `clone_failed` and excluded from the rate denominator.
