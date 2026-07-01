# Benchmark & validation

The benchmark suite is small on purpose. It answers two questions: did the rule catch
the leak, and did it stay quiet on the clean version?

## Fixture suite

Each rule has a leaky fixture and a clean fixture under
`tests/fixtures/{leaky,clean}/<RULE_ID>__<slug>.py`.

Regenerate them with:

```bash
python tests/fixtures/_gen.py
```

## Per-rule precision / recall

Computed in CI on the fixture suite:

```bash
python -m tests.corpus.metrics
```

Because these are paired fixtures, precision and recall should stay at 1.00. CI fails
when a rule misses its leaky fixture or fires on its clean fixture. It also tracks
gateable false positives: clean findings that would fail the severity/confidence gate.

| metric | value |
|--------|-------|
| rules with paired fixtures | 30 / 30 |
| macro recall (leaky set) | 1.00 |
| clean-set false-positive rate | 0.00 |
| clean-set gateable false-positive rate | 0.00 |

## Corpus audit

Corpus audits run the same scanner on public repos:

```json
{
  "repos": [
    {
      "name": "example-tabular",
      "url": "https://github.com/owner/repo",
      "ref": "optional pinned commit/tag",
      "domain": "tabular",
      "expected_rule_ids": ["P001"],
      "expected_false_positives": []
    },
    {"name": "example-nlp", "url": "https://github.com/owner/repo2", "domain": "nlp"}
  ]
}
```

Run:

```bash
leakproof corpus --manifest manifest.json --output corpus-report.json
```

The report aggregates findings by rule and domain. Repos that fail to clone are reported
as `clone_failed`.
