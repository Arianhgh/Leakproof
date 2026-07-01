# Benchmark & validation

The benchmark suite is small on purpose. It answers two questions: did the rule catch
the leak, and did it stay quiet on the clean version?

## Per-rule precision / recall

Computed on the labeled validation corpus. The suite keeps paired leaky and clean cases
for each rule, then tracks recall, clean false positives, and gateable false positives:
clean findings that would fail the severity/confidence gate.

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
