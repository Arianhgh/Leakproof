# leakproof corpus audit

Static-only scan of **5/5** public ML repos. **Leakage rate: 80%** (repos with ≥1 MEDIUM+ finding).

## Findings by rule

| Rule | Count |
|------|------|
| R001 | 200 |
| M004 | 113 |
| S002 | 81 |
| C002 | 43 |
| LP000 | 30 |
| M002 | 23 |
| P001 | 14 |
| C003 | 14 |
| C001 | 12 |
| M003 | 11 |
| C005 | 6 |
| C006 | 5 |
| T002 | 4 |
| P002 | 2 |
| S001 | 2 |

## Per repo

| Repo | Domain | Findings | Leakage? |
|------|--------|----------|----------|
| scikit-learn-videos | tabular | 18 | yes |
| pycon-2016-tutorial | nlp | 10 | yes |
| intro-to-ml-sklearn | tabular | 198 | yes |
| ml-with-python | mixed | 308 | yes |
| ds-from-scratch | tabular | 26 | no |
