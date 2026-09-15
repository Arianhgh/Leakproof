# Leakproof corpus audit

This is the Markdown companion to [corpus-report.json](corpus-report.json),
generated with the immutable pins in [corpus-manifest.json](corpus-manifest.json)
using report schema `2.0`.

Three repositories completed analysis, two repositories were partial, and none
failed to clone or failed at the analysis level. The observed gateable-finding
rate was **33.3% (1/3)** under the CI severity/confidence gate. Partial scans
remain visible below but are excluded from this denominator. The manifest
contains no expected labels, so this run is an operational corpus statistic
rather than a labeled precision/recall benchmark.

Environment: pandas 3.0.3, NumPy 2.4.4, scikit-learn 1.9.0, SciPy 1.17.1,
nbformat 5.10.4. Configuration: profile `ci`, `fail_on = high`, gate confidence
`0.75`. Manifest SHA-256:
`ea181a84f43578d05d4ae1921b6f5b96708876c88af0ef5650723d8112bf6d56`.

## Findings by rule

| Rule | Count |
|---|---:|
| M004 | 113 |
| R001 | 84 |
| S002 | 44 |
| C002 | 31 |
| C001 | 21 |
| M002 | 18 |
| P001 | 11 |
| M003 | 9 |
| C005 | 6 |
| T002 | 6 |
| C006 | 5 |
| C003 | 5 |
| S001 | 3 |
| S004 | 2 |
| P002 | 1 |

## Per repository

| Repository | Domain | Resolved commit | Completion | Findings | Gateable? |
|---|---|---|---|---:|---|
| scikit-learn-videos | tabular | `8545c74961398def7724501648fd504dbf061b41` | complete | 11 | no |
| pycon-2016-tutorial | nlp | `5339710562cc46c4a32a369d63f1e8751a18a54e` | complete | 7 | no |
| intro-to-ml-sklearn | tabular | `ea60cf6cf791553b6cca7cf31802c68cb3798ebb` | complete | 97 | yes |
| ml-with-python | mixed | `87cbe5caa5ce6c219ad92bb0c720f06aaf9c8773` | partial | 240 | n/a |
| ds-from-scratch | tabular | `d5d0f117f41b3ccab3b07f1ee1fa21cfcf69afa1` | partial | 2 | n/a |

Similarity, target-predictivity, MI, and other advisory probes are not included
in the gateable rate. Partial repository analysis remains visible in the JSON
report and is not equivalent to a complete clean result. The machine-readable
report records `n_analyzed = 3`, `n_partial = 2`, and `n_failed = 0`.
