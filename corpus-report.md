# Leakproof corpus audit

Companion to `corpus-report.json`, using the immutable commits in `corpus-manifest.json`.

This source-only run inspected 7 repositories: 2 complete, 5 partial, and 0 failed. The five partial repositories contain skipped notebook magic/shell code or legacy Python parse failures. Partial scans are excluded from the complete-analysis denominator. Neither complete repository crossed the configured high-severity gate, so the gateable-repository rate is 0%. No repository was executed.

This operational scan is separate from the 21 reviewed source scopes in `corpus-labeled.json`; those labeled checks cover seven repositories and pass independently. Full-repository findings are not exhaustively labeled accuracy evidence, and the reviewed scopes remain too deliberately selected and too small for broad precision/recall claims.

## Environment

pandas 3.0.5, numpy 2.5.3, sklearn 1.9.1, scipy 1.18.1, nbformat 5.11.1.

Profile: `ci`; manifest SHA-256: `3da99fe65ca6e52d531977df20c80f77b18fd09ffa7e41c1879be120a1efb642`.

## Per repository

| Repository | Commit | Completion | Findings | Diagnostics |
|---|---|---|---:|---:|
| scikit-learn-videos | `8545c74961398def7724501648fd504dbf061b41` | partial | 10 | 5 |
| pycon-2016-tutorial | `5339710562cc46c4a32a369d63f1e8751a18a54e` | partial | 0 | 4 |
| intro-to-ml-sklearn | `ea60cf6cf791553b6cca7cf31802c68cb3798ebb` | partial | 28 | 11 |
| ml-with-python | `87cbe5caa5ce6c219ad92bb0c720f06aaf9c8773` | partial | 98 | 94 |
| ds-from-scratch | `d5d0f117f41b3ccab3b07f1ee1fa21cfcf69afa1` | partial | 2 | 23 |
| category-encoders | `fafdccced2c5154478a5f10cd761e138daa56d21` | complete | 0 | 0 |
| pytorch-examples | `acc295dc7b90714f1bf47f06004fc19a7fe235c4` | complete | 76 | 0 |

## Findings by rule

| Rule | Count |
|---|---:|
| R001 | 88 |
| R002 | 76 |
| P001 | 12 |
| M003 | 9 |
| C001 | 8 |
| T002 | 6 |
| C006 | 5 |
| S002 | 4 |
| C005 | 2 |
| S001 | 2 |
| M004 | 1 |
| P002 | 1 |

Diagnostics and skipped cells are recorded in the JSON report. Advisory findings never fail the severity gate. The corpus command returns exit code `2` for these partial repository scans, while retaining their full report.
