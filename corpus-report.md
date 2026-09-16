# Leakproof corpus audit

Companion to `corpus-report.json`, using the immutable commits in `corpus-manifest.json`.

This source-only run analyzed 5 repositories: 0 complete, 5 partial, and 0 failed. All five contain skipped notebook magic/shell code or legacy Python parse failures. Partial scans are excluded from the complete-analysis denominator, so the gateable-repository rate is undefined. No repository was executed.

This operational scan is separate from the ten reviewed source scopes in `corpus-labeled.json`; those labeled checks cover five repositories and pass independently. Full-repository findings are not exhaustively labeled accuracy evidence.

## Environment

pandas 3.0.5, numpy 2.5.3, sklearn 1.9.1, scipy 1.18.1, nbformat 5.11.1.

Profile: `ci`; manifest SHA-256: `ea181a84f43578d05d4ae1921b6f5b96708876c88af0ef5650723d8112bf6d56`.

## Per repository

| Repository | Commit | Completion | Findings | Diagnostics |
|---|---|---|---:|---:|
| scikit-learn-videos | `8545c74961398def7724501648fd504dbf061b41` | partial | 10 | 5 |
| pycon-2016-tutorial | `5339710562cc46c4a32a369d63f1e8751a18a54e` | partial | 0 | 4 |
| intro-to-ml-sklearn | `ea60cf6cf791553b6cca7cf31802c68cb3798ebb` | partial | 55 | 11 |
| ml-with-python | `87cbe5caa5ce6c219ad92bb0c720f06aaf9c8773` | partial | 194 | 94 |
| ds-from-scratch | `d5d0f117f41b3ccab3b07f1ee1fa21cfcf69afa1` | partial | 2 | 23 |

## Findings by rule

| Rule | Count |
|---|---:|
| M004 | 124 |
| R001 | 88 |
| P001 | 12 |
| M003 | 9 |
| C001 | 8 |
| T002 | 6 |
| C006 | 5 |
| S002 | 4 |
| C005 | 2 |
| S001 | 2 |
| P002 | 1 |

Diagnostics and skipped cells are recorded in the JSON report. Advisory findings never fail the severity gate. The corpus command returns exit code `2` for these partial repository scans, while retaining their full report.
