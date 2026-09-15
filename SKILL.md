---
name: leakproof
description: Audit ML and data-science code for data leakage and evaluation-rigor risks, including split mistakes, broken CV, preprocessing leakage, test-set reuse, metric misuse, and determinism gaps.
---

# Leakproof usage notes

Leakproof provides deterministic evidence about common ML evaluation risks. Treat
findings as evidence to investigate, preserve diagnostics and coverage, and do
not call a clean scan proof that the methodology is valid.

## When to use

- The user asks to audit ML code for leakage, broken cross-validation, test-set
  reuse, metric misuse, or reproducibility.
- Reported metrics look suspiciously strong.
- A change touches splitting, preprocessing, training, model selection, or evaluation.

## Procedure

1. Locate training/evaluation scripts and notebooks.
2. Run the static layer:

   ```bash
   ml-leakproof check <paths> --format json
   ```

3. If explicit data splits are available, run:

   ```bash
   ml-leakproof audit-data --train train.csv --test test.csv --target y [--group g] [--time t] --format json
   ```

4. If the training script can run safely, use:

   ```bash
   ml-leakproof run train.py --format json
   ```

5. Inspect `findings`, `diagnostics`, `coverage`, and `completion`. Use
   `ml-leakproof explain RULE_ID` for rationale and paired examples.
6. Apply only the requested or explicitly approved changes. The built-in
   `--fix` path is limited to verified deterministic seed edits.
7. Re-run the relevant layer and verify the exit status.

## Result and exit semantics

Result APIs return `AnalysisResult`. The list wrappers `check` and `audit_data`
raise `AnalysisError` on incomplete analysis unless `allow_partial=True` is
explicit. CLI exit codes are `0` clean, `1` gateable finding, and `2` usage,
configuration, incomplete/failed analysis, or a non-zero script exit.

`--min-confidence` changes presentation only; `--gate-confidence` and
`--fail-on` determine gating. Findings marked `advisory_only` never gate.

## Guardrails

- Do not invent findings or dismiss them without project evidence.
- Preserve and report operational diagnostics; a partial result is not complete.
- Similarity, target-predictivity, MI, and imbalance results are hypotheses to
  review, not proof by themselves.
- Notebook analysis follows stored cell order. It does not reconstruct
  out-of-order execution history.
- Runtime automatic hooks are most reliable when split/CV helpers are imported
  inside `watch()`; use explicit runtime registration for unsupported code.
- The optional LLM layer explains deterministic findings and triages parse
  failures only. It cannot create, replace, or delete deterministic findings.

## Suppression

Use a reviewed inline suppression such as:

```python
# leakproof: ignore[P001]
```

Use `# leakproof: ignore-file` only when the whole file is intentionally out of
scope. Unknown or malformed suppression directives remain diagnostics.
