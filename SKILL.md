---
name: leakproof
description: Audit ML / data-science code for data leakage and evaluation-rigor bugs (broken CV, test-set reuse, metric misuse, preprocessing leakage). Use whenever a user asks to review, audit, or sanity-check ML training/evaluation code for methodology problems, or before trusting reported model metrics.
---

# leakproof usage notes

`leakproof` is a deterministic, multi-layer leakage/evaluation checker. Treat its
findings as the authoritative scan output, then add project-specific explanation and
fixes on top. Do not assert leakage that the tool did not flag.

## When to use

- The user asks to audit/review ML code for leakage, broken cross-validation,
  test-set reuse, metric misuse, or reproducibility.
- The user reports "suspiciously good" metrics and wants to know if they're trustworthy.
- Before merging a PR that changes training/evaluation/splitting code.

## Procedure

1. **Locate ML files.** Find training/evaluation scripts and notebooks (look for
   `train_test_split`, `cross_val_score`, `fit`, `.score`, `GridSearchCV`, etc.).
2. **Run the static layer:**
   ```bash
   leakproof check <paths> --format json
   ```
   Parse the JSON `findings` array. Each finding has `rule_id`, `severity`, `layer`,
   `message`, `location`, `fix`, `confidence`, and `evidence`.
3. **If datasets/splits are available**, run the data layer for overlap/target leakage:
   ```bash
   leakproof audit-data --train train.csv --test test.csv --target y [--group g] [--time t] --format json
   ```
4. **If you can run the training script**, run the runtime layer to catch fits that
   touch held-out rows:
   ```bash
   leakproof run train.py --format json
   ```
5. **For each finding**, explain it in the codebase's own terms and propose a concrete
   patch. Use `leakproof explain <RULE_ID>` for the rationale and a worked leaky/clean
   example.
6. **Apply fixes** only with user approval: `leakproof check <paths> --fix` for the
   autofixable subset, such as adding `random_state=`. Make other edits manually from the
   suggested fix.
7. **Re-run** to confirm zero findings at or above the gate:
   ```bash
   leakproof check <paths> --fail-on high
   ```
   Exit code `0` = clean (or only below gate), `1` = findings at/above gate, `2` = error.

## Guardrails

- **Deterministic findings are authoritative.** Do not invent leakage the tool did not
  report, and do not dismiss a reported finding without evidence. You may add explanation,
  prioritize, and propose fixes.
- **Confidence matters.** Findings carry a `confidence` (static heuristics may be < 1).
  Lead with `critical`/`high` findings; treat low-confidence `medium` findings as "worth
  a look," not certainties.
- **Notebooks** are analyzed as concatenated cells; out-of-order execution is not modeled
  (findings note this).
- **The LLM layer is explanation-only** (`--explain`); it never creates findings.

## Tips

- `leakproof rules` lists every rule with id/category/severity/layers.
- Narrow scope with `--select`/`--ignore` (rule globs, e.g. `--select "P*" "C*"`).
- Suppress a confirmed false positive inline with `# leakproof: ignore[RULE_ID]`.
