# Runnable examples

These examples use the local `ml-leakproof` package and require the runtime/data
extras:

```bash
python -m pip install -e '.[data,runtime]'
```

The example config removes the repository-wide `examples/` exclusion so the
static command can inspect these files directly.

## Static analysis

The intentionally unsafe workflow should report preprocessing leakage:

```bash
python -m ml_leakproof check examples/leaky_workflow.py \
  --config examples/leakproof.toml --select C001 \
  --gate-confidence 0.60 --format json
```

Expect exit code `1` and a `C001` finding explaining why the preprocessing
must be inside the CV pipeline. The clean workflow can be checked with the
same command (without `--select`/`--gate-confidence`) and returns exit code
`0`.

## Runtime analysis

The clean workflow fits its scaler only on the training partition:

```bash
python -m ml_leakproof run examples/clean_workflow.py \
  --config examples/leakproof.toml --format json
```

Expect exit code `0`, `completion: "complete"`, and no findings. The leaky
workflow can be executed the same way and should return exit code `1` with
runtime findings. Script print output is sent to `stderr`, leaving the JSON
report on `stdout` for piping into tools such as `jq`.

## Data analysis

The CSV example contains one row duplicated across the train/test boundary:

```bash
python -m ml_leakproof audit-data \
  --train examples/data/train.csv \
  --test examples/data/test.csv \
  --target target \
  --config examples/leakproof.toml --format json
```

Expect exit code `1` and an exact-overlap `D001` finding. The JSON output also
shows the completion status, coverage, diagnostics, and rule summary.
