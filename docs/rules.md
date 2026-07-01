# Rule catalog

30 rules. Severity is the default; override per-rule via config.

| ID | Name | Category | Severity | Layers |
|----|------|----------|----------|--------|
| C001 | preprocessing fit outside CV loop | cross_validation | high | static, runtime |
| C002 | plain KFold on grouped data | cross_validation | high | static, data |
| C003 | plain KFold on time series | cross_validation | high | static, data |
| C004 | tuning on the test set | cross_validation | high | static, runtime |
| C005 | model selection without nested CV | cross_validation | medium | static |
| C006 | target/mean encoding without fold isolation | cross_validation | high | static, data |
| D001 | exact train/test row overlap | data_overlap | critical | data |
| D002 | near-duplicate across splits | data_overlap | high | data |
| D003 | group/entity in both splits | data_overlap | critical | data |
| D005 | single feature near-perfectly predicts target | target_leakage | high | data |
| D006 | suspiciously high feature-target MI | target_leakage | medium | data |
| D007 | duplicate rows inflating test | data_overlap | medium | data |
| M001 | accuracy on imbalanced target | metric | medium | data |
| M002 | decision threshold tuned on test | metric | medium | static, runtime |
| M003 | no variance/CI across folds or seeds | metric | low | static |
| M004 | metric computed on training data | metric | medium | static, runtime |
| P001 | fit_transform on full X before split | preprocessing | high | static, runtime |
| P002 | fit on concat([train, test]) | preprocessing | high | static, runtime |
| P003 | imputation statistics from full data | preprocessing | high | static, runtime |
| R001 | missing random_state/seed | determinism | low | static |
| R002 | nondeterministic framework ops | determinism | low | static |
| S001 | transformer fit on full data before split | split | high | static, runtime |
| S002 | train_test_split(shuffle=True) on temporal data | temporal | medium | static, data |
| S003 | resampling/augmentation before split | split | high | static, runtime |
| S004 | feature selection on full data before split | split | high | static, runtime |
| T001 | test set evaluated multiple times | adaptivity | medium | runtime |
| T002 | early stopping / selection on test | adaptivity | high | static, runtime |
| T003 | best-of-N seeds reported | adaptivity | medium | static |
| TM001 | train/test temporal overlap | temporal | high | data |
| TM002 | look-ahead feature across split | temporal | high | static |

## Rationale

### C001 — preprocessing fit outside CV loop

Fitting a transformer once on all data and reusing it across CV folds leaks validation-fold statistics into training. Put the transformer in a Pipeline passed to cross_val_score/GridSearchCV so it is refit per fold.

References: https://scikit-learn.org/stable/common_pitfalls.html, https://scikit-learn.org/stable/modules/cross_validation.html

### C002 — plain KFold on grouped data

When the same entity (patient/user/session) has multiple rows, a plain KFold or train_test_split can place rows from one entity on both sides of the split. Use GroupKFold/StratifiedGroupKFold.

References: https://scikit-learn.org/stable/modules/cross_validation.html

### C003 — plain KFold on time series

Random KFold on time-ordered data trains on the future to predict the past. Use TimeSeriesSplit.

References: https://scikit-learn.org/stable/modules/cross_validation.html

### C004 — tuning on the test set

Fitting a hyperparameter search directly on the held-out test split selects hyperparameters using the data you report results on.

References: https://doi.org/10.1016/j.patter.2023.100804, https://scikit-learn.org/stable/common_pitfalls.html

### C005 — model selection without nested CV

Reporting a hyperparameter search's best_score_ as model performance reuses the selection folds for evaluation, biasing the estimate upward. Use nested CV.

References: https://doi.org/10.1016/j.patter.2023.100804, https://scikit-learn.org/stable/modules/cross_validation.html

### C006 — target/mean encoding without fold isolation

Computing target/mean/count encodings over data spanning the evaluation split lets each row's encoding depend on its own (and the held-out rows') target. Compute encodings within CV folds (out-of-fold).

References: https://doi.org/10.1016/j.patter.2023.100804, https://scikit-learn.org/stable/common_pitfalls.html

### D001 — exact train/test row overlap

Identical rows in train and test mean the model is evaluated on examples it trained on; reported scores are invalid.

References: https://doi.org/10.1145/2382577.2382579

### D002 — near-duplicate across splits

Near-duplicate rows across the split (paraphrased text, augmented images, near-identical feature vectors) leak almost the same information as exact duplicates.

References: https://doi.org/10.1145/2382577.2382579

### D003 — group/entity in both splits

When the same entity (patient/user/session) appears in both train and test, the model can memorize entity-specific signal. Split by group.

References: https://scikit-learn.org/stable/common_pitfalls.html

### D005 — single feature near-perfectly predicts target

A lone feature that predicts the target almost perfectly is usually a proxy/leak (an id, a post-outcome field, or a transformed copy of the label).

References: https://doi.org/10.1145/2382577.2382579, https://doi.org/10.1016/j.patter.2023.100804

### D006 — suspiciously high feature-target MI

A feature whose mutual information with the target is far above the rest of the cohort may be a leaked proxy. Advisory; confirm against the data dictionary.

References: https://doi.org/10.1145/2382577.2382579

### D007 — duplicate rows inflating test

Internal duplicate rows in the test set overstate the effective sample size and let a few patterns dominate the score.

References: https://doi.org/10.1145/2382577.2382579

### M001 — accuracy on imbalanced target

On a heavily imbalanced target, accuracy is dominated by the majority class and can look high while the model is useless. Prefer balanced accuracy / F1 / AUC / PR-AUC.

References: https://scikit-learn.org/stable/common_pitfalls.html

### M002 — decision threshold tuned on test

Selecting a classification threshold from a curve computed on the test labels tunes the decision rule on the data you report on. Pick the threshold on a validation split.

References: https://doi.org/10.1016/j.patter.2023.100804

### M003 — no variance/CI across folds or seeds

A single point estimate hides run-to-run variance. Report spread (std / CI) across CV folds or seeds.

References: https://doi.org/10.1016/j.patter.2023.100804

### M004 — metric computed on training data

Reporting a score on the training data measures memorization, not generalization. Evaluate on the held-out split.

References: https://scikit-learn.org/stable/common_pitfalls.html

### P001 — fit_transform on full X before split

Calling fit_transform on the entire feature matrix before splitting lets statistics from the test rows (means, variances, components) leak into the transform applied to training, inflating measured performance.

References: https://scikit-learn.org/stable/common_pitfalls.html, https://doi.org/10.1145/2382577.2382579

### P002 — fit on concat([train, test])

Fitting a transformer on data concatenated from both train and test splits directly exposes the model to test-set distribution information.

References: https://scikit-learn.org/stable/common_pitfalls.html, https://doi.org/10.1145/2382577.2382579

### P003 — imputation statistics from full data

Imputation parameters (mean/median/mode/neighbors) learned across the split encode test-set values into the filled-in training data.

References: https://scikit-learn.org/stable/common_pitfalls.html, https://doi.org/10.1145/2382577.2382579

### R001 — missing random_state/seed

Without a fixed seed, splits and stochastic estimators vary run to run, making results irreproducible (and inviting seed cherry-picking).

### R002 — nondeterministic framework ops

Torch/cuDNN operations are nondeterministic by default. Set a seed and torch.use_deterministic_algorithms(True) for reproducible training.

### S001 — transformer fit on full data before split

A scaler/encoder/PCA fit on data that predates (or crosses) the split learns parameters from the held-out rows. Fit on the training split only, ideally inside a Pipeline passed to the cross-validation utility.

References: https://scikit-learn.org/stable/common_pitfalls.html, https://doi.org/10.1145/2382577.2382579

### S002 — train_test_split(shuffle=True) on temporal data

Shuffling before splitting time-ordered data lets the model train on the future and evaluate on the past. Use shuffle=False or TimeSeriesSplit when a datetime column/index is present.

References: https://scikit-learn.org/stable/modules/cross_validation.html

### S003 — resampling/augmentation before split

SMOTE/over/under-sampling applied to the full dataset before splitting places synthetic neighbours of test rows into the training set (and vice versa).

References: https://scikit-learn.org/stable/common_pitfalls.html, https://doi.org/10.1145/2382577.2382579

### S004 — feature selection on full data before split

SelectKBest/RFE/VarianceThreshold fit on the full feature matrix chooses features using the test labels/values, a classic source of optimistic bias.

References: https://scikit-learn.org/stable/common_pitfalls.html, https://doi.org/10.1145/2382577.2382579

### T001 — test set evaluated multiple times

Scoring against the same held-out split repeatedly within a run is adaptive data analysis: each peek erodes the validity of the held-out estimate. Touch the test set once.

References: https://doi.org/10.1126/science.aaa9375

### T002 — early stopping / selection on test

Using the test split as an eval_set for early stopping (or to pick the best epoch) selects the model against the data you report on. Use a separate validation split.

References: https://doi.org/10.1016/j.patter.2023.100804

### T003 — best-of-N seeds reported

Looping over seeds/configs and keeping only the maximum score reports a biased best-case rather than the distribution. Report mean ± std across runs.

References: https://doi.org/10.1126/science.aaa9375, https://doi.org/10.1016/j.patter.2023.100804

### TM001 — train/test temporal overlap

If the latest training timestamp is at or after the earliest test timestamp, the model trains on data from the test period — a temporal leak.

References: https://doi.org/10.1016/j.patter.2023.100804

### TM002 — look-ahead feature across split

A negative shift pulls future values into the present row, and a rolling/expanding window computed before splitting can summarize rows that belong to the test period. Compute time features within each split's window only.

References: https://doi.org/10.1016/j.patter.2023.100804
