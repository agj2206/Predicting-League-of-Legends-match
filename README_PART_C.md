# Part C — Model Evaluation and Interpretation

**Research Question:** Can we predict the winning team in a League of Legends match using only statistics available after the first 10 minutes?

## Completed final evaluation

The authorized final workflow completed successfully on the original **1,976
held-out Diamond matches**. The original dataset contains 9,879 matches;
7,903 were used for training. Blue (`blueWins=1`) is the positive class.
The SVM selected through nested cross-validation in Part B remains the main
model. No model was refitted, and no pipeline, feature, hyperparameter,
classification threshold, or family-selection rule was changed.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy (baseline) | 0.5010 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Decision Tree | 0.7277 | 0.7398 | 0.7008 | 0.7198 | 0.7994 |
| KNN | 0.7262 | 0.7263 | 0.7241 | 0.7252 | 0.8081 |
| SVM | 0.7358 | 0.7427 | 0.7201 | 0.7312 | 0.8149 |

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Dummy (baseline) | 990 | 0 | 986 | 0 |
| Decision Tree | 747 | 243 | 295 | 691 |
| KNN | 721 | 269 | 272 | 714 |
| SVM | 744 | 246 | 276 | 710 |

Rows of each confusion matrix are actual classes; columns are predicted
classes, both ordered `[Red (0), Blue (1)]`. Thus the matrix is
`[[TN, FP], [FN, TP]]`. Each matrix sums to 1,976. Metric tables use four
decimal places; JSON and CSV retain full precision.

**Answer:** Yes, within this Diamond sample, ten-minute statistics support
above-baseline prediction. The preselected linear SVM achieves **73.58% test
accuracy** and **0.8149 ROC-AUC**, compared with **50.10% accuracy** for Dummy:
a gain of **23.48 percentage points**. It correctly predicts 1,454 winners and
makes 522 errors. This supports predictive value within the evaluated sample,
not causal effects or proven transfer to other ranks, regions, or future patches.

## Reproduce displays from saved results

Use Python 3.11 and install `requirements-part-b.txt` as described in the
[project README](README.md#installation). From the repository root in a
disposable copy:

```bash
# Refresh figures and findings from saved aggregate results.
.venv/bin/python -B scripts/evaluate_models.py --render-saved-results

# Refresh the executed notebook from those aggregates and figures.
.venv/bin/python -B scripts/execute_evaluation_notebook.py
```

The final evaluation is already complete. The saved status and the exclusive
`evaluation_started.json` reservation guard the original held-out run.

### Tests and historical audit

The evaluation tests use artificial observations and frozen pipelines. The
preflight test additionally requires the original source CSV and the local
Part A split and Part B CV manifests. It also compares files to historical
Part A hashes. The final integration snapshot already differed from that
baseline for `README.md`, `.gitignore`, `requirements.txt` and the Part A
notebook. The [checksum investigation](docs/publication_review.md#historical-preflight) explains the provenance difference. For a fresh integration checkout, run the 30 evaluation tests that
exercise synthetic data, artifacts and the saved-result workflow without that
historical preflight, for example:

```bash
PYTHONPATH=tests .venv/bin/python -B -m unittest -v \
  test_model_evaluation.MetricTests \
  test_model_evaluation.ContractTests \
  test_model_evaluation.WorkflowTests \
  test_model_evaluation.FrozenArtifactTests.test_saved_models_accept_synthetic_schema_without_fitting_or_mutation \
  test_model_evaluation.FrozenArtifactTests.test_saved_parameters_and_feature_order \
  test_model_evaluation.FrozenArtifactTests.test_real_svm_algebra_on_synthetic_points \
  test_model_evaluation.FrozenArtifactTests.test_synthetic_parameters_have_hand_calculated_primal_weights \
  test_model_evaluation.FrozenArtifactTests.test_interpretation_rejects_wrong_kernel_schema_or_scale
```

The historical [prediction-cache verifier](scripts/verify_evaluation_results.py)
checks metrics against the original private `test_predictions.csv`, the local
split manifest, the dataset checksum, and all files in the [protected hash
manifest](reports/model_evaluation/diamond_5x3_seed42/protected_file_hashes.json).
Several of those files are local-only. The verifier writes
`quality_assurance.json`; run it only in a copy of the original evaluation
workspace with its original protected files and private inputs. The [saved
audit](reports/model_evaluation/diamond_5x3_seed42/quality_assurance.json)
documents the checks performed at the time of evaluation.

## Frozen methodology

The workflow reconstructs the unchanged Part A split using ascending `gameId`
ordering, `test_size=0.20`, stratification, and `random_state=42`. It verifies
exact original split membership and the training-data fingerprint. The
original 12 features enter the saved pipelines **unscaled**. StandardScaler
inside KNN/SVM was already fitted on training records; it is never refitted.

Accuracy, precision, recall, and F1 use the existing `predict()` labels.
ROC-AUC uses SVM `decision_function()` and the column explicitly associated
with class `1` in `predict_proba()` for Dummy, Decision Tree, and KNN.
Precision, recall, and F1 are binary metrics with `pos_label=1` and
`zero_division=0`. AUC requires both real classes and finite scores.
ROC thresholds are descriptive and were not optimized.

The five outer-fold CV means and sample SD from Part B provide descriptive
context. Final inner-search scores are not substituted for outer CV results.
SVM outer-CV accuracy is 0.7295 (sample SD 0.0167), compared with 0.7358 on
the independent test: approximately +0.64 percentage points. SD is not a
confidence interval, and this difference is not a significance test.

## Numerically verified SVM interpretation

The SVM is linear (`C=10`, `probability=False`). Signed dual coefficients
and support vectors are combined coordinate by coordinate using `math.fsum`.
This avoids the local `matmul` path that emits warnings when `SVC.coef_` is
read. The warnings are retained in the preliminary audit, not globally hidden.

Coefficient finiteness and agreement with the public property, NumPy `dot`,
and NumPy `matmul` were checked before held-out evaluation. On **539 artificial
points**, reconstructed decision scores agree with the saved pipeline's
`decision_function` within a maximum absolute error of approximately
**2.54e-11**, in both standardized and original coordinates. The fixed
tolerances are coefficient `atol=1e-8`, score `atol=1e-7`, and `rtol=1e-8`.
The low-level library or hardware cause of the warning remains unestablished.

With training means `mu`, training scales `s`, and standardized weights `w`:

```text
f(x) = b + sum_j w_j * (x_j - mu_j) / s_j
original_unit_weight_j = w_j / s_j
original_unit_intercept = b - sum_j w_j * mu_j / s_j
```

The largest absolute standardized coefficients are `blueGoldDiff` (+1.1238),
`blueExperienceDiff` (+0.4151), and `dragonsDiff` (+0.2432). Increasing these
features increases the Blue decision score, holding other predictors fixed.
Negative coefficients for correlated combat, farming, or tower differences
do not imply that obtaining those resources causes a team to lose. These
are conditional associations in the fitted model. Decision scores are not
calibrated probabilities or odds ratios.

## Files and evidence

| Path | Purpose |
|---|---|
| `src/lol_model_evaluation.py` | Frozen evaluation, metrics, reliable coefficients, integrity checks, and saved-result rendering |
| `scripts/evaluate_models.py` | Preliminary workflow, explicit final option, and saved-result rendering |
| `scripts/execute_evaluation_notebook.py` | Executes Part C without model inference |
| `scripts/verify_evaluation_results.py` | Independent audit of cached predictions and final displays |
| `notebooks/03_model_evaluation.ipynb` | Executed methods, metrics, figures, interpretation, limitations, and conclusion |
| `tests/test_model_evaluation.py` | Synthetic metrics, edge cases, schema, frozen parameters, and no-fitting tests |
| `reports/part_c_findings.md` | Final results and evidence-based answer to the Research Question |
| `reports/model_evaluation/diamond_5x3_seed42/` | Aggregate metrics, numerical audit, CV comparison, metadata, and verification |
| `reports/figures/part_c/diamond_5x3_seed42/` | Four final figures |
| `data/processed/part_c/diamond_5x3_seed42/test_predictions.csv` | Local ignored cache of individual predictions and scores |

The aggregate directory includes `test_metrics.json/.csv`,
`confusion_matrices.json/.csv`, `roc_curves.json`,
`cv_test_comparison.json/.csv`, `svm_coefficients.json/.csv`,
`preflight.json`, `test_verification.json`, `evaluation_metadata.json`,
`evaluation_started.json`, `verification.json`, and final quality-assurance
evidence. Strict JSON represents the initial infinite ROC threshold as `null`,
with an explanatory field. Machine-readable model keys retain the original
`dummy`, `decision_tree`, `knn`, and `svm` identifiers.

Figures are `confusion_matrices.png`, `roc_curves.png`,
`cv_test_accuracy.png`, and `svm_standardized_coefficients.png`.
Matplotlib's noninteractive Agg renderer is used. After evaluation, changes
to rendering labels and explanatory text use only saved evidence and do not
alter the evaluated prediction or metric functions. Original evaluation
provenance remains in `preflight.json`.

The raw CSV and prediction cache belong in ignored local paths. The notebook
and evaluation tables report aggregate evidence. The [project README](README.md)
provides the integrated results and publication status.

## Limitations

- Player IDs and patch/date/region metadata are unavailable, so repeated-player
  independence, temporal robustness, and population transfer cannot be established.
- Correlated predictors limit isolated coefficient interpretation; this is
  observational prediction rather than causal inference.
- Suspicious ward counts remain unchanged. Relative differences omit absolute
  game tempo and some tied objective states.
- Model-family selection introduces optimism in the selected CV score. A single
  holdout has sampling uncertainty; five-fold SD is not a confidence interval.
- KNN's `k=81` and SVM's `C=10` are grid-boundary choices. The test does not justify
  more tuning, new features, new thresholds, or a different selected family.
- SVM scores are uncalibrated. Small differences between learned models do not
  establish statistically significant superiority.
