# Part B — Model Training (CDS3004)

**Research Question:** Can we predict the winning team in a League of Legends match
using only statistics available after the first 10 minutes?

This stage uses only the original Diamond CSV and reuses the preprocessing
implemented in Part A without changing its code, features, data split, or
exploratory data analysis (EDA). The target is `blueWins`: `1` denotes a Blue
win and `0` a Red win. The 12 predictors represent Blue − Red differences
measured at the 10-minute mark.

## Local reproduction

Use Python 3.11 and the existing `.venv` environment. Additional dependencies
for Part B are pinned in `requirements-part-b.txt`, which includes the original
`requirements.txt`. There is no need to rerun the Part A notebook.

When setting up an environment on another computer, install the dependencies
with `python -m pip install -r requirements-part-b.txt` inside a virtual environment.
The training workflow does not download data or install packages.

Each group member must have a local copy of the same
`high_diamond_ranked_10min.csv` file, retaining its original name. Place it in
`data/raw/`, use the `Local_Datasets` directory alongside the repository, or pass
`--data-path /path/to/high_diamond_ranked_10min.csv`.
The Part A interface also supports `LOL_DATA_PATH`.
The expected checksum is
`82b2e23279097c2dac616e966eb2911711c428d41b6382b4cdb15da4892f9ca6`.
Do not change the checksum or the data split to accommodate a different dataset.

Run the following commands from the repository root:

```bash
# 1. Run the quick Part B tests on a subset of the Diamond training set.
.venv/bin/python -B -m unittest discover -s tests -p test_model_training.py -v

# 2. Run the complete training workflow: 5 outer folds, 3 inner folds; training data only.
.venv/bin/python -B scripts/train_models.py --n-jobs 2

# 3. Execute the notebook to display results, explanations, and figures without retraining.
.venv/bin/python -B scripts/execute_training_notebook.py

# 4. Run the complete test suite, including the original Part A tests.
.venv/bin/python -B -m unittest discover -s tests -v
```

The default run is named `diamond_5x3_seed42`. Existing results are never
overwritten. To reproduce the experiment under a separate run name, use:

```bash
.venv/bin/python -B scripts/train_models.py --run-id diamond_reproduction --n-jobs 2
.venv/bin/python -B scripts/execute_training_notebook.py --run-id diamond_reproduction
```

Use `--n-jobs 1` to run without parallel workers. The outer loop always runs
sequentially, and the number of internal BLAS/OpenMP threads is also limited.
Changing `n_jobs` does not change the folds, search grids, or selection criterion.
An interrupted run has status `running` or `failed` and must not be treated as
a completed result. To restart, choose a new `--run-id`; automatic resumption is
not implemented.

## Methodology and model selection

`load_training_data()` calls `prepare_data()` and returns only `X_train`,
`y_train`, and provenance metadata. The CSV contains 9,879 matches, but this stage
uses only the 7,903 matches in the training partition. The original 80/20 split,
feature representation, and `random_state=42` remain fixed. Data preparation
creates the test objects as part of the existing interface and may verify their
IDs in the split manifest. However, the values in `X_test` and `y_test` are not
passed to fitting, scoring, tuning, or model selection.

Nested stratified cross-validation uses 5 outer folds and 3 inner folds, with
`shuffle=True` and `random_state=42`. Within each outer training fold,
`GridSearchCV` selects hyperparameters using only the inner folds. The selected
pipeline is then refitted on the entire outer training fold and evaluated on the
outer validation fold, which was excluded from tuning. All algorithms use the
same fold assignments.

| Model family | Pipeline and hyperparameters | Configurations |
|---|---|---:|
| Baseline | `DummyClassifier(strategy="most_frequent", random_state=42)` | No tuning |
| Decision Tree | No scaling; `criterion=[gini, entropy]`, `max_depth=[3,5,8,12,None]`, `min_samples_leaf=[1,5,10,20]` | 40 |
| KNN | `StandardScaler` → KNN; `n_neighbors=[3,5,11,21,41,81]`, `weights=[uniform,distance]`, Minkowski `p=[1,2]` | 24 |
| SVM | `StandardScaler` → SVC; linear kernel with `C=[0.01,0.1,1,10]`; RBF kernel with the same `C` values and `gamma=[scale,0.01,0.1,1]` | 20 |

The baseline predicts the most frequent class in its training fold. It does not
learn relationships between the predictors and the target. It serves as a
diagnostic reference and is excluded from selection among Decision Tree, KNN,
and SVM. If the selected learned model fails to outperform the baseline, the
workflow explicitly reports this outcome.

For KNN and SVM, feature means and standard deviations are estimated within each
pipeline fit. The full `X_train` is never scaled before cross-validation.
Decision Tree and the baseline do not include a scaler. Lecture L7 also presents
scaling to `[0,1]`; this project uses `StandardScaler` as specified, which
standardizes features rather than constraining them to that interval.

The primary metric is **accuracy**, because the training classes are nearly
balanced and no asymmetric error costs have been specified. **Precision, recall,
F1, and ROC-AUC** are also recorded. Blue (`1`) is the positive class;
precision, recall, and F1 are binary metrics rather than macro averages.
When no positive predictions are made, precision and F1 are reported as zero
using `zero_division=0`. This convention is explicit rather than left to an
undefined-metric warning.
ROC-AUC uses class-`1` probabilities for Tree/KNN/Dummy and
`decision_function` for SVM. With `probability=False`, no probability calibration
is performed, and no additional classification threshold is learned. The
remaining metrics use `predict`.

Model comparison uses the mean and **sample standard deviation (ddof=1)** across
the 5 outer folds. The standard deviation describes fold-to-fold variability;
it is not a confidence interval. The family with the highest mean outer-fold
accuracy is selected. Exact ties are resolved by lower standard deviation,
followed by the fixed order Decision Tree → KNN → SVM. Within each grid search,
exact ties follow the deterministic ordering of `ParameterGrid`.

The family selection is saved in `selection.json` **before** the final searches.
Each learned family then undergoes a final grid search with **3 folds over the
entire training set**, using `refit="accuracy"`. The baseline is fitted directly
on the entire training set. Four final models are saved. Scores from these final
inner searches support hyperparameter selection; they do not change the family
already selected using outer CV.
The full workflow comprises 1,536 fits: 1,260 candidate fits within the nested
searches + 20 outer-fold refits, followed by 252 candidate fits in the final
searches + 4 final fits. No model is trained on the test set or the full dataset.

## Files and artifacts

| Path | Contents |
|---|---|
| `src/lol_model_training.py` | Reusable API for pipelines, CV, tuning, scoring, saving, and loading |
| `scripts/train_models.py` | Entry point for the complete training workflow |
| `scripts/execute_training_notebook.py` | Executes and saves only the Part B notebook using the current interpreter |
| `notebooks/02_model_training.ipynb` | Methodology, CV results, figures, and limitations; does not read the test set |
| `tests/test_model_training.py` | Quick tests using only Diamond training records |
| `requirements-part-b.txt` | Pinned additional dependencies, leaving Part A unchanged |
| `reports/model_training_handoff.md` | Instructions for the subsequent model evaluation stage |
| `reports/model_training/<run_id>/` | Run results and metadata |
| `models/part_b/<run_id>/` | `dummy.joblib`, `decision_tree.joblib`, `knn.joblib`, `svm.joblib` |
| `data/processed/part_b/<run_id>/cv_manifest.json` | Training-record IDs assigned to each fold; no labels or feature values |

Each run saves `nested_cv_scores`, `model_comparison`, and `cv_results_*` in
**CSV and JSON** formats. The `inner_cv/` directory contains all 15 searches
performed within the outer folds, also in both formats. Complete search results
include parameters, fold scores, means, standard deviations, metric-specific
ranks, and timings. The scikit-learn column pattern `split*_test_*` refers to
validation within CV: **it does not refer to the held-out test set from Part A**.

`best_params.json` contains the final hyperparameters, and `selection.json`
records the selected family. `run_metadata.json` records run status, seed, search
grids, feature schema, software versions, a training-data fingerprint, checksums
for the source data, models, and fold manifest, elapsed time, warnings, and the
absence of test-set use. Timings and timestamps are not deterministic results:
reproducibility concerns the splits, parameters, predictions, and metrics in the
same software environment. Binary files need not be identical byte for byte
across platforms.

## Subsequent sharing through GitHub

The Part B workflow does not require Git commands or modify `.gitignore`.
The existing ignore rules exclude **all CSV files**, `.joblib` files, and
`data/` by default. However, the four trained models for `diamond_5x3_seed42`
have already been explicitly added with `git add -f` and uploaded to the
`model-training` branch. They are therefore tracked and available on GitHub
despite the matching ignore rule. Force-adding these files did not change
`.gitignore`; the rule still applies to other untracked `.joblib` files.

The published model files are located in
`models/part_b/diamond_5x3_seed42/`:

- `dummy.joblib`
- `decision_tree.joblib`
- `knn.joblib`
- `svm.joblib`

**Member 3 can use these models without repeating training.** On GitHub, select
the `model-training` branch and download that branch as a ZIP archive to obtain
the models together with the corresponding code and results. Preserve the
repository's directory structure when extracting it. If downloading files
individually, use GitHub's raw-file download option for each model and retain
the corresponding files in `reports/model_training/diamond_5x3_seed42/`,
including `run_metadata.json`, `model_comparison.json`, `nested_cv_scores.json`,
and `best_params.json`, which the loading functions require.

Member 3 should use Python 3.11 and the pinned dependencies, then load the models
with `load_final_model()` using `run_id="diamond_5x3_seed42"` and the model name
`dummy`, `decision_tree`, `knn`, or `svm`. The original
Diamond CSV is still required locally to reconstruct the fixed test partition
with `prepare_data()`. For final evaluation, pass unscaled test features directly
to the saved pipelines without refitting the models or their scalers. The model
family selected through CV remains fixed; test results must not reopen selection.

CSV results and local fold manifests remain excluded by default. The JSON
files under `reports/model_training/`, executed notebook, source code, tests,
and documentation can also be shared under the current rules.
The JSON files contain only aggregate results and configurations, without
dataset rows or individual predictions. They provide complete search results
that remain readable without the CSV files. Fold IDs remain in the ignored
manifest.

If regeneration or additional local artifacts are needed, the following options
remain available:

1. **Regenerate the artifacts:** use the same original CSV and Python 3.11,
   install the pinned dependencies, run the quick tests, and then execute
   `scripts/train_models.py`. The workflow reconstructs the Part A split and
   the Part B training artifacts. If shared results already exist for that run,
   choose a new `--run-id`.
2. **Receive the local artifacts:** privately share the three directories
   `models/part_b/<run_id>/`, `reports/model_training/<run_id>/`, and
   `data/processed/part_b/<run_id>/`, preserving their relative paths.
   CSV results may be included in this transfer. The recipient must still have
   a local copy of the original CSV for the subsequent evaluation.

The published models do not need to be force-added again, and the dataset does
not need to be added to Git. Before loading downloaded or transferred models,
verify that they come from the project repository or the intended group member,
because `joblib` serializes Python objects. `load_final_model()` checks the checksum, feature
schema, package versions, and Python major/minor version before loading.

## Relationship to the course and limitations

Lectures L4–L5 cover classification, decision trees, Gini impurity, and entropy.
L6, slides 25–26, covers pre-pruning; L6, slides 44–51, covers cross-validation
and nested CV. L7, slides 9–24, covers KNN, scaling, SVM, C, and kernels; L7,
slides 27–34, discusses thresholds, ROC curves, and fair model comparison.

CV performance estimates apply to the Diamond sample and this representation
of 12 features. Selecting a family from several candidates introduces additional
selection bias: the selected family's outer score does not replace an independent
final evaluation. Five folds do not establish statistically significant
differences between closely performing models.
The search grid is finite; a boundary optimum does not establish a global
optimum or justify adjustments after inspecting the test set. Scaling does not
remove feature correlations or extreme values. The limitations identified in
Part A remain: suspicious ward counts, unavailable player IDs and patch/time
metadata, and the omission of absolute game tempo. No filtering, feature
selection, resampling, or additional EDA has been introduced to address these
limitations.

The subsequent model evaluation stage will apply the frozen models to the test
set once. It may compare their results descriptively, but those results must not
be used to reopen tuning or change the selected family.
