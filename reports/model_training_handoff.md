# Part B → Model Evaluation

This document accompanies `README_PART_B.md`. Part B does not compute test-set
performance; the subsequent model evaluation stage remains to be completed.

## Delivered artifacts

The default run is `diamond_5x3_seed42`. The directory
`reports/model_training/<run_id>/` contains `run_metadata.json`,
`selection.json`, `best_params.json`, `model_comparison.json`,
`nested_cv_scores.json`, the three final `cv_results_*.json` files, and the
15 search results in `inner_cv/`. CSV files contain the same results and may be
shared locally.

`models/part_b/<run_id>/` contains four pipelines already fitted on the 7,903
training matches: `dummy`, `decision_tree`, `knn`, and `svm`. KNN and SVM
include their fitted scalers; the other two models do not require scaling.
The single SVM artifact contains the kernel selected from linear and RBF
candidates during the final grid search. No further fitting is required.

`data/processed/part_b/<run_id>/cv_manifest.json` records fold assignments
using only training-set match IDs. The Part A manifest remains at
`data/processed/split_manifest.json`. Neither type of manifest contains labels
or feature values.

## Environment preparation and loading

Use the same Diamond dataset, Python 3.11, and the versions pinned in
`requirements-part-b.txt`. The following loading example should be run from
the repository root:

```python
from src.lol_preprocessing import prepare_data
from src.lol_model_training import load_final_model, load_run_results

run_id = "diamond_5x3_seed42"
results = load_run_results(run_id=run_id)
selected = results["metadata"]["selected_model"]
model = load_final_model(selected, run_id=run_id)
prepared = prepare_data()  # Reconstruct the original split without refitting anything.
```

These instructions do not perform model evaluation. When that stage begins,
use `prepared.X_test` **without scaling it separately**, retaining the original
columns in the recorded order. Pass it directly to `model.predict`; do not
call `fit` or `fit_transform`, or introduce a new scaler. For ROC-AUC, use
`decision_function` when available (SVM); otherwise, use `predict_proba` and
select the column corresponding to class `1`.

The positive class is `blueWins=1` (Blue wins). CV precision, recall, and F1
are binary metrics with `zero_division=0`. The final evaluation may also include
a confusion matrix and results for both classes, while retaining the five
common metrics. Accuracy is the primary criterion specified before the
experiment.

Confirm that `run_metadata.json` reports `status="complete"` and
`held_out_test_used=false`, and that the source checksum matches the original
file. `load_final_model` checks the model checksum, feature schema, package
versions, and Python major/minor version. If the binary artifacts are missing,
the aggregate JSON results remain readable but cannot generate predictions;
regenerate the artifacts.

## Rules for final evaluation

The selected family is frozen in `selection.json`, based on mean outer-fold
accuracy from nested CV 5×3, before the final tuning searches. The three learned
model families used the same fold structure and their documented search grids;
the baseline was not tuned.
The final test comparison may include all four models, but its results must
not be used to change the selected family, hyperparameters, thresholds,
features, or preprocessing rules.

`cv_results_*` describes the final 3-fold grid search used for hyperparameter
selection. The scikit-learn field pattern `split*_test_*` refers to an internal
validation fold, not the independent test set. To present the CV comparison,
use `model_comparison.json` and `nested_cv_scores.json`, which summarize the
outer folds. Reported outer-fold standard deviations are sample standard
deviations across five folds, not confidence intervals.

When models perform similarly, avoid claims of definitive superiority.
The test set will provide a separate final estimate for the held-out sample;
it will not establish generalization to other ranks, players, or patches.
Do not train on all 9,879 records before evaluation, as this would compromise
test-set independence.

## GitHub sharing and transfer of ignored files

The original `.gitignore` excludes `.joblib` files, all `.csv` files, and
`data/`; it has not been changed. Aggregate JSON files, this document,
`README_PART_B.md`, tests, source code, and the executed notebook may be shared
subsequently through GitHub. The shareable JSON files and the Part B notebook
contain no original CSV contents, dataset rows, or individual predictions.

To regenerate the artifacts, follow the commands in `README_PART_B.md`:
run the quick tests, complete training, and notebook execution. If the shared
run already exists, choose a new `--run-id` and use that same ID when loading
the results. Existing results are not overwritten; do not change the seed or
search grid to obtain a more favorable outcome.

Alternatively, the group member who performed training can privately transfer
`models/part_b/<run_id>/`, `reports/model_training/<run_id>/`, and
`data/processed/part_b/<run_id>/`, preserving their relative paths. There is
no need to include binary artifacts in Git or change `.gitignore`. Load only
`joblib` artifacts received from the intended group member. The recipient
must also have a local copy of the original dataset.
