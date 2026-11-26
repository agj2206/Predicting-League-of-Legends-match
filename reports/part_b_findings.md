# Part B: Empirical Findings

Run: `diamond_5x3_seed42`. Dataset: **Diamond only**. Part A was reused without modification.

**Research Question:** Can we predict the winning team in a League of Legends match using only statistics available after the first 10 minutes?

## Methodology

The workflow uses 7,903 training matches and 12 initially unscaled features. Nested stratified cross-validation uses 5 outer folds and 3 inner folds, with shuffle=True and random_state=42. The test set was not used for training, scoring, tuning, or model selection.

Accuracy is the primary criterion, with Blue (`1`) as the positive class. Results are reported as mean ± sample standard deviation (ddof=1) across the 5 outer folds. The following values are not test-set metrics.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| dummy | 0.5009 ± 0.0002 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 |
| decision_tree | 0.7158 ± 0.0206 | 0.7216 ± 0.0315 | 0.7051 ± 0.0441 | 0.7121 ± 0.0212 | 0.7940 ± 0.0132 |
| knn | 0.7162 ± 0.0145 | 0.7145 ± 0.0188 | 0.7193 ± 0.0218 | 0.7167 ± 0.0141 | 0.7986 ± 0.0130 |
| svm | 0.7295 ± 0.0167 | 0.7284 ± 0.0160 | 0.7302 ± 0.0215 | 0.7292 ± 0.0175 | 0.8089 ± 0.0136 |

## Model selection and final hyperparameters

**Selected family: SVM**, based on mean outer-fold accuracy. This choice was frozen before final hyperparameter tuning. The improvement over the baseline is **22.85 percentage points**.

| Model | Final hyperparameters |
|---|---|
| dummy | `{"classifier__strategy": "most_frequent"}` |
| decision_tree | `{"classifier__criterion": "entropy", "classifier__max_depth": 5, "classifier__min_samples_leaf": 20}` |
| knn | `{"classifier__n_neighbors": 81, "classifier__p": 1, "classifier__weights": "distance"}` |
| svm | `{"classifier__C": 10, "classifier__kernel": "linear"}` |

Final hyperparameters were selected using 3-fold CV over the entire training set. All four saved models have been fitted on all 7,903 training matches. KNN and SVM include StandardScaler within their pipelines. The final SVM uses a linear kernel; the RBF kernel was also evaluated in the same search.

## Interpretation for the project presentation

The baseline always predicts Red, the slightly more frequent class, and achieves approximately 50% accuracy. The learned models improve on this reference, reaching approximately 73% with SVM: on average across the validation folds, about 73 of every 100 match winners are predicted correctly. Statistics available after the first ten minutes therefore show predictive value in the Diamond sample examined. Independent confirmation on the held-out test set remains the responsibility of the model evaluation stage.

The baseline has zero precision, recall, and F1 for Blue because it never predicts Blue; these values are expected rather than calculation errors. Its ROC-AUC is 0.5. The SVM's ROC-AUC is approximately 0.809: it measures ranking performance, not the probability of victory in an individual match.

Decision Tree and KNN have very similar accuracy. SVM ranks first in all five outer folds, but this observation does not establish statistical significance or guarantee performance on future matches.

## Limitations and issues requiring explicit acknowledgment

- KNN selects k=81 and SVM selects C=10, both at the upper boundary of the search grid. A global optimum has not been established. The predefined grid is retained and is not expanded using test-set results.
- For SVM, the mean accuracy in the final tuning search is 0.729596 with C=10 and 0.729216 with C=1; the difference is very small. The predeclared selection rule is retained without an additional post hoc decision.
- Outer scores were used to select the model family and do not replace an independent test evaluation. Standard deviations are not confidence intervals.
- The limitations identified in Part A remain: suspicious ward counts, correlated features, unavailable player IDs and patch/time metadata, and the omission of absolute game tempo. No additional cleaning, EDA, or feature selection was performed.

## Execution and verification

The complete training workflow performed **1,536 fits** in approximately **119.5 seconds** on this computer; execution time depends on the machine. No runtime warnings were recorded.

The 8 quick Part B tests passed before the full training run. Subsequently, all 21 tests in the complete suite passed (8 from Part B + 13 from Part A). The reduced integration tests verify repeatability across two runs, scaling within each fold, and fitting restricted to the permitted training folds. The full models were reloaded and checked without making predictions on the test set. Checks involving match IDs assess only the structure of the data partitions.

The artifact audit passed 58 checks; details are available in `model_training/diamond_5x3_seed42/artifact_verification.json`.

## Reproduction and handoff

Commands, Git-ignored files, regeneration, and private transfer of artifact directories are documented in [README_PART_B.md](../README_PART_B.md) and [model_training_handoff.md](model_training_handoff.md). The aggregate JSON files retain all CV search results even when CSV files and models are excluded by .gitignore. The complete list of installed packages is recorded in `environment_versions.txt` within the run directory.

The notebook `notebooks/02_model_training.ipynb` reads these results, displays tables and figures, and does not rerun training. The model evaluation stage must apply the saved pipelines to unscaled X_test without refitting them or changing the selected family.

The notebook was executed and saved with all 9 code cells completed, one embedded
figure, and no error outputs. An initial-to-final SHA-256 comparison confirmed
that all 18 pre-existing files checked remained identical, including the Part A
code, tests, notebook, manifest, and results. Details are available in
`model_training/diamond_5x3_seed42/part_a_integrity.json`.

Two technical issues encountered during notebook execution were resolved.
Initially, the sandbox blocked Jupyter's local communication ports; execution
of the local kernel was authorized separately. pandas Styler formatting required
Jinja2, which was not installed, so it was replaced with ordinary DataFrame
formatting without installing packages. The initial stalled execution was
terminated. Neither issue affected model training or its results.
