# Predicting-League-of-Legends-match
CDS3004 Data Mining Group Project - Predicting League of Legends match outcomes using early-game statistics.

## Part A: preprocessing and exploratory analysis

Research question: **Can we predict the winning team using only statistics available after the first 10 minutes?**
The response is `blueWins` (`1` = Blue wins; `0` = Red wins). Part A uses only
the original [League of Legends Diamond Ranked Games (10 min) dataset](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min)
by bobbyscience. No models are trained in this stage.

### Files

- `notebooks/01_data_preprocessing.ipynb`: complete workflow, explanations, executed tables, and embedded charts.
- `src/lol_preprocessing.py`: source discovery, validation, fixed feature policy, and reproducible unscaled split.
- `reports/figures/`: eight labeled EDA figures. All use training data except the full-source target-count chart.
- `reports/part_a_findings.md`: aggregate findings generated from the executed notebook.
- `scripts/execute_notebook.py`: run the notebook using the invoking Python, without installing a global Jupyter kernel.
- `tests/test_preprocessing.py`: source, validation, feature-exclusion, and split-reproducibility checks.
- `requirements.txt`: exact direct dependency versions tested with Python 3.11.
- `data/processed/split_manifest.json`: generated **locally and ignored**; source checksum, feature order, and split membership without labels or predictor values.

### Environment and reproduction

Use Python 3.11. A project-local `.venv` was necessary on the inspected machine:
the default Python's scientific libraries had incompatible CPU architectures,
and the existing Anaconda interpreter could not run on the machine's CPU.
The system and Anaconda installations were left unchanged.

From the repository root, for a fresh checkout:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/execute_notebook.py
.venv/bin/python -m unittest discover -s tests -v
```

If `.venv` already exists, use it without recreating it. In VS Code, select
`.venv/bin/python` as the notebook kernel, then run all cells. The notebook also
works when its working directory is `notebooks/`. The execution script supplies
its own temporary kernel, so it does not depend on a globally registered kernel.

Keep the raw file named `high_diamond_ranked_10min.csv` locally. Discovery checks
`data/raw/` and sibling directories named `Local_Datasets` after trimming surrounding
whitespace. In the inspected workspace the file is outside the repository at:

```text
../Local_Datasets /high_diamond_ranked_10min.csv
```

There is a space immediately after `Local_Datasets`. Alternatively set
`LOL_DATA_PATH` to the file's absolute path, or pass that path to `prepare_data()`.
An ambiguous search fails rather than choosing an arbitrary dataset. No download
or raw-data copying is performed. `.gitignore` excludes datasets, processed data,
environments, caches, and common large intermediate formats. The local source SHA-256 is:

```text
82b2e23279097c2dac616e966eb2911711c428d41b6382b4cdb15da4892f9ca6
```

`prepare_data()` rejects different file bytes to prevent silent source/version
changes. This fingerprints the supplied local file; it is not independent
verification against a second Kaggle download. Investigate a mismatch rather
than replacing the checksum or regenerating the test set to obtain favorable results.

### Fixed feature representation

The module keeps `blueGoldDiff` and `blueExperienceDiff`, and creates ten
Blue-minus-Red differences:

```text
killsDiff, assistsDiff, minionsDiff, jungleMinionsDiff,
wardsPlacedDiff, wardsDestroyedDiff, firstBloodDiff,
dragonsDiff, heraldsDiff, towersDestroyedDiff
```

Positive differences favor Blue. The 12-column policy is domain-based and fixed
before target-association EDA. `feature_decisions()` documents the action and
reason for every original column. The audit verifies mirrored kills/deaths,
opposite team differences, gold/experience difference identities, per-minute
rescalings, and elite-monster sums before relying on their redundancy.
Average level is excluded as a coarse experience-related summary, not as an
exact duplicate. Relative differences deliberately omit absolute game tempo and
can merge different tied objective states. These are documented simplicity
tradeoffs, not evidence that the excluded columns have no predictive information.
No target-based feature selection is performed.

### Reuse in Part B

From the repository root:

```python
from src.lol_preprocessing import FEATURE_NAMES, TARGET, prepare_data, save_split_manifest

prepared = prepare_data()  # optionally pass the absolute CSV path
save_split_manifest(prepared)  # verifies an existing local manifest, or creates it
X_train, X_test = prepared.X_train, prepared.X_test
y_train, y_test = prepared.y_train, prepared.y_test
```

`X_train` has **7,903 rows × 12 columns** and `X_test` has **1,976 rows × 12 columns**.
The label shapes are `(7903,)` and `(1976,)`. Training contains 3,959 Red wins and
3,944 Blue wins; test contains 990 Red wins and 986 Blue wins. All four objects
use `gameId` as the row index for traceability; neither `gameId` nor `blueWins`
appears in the predictor columns. Algorithms must consume the predictor columns,
without resetting the match ID into an additional feature.

Splitting uses `test_size=0.20`, `stratify=y`, and `random_state=42`, after sorting
by unique `gameId`. The low-level split is stable even if input rows are reordered;
the public `prepare_data()` interface additionally requires the exact source bytes.
The manifest refuses to overwrite different source/feature/membership information.
The source schema, expected fingerprint, feature policy, sorting, and seed together
define the reproducible contract; do not alter them during the three-model comparison.

Decision Tree, KNN, and SVM should reuse these same four objects. Part B may fit
models and choose hyperparameters using training cross-validation. Put scaling
inside the KNN/SVM scikit-learn pipelines so each fold learns its own scaling
parameters. A Decision Tree does not require scaling. Any learned imputation or
supervised feature selection must also be inside the training-fold pipeline.
After all decisions are fixed, use the held-out test set for final evaluation.

### Data quality and limitations

The supplied file has **9,879 matches and 40 columns**, no missing values, no
identical duplicate records, no repeated match IDs, and no detected invalid
numeric, binary, count, or team-arithmetic values. No records or values are changed.
Validation stops on future anomalies rather than silently imputing, deduplicating,
or correcting them. Outlier flags are descriptive and are never cleaning rules.
Very large ward counts remain suspicious and unresolved without match timelines
or extraction details; they are retained and explicitly discussed in the notebook.

Full-source inspection is limited to basic descriptive summaries and structural
checks. Predictor distributions, outlier analysis, target relationships, and
correlation analysis use **training data only**. Correlations and binned win
proportions are descriptive associations, not causal effects, feature-selection
decisions, or model performance. No preprocessing is fitted in Part A.

The snapshot timestamp is based on the original dataset's description and schema,
not independently reconstructed timelines. Player identities and patch/time
metadata are unavailable here, so this random split cannot establish independence
across repeated players or generalization to later patches. The source CSV and
generated data remain local; no commits, pushes, or branch changes are needed to
reproduce Part A.

---

## Project status and work division

One research question, three sequential stages.

| Stage | Scope | Owner | Status |
|---|---|---|---|
| **1** | Data preprocessing, validation, split, EDA | Cesar | **done** |
| **2** | Model training — Decision Tree, KNN, SVM, nested CV, hyperparameter tuning | Andrea | **done** |
| **3** | Model evaluation — confusion matrix, accuracy, precision, recall, F1, ROC-AUC, ROC curves, interpretation | Christian | next |

We deliberately keep to the single question above. Clustering and association-rule mining were
considered as secondary questions and dropped (decision D07) so that one question is answered
properly rather than three superficially.

### Stage 2 result

Nested cross-validation, outer 5-fold × inner 3-fold, against a majority-class baseline:

| model | accuracy | ROC-AUC |
|---|---|---|
| dummy baseline | 0.5009 ± 0.0002 | 0.5000 |
| decision tree | 0.7158 ± 0.0206 | 0.7940 ± 0.0132 |
| KNN | 0.7162 ± 0.0145 | 0.7986 ± 0.0130 |
| **SVM (linear, C=10)** | **0.7295 ± 0.0167** | **0.8089 ± 0.0136** |

The SVM was selected on outer mean accuracy under a rule frozen before the final searches.
**The held-out test set has not been used** — it is spent once, in Stage 3.

### How each stage consumes the previous one

Call the functions; never read a stored file, so nobody works from a stale artefact.

```python
from src.lol_preprocessing import prepare_data
from src.lol_model_training import load_final_model, load_run_results

data = prepare_data()                 # X_train (7903, 12), X_test (1976, 12)

run_id   = "diamond_5x3_seed42"
results  = load_run_results(run_id=run_id)
model    = load_final_model(results["metadata"]["selected_model"], run_id=run_id)
```

Pass `data.X_test` **unscaled** straight to `model.predict` — the scaler lives inside the
Pipeline. Do not call `fit` or `fit_transform`, and do not recompute the split: Stage 1 split
before computing any statistic (decision D04), and the held-out set is only honest untouched.

See `SPRINTS.md` for the stage breakdowns, `docs/decisions.md` for every choice and its
evidence, and `docs/agent_validation.md` for mistakes caught during the work.

### Tests

```
python -m unittest discover -s tests
```

## Team

Workload is evidenced by commits and by each member's agent-use trajectory, as described in the
course's Special Tutorial. Trajectories are kept locally and submitted in the final ZIP; they are
git-ignored and never committed.

## Scope and limitations

9,879 ranked solo-queue matches, described by the dataset's author as high Diamond to low
Master. **Patch, date and region are unknown** — the CSV carries no such columns, so we cannot
say the matches share a patch window or a server, and we do not claim it. That limits what the
result generalises to: a model fit here is not shown to hold on a later patch or another region.
Observational data: a gold lead predicting a win is not evidence that chasing gold early causes
wins. Conclusions are not generalised beyond this scope.
