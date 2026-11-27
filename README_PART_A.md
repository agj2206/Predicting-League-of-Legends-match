# Part A — Data Preprocessing, Feature Engineering and EDA

Research question: **Can we predict the winning team using only statistics available after the first 10 minutes?**
The response is `blueWins` (`1` = Blue wins; `0` = Red wins). Part A uses only
the original [League of Legends Diamond Ranked Games (10 min) dataset](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min)
by bobbyscience. No models are trained in this stage.

## Files

- [`notebooks/01_data_preprocessing.ipynb`](notebooks/01_data_preprocessing.ipynb) — complete workflow, explanations, executed tables, and embedded charts.
- [`src/lol_preprocessing.py`](src/lol_preprocessing.py) — source discovery, validation, fixed feature policy, and reproducible unscaled split.
- [`reports/figures/`](reports/figures/) — eight labeled EDA figures. All use training data except the full-source target-count chart.
- [`reports/part_a_findings.md`](reports/part_a_findings.md) — aggregate findings generated from the executed notebook.
- [`scripts/execute_notebook.py`](scripts/execute_notebook.py) — run the notebook using the invoking Python, without installing a global Jupyter kernel.
- [`tests/test_preprocessing.py`](tests/test_preprocessing.py) — source, validation, feature-exclusion, and split-reproducibility checks.
- [`requirements.txt`](requirements.txt) — exact direct dependency versions tested with Python 3.11.
- `data/processed/split_manifest.json`: generated **locally and ignored**; source checksum, feature order, and split membership without labels or predictor values.

## Environment and reproduction

Use Python 3.11 and a project-local virtual environment. See the [project README](README.md#installation)
for installation across all three parts.

From the repository root, for a fresh checkout:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/execute_notebook.py
.venv/bin/python -B -m unittest discover -s tests -p test_preprocessing.py -v
```

If `.venv` already exists, use it without recreating it. In VS Code, select
`.venv/bin/python` as the notebook kernel, then run all cells. The notebook also
works when its working directory is `notebooks/`. The execution script supplies
its own temporary kernel, so it does not depend on a globally registered kernel.
The script updates the notebook, figures and Part A findings in place.

Download the original file from Kaggle and keep it named `high_diamond_ranked_10min.csv`.
The loader checks `data/raw/` and sibling directories named `Local_Datasets` after trimming surrounding
whitespace. Set `LOL_DATA_PATH` or pass an explicit path to `prepare_data()` to select a specific local copy.
Discovery requires a unique match. The loader performs no download or raw-data copying.
The expected source SHA-256 is:

```text
82b2e23279097c2dac616e966eb2911711c428d41b6382b4cdb15da4892f9ca6
```

`prepare_data()` rejects different file bytes to prevent silent source/version
changes. This fingerprints the supplied local file; it is not independent
verification against a second Kaggle download. Investigate a mismatch rather
than replacing the checksum or regenerating the test set to obtain favorable results.

## Fixed feature representation

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

## Reuse in Part B

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

## Data quality and limitations

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
