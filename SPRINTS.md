# Workstreams

**Research question:** can we predict the winning team using only statistics available after the
first 10 minutes? One question, three sequential stages.

**Presentation: Friday 20 Nov 2026. Project materials: 6 Dec 2026.**

| Stage | Owner | Status | Branch |
|---|---|---|---|
| 1 — Data preprocessing + EDA | Cesar | **done** | `data-preprocessing` |
| 2 — Model training | Andrea | **done** | `model-training` |
| 3 — Model evaluation | Christian | next | `model-evaluation` |

---

## Stage 1 — preprocessing and EDA  ·  DONE

`src/lol_preprocessing.py`, `notebooks/01_data_preprocessing.ipynb`, 8 figures,
`reports/part_a_findings.md`, 13 passing tests.

```python
from src.lol_preprocessing import prepare_data, FEATURE_NAMES, TARGET

data = prepare_data()
data.X_train   # (7903, 12)
data.X_test    # (1976, 12)
data.y_train, data.y_test
```

**The 12 predictors**, all blue-minus-red differences: `blueGoldDiff`, `blueExperienceDiff`,
`killsDiff`, `assistsDiff`, `minionsDiff`, `jungleMinionsDiff`, `wardsPlacedDiff`,
`wardsDestroyedDiff`, `firstBloodDiff`, `dragonsDiff`, `heraldsDiff`, `towersDestroyedDiff`.

| | |
|---|---|
| Source | 9,879 matches, 40 columns, 0 missing, 0 duplicate `gameId` |
| Target | blue wins 49.904% — near balanced |
| Split | 80/20 stratified, `random_state=42`, nothing fit before splitting |
| Strongest predictors | `blueGoldDiff` 0.511, `blueExperienceDiff` 0.489, `killsDiff` 0.479 |
| Correlated pairs ≥ 0.75 | gold/kills 0.918, gold/experience 0.896, kills/assists 0.831 |

**Left honestly unresolved:** ward counts reach 276 in training. Implausible at ten minutes, but
the CSV cannot establish whether they are real or extraction artefacts, so nothing was clipped
(D06). `wardsPlacedDiff` is the weakest predictor at 0.016 — treat it as the least trustworthy
feature and say so in the report.

---

## Stage 2 — model training  ·  DONE

`src/lol_model_training.py`, `notebooks/02_model_training.ipynb`, `tests/test_model_training.py`,
results under `reports/model_training/diamond_5x3_seed42/`.

Nested cross-validation, outer 5-fold × inner 3-fold, `random_state=42`. A `DummyClassifier`
baseline was included so every gain is measured against predicting the majority class.

**Nested CV results, outer-fold mean ± std:**

| model | accuracy | ROC-AUC | F1 | |
|---|---|---|---|---|
| dummy | 0.5009 ± 0.0002 | 0.5000 ± 0.0000 | 0.0000 | baseline |
| decision tree | 0.7158 ± 0.0206 | 0.7940 ± 0.0132 | 0.7121 ± 0.0212 | |
| KNN | 0.7162 ± 0.0145 | 0.7986 ± 0.0130 | 0.7167 ± 0.0141 | |
| **SVM** | **0.7295 ± 0.0167** | **0.8089 ± 0.0136** | **0.7292 ± 0.0175** | **selected** |

Selected hyperparameters: SVM with a **linear** kernel, `C=10`. Decision tree: entropy,
`max_depth=5`, `min_samples_leaf=20`. KNN: `n_neighbors=81`, `p=1`, distance weighting.

The selection rule was frozen before the final searches: maximum outer mean accuracy, ties
broken by lowest standard deviation. **The held-out test set was not touched**
(`held_out_test_used: false`).

That the linear kernel won is itself worth a sentence in the report: the 12 difference features
separate the classes about as well as a straight boundary allows, and the extra flexibility of
an RBF kernel bought nothing.

---

## Stage 3 — model evaluation  ·  NEXT

Deliverable: `notebooks/03_model_evaluation.ipynb` plus figures in `reports/figures/`.

| | Sprint |
|---|---|
| **C1** | Load the selected model and the CV results; verify the run metadata |
| **C2** | Score once on `X_test` — confusion matrix, accuracy, precision, recall, F1, ROC-AUC |
| **C3** | ROC curve, and a comparison of all four models on the test set |
| **C4** | Interpretation — which early advantages drive the prediction, in plain language |
| **C5** | Limitations and what the result does not show |

### How to load what Stage 2 produced

```python
from src.lol_preprocessing import prepare_data
from src.lol_model_training import load_final_model, load_run_results

run_id = "diamond_5x3_seed42"
results  = load_run_results(run_id=run_id)
selected = results["metadata"]["selected_model"]   # "svm"
model    = load_final_model(selected, run_id=run_id)
prepared = prepare_data()
```

### Rules that break quietly if ignored

- **Pass `prepared.X_test` unscaled and with the original column order.** The scaler is inside
  the Pipeline, so `model.predict(X_test)` is correct. Calling `fit`, `fit_transform` or adding
  a new scaler would both leak and double-scale.
- **For ROC-AUC use `decision_function`** when available — the selected SVM has no
  `predict_proba`. Fall back to `predict_proba`'s class-`1` column for the other models.
- **Score the test set once.** Scoring repeatedly and keeping the best result is how a test set
  stops being a test set. If the test number disappoints, report it — do not go back and retune.
- The positive class is `blueWins=1`. CV precision, recall and F1 were binary with
  `zero_division=0`; keep that consistent so the test numbers are comparable.
- Check `run_metadata.json` has `status="complete"` and that the source checksum matches.

### Expect roughly these numbers

Nested CV put the SVM at accuracy 0.7295 and ROC-AUC 0.8089. A test result **close to that** is
the healthy outcome and confirms the CV estimate was honest. A result far above it means
something leaked; far below means the model does not generalise and that is a finding, not a
failure. Either way, report what came out.

---

## Non-negotiables

See CLAUDE.md §3. The ones that break quietly:

- `gameId` is never a feature.
- Never recompute the split or fit anything on `X_test`.
- Scaling goes inside Pipelines.
- `random_state=42` everywhere.
- Run `python -m unittest discover -s tests` before committing.
- No number in a slide, doc or markdown cell that did not come from executed code.
