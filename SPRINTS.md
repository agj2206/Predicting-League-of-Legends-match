# Workstreams

**Presentation: Friday 20 Nov 2026. Project materials: 6 Dec 2026.**

Part A is done. B and C can both start now and do not depend on each other.

---

## A — preprocessing and EDA  ·  DONE

`src/lol_preprocessing.py`, `notebooks/01_data_preprocessing.ipynb`, 8 figures,
`reports/part_a_findings.md`, 13 passing tests.

**What it hands over.** Call `prepare_data()` — do not read a stored file:

```python
import sys; sys.path.insert(0, "src")
from lol_preprocessing import prepare_data, FEATURE_NAMES, TARGET

data = prepare_data()
data.X_train   # (7903, 12)
data.X_test    # (1976, 12)
data.y_train, data.y_test
```

**The 12 predictors**, all blue-minus-red differences:
`blueGoldDiff`, `blueExperienceDiff`, `killsDiff`, `assistsDiff`, `minionsDiff`,
`jungleMinionsDiff`, `wardsPlacedDiff`, `wardsDestroyedDiff`, `firstBloodDiff`, `dragonsDiff`,
`heraldsDiff`, `towersDestroyedDiff`.

**Numbers you can rely on** (training data only — the split happened first):

| | |
|---|---|
| Source | 9,879 matches, 40 columns, 0 missing, 0 duplicate `gameId` |
| Target | blue wins 49.904% — near balanced |
| Split | 80/20 stratified, `random_state=42`, nothing fit before splitting |
| Strongest predictors | `blueGoldDiff` 0.511, `blueExperienceDiff` 0.489, `killsDiff` 0.479 |
| Weakest | `wardsPlacedDiff` 0.016 |
| Correlated pairs ≥ 0.75 | gold/kills 0.918, gold/experience 0.896, kills/assists 0.831 |

**Open item A left honestly unresolved:** ward counts reach 276 in training. Hundreds of wards at
ten minutes is implausible, but the CSV cannot establish whether they are real or extraction
artefacts, so all rows were retained and nothing was clipped. Worth one line in the report.

---

## B — Q1 classification and evaluation

Deliverable: `notebooks/02_classification.ipynb` plus an evaluation module in `src/`.

| | Sprint |
|---|---|
| **B1** | Nested CV harness — outer 5-fold stratified, inner 3-fold for hyperparameters |
| **B2** | Feature selection three ways — correlation, mutual information, tree importance — and where they disagree |
| **B3** | Decision Tree, KNN, SVM, all inside Pipelines |
| **B4** | Compare, choose, evaluate **once** on `X_test` |
| **B5** | Interpretation — depth-limited tree plot, feature importances, plain language |

**Why nested CV.** If hyperparameters are chosen on the same folds that then report the score,
the score is optimistic — the choice has already seen the data it is graded on. The rubric names
nested cross-validation explicitly, so a plain `cross_val_score` leaves marks on the table.
Report precision, recall, F1 and ROC-AUC as mean ± std across the outer folds.

**Pipelines, not pre-scaled data.** KNN and SVM need scaling; the tree does not. Putting the
scaler inside the Pipeline means it is fit per fold rather than on everything, and makes that
difference explicit instead of hidden.

**No Naive Bayes** — the course did not teach it (D03).

**Interpretation caveat.** `blueGoldDiff` and `killsDiff` correlate at 0.918, so whichever the
tree splits on first will look far more important than the other. Any claim like "gold matters
more than kills" has to survive that.

**Measured timings on this data** — nothing here is slow, one fold is ~7,900 rows:

| estimator | one fit |
|---|---|
| Decision Tree | full nested CV in ~3 s |
| KNN, k=25 | 0.05 s |
| SVC, RBF kernel | 0.91 s |

The RBF SVM is affordable — use it, not `LinearSVC`.

**Sanity tripwire.** A depth-limited tree under nested CV lands around **ROC-AUC 0.80** on this
data. Much higher means something leaked; check before celebrating.

---

## C — Q2 clustering and Q3 association rules

Deliverables: `notebooks/03_clustering.ipynb`, `notebooks/04_association_rules.ipynb`, plus
helpers in `src/`. Add `mlxtend` from `requirements.txt`.

| | Sprint |
|---|---|
| **C1** | Scale and log-transform skewed features; explain why scaling is required for distance methods |
| **C2** | K-means — choose K by elbow **and** silhouette over K = 2..10 |
| **C3** | Compare with Hierarchical (Ward, dendrogram on a sample) and DBSCAN (eps from a k-distance plot); pick one and justify it |
| **C4** | Profile each cluster, give it a business name — these are the game archetypes |
| **C5** | Discretise: compare equal-width vs equal-frequency, choose one, justify it |
| **C6** | Apriori with `blueWins` as the consequent; justify minsup and minconf; prune redundant rules |
| **C7** | Cross-check the top rules against B's tree splits — agreements and disagreements both matter |

Clustering is unsupervised, so it can use the 12 predictors without the target. Keep the target
out of the clustering input and bring it back only to profile the resulting clusters.

---

## Non-negotiables

See CLAUDE.md §3. The ones that break quietly:

- `gameId` is never a feature.
- Never refit the split or compute statistics on `X_test` — Part A already split, use it as given.
- Scaling and imputation go inside Pipelines.
- `random_state=42` everywhere.
- Run `python -m unittest discover -s tests` before committing.
- No number in a slide, doc or markdown cell that did not come from executed code.
