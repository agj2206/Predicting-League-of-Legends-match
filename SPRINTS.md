# Workstreams — two days to the presentation

Full plan, including commit messages and the Friday slide split, is in the team doc.
This file is the offline copy of who does what.

**Presentation: Friday 20 Nov 2026. Project materials (code, report, trajectories): 6 Dec 2026.**

Dependency: A ships a minimal processed table tonight. B and C build against it from Thursday
morning. B and C are not blocked tonight — reading the brief and drafting their approach are real
commits.

---

## A — data, EDA, tier transfer

Owns `load.py`, `clean.py`, `features.py`, `viz.py` · notebooks 01, 02, 06

1. Load both tiers. **Measure** the schemas; do not assume they match.
2. Reconcile columns, merge, dedupe on `gameId` across tiers, record what differed.
3. Redundancy audit: quantify how many of the 38 features are mirrors or arithmetic
   restatements of each other. This is the headline finding — measure it, do not estimate it.
4. Quality: ranges, implausible values, outlier policy. Check `blueWardsPlaced` in particular.
5. EDA figures for the slides, 150 dpi, into `reports/figures/`.
6. Ship `data/processed/games.parquet` — minimal first, refined after.
7. Q4: apply B's final model across tiers and report what changes.

## B — Q1 classification and evaluation

Owns `evaluate.py` · notebook 03

1. Preprocessing inside sklearn Pipelines so nothing is fit outside the CV folds.
2. Feature selection: correlation filtering, mutual information, tree importance. Report what
   was removed and why.
3. Decision Tree, KNN, SVM. No Naive Bayes (see decision D03).
4. Cross-validation, stratified. Report precision, recall, F1 and ROC-AUC with mean ± std.
5. Pick a final model, retrain, evaluate once on the held-out set.
6. Interpretation: depth-limited tree plot, feature importances, in plain language.

## C — Q2 clustering and Q3 association rules

Owns `segments.py`, `rules.py` · notebooks 04, 05

1. Scale and log-transform skewed features; explain why scaling is required for distance methods.
2. K-means: choose K by elbow **and** silhouette over K = 2..10.
3. Compare with Hierarchical (Ward, dendrogram on a sample) and DBSCAN (eps from a k-distance
   plot). Pick one, justify it.
4. Profile each cluster and give it a business name. These are the game archetypes.
5. Discretise: compare equal-width vs equal-frequency, choose one, justify it.
6. Apriori with `blue_wins` as the consequent. Justify minsup and minconf. Prune redundant rules.
7. Cross-check the top rules against B's tree splits. Agreements and disagreements both matter.

---

## Non-negotiables

See CLAUDE.md §3. The ones that break quietly:

- `gameId` is never a feature.
- **10-minute data only** — the Challenger source also ships a 15-minute file.
- Dedupe on `gameId` across tiers *before* splitting.
- Scaling and imputation go inside Pipelines.
- Drop mirror columns before reading feature importances.
- `random_state=42` everywhere.
- No number in a slide, doc or markdown cell that did not come from executed code.
