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

### A's sprints

| | Sprint | Status | Blocked by |
|---|---|---|---|
| **A1** | Pipeline, quality audit, `games.parquet` | **done** — PR #1 | — |
| **A2** | Feature–target EDA: distributions, correlation structure, which of the 14 carry signal | next | — |
| **A3** | Q4 transfer harness — train-on-Diamond / test-on-Challenger, taking any fitted estimator | next | — |
| **A4** | Q4 results with B's final model | Thursday evening | B's A1 model |
| **A5** | A's slides (1, 2, 3, 4, 11) and report sections | Thursday night | A2–A4 |

**A1 — done.** `load.py`, `clean.py`, `features.py`, `viz.py`, notebook 01, four figures,
D01–D10, V01–V04. 33,755 games, 14 features, zero missing values.

**A2 — feature–target EDA.** The collapse proved the 14 features are linearly independent of
each other; it did not show which of them say anything about the result. Deliverable:
notebook 02 plus figures showing each feature's distribution, its relationship to `blue_wins`,
and the residual correlation structure. This also hands B a head start on feature selection —
but B decides what the model uses; A only describes what is there.

**A3 — transfer harness.** `src/lol/transfer.py`: a function taking any fitted sklearn
estimator, training on Diamond, evaluating on Challenger, and reporting the metric deltas.
Built before B's model exists so A4 is a swap, not a build. Notebook 06 is scaffolded with a
placeholder estimator so it runs end to end from the start.

**A4 — Q4 results.** Swap in B's chosen model. The question: does a model fit on Diamond hold
at Challenger, and do the same features matter at both skill levels?

**A5 — communication.** Slides 1–4 and 11 per the team doc, plus A's report sections.

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
