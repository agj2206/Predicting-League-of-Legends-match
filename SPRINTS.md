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

### B's sprints  ·  owner: Andrea

Q1 trains and tests **within Diamond only** (9,879 games, 14 features) — decision D10.
Challenger is held out entirely as Q4's transfer target. Read `docs/decisions.md` D09–D12 and
notebook 02 before starting; A already measured things you would otherwise rediscover.

| | Sprint | Deliverable |
|---|---|---|
| **B1** | `src/lol/evaluate.py` — nested CV harness, metrics, ROC | module + commit |
| **B2** | Feature selection, three ways, and where they disagree | notebook 03 section |
| **B3** | Decision Tree, KNN, SVM, all inside Pipelines | notebook 03 section |
| **B4** | Compare, choose, evaluate once on held-out | comparison table + figures |
| **B5** | Interpretation: tree plot, importances, plain language | figures + write-up |

**B1 — nested CV.** Outer 5-fold stratified for an honest estimate, inner 3-fold for
hyperparameters. The rubric names nested cross-validation explicitly, so a plain
`cross_val_score` leaves marks on the table. Report precision, recall, F1 and ROC-AUC as
mean ± std across the outer folds.

**B2 — feature selection.** Correlation filter, mutual information, tree importance. Report
where they disagree rather than just the union. A2 predicts every method drops
`total_gold_both` and `total_kills_both` (correlation 0.008 and 0.005 with the result) — if one
method keeps them, that method is the suspect, not the features.

**B3 — the three models.** No Naive Bayes (D03: it was not taught in this course). All
preprocessing inside sklearn Pipelines so scaling is fit per fold and never on the full data.
KNN and SVM need scaling; the tree does not — the Pipeline makes that difference explicit
instead of hidden.

**B4 — choose once, test once.** Retrain the winner on the full training split and score it a
single time on the held-out set. Scoring repeatedly and picking the best is how a test set
stops being a test set.

**B5 — interpretation, with A2's caveat attached.** `kills_diff` and `total_gold_diff` correlate
at 0.92, so whichever the tree splits on first will look far more important than the other.
Any claim like "gold matters more than kills" has to survive that.

**Known risk.** RBF SVM under nested CV on ~7,900 rows is roughly 90 fits — minutes, not
seconds. Keep the grid small. If it drags, `LinearSVC` is an allowed fallback (CLAUDE.md §3.4)
provided the swap is logged in `docs/decisions.md`.

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
