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

### B's sprints  ·  owner: Andrea (own machine, Codex)

#### Getting set up

The repo has no data in it — `data/` is git-ignored and always will be. You regenerate the
modelling table locally.

```
git clone https://github.com/agj2206/Predicting-League-of-Legends-match
cd Predicting-League-of-Legends-match
git checkout -b model-training origin/main
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then download both CSVs into `data/raw/` (links in README.md) and run
`notebooks/01_data_quality.ipynb` top to bottom. Its last cell writes
`data/processed/games.parquet` — **33,755 games, 14 features, zero missing values.** That file
is your input; you should not need to touch `load.py`, `clean.py` or `features.py`.

Check `git config user.email` matches the email on your GitHub account before your first
commit, or GitHub attributes your work to nobody.

**Your trajectory.** Export your Codex conversation history at the end of each session and keep
it locally — it goes in the final submission ZIP, not in the repo (`*.jsonl` and friends are
git-ignored on purpose). Together with your commits it is how your individual workload is
graded, so export as you go rather than on 6 December.

#### What to build

Q1 trains and tests **within Diamond only** (9,879 of the 33,755 games — filter `tier ==
"diamond"`). Challenger is held out entirely as Q4's transfer target. That is decision D10;
read `docs/decisions.md` D09–D13 and `notebooks/02_feature_eda.ipynb` before starting, because
A already measured several things you would otherwise rediscover.

| | Sprint | Deliverable |
|---|---|---|
| **B1** | `src/lol/evaluate.py` — nested CV harness, metrics, ROC | module + commit |
| **B2** | Feature selection, three ways, and where they disagree | notebook 03 section |
| **B3** | Decision Tree, KNN, SVM, all inside Pipelines | notebook 03 section |
| **B4** | Compare, choose, evaluate once on held-out | comparison table + figures |
| **B5** | Interpretation: tree plot, importances, plain language | figures + write-up |

**B1 — nested CV.** Outer 5-fold stratified for an honest estimate, inner 3-fold for
hyperparameters. If hyperparameters are chosen on the same folds that then report the score,
the score is optimistic — the choice has already seen the data it is graded on. The rubric
names nested cross-validation explicitly, so a plain `cross_val_score` leaves marks on the
table. Report precision, recall, F1 and ROC-AUC as mean ± std across the outer folds.

**B2 — feature selection.** Correlation, mutual information, tree importance. Report where they
disagree, not just the union — see D13, where they disagreed sharply and the interesting
feature was the one two methods called worthless.

**B3 — the three models.** No Naive Bayes (D03: not taught in this course). All preprocessing
inside sklearn Pipelines so scaling is fit per fold, never on the full data — fitting a scaler
before splitting leaks the test fold's distribution into training. KNN and SVM need scaling;
the tree does not, and the Pipeline makes that difference explicit rather than hidden.

**B4 — choose once, test once.** Retrain the winner on the full training split and score it a
single time on the held-out set. Scoring a test set repeatedly and keeping the best result is
how a test set stops being a test set.

**B5 — interpretation, with A's caveat attached.** `kills_diff` and `total_gold_diff` correlate
at 0.92, so whichever the tree splits on first will look far more important than the other. Any
claim like "gold matters more than kills" has to survive that.

#### Timings, already measured on this data

Nothing here is slow. One fold of Diamond is 7,903 rows.

| estimator | one fit | nested CV estimate |
|---|---|---|
| Decision Tree (full nested CV, 6-point grid) | — | **2.7 s total** |
| KNN, k=25 | 0.05 s | well under a minute |
| SVC, RBF kernel | 0.91 s | a few minutes |

**The RBF SVM is affordable — use it.** The `LinearSVC` fallback in CLAUDE.md is not needed and
would be a worse model for no gain.

#### Handy numbers from A's work

* Diamond: 9,879 games, blue wins 0.4990 — near balanced, so accuracy is readable here, but
  never report it alone.
* A depth-limited tree under nested CV reaches **ROC-AUC ≈ 0.80** on Diamond. If a model comes
  in far above that, suspect a leak before celebrating.
* The strongest single feature is `total_gold_diff` (correlation 0.511 with the result on
  Diamond), followed by `kills_diff`, `avg_level_diff` and `assists_diff` — all measuring the
  same underlying lead, which is why they correlate so tightly.

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
