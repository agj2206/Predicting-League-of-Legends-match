# Decision log

Every non-trivial choice (filters, thresholds, K, minsup, features dropped) gets one entry
(CLAUDE.md §3.6). Newest at the bottom. **Append only** — never edit someone else's entry.
Never cite a number that did not come from executed code.

Format:

```
### D<nn> — <short title>
- **Date:** YYYY-MM-DD · **Author:** <member>
- **Decision:** what was chosen.
- **Alternatives:** what else was on the table.
- **Why:** the reason, with the evidence behind it.
- **Affects:** which questions / files depend on this.
```

---

### D01 — Project topic: predicting the winner from the 10-minute state
- **Date:** 2026-11-18 · **Author:** A
- **Decision:** the group project asks whether the winning team can be predicted from statistics
  available after ten minutes, on the Kaggle Diamond–Master dataset.
- **Alternatives:** (a) an Olist e-commerce review-risk project, taken to a completed data-quality
  audit before being dropped; (b) the "Name That LLM" open pool (212k passages) — rejected because
  TF-IDF and bag-of-words are not taught in this course, so the single most important
  transformation would have been off-syllabus, and the tabular data-quality work the rubric grades
  barely exists in it.
- **Why:** team decision, taken with the 20 Nov presentation date known.
- **Affects:** everything.

### D02 — One tier only: Diamond–Master
- **Date:** 2026-11-19 · **Author:** A (team decision)
- **Decision:** use `high_diamond_ranked_10min.csv` alone. No second rank tier, and no cross-tier
  transfer study.
- **Alternatives:** add the Challenger 10-minute file for ~33,700 games and a Q4 asking whether a
  Diamond-fit model holds at Challenger. That path was built and then dropped.
- **Why:** scope, with one working day before the presentation. The cost is real and should be
  stated in the report rather than hidden: the rubric rewards "collect dataset**s**" and "load
  **large** datasets", and 9,879 rows is not large. What was gained is a Part A that is finished,
  tested and internally consistent.
- **Affects:** everything. The Challenger work is archived outside the repository.

### D03 — Naive Bayes is not in the model set
- **Date:** 2026-11-18 · **Author:** A
- **Decision:** Q1 compares Decision Tree, KNN and SVM.
- **Alternatives:** include Naive Bayes as a fourth baseline.
- **Why:** measured, not assumed — "naive bayes"/"bayes" returns one hit across all ten lecture
  decks, a passing mention in L7. L4–L5 are decision trees, L6 is evaluation, L7 is KNN and SVM.
  CLAUDE.md §3.4 limits us to course techniques.
- **Affects:** Q1, notebook 02.

### D04 — The split happens before any statistic is computed
- **Date:** 2026-11-19 · **Author:** A
- **Decision:** 80/20 stratified split, `random_state=42`, `gameId` sorted ascending beforehand.
  No scaler, imputer, selector or model is fit before the split, and every descriptive statistic
  in Part A — correlations, extreme-value summaries, all figures but one — uses training data only.
- **Alternatives:** describe the full dataset first and split later, which is the more common
  order in tutorials.
- **Why:** describing the test set is a mild form of looking at it. Decisions made after seeing
  test-set statistics are no longer independent of it, and the final held-out number stops being
  honest. The one exception is the class-distribution figure, which is explicitly labelled as
  full-source.
- **Affects:** every question. B and C must use the split as given and never recompute it.

### D05 — Twelve predictors, all blue-minus-red differences
- **Date:** 2026-11-19 · **Author:** A
- **Decision:** retain `blueGoldDiff` and `blueExperienceDiff` as published, and engineer ten more
  differences: kills, assists, minions, jungle minions, wards placed, wards destroyed, first blood,
  dragons, heralds, towers destroyed. Forty source columns become twelve predictors.
- **Alternatives:** keep both teams' absolute values (40 columns), or keep blue-side values only.
- **Why:** the published columns restate each other exactly — `blueDeaths` equals `redKills`,
  `blueCSPerMin` equals `blueTotalMinionsKilled`/10, `blueGoldDiff` equals the negative of
  `redGoldDiff`. Keeping both sides doubles the column count without adding information and makes
  feature importances unreadable. Per-column exclusion reasons are in `feature_decisions()` and
  the notebook; 13 arithmetic consistency checks confirm the identities hold on all 9,879 rows.
- **Affects:** Q1, Q2, Q3.

### D06 — Ward outliers are retained, not clipped
- **Date:** 2026-11-19 · **Author:** A
- **Decision:** no outlier filtering, clipping or replacement anywhere in Part A.
- **Alternatives:** winsorise the ward columns at a physical bound, or drop the affected rows.
- **Why:** ward counts reach 276 in training, and hundreds of wards at ten minutes is implausible
  — but the CSV cannot establish whether they are real or extraction artefacts, and an IQR flag is
  not evidence of an error. Removing data on a suspicion is a larger intervention than keeping it
  with a caveat. Gold, combat, experience and minion extremes are all plausible for real games.
- **Affects:** every question. `wardsPlacedDiff` is the weakest predictor (0.016) and should be
  treated as the least trustworthy feature in the set. This belongs in the report's limitations.

### D07 — One research question: clustering and association rules are dropped
- **Date:** 2026-11-19 · **Author:** team decision
- **Decision:** the project answers a single question — can the winner be predicted from the
  first ten minutes? Supervised binary classification only. No K-means, hierarchical clustering,
  DBSCAN or Apriori. `mlxtend` is removed from `requirements.txt`.
- **Alternatives:** keep two secondary questions — clustering for "game archetypes at ten
  minutes" and association rules for "which combinations of early conditions precede a win" —
  which would have exercised more of the syllabus.
- **Why:** one question answered properly beats three answered superficially, and the work
  already done on this one is deep: a validated split taken before any statistic, a 40-to-12
  feature reduction with arithmetic proofs, nested cross-validation against a baseline, and a
  held-out set still untouched. Spreading the remaining time across clustering and rule mining
  would have thinned all three.
- **The cost, stated rather than hidden:** the rubric's LO3 row rewards applying different data
  mining techniques, and we now demonstrate one family rather than three. The report should name
  this as a deliberate scope choice and point at the depth bought with it, not leave it to be
  read as an omission.
- **Affects:** `README.md`, `CLAUDE.md`, `SPRINTS.md`, `requirements.txt`. Stage 3 becomes model
  evaluation rather than clustering and rules.
