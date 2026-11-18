# CLAUDE.md — LoL Early-Game Project (CDS3004 Data Mining, Lingnan University)

Read by Claude Code at the start of every session. Follow it strictly.

## 1. Project context

In League of Legends ranked play, teams often claim a match is "already decided" by the 10-minute
mark. This project tests that claim on data, and asks whether it holds the same way at different
skill levels.

**Central question:** How much of a ranked match outcome is already determined at 10 minutes, which
early advantages actually carry that signal, and does the answer change with player skill?

| ID | Business question | Technique | Decision it enables |
|---|---|---|---|
| Q1 (main) | Can the 10-minute state predict the winner, and which advantages matter? | Classification (Decision Tree, KNN, SVM) | What coaching and draft guidance should prioritise early |
| Q2 | What game archetypes exist at 10 minutes? | Clustering (K-means vs Hierarchical vs DBSCAN) | Describing match state beyond "ahead / behind" |
| Q3 | Which *combinations* of early conditions precede a win? | Association rules (Apriori, consequent = `blue_wins`) | Interpretable rules, cross-checked against the tree |
| Q4 (transfer) | Does a model fit on Diamond hold at Challenger? | Apply the Q1 model across tiers | Whether early-game advice generalises across skill |

Q2's archetype label can be used as a feature in Q1. Q3's rules validate what the tree learned.
Q4 is the project's differentiator and supplies the rubric's "decision making on new examples".

**Constraints that must be respected and discussed in the report:**
- **Redundancy, not leakage, is the main data trap.** Roughly half the 38 team features are exact
  mirrors (`blueGoldDiff` = −`redGoldDiff`, `blueDeaths` = `redKills`, `blueKills` = `redDeaths`),
  and several more are deterministic restatements (`blueCSPerMin` = `blueTotalMinionsKilled`/10;
  `blueGoldPerMin` likewise; `blueAvgLevel` is near-deterministic in `blueTotalExperience`).
  Quantifying this collapse is a graded deliverable, not a footnote.
- **Balanced classes.** `blueWins` is close to 50/50, so accuracy is readable here — but still
  report precision, recall, F1 and ROC-AUC, and show the confusion matrix.
- **Observational data.** Correlation at 10 minutes is not causation. A gold lead predicting a win
  does not prove that chasing gold early causes wins.
- **Scope.** Ranked solo queue, high Diamond to low Master (plus the Challenger tier), one patch
  window, one region. Do not generalise beyond it.

**Alternatives considered and rejected** (mention in the report): predicting from post-10-minute
state (trivially easy, decides nothing anyone can act on), champion-level draft analysis (the data
has no champion identifiers), and regression on game duration (a different question).

## 2. Data

Two Kaggle datasets, both in `data/raw/`:

| Tier | Source | File |
|---|---|---|
| Diamond–Master | `bobbyscience/league-of-legends-diamond-ranked-games-10-min` | `high_diamond_ranked_10min.csv` |
| Challenger | `gyejr95/league-of-legends-challenger-rank-game10min15min` | the **10-minute** file only |

- `data/` is git-ignored. Never commit raw or processed data.
- If `data/raw/` has no LoL CSV, STOP and ask the human to download it. Never ask for or handle
  Kaggle credentials.
- Every fact about shape, columns and dtypes must be **measured from the file**, never assumed from
  the Kaggle description. The two tiers were built by different authors and their schemas differ.

## 3. Non-negotiable rules for the agent

1. **Plan before code.** At the start of each workstream, inspect the current state, propose a plan,
   and wait for human approval before writing code.
2. **Validity rules:**
   - `gameId` is an identifier, never a feature. Drop it before any model.
   - **Only 10-minute data.** The Challenger source ships a 15-minute file as well. Merging or
     using any 15-minute column would mix two different decision points and invalidate Q1.
   - **Deduplicate across tiers before splitting.** A `gameId` present in both files would put the
     same match in train and test.
   - All imputation, scaling and encoding happen **inside** sklearn Pipelines, so nothing is fit
     outside the CV folds.
   - Mirror columns must be removed or reduced to differences before interpreting any feature
     importance. Keeping both sides double-counts and makes the importances unreadable.
3. **Reproducibility:** `random_state=42` everywhere. Pin dependencies in `requirements.txt`.
   Notebooks must run top-to-bottom ("Run All") without errors.
4. **Course techniques only:** scikit-learn and mlxtend, limited to what the course covered —
   preprocessing, discretization, similarity and distance, PCA, decision trees, KNN, SVM, model
   evaluation and cross-validation, K-means, hierarchical clustering, DBSCAN, Apriori. Note that
   **Naive Bayes was not taught in this course** (it appears once across all ten lectures), so do
   not include it. Anything beyond the syllabus must be justified in `docs/decisions.md`.
5. **Reusable logic goes in `src/lol/`.** Notebooks import from there and carry narrative, results
   and figures.
6. **Log decisions:** every non-trivial choice (filters, thresholds, K, minsup, features dropped)
   gets one entry in `docs/decisions.md` (date, decision, alternatives, why).
7. **Log validation:** whenever the human or the agent catches an agent mistake (a wrong join, a
   bad metric, a hallucinated column, an invented number), add an entry to
   `docs/agent_validation.md`. The course grades this.
8. **Commits:** small and descriptive (`feat(features): tier-merged 10-minute table`). Each member
   commits their own work from their own GitHub account. Commit only when the human asks. Never
   rewrite history.
9. **Figures:** save to `reports/figures/` as PNG, 150 dpi, labelled axes, a title, readable fonts.
   They go straight into Friday's slides.
10. **Never invent numbers** in markdown cells, docs or slides. Every number cited must come from
    executed code output.

## 4. Repository structure

```
├── CLAUDE.md
├── README.md
├── SPRINTS.md
├── requirements.txt
├── data/
│   ├── raw/             # git-ignored, the LoL CSVs
│   └── processed/       # git-ignored, parquet outputs
├── src/lol/
│   ├── __init__.py
│   ├── load.py          # read CSVs, dtypes, schema reconciliation across tiers
│   ├── clean.py         # dedup, outlier policy, tier merge
│   ├── features.py      # redundancy audit, difference features, archetype join
│   ├── segments.py      # Q2 clustering helpers (fit/predict)
│   ├── rules.py         # discretization + Apriori helpers
│   ├── evaluate.py      # CV, metrics, ROC, model comparison
│   └── viz.py           # shared plotting style
├── notebooks/
│   ├── 01_data_quality.ipynb
│   ├── 02_features.ipynb
│   ├── 03_classification.ipynb
│   ├── 04_clustering.ipynb
│   ├── 05_association_rules.ipynb
│   └── 06_tier_transfer.ipynb
├── reports/
│   ├── figures/
│   └── report_outline.md
└── docs/
    ├── decisions.md
    ├── agent_validation.md
    └── trajectories/    # each member's exported agent history
```

## 5. Team ownership

Each member runs **their own** Claude Code sessions, commits from **their own** GitHub account, and
exports their own trajectory to `docs/trajectories/<name>/`. Workload is graded from commits and
trajectories (Special Tutorial, slide 9).

| Member | Owns | Files |
|---|---|---|
| **A** | Data, EDA, visualization, Q4 tier transfer | `load.py`, `clean.py`, `features.py`, `viz.py` · nb 01, 02, 06 |
| **B** | Q1 classification and its evaluation | `evaluate.py` · nb 03 |
| **C** | Q2 clustering + Q3 association rules | `segments.py`, `rules.py` · nb 04, 05 |

Shared files (`docs/decisions.md`, `docs/agent_validation.md`) are **append-only**: add at the
bottom, pull right before you write, and if git reports a conflict there, keep both sides.

## 6. Deadlines

- **Presentation: Friday 20 Nov 2026.** Slides must be built from executed figures only.
- **Project materials (code, report, trajectories): 23:59 on 6 Dec 2026.** Submitted as one ZIP
  with a README containing a **public** GitHub repo URL, the code folder, the report as PDF with
  each member's contributions listed, and each member's agent-use trajectory.
