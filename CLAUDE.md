# CLAUDE.md — LoL Early-Game Project (CDS3004 Data Mining, Lingnan University)

Read at the start of every session. Follow it strictly.

## 1. Project context

Players claim a ranked League of Legends match is "already decided" by the 10-minute mark.
This project tests that on data.

**Research question — one, deliberately:**

> Can we predict the winning team using only statistics available after the first 10 minutes?

The target is `blueWins` (1 = Blue wins, 0 = Red wins). This is a supervised binary
classification problem and the whole project serves it. We do **not** run clustering or
association-rule mining — see decision D07. Depth on one question beats breadth across three.

**Constraints that must be respected and discussed in the report:**
- **Redundancy is the main data trap.** Many of the 40 published columns restate each other
  (`blueGoldDiff` = −`redGoldDiff`, `blueDeaths` = `redKills`, `blueCSPerMin` =
  `blueTotalMinionsKilled`/10). Part A reduced them to 12 predictors, all blue-minus-red
  differences. Quantifying that collapse is a graded deliverable, not a footnote.
- **Balanced classes.** Blue wins 49.904% of games, so accuracy is readable here — the only
  condition under which it means anything. Still never report it alone.
- **Observational data.** A gold lead predicting a win is not evidence that chasing gold early
  causes wins.
- **Scope, and what we cannot claim.** 9,879 ranked solo-queue matches, described by the
  dataset's author as high Diamond to low Master. The CSV has no patch, date or region column,
  so we do **not** assert that the matches share a patch window or a server — that was an
  unverified claim and it was removed (V07). Generalisation beyond this sample is unestablished.

**Alternatives considered and rejected** (mention in the report): predicting from post-10-minute
state (trivially easy, decides nothing anyone can act on); a second rank tier for a cross-tier
transfer study (D02); clustering and association rules as secondary questions (D07);
champion-level draft analysis (the data has no champion identifiers).

## 2. Data

One Kaggle dataset: `bobbyscience/league-of-legends-diamond-ranked-games-10-min`, file
`high_diamond_ranked_10min.csv` — **9,879 matches, 40 columns**, SHA-256
`82b2e23279097c2dac616e966eb2911711c428d41b6382b4cdb15da4892f9ca6`.

- `data/` is git-ignored. Never commit raw or derived data.
- If the CSV is absent, STOP and ask the human to download it. Never ask for or handle Kaggle
  credentials.
- Every claim about shape, columns or dtypes must be **measured from the file**.

## 3. Non-negotiable rules for the agent

1. **Plan before code.** Propose a plan and wait for human approval before writing code.
2. **Validity rules:**
   - `gameId` is an identifier, never a feature.
   - The train/test split happened in Part A, **before** any statistic, scaler, imputer or
     selector was fit. Use it as given. Never recompute it.
   - **The held-out test set is untouched so far** (`held_out_test_used: false`). It is spent
     once, in Model Evaluation, and never revisited to pick a better number.
   - All preprocessing goes **inside** sklearn Pipelines, so nothing is fit outside the CV folds.
   - Mirror columns are already removed. Do not reintroduce both sides of a difference.
3. **Reproducibility:** `random_state=42` everywhere. Pin dependencies. Notebooks must run
   top-to-bottom without errors.
4. **Course techniques only:** scikit-learn, limited to what the course covered — preprocessing,
   discretization, similarity and distance, PCA, decision trees, KNN, SVM, model evaluation and
   cross-validation. **Naive Bayes was not taught** (one passing mention across ten lecture
   decks) — do not use it. Anything beyond the syllabus must be justified in `docs/decisions.md`.
5. **Reusable logic goes in `src/`.** Notebooks carry narrative, results and figures.
6. **Log decisions:** every non-trivial choice gets one entry in `docs/decisions.md`.
7. **Log validation:** every caught agent mistake gets an entry in `docs/agent_validation.md`.
   The course grades this; an empty log reads as "we never checked".
8. **Tests:** run them before committing — `python -m unittest discover -s tests`.
9. **Commits:** small and descriptive. Commit only when the human asks. Never rewrite history.
10. **Figures:** `reports/figures/` as PNG, labelled axes, titles, readable fonts.
11. **Never invent numbers.** Every number cited must come from executed code output.

## 4. Repository structure

```
├── CLAUDE.md · SPRINTS.md · README.md · requirements.txt
├── data/                     # git-ignored; the Kaggle CSV goes in data/raw/
├── src/
│   ├── lol_preprocessing.py      # Part A: validation, feature policy, split
│   └── lol_model_training.py     # Part B: nested CV, tuning, model selection
├── notebooks/
│   ├── 01_data_preprocessing.ipynb   # Part A (done)
│   ├── 02_model_training.ipynb       # Part B (done)
│   └── 03_model_evaluation.ipynb     # Part C (next)
├── models/part_b/<run_id>/       # fitted Pipelines, git-ignored if large
├── scripts/
├── tests/
├── reports/
│   ├── figures/
│   ├── model_training/<run_id>/  # CV results as JSON
│   ├── part_a_findings.md
│   ├── part_b_findings.md
│   └── model_training_handoff.md
└── docs/
    ├── decisions.md
    ├── agent_validation.md
    └── trajectories/             # local only, git-ignored; goes in the submission ZIP
```

## 5. Team

| Member | Owns | Status |
|---|---|---|
| **1 — Cesar** | Data preprocessing, validation, split, EDA | **done** |
| **2 — Andrea** | Model training: Decision Tree, KNN, SVM, nested CV, hyperparameter tuning | **done** |
| **3 — Christian** | Model evaluation: confusion matrix, accuracy, precision, recall, F1, ROC-AUC, ROC curves, interpretation | next |

The three stages run in sequence and each hands the next a fixed artefact:

- **1 → 2:** `prepare_data()` returns `X_train` (7,903 × 12), `X_test` (1,976 × 12) and aligned
  labels. Called, never read from a stored file, so nobody works from a stale artefact.
- **2 → 3:** `load_final_model()` returns the selected fitted Pipeline; `load_run_results()`
  returns the nested-CV results. The selected family is frozen in `selection.json`.

Workload is evidenced by commits and per-member agent trajectories (Special Tutorial, slide 9).
Shared files (`docs/decisions.md`, `docs/agent_validation.md`) are **append-only**.

## 6. Deadlines

- **Presentation: Friday 20 Nov 2026.** Slides built from executed figures only.
- **Project materials: 23:59 on 6 Dec 2026** — one ZIP with a README containing the public GitHub
  repo URL, the code, the report as PDF with each member's contributions, and each member's
  agent-use trajectory.
