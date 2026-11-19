# CLAUDE.md — LoL Early-Game Project (CDS3004 Data Mining, Lingnan University)

Read at the start of every session. Follow it strictly.

## 1. Project context

Players claim a ranked League of Legends match is "already decided" by the 10-minute mark.
This project tests that on data.

**Central question:** can we predict the winning team using only statistics available after the
first 10 minutes?

| ID | Question | Technique | Owner |
|---|---|---|---|
| Q1 (main) | Can the 10-minute state predict the winner, and which advantages matter? | Classification — Decision Tree, KNN, SVM | B |
| Q2 | What game archetypes exist at 10 minutes? | Clustering — K-means vs Hierarchical vs DBSCAN | C |
| Q3 | Which combinations of early conditions precede a win? | Association rules — Apriori | C |

**Constraints that must be respected and discussed in the report:**
- **Redundancy is the main data trap.** Many of the 40 published columns restate each other
  (`blueGoldDiff` = −`redGoldDiff`, `blueDeaths` = `redKills`, `blueCSPerMin` =
  `blueTotalMinionsKilled`/10). Part A reduced them to 12 predictors, all blue-minus-red
  differences. Quantifying that collapse is a graded deliverable, not a footnote.
- **Balanced classes.** Blue wins 49.904% of games, so accuracy is readable here — the only
  condition under which it means anything. Still never report it alone.
- **Observational data.** A gold lead predicting a win is not evidence that chasing gold early
  causes wins.
- **Scope.** Ranked solo queue, high Diamond to low Master, one patch window, one region.

**Alternatives considered and rejected** (mention in the report): predicting from post-10-minute
state (trivially easy, decides nothing anyone can act on), adding a second rank tier for a
cross-tier transfer study (dropped on 2026-11-19 for scope — see D02), champion-level draft
analysis (the data has no champion identifiers).

## 2. Data

One Kaggle dataset: `bobbyscience/league-of-legends-diamond-ranked-games-10-min`, file
`high_diamond_ranked_10min.csv` — **9,879 matches, 40 columns**, SHA-256
`82b2e23279097c2dac616e966eb2911711c428d41b6382b4cdb15da4892f9ca6`.

- `data/` is git-ignored. Never commit raw or derived data.
- If the CSV is absent, STOP and ask the human to download it. Never ask for or handle Kaggle
  credentials.
- Every claim about shape, columns or dtypes must be **measured from the file**, never assumed
  from the Kaggle description.

## 3. Non-negotiable rules for the agent

1. **Plan before code.** Propose a plan and wait for human approval before writing code.
2. **Validity rules:**
   - `gameId` is an identifier, never a feature.
   - The train/test split happens **before** any statistic, scaler, imputer or selector is fit.
     Part A already did this; descriptive statistics are computed on training data only.
   - All preprocessing goes **inside** sklearn Pipelines, so nothing is fit outside the CV folds.
   - Mirror columns are already removed. Do not reintroduce both sides of a difference.
3. **Reproducibility:** `random_state=42` everywhere. Pin dependencies in `requirements.txt`.
   Notebooks must run top-to-bottom without errors.
4. **Course techniques only:** scikit-learn and mlxtend, limited to what the course covered —
   preprocessing, discretization, similarity and distance, PCA, decision trees, KNN, SVM, model
   evaluation and cross-validation, K-means, hierarchical clustering, DBSCAN, Apriori.
   **Naive Bayes was not taught** (one passing mention across ten lecture decks) — do not use it.
   Anything beyond the syllabus must be justified in `docs/decisions.md`.
5. **Reusable logic goes in `src/`.** Notebooks carry narrative, results and figures.
6. **Log decisions:** every non-trivial choice gets one entry in `docs/decisions.md`.
7. **Log validation:** every caught agent mistake gets an entry in `docs/agent_validation.md`.
   The course grades this; an empty log reads as "we never checked".
8. **Tests:** `tests/` holds the data-contract and split-reproducibility checks. Run them before
   committing: `python -m unittest discover -s tests`.
9. **Commits:** small and descriptive. Commit only when the human asks. Never rewrite history.
10. **Figures:** `reports/figures/` as PNG, labelled axes, titles, readable fonts — they go into
    the slides.
11. **Never invent numbers.** Every number cited must come from executed code output.

## 4. Repository structure

```
├── CLAUDE.md · SPRINTS.md · README.md · requirements.txt
├── data/                 # git-ignored; the Kaggle CSV goes in data/raw/
├── src/
│   ├── __init__.py
│   └── lol_preprocessing.py   # Part A: validation, feature policy, reproducible split
├── notebooks/
│   ├── 01_data_preprocessing.ipynb   # Part A (done)
│   ├── 02_classification.ipynb       # Part B — Q1
│   ├── 03_clustering.ipynb           # Part C — Q2
│   └── 04_association_rules.ipynb    # Part C — Q3
├── scripts/execute_notebook.py
├── tests/test_preprocessing.py
├── reports/
│   ├── figures/
│   └── part_a_findings.md
└── docs/
    ├── decisions.md
    ├── agent_validation.md
    └── trajectories/     # local only, git-ignored; goes in the submission ZIP
```

## 5. Team

| Member | Owns | Status |
|---|---|---|
| **A** | Data preprocessing, EDA, figures | **done** |
| **B** | Q1 classification and evaluation | next |
| **C** | Q2 clustering, Q3 association rules | next |

Part A hands over `prepare_data()` from `src/lol_preprocessing.py`, returning
`X_train` (7,903 × 12), `X_test` (1,976 × 12) and aligned labels. B and C call it rather than
reading a stored file, so nobody works from a stale artefact.

Workload is evidenced by commits and per-member agent trajectories (Special Tutorial, slide 9).
Shared files (`docs/decisions.md`, `docs/agent_validation.md`) are **append-only**.

## 6. Deadlines

- **Presentation: Friday 20 Nov 2026.** Slides built from executed figures only.
- **Project materials: 23:59 on 6 Dec 2026** — one ZIP with a README containing the public GitHub
  repo URL, the code, the report as PDF with each member's contributions, and each member's
  agent-use trajectory.
