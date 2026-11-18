# LoL Early-Game — CDS3004 Data Mining (Lingnan University)

**How much of a ranked League of Legends match is already decided at 10 minutes — and does the
answer change with skill level?**

Players routinely claim a game is "over" by the 10-minute mark. We test that on ~10k Diamond–Master
ranked games plus a Challenger tier, and ask whether the same early advantages carry the same
weight at different skill levels.

| ID | Question | Technique |
|---|---|---|
| Q1 | Can the 10-minute state predict the winner, and which advantages matter? | Classification (Decision Tree, KNN, SVM) |
| Q2 | What game archetypes exist at 10 minutes? | Clustering (K-means / Hierarchical / DBSCAN) |
| Q3 | Which combinations of early conditions precede a win? | Association rules (Apriori) |
| Q4 | Does a model fit on Diamond hold at Challenger? | Cross-tier application |

> Status: scaffolding. Written properly once the analysis is done.

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then download both datasets from Kaggle into `data/raw/` (`data/` is git-ignored — data is never
committed):

- [`bobbyscience/league-of-legends-diamond-ranked-games-10-min`](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min)
- [`gyejr95/league-of-legends-challenger-rank-game10min15min`](https://www.kaggle.com/datasets/gyejr95/league-of-legends-challenger-rank-game10min15min) — **the 10-minute file only**

Run the notebooks in order: `01_data_quality` → `02_features` → `03_classification` →
`04_clustering` → `05_association_rules` → `06_tier_transfer`.

## Repo map

Reusable logic lives in `src/lol/`; notebooks hold narrative and figures. Decisions are logged in
`docs/decisions.md`, agent mistakes in `docs/agent_validation.md`, and each member's agent-use
history in `docs/trajectories/<name>/`.

See `CLAUDE.md` for the project rules and `SPRINTS.md` for who owns what.

## Scope

Ranked solo queue, high Diamond to low Master plus Challenger, one patch window, one region.
Observational data — correlation at 10 minutes is not causation. Conclusions are not generalised
beyond this scope.

## Team

| Member | Owns |
|---|---|
| A | Data, EDA, visualization, Q4 tier transfer |
| B | Q1 classification and evaluation |
| C | Q2 clustering, Q3 association rules |

Contributions are listed in the final report. Workload is evidenced by per-member commits and
agent trajectories.
