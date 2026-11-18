"""Turn the canonical table into modelling features, and measure the collapse.

The headline finding of this project is that the published 38 team columns carry
far less information than their count suggests. This module does two things:

* `audit_redundancy` — measures it. Every claim is a computed relationship over
  the real rows, never an assertion about what the columns "should" mean.
* `build_features` — applies it, representing each contested quantity once, as a
  signed blue-minus-red difference.

Why differences rather than both sides: for every concept the two sides are
mirrors, so `blue_x` and `red_x` together carry exactly one quantity's worth of
information plus the match total. The difference keeps the part that decides the
game and drops the duplication that makes feature importances unreadable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .clean import COLUMN_MAP

#: The concepts that exist in both tiers, so every feature below is cross-tier safe.
CONCEPTS = list(COLUMN_MAP)


def audit_redundancy(df: pd.DataFrame, cols: list[str], tol: float = 1e-12) -> pd.DataFrame:
    """Find pairs of columns that are exactly linearly dependent.

    Returns one row per dependent pair with the relationship found. Uses exact
    equality on the fitted line rather than a correlation threshold, so a pair is
    only reported when it genuinely carries no independent information.
    """
    found = []
    work = df[cols].astype("float64")
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            x, y = work[a].to_numpy(), work[b].to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 2 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
                continue
            slope = np.polyfit(x[ok], y[ok], 1)
            residual = np.max(np.abs(np.polyval(slope, x[ok]) - y[ok]))
            if residual <= tol * max(1.0, np.max(np.abs(y[ok]))):
                found.append({
                    "column_a": a, "column_b": b,
                    "relationship": f"{b} = {slope[0]:.6g}·{a} + {slope[1]:.6g}",
                    "max_residual": residual,
                })
    return pd.DataFrame(found)


def independent_set(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Greedily keep columns that add rank — the measured independent subset."""
    keep: list[str] = []
    for c in cols:
        trial = keep + [c]
        block = df[trial].astype("float64").dropna()
        if block.empty:
            continue
        centred = block - block.mean()
        if np.linalg.matrix_rank(centred.to_numpy(), tol=1e-8) == len(trial):
            keep = trial
    return keep


def build_features(games: pd.DataFrame) -> pd.DataFrame:
    """One signed difference per concept, plus the identifiers and the target."""
    out = pd.DataFrame({
        "game_id": games["game_id"],
        "tier": games["tier"],
        "blue_wins": games["blue_wins"],
    })

    for concept in CONCEPTS:
        blue, red = games[f"blue_{concept}"], games[f"red_{concept}"]
        out[f"{concept}_diff"] = blue - red

    # Totals that are genuinely shared rather than contested: how farmed the game
    # was overall, not who is ahead. These are not differences and are not mirrors.
    out["total_gold_both"] = games["blue_total_gold"] + games["red_total_gold"]
    out["total_kills_both"] = games["blue_kills"] + games["red_kills"]

    return out


#: Dropped because `audit_redundancy` proved it carries nothing new, not because
#: anyone judged it uninteresting: blue_deaths equals red_kills exactly in both
#: tiers, so deaths_diff is identically -kills_diff. Built by `build_features`
#: so the notebook can show the audit catching it, excluded from modelling here.
EXCLUDED_AFTER_AUDIT = {"deaths_diff"}


def feature_columns(df: pd.DataFrame, include_dependent: bool = False) -> list[str]:
    """Modelling columns only — never `game_id`, never `tier`, never the target.

    `tier` is deliberately excluded: letting a model see it would let it learn
    which file a row came from instead of what the early game says, and would
    make the Q4 transfer result meaningless.
    """
    drop = {"game_id", "tier", "blue_wins"}
    if not include_dependent:
        drop |= EXCLUDED_AFTER_AUDIT
    return [c for c in df.columns if c not in drop]
