"""Reconcile the two tiers onto one canonical schema.

The story this module implements, in order:

1.  The Challenger file is not purely a 10-minute snapshot. About 9.5% of its
    rows record a much later game state (D04, V02). Those rows are removed first,
    because several other anomalies turn out to be symptoms of the same cause
    (V03) and disappear once it is gone.
2.  Challenger carries 55 duplicate `gameId` values, and a `redWins` column that
    is exactly `1 - blueWins`. Both go.
3.  The two authors used different names for the same thirteen quantities. They
    are renamed to one canonical vocabulary (D06, D07).
4.  A residual ward tail survives in BOTH files, in otherwise-normal games. It is
    capped, not dropped, with one bound shared across tiers (D08).

Nothing here is tier-dependent except step 1, which has to be: only one of the
two files has the contamination.
"""

from __future__ import annotations

import pandas as pd

from . import load

# ---------------------------------------------------------------- D04 filter

#: Physical ceilings for a true 10-minute state. Chosen to sit inside the empty
#: gap between the two populations in the Challenger file (level p90 = 7.60,
#: p95 = 10.80; gold p90 = 21,101, p95 = 33,073), not to trim a legitimate tail.
#: The Diamond file, internally consistent, caps at level 8.0 and 23,701 gold.
MAX_AVG_LEVEL_AT_10MIN = 8.5
MAX_TEAM_GOLD_AT_10MIN = 26_000

#: Shared ward ceiling (D08). Five players over ten minutes cannot place this
#: many; the value is generous on purpose so the cap touches only the broken tail.
MAX_WARDS = 60


#: canonical name -> (Diamond column, Challenger column)
COLUMN_MAP = {
    "wards_placed":         ("WardsPlaced",              "WardPlaced"),
    "wards_destroyed":      ("WardsDestroyed",           "WardKills"),
    "first_blood":          ("FirstBlood",               "FirstBlood"),
    "kills":                ("Kills",                    "Kill"),
    "deaths":               ("Deaths",                   "Death"),
    "assists":              ("Assists",                  "Assist"),
    "dragons":              ("Dragons",                  "Dragon"),
    "heralds":              ("Heralds",                  "RiftHeralds"),
    "towers_destroyed":     ("TowersDestroyed",          "TowerKills"),
    "total_gold":           ("TotalGold",                "TotalGolds"),
    "avg_level":            ("AvgLevel",                 "AvgLevel"),
    "total_minions":        ("TotalMinionsKilled",       "TotalMinionKills"),
    "total_jungle_minions": ("TotalJungleMinionsKilled", "TotalJungleMinionKills"),
}

#: Present in Diamond only — Challenger has no experience columns at all.
#: Kept on Diamond rows, NaN on Challenger rows, and excluded from anything
#: that crosses tiers.
DIAMOND_ONLY = {"total_experience": "TotalExperience"}


def flag_not_ten_minutes(challenger: pd.DataFrame) -> pd.Series:
    """True where a Challenger row cannot be a 10-minute state (D04)."""
    return (
        (challenger["blueAvgLevel"] > MAX_AVG_LEVEL_AT_10MIN)
        | (challenger["redAvgLevel"] > MAX_AVG_LEVEL_AT_10MIN)
        | (challenger["blueTotalGolds"] > MAX_TEAM_GOLD_AT_10MIN)
        | (challenger["redTotalGolds"] > MAX_TEAM_GOLD_AT_10MIN)
        | (challenger["blueInhibitor"] > 0)
        | (challenger["redInhibitor"] > 0)
    )


def _to_canonical(df: pd.DataFrame, which: int, tier: str) -> pd.DataFrame:
    """Rename one file's columns to the canonical vocabulary.

    `which` is 0 for the Diamond spelling, 1 for the Challenger spelling.
    """
    out = pd.DataFrame({"game_id": df["gameId"], "blue_wins": df["blueWins"]})
    for canon, spellings in COLUMN_MAP.items():
        src = spellings[which]
        for side in ("blue", "red"):
            out[f"{side}_{canon}"] = df[f"{side}{src}"]
    for canon, src in DIAMOND_ONLY.items():
        for side in ("blue", "red"):
            out[f"{side}_{canon}"] = df[f"{side}{src}"] if which == 0 else pd.NA
    out["tier"] = tier
    return out


def cap_wards(df: pd.DataFrame, bound: int = MAX_WARDS) -> tuple[pd.DataFrame, int]:
    """Winsorise ward columns at `bound`. Returns the frame and the cells capped."""
    cols = [c for c in df.columns if "wards_" in c]
    capped = int((df[cols] > bound).sum().sum())
    df = df.copy()
    df[cols] = df[cols].clip(upper=bound)
    return df, capped


def build_games(verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Produce the canonical table, and a provenance record of what was removed.

    The second return value is the audit trail: every count below came from this
    run, so the notebook and the slides can quote it without anyone retyping it.
    """
    D = load.load_diamond()
    C = load.load_challenger()
    report: dict[str, int] = {
        "diamond_rows_in": len(D),
        "challenger_rows_in": len(C),
    }

    # 1. stage contamination (D04) ------------------------------------------
    bad = flag_not_ten_minutes(C)
    report["challenger_not_10min"] = int(bad.sum())
    C = C.loc[~bad].copy()

    # 2. duplicates and the redundant target mirror -------------------------
    dup = C["gameId"].duplicated()
    report["challenger_duplicate_ids"] = int(dup.sum())
    C = C.loc[~dup].copy()
    report["red_wins_is_mirror"] = int((C["redWins"] == 1 - C["blueWins"]).all())

    dup_d = D["gameId"].duplicated()
    report["diamond_duplicate_ids"] = int(dup_d.sum())
    D = D.loc[~dup_d].copy()

    # 3. canonical vocabulary (D06, D07) ------------------------------------
    games = pd.concat(
        [_to_canonical(D, 0, "diamond"), _to_canonical(C, 1, "challenger")],
        ignore_index=True,
    )
    report["cross_tier_id_overlap"] = int(
        games.loc[games.tier == "diamond", "game_id"]
        .isin(games.loc[games.tier == "challenger", "game_id"])
        .sum()
    )

    # 4. residual ward tail (D08) -------------------------------------------
    games, capped = cap_wards(games)
    report["ward_cells_capped"] = capped

    report["rows_out"] = len(games)
    report["blue_win_rate"] = round(float(games.blue_wins.mean()), 4)

    if verbose:
        for k, v in report.items():
            print(f"  {k:28s} {v:,}" if isinstance(v, int) else f"  {k:28s} {v}")
    return games, report
