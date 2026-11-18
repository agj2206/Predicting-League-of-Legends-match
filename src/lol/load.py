"""Readers for the two raw 10-minute files.

Loading only — no filtering, no renaming, no derived columns. The data-quality
audit (notebook 01) has to be able to see the files exactly as they were
published, including the parts that turned out to be wrong (see docs/decisions.md
D04 and docs/agent_validation.md V02).

Two files, two authors, two different vocabularies for the same game:

    high_diamond_ranked_10min.csv        9,879 games · 40 columns · Diamond–Master
    Challenger_Ranked_Games_10minute.csv 26,409 rows · 51 columns · Challenger

`clean.py` reconciles them. Nothing here assumes they agree.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DIAMOND_FILE = "high_diamond_ranked_10min.csv"
CHALLENGER_FILE = "Challenger_Ranked_Games_10minute.csv"

#: Columns that must exist, per file. Verified against the published files on
#: 2026-11-18. A mismatch means the source changed and the audit must be re-run.
DIAMOND_COLUMNS = [
    "gameId", "blueWins",
    "blueWardsPlaced", "blueWardsDestroyed", "blueFirstBlood", "blueKills",
    "blueDeaths", "blueAssists", "blueEliteMonsters", "blueDragons",
    "blueHeralds", "blueTowersDestroyed", "blueTotalGold", "blueAvgLevel",
    "blueTotalExperience", "blueTotalMinionsKilled",
    "blueTotalJungleMinionsKilled", "blueGoldDiff", "blueExperienceDiff",
    "blueCSPerMin", "blueGoldPerMin",
    "redWardsPlaced", "redWardsDestroyed", "redFirstBlood", "redKills",
    "redDeaths", "redAssists", "redEliteMonsters", "redDragons", "redHeralds",
    "redTowersDestroyed", "redTotalGold", "redAvgLevel", "redTotalExperience",
    "redTotalMinionsKilled", "redTotalJungleMinionsKilled", "redGoldDiff",
    "redExperienceDiff", "redCSPerMin", "redGoldPerMin",
]

CHALLENGER_COLUMNS = [
    "gameId", "blueWins",
    "blueTotalGolds", "blueCurrentGolds", "blueTotalLevel", "blueAvgLevel",
    "blueTotalMinionKills", "blueTotalJungleMinionKills", "blueFirstBlood",
    "blueKill", "blueDeath", "blueAssist", "blueWardPlaced", "blueWardKills",
    "blueFirstTower", "blueFirstInhibitor", "blueFirstTowerLane",
    "blueTowerKills", "blueMidTowerKills", "blueTopTowerKills",
    "blueBotTowerKills", "blueInhibitor", "blueFirstDragon", "blueDragnoType",
    "blueDragon", "blueRiftHeralds",
    "redWins", "redTotalGolds", "redCurrentGolds", "redTotalLevel",
    "redAvgLevel", "redTotalMinionKills", "redTotalJungleMinionKills",
    "redFirstBlood", "redKill", "redDeath", "redAssist", "redWardPlaced",
    "redWardKills", "redFirstTower", "redFirstInhibitor", "redFirstTowerLane",
    "redTowerKills", "redMidTowerKills", "redTopTowerKills", "redBotTowerKills",
    "redInhibitor", "redFirstDragon", "redDragnoType", "redDragon",
    "redRiftHeralds",
]

#: Columns that are text, not numbers. Everything else is numeric.
#: `DragnoType` is the source's own spelling; it holds a stringified Python list
#: such as "['WATER_DRAGON', 'FIRE_DRAGON']", and `FirstTowerLane` a lane name.
_CHALLENGER_TEXT = ["blueDragnoType", "redDragnoType",
                    "blueFirstTowerLane", "redFirstTowerLane"]


def repo_root(start: Path | str | None = None) -> Path:
    """Walk up from `start` until a directory containing CLAUDE.md is found."""
    here = Path(start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "CLAUDE.md").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate the repository root (no CLAUDE.md found above "
        f"{here}). Run from inside the project."
    )


def raw_dir() -> Path:
    return repo_root() / "data" / "raw"


def processed_dir() -> Path:
    path = repo_root() / "data" / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_raw_files() -> pd.DataFrame:
    """Report which raw files are present, and how big they are."""
    rows = []
    for name in (DIAMOND_FILE, CHALLENGER_FILE):
        path = raw_dir() / name
        rows.append({
            "file": name,
            "present": path.exists(),
            "size_mb": round(path.stat().st_size / 1e6, 2) if path.exists() else None,
        })
    return pd.DataFrame(rows)


def require_raw_files() -> None:
    """Fail with instructions rather than a confusing pandas error."""
    missing = [n for n in (DIAMOND_FILE, CHALLENGER_FILE)
               if not (raw_dir() / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing from {raw_dir()}: {', '.join(missing)}.\n"
            "Download them from Kaggle (see README.md). Never commit the data."
        )


def _read(name: str, expected: list[str], text_cols: list[str] | None = None) -> pd.DataFrame:
    """Read one raw CSV and assert its schema matches what we audited."""
    require_raw_files()
    dtypes = {c: "string" for c in (text_cols or [])}
    df = pd.read_csv(raw_dir() / name, dtype=dtypes)

    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    if missing or extra:
        raise ValueError(
            f"{name} schema changed.\n  missing: {missing}\n  unexpected: {extra}\n"
            "Re-run the audit in notebook 01 before trusting anything downstream."
        )
    return df


def load_diamond() -> pd.DataFrame:
    """Diamond–Master games. Verified clean: no missing values, no duplicate ids."""
    return _read(DIAMOND_FILE, DIAMOND_COLUMNS)


def load_challenger() -> pd.DataFrame:
    """Challenger games, exactly as published.

    About 9.5% of these rows are NOT 10-minute states — see V02. This function
    returns them anyway, on purpose: the audit has to be able to show the problem
    before `clean.py` removes it.
    """
    return _read(CHALLENGER_FILE, CHALLENGER_COLUMNS, _CHALLENGER_TEXT)


def load_raw() -> dict[str, pd.DataFrame]:
    """Both files, keyed by tier."""
    return {"diamond": load_diamond(), "challenger": load_challenger()}
