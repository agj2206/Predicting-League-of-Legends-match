"""Deterministic Part A preprocessing; no model fitting or learned transforms.

Call prepare_data() from the repository root to reproduce the Part A split.
Match IDs are row indices for traceability, never predictor columns.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_FILENAME = "high_diamond_ranked_10min.csv"
TARGET = "blueWins"
RANDOM_STATE = 42
TEST_SIZE = 0.20
ORIGINAL_SHA256 = "82b2e23279097c2dac616e966eb2911711c428d41b6382b4cdb15da4892f9ca6"

TEAM_STATISTICS = (
    "WardsPlaced", "WardsDestroyed", "FirstBlood", "Kills", "Deaths",
    "Assists", "EliteMonsters", "Dragons", "Heralds", "TowersDestroyed",
    "TotalGold", "AvgLevel", "TotalExperience", "TotalMinionsKilled",
    "TotalJungleMinionsKilled", "GoldDiff", "ExperienceDiff", "CSPerMin",
    "GoldPerMin",
)
EXPECTED_COLUMNS = (
    "gameId", TARGET,
    *("blue" + name for name in TEAM_STATISTICS),
    *("red" + name for name in TEAM_STATISTICS),
)
BINARY_COLUMNS = (
    TARGET, "blueFirstBlood", "redFirstBlood", "blueDragons", "redDragons",
    "blueHeralds", "redHeralds",
)
DIFFERENCE_SOURCES = {
    "killsDiff": "Kills",
    "assistsDiff": "Assists",
    "minionsDiff": "TotalMinionsKilled",
    "jungleMinionsDiff": "TotalJungleMinionsKilled",
    "wardsPlacedDiff": "WardsPlaced",
    "wardsDestroyedDiff": "WardsDestroyed",
    "firstBloodDiff": "FirstBlood",
    "dragonsDiff": "Dragons",
    "heraldsDiff": "Heralds",
    "towersDestroyedDiff": "TowersDestroyed",
}
FEATURE_NAMES = ("blueGoldDiff", "blueExperienceDiff", *DIFFERENCE_SOURCES)


@dataclass
class PreparedData:
    """Unscaled predictors and labels; all four indices contain gameId values."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    source_path: Path
    source_sha256: str


def resolve_data_path(path=None):
    """Use an explicit path, LOL_DATA_PATH, or one unambiguous local CSV.

    Only the repository's data/raw/ and sibling Local_Datasets folders are
    searched. strip() handles this workspace's trailing space in the folder name.
    Nothing is downloaded, copied, or uploaded.
    """
    configured_path = path if path is not None else os.environ.get("LOL_DATA_PATH")
    if configured_path is not None:
        candidate = Path(configured_path).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Dataset file does not exist: {candidate}")
        if candidate.name != DATASET_FILENAME:
            raise ValueError(f"Expected the original filename {DATASET_FILENAME}.")
        return candidate

    candidates = [REPOSITORY_ROOT / "data" / "raw" / DATASET_FILENAME]
    for folder in REPOSITORY_ROOT.parent.iterdir():
        if folder.is_dir() and folder.name.strip() == "Local_Datasets":
            candidates.append(folder / DATASET_FILENAME)
    matches = sorted({p.resolve() for p in candidates if p.is_file()})
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Found {len(matches)} candidate datasets. Pass a path to prepare_data() "
            "or set LOL_DATA_PATH to the original CSV."
        )
    return matches[0]


def consistency_checks(frame):
    """Verify arithmetic identities with a small floating-point tolerance."""
    relationships = {
        "blueKills = redDeaths": (frame["blueKills"], frame["redDeaths"]),
        "redKills = blueDeaths": (frame["redKills"], frame["blueDeaths"]),
        "first-blood flags sum to 1": (
            frame["blueFirstBlood"] + frame["redFirstBlood"],
            np.ones(len(frame)),
        ),
        "blueGoldDiff = blueTotalGold - redTotalGold": (
            frame["blueGoldDiff"], frame["blueTotalGold"] - frame["redTotalGold"],
        ),
        "redGoldDiff = -blueGoldDiff": (
            frame["redGoldDiff"], -frame["blueGoldDiff"],
        ),
        "blueExperienceDiff = blueTotalExperience - redTotalExperience": (
            frame["blueExperienceDiff"],
            frame["blueTotalExperience"] - frame["redTotalExperience"],
        ),
        "redExperienceDiff = -blueExperienceDiff": (
            frame["redExperienceDiff"], -frame["blueExperienceDiff"],
        ),
    }
    for team in ("blue", "red"):
        relationships[f"{team}GoldPerMin = {team}TotalGold / 10"] = (
            frame[team + "GoldPerMin"], frame[team + "TotalGold"] / 10,
        )
        relationships[f"{team}CSPerMin = {team}TotalMinionsKilled / 10"] = (
            frame[team + "CSPerMin"], frame[team + "TotalMinionsKilled"] / 10,
        )
        relationships[f"{team}EliteMonsters = Dragons + Heralds"] = (
            frame[team + "EliteMonsters"],
            frame[team + "Dragons"] + frame[team + "Heralds"],
        )
    records = []
    for name, (left, right) in relationships.items():
        difference = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
        records.append({
            "relationship": name,
            "violations": int((~np.isclose(left, right, atol=1e-8, rtol=0)).sum()),
            "maximum_absolute_error": float(np.abs(difference).max()),
        })
    return pd.DataFrame(records).set_index("relationship")


def audit_data(frame):
    """Inspect all records without changing values or dropping observations.

    Dataset-wide checks are structural only. This function does not rank
    predictors, choose outlier thresholds, or learn any preprocessing.
    """
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    extra_columns = sorted(set(frame.columns) - set(EXPECTED_COLUMNS))
    errors = []
    if missing_columns or extra_columns or frame.columns.duplicated().any():
        errors.append(f"Unexpected schema: missing={missing_columns}, extra={extra_columns}.")
    if frame.empty:
        errors.append("The dataset is empty.")
    missing = frame.isna().sum()
    if missing.sum():
        errors.append("Missing values require investigation; no automatic imputation is applied.")
    duplicate_rows = int(frame.duplicated().sum())
    duplicate_ids = int(frame["gameId"].duplicated().sum()) if "gameId" in frame else None
    if duplicate_rows or duplicate_ids:
        errors.append("Duplicate rows or match IDs require investigation before splitting.")
    numeric_columns = [c for c in frame if is_numeric_dtype(frame[c])]
    nonnumeric = [c for c in frame if c not in numeric_columns]
    if nonnumeric:
        errors.append(f"Non-numeric columns: {nonnumeric}.")
    finite = bool(np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all())
    if not finite:
        errors.append("Non-finite values found.")
    validity = []
    for column in frame:
        if column not in numeric_columns:
            continue
        values = frame[column].dropna()
        negative_count = int((values < 0).sum()) if not column.endswith("Diff") else 0
        invalid_binary = int((~values.isin([0, 1])).sum()) if column in BINARY_COLUMNS else 0
        is_count = not column.endswith(("AvgLevel", "CSPerMin", "GoldPerMin"))
        fractional_count = int((values % 1 != 0).sum()) if is_count else 0
        invalid_level = int((~values.between(1, 18)).sum()) if column.endswith("AvgLevel") else 0
        invalid_id = int((values <= 0).sum()) if column == "gameId" else 0
        validity.append({"column": column, "negative_non_difference": negative_count,
                         "invalid_binary": invalid_binary, "fractional_count": fractional_count,
                         "invalid_level": invalid_level, "invalid_id": invalid_id})
        if negative_count or invalid_binary or fractional_count or invalid_level or invalid_id:
            errors.append(f"Invalid values in {column}.")
    relationships = pd.DataFrame()
    if not missing_columns and not nonnumeric and not frame.empty and finite:
        relationships = consistency_checks(frame)
        if relationships["violations"].sum():
            errors.append("Team arithmetic relationships failed; inspect before proceeding.")
    if TARGET in frame and set(frame[TARGET].dropna().unique()) != {0, 1}:
        errors.append("The dataset must contain both target classes 0 and 1.")
    return {
        "shape": frame.shape, "missing_by_column": missing,
        "total_missing": int(missing.sum()), "duplicate_rows": duplicate_rows,
        "duplicate_game_ids": duplicate_ids, "nonnumeric_columns": nonnumeric,
        "validity": pd.DataFrame(validity).set_index("column") if validity else pd.DataFrame(),
        "relationships": relationships, "errors": errors,
    }


def validate_data(frame):
    """Fail visibly on uninvestigated issues rather than silently cleaning data."""
    report = audit_data(frame)
    if report["errors"]:
        raise ValueError("Data validation failed:\n- " + "\n- ".join(report["errors"]))
    return report


def load_data(path=None):
    """Load the local original CSV and validate its schema and gameplay values."""
    frame = pd.read_csv(resolve_data_path(path))
    validate_data(frame)
    return frame


def prepare_features(frame):
    """Apply the fixed domain-based policy; no target values are read.

    All engineered differences are Blue minus Red. Positive values indicate
    a Blue lead. This deliberately summarizes relative advantages and loses
    absolute game tempo; it is a compact baseline, not lossless compression.
    """
    features = frame.loc[:, ["blueGoldDiff", "blueExperienceDiff"]].copy()
    for feature, statistic in DIFFERENCE_SOURCES.items():
        features[feature] = frame["blue" + statistic] - frame["red" + statistic]
    features.index = pd.Index(frame["gameId"].to_numpy(), name="gameId")
    return features.loc[:, list(FEATURE_NAMES)]


def create_train_test_split(frame):
    """Return X_train, X_test, y_train, y_test with fixed 80/20 stratification.

    Sort on the unique identifier first so input CSV row order cannot change
    split membership. Identifiers do not enter the predictor matrix. Never
    change the seed, test size, or sorting policy during model comparison.
    """
    validate_data(frame)
    ordered = frame.sort_values("gameId").reset_index(drop=True)
    features = prepare_features(ordered)
    target = pd.Series(ordered[TARGET].to_numpy(), index=features.index, name=TARGET)
    return train_test_split(
        features, target, test_size=TEST_SIZE, stratify=target, random_state=RANDOM_STATE,
    )


def prepare_data(path=None):
    """Public interface for Part B; returns the same unscaled split every time."""
    source = resolve_data_path(path)
    source_hash = sha256(source.read_bytes()).hexdigest()
    if source_hash != ORIGINAL_SHA256:
        raise ValueError(
            "CSV checksum differs from the original Part A source. Investigate "
            "the file/version before reusing the held-out split."
        )
    frame = load_data(source)
    X_train, X_test, y_train, y_test = create_train_test_split(frame)
    return PreparedData(X_train, X_test, y_train, y_test, source, source_hash)


def split_manifest(prepared):
    """Describe source, feature order, and membership without any label values."""
    return {
        "source_sha256": prepared.source_sha256,
        "target": TARGET, "feature_names": list(FEATURE_NAMES),
        "random_state": RANDOM_STATE, "test_size": TEST_SIZE,
        "ordering": "gameId ascending before sklearn train_test_split",
        "train_game_ids": [int(i) for i in prepared.X_train.index],
        "test_game_ids": [int(i) for i in prepared.X_test.index],
    }


def save_split_manifest(prepared, path=None):
    """Save once locally; fail if an existing split would be overwritten differently."""
    destination = Path(path) if path is not None else REPOSITORY_ROOT / "data/processed/split_manifest.json"
    manifest = split_manifest(prepared)
    if destination.exists():
        if json.loads(destination.read_text()) != manifest:
            raise ValueError("Existing split manifest differs. Investigate; do not silently replace it.")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


def feature_decisions():
    """One documented decision for every original column."""
    records = []
    for column in EXPECTED_COLUMNS:
        if column == TARGET:
            action, reason = "Target only", "Final result; never a predictor."
        elif column == "gameId":
            action, reason = "Exclude from X", "Match identifier; used only as a row index and to fix split order."
        elif column in ("blueGoldDiff", "blueExperienceDiff"):
            action, reason = "Retain", "Existing signed Blue-minus-Red advantage at 10 minutes."
        elif column in ("redGoldDiff", "redExperienceDiff"):
            action, reason = "Exclude", "Exact negative of the retained Blue difference (verified)."
        elif column.endswith("Deaths"):
            action, reason = "Exclude", "Exact copy of opposing kills (verified); killsDiff already summarizes combat."
        elif column.endswith("EliteMonsters"):
            action, reason = "Exclude", "Exactly Dragons + Heralds (verified); retain the separate objective differences."
        elif column.endswith(("GoldPerMin", "CSPerMin")):
            action, reason = "Exclude", "Exactly the corresponding total / 10 (verified); no additional measurement."
        elif column.endswith(("TotalGold", "TotalExperience")):
            action, reason = "Replace by difference", "Gold/experience lead retained; absolute tempo deliberately omitted."
        elif column.endswith("AvgLevel"):
            action, reason = "Exclude", "Coarse summary related to experience; not an exact duplicate. Domain simplification, not target-based selection."
        else:
            statistic = column[4:] if column.startswith("blue") else column[3:]
            feature = next(name for name, source in DIFFERENCE_SOURCES.items() if source == statistic)
            action, reason = "Replace by difference", f"Contributes to {feature} = Blue - Red; absolute totals omitted."
        records.append({"original_column": column, "decision": action, "reason": reason})
    return pd.DataFrame(records)
