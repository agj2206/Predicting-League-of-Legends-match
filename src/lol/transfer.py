"""Q4: does a model fit on Diamond still work at Challenger?

Built before B's model exists, so plugging the real one in is a swap rather
than a build. Everything here takes an *unfitted* sklearn estimator or pipeline
and does the fitting itself, so the split discipline lives in one place instead
of being re-implemented per notebook.

The question Q4 actually asks is narrower than "is the model good". It is:
*does the relationship between early-game state and the result hold at a
different skill level?* Two things can go wrong, and they mean different things:

  * **Discrimination drops** (ROC-AUC falls) — the features rank games worse at
    the new tier, so the relationship itself differs.
  * **Calibration drifts** (accuracy falls while AUC holds) — the ranking still
    works but the decision threshold no longer sits in the right place.

`transfer_report` separates them, because "it got worse" is not a finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from .features import feature_columns

RANDOM_STATE = 42


def split_by_tier(games: pd.DataFrame, tier: str = "diamond"):
    """Rows of one tier, and everything else, as (X, y) pairs.

    `tier` never reaches the feature matrix — `feature_columns` excludes it.
    A model allowed to see which file a row came from would learn the file
    rather than the game, which is exactly what Q4 must not measure.
    """
    cols = feature_columns(games)
    home, away = games[games.tier == tier], games[games.tier != tier]
    return (home[cols], home["blue_wins"]), (away[cols], away["blue_wins"])


def _metrics(y_true, y_pred, y_score) -> dict[str, float]:
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_score),
        "n":         int(len(y_true)),
    }


def _scores(model, X) -> np.ndarray:
    """Continuous scores, whether the estimator offers probabilities or margins."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def transfer_report(model, games: pd.DataFrame, home_tier: str = "diamond",
                    test_size: float = 0.25) -> pd.DataFrame:
    """Fit on the home tier, then score three ways.

    Returns one row per evaluation:

      * `home (train)`      — in-sample, shown only to make overfitting visible
      * `home (held out)`   — the honest within-tier number
      * `away (transfer)`   — the Q4 result

    The home tier is split so the within-tier number is never in-sample; without
    that, the transfer drop would be confounded by memorisation.
    """
    (Xh, yh), (Xa, ya) = split_by_tier(games, home_tier)
    Xtr, Xte, ytr, yte = train_test_split(
        Xh, yh, test_size=test_size, random_state=RANDOM_STATE, stratify=yh
    )

    fitted = clone(model).fit(Xtr, ytr)
    rows = {
        f"{home_tier} (train)":    _metrics(ytr, fitted.predict(Xtr), _scores(fitted, Xtr)),
        f"{home_tier} (held out)": _metrics(yte, fitted.predict(Xte), _scores(fitted, Xte)),
        "away (transfer)":         _metrics(ya,  fitted.predict(Xa),  _scores(fitted, Xa)),
    }
    out = pd.DataFrame(rows).T
    out["n"] = out["n"].astype(int)
    return out


def diagnose(report: pd.DataFrame, home_tier: str = "diamond") -> str:
    """Name what the numbers mean, so the write-up is not left to eyeballing."""
    held, away = report.loc[f"{home_tier} (held out)"], report.loc["away (transfer)"]
    d_auc = away.roc_auc - held.roc_auc
    d_acc = away.accuracy - held.accuracy
    verdict = f"ROC-AUC {d_auc:+.3f}, accuracy {d_acc:+.3f}"

    if abs(d_auc) < 0.02 and abs(d_acc) < 0.02:
        return (f"Transfers cleanly ({verdict}). The early-game relationship holds at the "
                "other skill level; nothing needs re-fitting.")
    if d_auc >= 0.02:
        return (f"Transfers better than it scores at home ({verdict}). The relationship holds "
                "and is if anything cleaner at the other tier — plausible when the away tier is "
                "larger or its play more consistent. Not evidence the model is 'better there': "
                "the two tiers are different populations, so the numbers are not a like-for-like "
                "contest.")
    if abs(d_auc) < 0.02:
        direction = "improves" if d_acc > 0 else "degrades"
        advice = ("No action needed; the home threshold is if anything conservative away."
                  if d_acc > 0 else
                  "The threshold needs re-fitting per tier, the model does not.")
        return (f"Ranking holds while the operating point {direction} ({verdict}) — a "
                f"calibration shift, not a broken relationship. {advice}")
    return (f"Discrimination drops ({verdict}). The features rank games differently at the other "
            "tier, so the relationship itself differs — this is the interesting negative result.")


def feature_importance_shift(model, games: pd.DataFrame) -> pd.DataFrame | None:
    """Fit the same estimator on each tier and compare what it leaned on.

    Returns None for estimators that expose neither importances nor coefficients
    (KNN and an RBF SVM, for instance).
    """
    cols = feature_columns(games)
    out: dict[str, pd.Series] = {}
    for tier in games.tier.unique():
        block = games[games.tier == tier]
        fitted = clone(model).fit(block[cols], block["blue_wins"])
        if hasattr(fitted, "feature_importances_"):
            vals = fitted.feature_importances_
        elif hasattr(fitted, "coef_"):
            vals = np.abs(np.ravel(fitted.coef_))
        else:
            return None
        out[tier] = pd.Series(vals / vals.sum(), index=cols)
    table = pd.DataFrame(out)
    if table.shape[1] == 2:
        a, b = table.columns
        table["shift"] = (table[a] - table[b]).abs()
        table = table.sort_values("shift", ascending=False)
    return table
