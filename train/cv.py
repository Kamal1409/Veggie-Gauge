"""Leave-one-item-out cross-validation shared by training and evaluation.

For each physical vegetable, every model is refitted on the singles of all OTHER items
and then predicts every photo row of the held-out item (in any photo set). The result is
an out-of-fold (OOF) prediction for every row, from models that never saw that vegetable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import crop_classifier, size_classifier, weight_model
from src.features import CROP_FEATURES
from src.size_classifier import ML_FEATURES

CROP_KINDS = list(crop_classifier.CANDIDATES)
WEIGHT_NAMES = ["proxy", "proxy+intercept", "ridge_log"]
SIZE_NAMES = list(size_classifier.CANDIDATES)


def size_thresholds(gt: pd.DataFrame) -> dict[str, float]:
    """The weight that separates SMALL from BIG for each crop.

    It is the median weight of the bought items: a documented constant that can be
    swapped for a market or ARL grading standard.

    Args:
        gt: Ground-truth table of weighed items.

    Returns:
        The BIG threshold in grams for each crop.
    """
    return {str(c): float(np.median(np.asarray(g["weight_g"], dtype=float))) for c, g in gt.groupby("crop")}


def fit_weight(train: pd.DataFrame, crop: str, name: str):
    """Fit one weight-model candidate for a crop.

    Args:
        train: Training rows of that crop.
        crop: Crop name, which decides the volume proxy.
        name: Candidate name ("proxy", "proxy+intercept" or "ridge_log").

    Returns:
        The fitted model and the feature names it expects, in order.
    """
    feats, factory = weight_model.candidates(crop)[name]
    return factory().fit(train[feats].to_numpy(float), train["weight_g"].to_numpy(float)), feats


def fit_size(train: pd.DataFrame, name: str, threshold: float, est_weight: np.ndarray):
    """Fit one Small/Big candidate for a crop.

    Args:
        train: Training rows of that crop.
        name: Candidate name ("weight_threshold", "logreg" or "svm").
        threshold: The crop's BIG threshold in grams.
        est_weight: Estimated weight of each training row, used as a feature.

    Returns:
        The fitted model, or None for the threshold rule (nothing to fit) or when the
        training rows are all one class.
    """
    if size_classifier.CANDIDATES[name] is None:
        return None
    X = train.assign(est_weight_g=est_weight)[ML_FEATURES].to_numpy(float)
    y = train["weight_g"].to_numpy(float) >= threshold
    if len(set(y)) < 2:
        return None
    return size_classifier.CANDIDATES[name]().fit(X, y)


def predict_fold(train: pd.DataFrame, test: pd.DataFrame, crop: str, threshold: float) -> pd.DataFrame:
    """Fit every model on one fold's training rows and predict the held-out item.

    Covers the crop classifier (both kinds), every weight candidate and every
    Small/Big candidate paired with each weight candidate.

    Args:
        train: Singles of every item except the held-out one.
        test: All rows of the held-out item, from any photo set.
        crop: The held-out item's crop.
        threshold: That crop's BIG threshold in grams.

    Returns:
        A frame aligned with `test`, with one column per prediction
        (`crop_pred_<kind>`, `crop_prob_<kind>`, `weight_<model>`, `size_<model>_<weight>`).
    """
    out = pd.DataFrame(index=test.index)
    for kind in CROP_KINDS:
        model = crop_classifier.CANDIDATES[kind]().fit(train[CROP_FEATURES].to_numpy(float), train["crop"])
        proba = model.predict_proba(test[CROP_FEATURES].to_numpy(float))
        out[f"crop_pred_{kind}"] = model.classes_[proba.argmax(1)]
        out[f"crop_prob_{kind}"] = proba.max(1)

    tr = train.loc[train["crop"] == crop]
    if len(set(tr["veg_id"])) < 2:
        return out
    for w in WEIGHT_NAMES:
        model, feats = fit_weight(tr, crop, w)
        out[f"weight_{w}"] = np.maximum(model.predict(test[feats].to_numpy(float)), 0)
        est_train = model.predict(tr[feats].to_numpy(float))
        X_test = test.assign(est_weight_g=out[f"weight_{w}"])[ML_FEATURES].to_numpy(float)
        for s in SIZE_NAMES:
            sm = fit_size(tr, s, threshold, est_train)
            if s == "weight_threshold":
                out[f"size_{s}_{w}"] = out[f"weight_{w}"] >= threshold
            elif sm is not None:
                out[f"size_{s}_{w}"] = sm.predict(X_test).astype(bool)
    return out


def oof_predictions(singles: pd.DataFrame, target: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """Leave-one-item-out predictions for every row of `target`.

    Each physical vegetable is held out in turn. Models are refitted on the singles of
    all the other items and then predict every row of the held-out vegetable,
    whichever photo it appears in. So every prediction comes from models that never
    saw that vegetable.

    Args:
        singles: Clean singles, the only rows models are trained on.
        target: Rows to predict; each needs a `veg_id` and `crop`.
        thresholds: BIG threshold in grams for each crop.

    Returns:
        `target` with the prediction columns from `predict_fold` added.
    """
    parts = []
    for vid, rows in target.groupby("veg_id"):
        crop = rows["crop"].iloc[0]
        train = singles.loc[singles["veg_id"] != vid]
        parts.append(predict_fold(train, rows, crop, thresholds[crop]))
    pred = pd.concat(parts).reindex(target.index)
    return target.join(pred)


def item_mae(df: pd.DataFrame, col: str) -> pd.Series:
    """Mean absolute weight error per physical vegetable: one CV fold per item.

    Args:
        df: Rows with out-of-fold predictions.
        col: Column holding the predicted weight.

    Returns:
        MAE in grams for each `veg_id`.
    """
    return (df[col] - df["weight_g"]).abs().groupby(df["veg_id"]).mean()


def select_simplest(per_item: dict[str, pd.Series], order: list[str]) -> str:
    """Pick the simplest model whose CV error is within one standard error of the best.

    Args:
        per_item: Per-item errors for each candidate, as returned by `item_mae`.
        order: Candidate names from simplest to most complex.

    Returns:
        The chosen candidate's name.
    """
    stats = {n: (s.mean(), s.std(ddof=1) / np.sqrt(len(s))) for n, s in per_item.items()}
    best = min(stats, key=lambda n: stats[n][0])
    limit = stats[best][0] + stats[best][1]
    return next(n for n in order if n in stats and stats[n][0] <= limit)
