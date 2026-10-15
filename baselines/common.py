"""Shared setup for the raw-image baselines (docs/PIPELINE.md, Stage 6).

They test the alternative idea fairly: learn SMALL / BIG straight from pixels, with the
same labels (BIG >= the crop's median weight) and the same leave-one-item-out folds as
the feature-based models.

Patches keep the size information. Each one covers the same physical square on the sheet
(`side_mm`, big enough for the longest item) and every patch is scaled by the same factor,
so a bigger vegetable always fills more of its patch. That is the most favourable setup
the baseline can get.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.calibrate import Calibration
from src.config import REPORTS_DIR, DataPaths
from src.data import imread, load_ground_truth
from train.cv import size_thresholds
from train.train_models import classification_scores, training_rows


def load_patches(dp: DataPaths, out_px: int, side_mm: float | None = None) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Cut a size-preserving patch around every clean single.

    Each patch covers the same physical square on the sheet and every patch is scaled
    by the same factor, so a bigger vegetable always fills more of its patch.

    Args:
        dp: Dataset locations; `features.csv` must already exist.
        out_px: Side of each patch after scaling, in pixels.
        side_mm: Side of the square each patch covers. By default it is just big enough
            for the longest item, plus a margin.

    Returns:
        The rows (one per patch), the patches as an `(N, out_px, out_px, 3)` BGR array,
        and a dict with the BIG thresholds and the side length used.
    """
    gt = load_ground_truth(dp)
    rows = training_rows(pd.read_csv(dp.features, keep_default_na=False, na_values=[""]))
    calib = Calibration.load(dp.calibration)
    if side_mm is None:
        side_mm = float(np.ceil(rows["length_mm"].max() * 1.1 / 10) * 10)
    patches = []
    for image, grp in rows.groupby("image", sort=False):
        img = imread(dp.raw / str(image))
        mmpp, _ = calib.for_image(img.shape)
        half = int(round(side_mm / mmpp / 2))
        sheet = np.median(img.reshape(-1, 3), axis=0).tolist()
        padded = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_CONSTANT, value=sheet)
        for idx, r in zip(grp.index, grp.to_dict("records")):
            x, y = int(round(r["cx"])) + half, int(round(r["cy"])) + half
            crop = padded[y - half : y + half, x - half : x + half]
            patches.append((idx, cv2.resize(crop, (out_px, out_px), interpolation=cv2.INTER_AREA)))
    order = [i for i, _ in patches]
    X = np.stack([p for _, p in patches])
    return rows.loc[order].reset_index(drop=True), X, {"thresholds": size_thresholds(gt), "side_mm": side_mm}


def leave_one_item_out(rows: pd.DataFrame, X: np.ndarray, thresholds: dict, fit_predict) -> pd.DataFrame:
    """Score a baseline with the same leave-one-item-out folds as the feature models.

    Per crop, each item is held out in turn: the baseline is fitted on the crop's other
    items and predicts the held-out item's patches.

    Args:
        rows: One row per patch.
        X: Inputs aligned with `rows` (patches or features derived from them).
        thresholds: BIG threshold in grams for each crop.
        fit_predict: Function `(X_train, y_train, X_test) -> predicted BIG flags`.

    Returns:
        `rows` with "big_true" and "big_pred" columns added.
    """
    rows = rows.copy()
    rows["big_true"] = [w >= thresholds[c] for w, c in zip(rows["weight_g"], rows["crop"])]
    rows["big_pred"] = False
    for _crop, grp in rows.groupby("crop"):
        for vid in grp["veg_id"].unique():
            test = grp.index[grp["veg_id"] == vid].to_numpy()
            train = grp.index[grp["veg_id"] != vid].to_numpy()
            y = rows.loc[train, "big_true"].to_numpy(bool)
            if len(set(y)) < 2:
                rows.loc[test, "big_pred"] = bool(y[0])
                continue
            rows.loc[test, "big_pred"] = fit_predict(X[train], y, X[test])
    return rows


def write_report(name: str, rows: pd.DataFrame, meta: dict, synthetic: bool, out_dir: Path = REPORTS_DIR) -> dict:
    """Save a baseline's scores and per-patch predictions.

    Args:
        name: Baseline name, used in the file names.
        rows: Output of `leave_one_item_out`.
        meta: Extra settings to record (patch size, model, ...).
        synthetic: Whether the data was synthetic.
        out_dir: Folder to write `baseline_<name>.json` and its predictions CSV into.

    Returns:
        The report: overall and per-crop accuracy, F1 and confusion counts.
    """
    t, p = rows["big_true"].to_numpy(bool), rows["big_pred"].to_numpy(bool)
    report = {
        "baseline": name,
        "overall": classification_scores(t, p),
        "per_crop": {c: classification_scores(g["big_true"].to_numpy(bool), g["big_pred"].to_numpy(bool)) for c, g in rows.groupby("crop")},
        "n_rows": len(rows),
        "synthetic_data": synthetic,
        **meta,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"baseline_{name}.json").write_text(json.dumps(report, indent=2))
    rows[["image", "veg_id", "crop", "weight_g", "big_true", "big_pred"]].to_csv(out_dir / f"baseline_{name}_predictions.csv", index=False)
    return report
