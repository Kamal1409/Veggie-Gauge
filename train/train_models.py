"""Stage 5: train the classic ML models on the singles, selected by leave-one-item-out CV.

  (a) crop classifier        logistic regression vs kNN on colour + shape features
  (b) weight regressor/crop  weight = a * volume_proxy (+ b)  vs  ridge on log features;
                             the simplest model within one standard error of the best is kept
  (c) Small/Big per crop     threshold on estimated weight vs logistic regression vs SVM;
                             an ML model replaces the threshold only if it beats it

    python -m train.train_models [--data data] [--models models] [--reports reports]

Run `python -m train.build_features` first.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import crop_classifier, size_classifier, weight_model
from src.config import MODELS_DIR, REPORTS_DIR, DataPaths
from src.data import load_ground_truth
from src.features import CROP_FEATURES
from train.cv import CROP_KINDS, SIZE_NAMES, WEIGHT_NAMES, fit_size, fit_weight, item_mae, oof_predictions, select_simplest, size_thresholds


def training_rows(features: pd.DataFrame) -> pd.DataFrame:
    """The rows models are trained on: singles that segmented cleanly.

    Clean means the photo's one detection was matched to its item and is not cut off
    by the frame.

    Args:
        features: The full feature table from `data/features.csv`.

    Returns:
        The training rows, re-indexed from 0.
    """
    df = features.loc[(features["set"] == "singles") & features["matched"].astype(bool) & ~features["truncated"].astype(bool)]
    return df.reset_index(drop=True)


def classification_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, F1 for the BIG class and the confusion counts.

    Args:
        y_true: True labels (True = BIG).
        y_pred: Predicted labels (True = BIG).

    Returns:
        A dict with "accuracy", "f1_big" and "confusion" (tp, fp, fn, tn).
    """
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {"accuracy": round((tp + tn) / len(y_true), 4), "f1_big": round(f1, 4), "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}


def weight_scores(y: np.ndarray, p: np.ndarray) -> dict:
    """Error metrics for weight estimates.

    Args:
        y: Measured weights in grams.
        p: Estimated weights in grams.

    Returns:
        A dict with "mae_g", "mape_pct" and "r2" (None if the measured weights don't vary).
    """
    ss = np.sum((y - y.mean()) ** 2)
    return {
        "mae_g": round(float(np.mean(np.abs(p - y))), 2),
        "mape_pct": round(float(100 * np.mean(np.abs(p - y) / y)), 2),
        "r2": round(float(1 - np.sum((y - p) ** 2) / ss), 4) if ss > 0 else None,
    }


def main() -> None:
    """Command-line entry point: train every model and write the training report."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DataPaths().root)
    p.add_argument("--models", type=Path, default=MODELS_DIR)
    p.add_argument("--reports", type=Path, default=REPORTS_DIR)
    args = p.parse_args()
    run(DataPaths(args.data), args.models, args.reports)


def run(dp: DataPaths, models_dir: Path, reports_dir: Path) -> dict:
    """Train and save the crop classifier, weight regressors and Small/Big classifiers.

    Every candidate is scored with leave-one-item-out CV on the singles first. The
    best crop classifier is kept. For weight, the simplest regressor within one
    standard error of the best is kept. For Small/Big, a learned model replaces the
    weight threshold only if it is strictly more accurate. The chosen models are then
    refitted on all the singles.

    Args:
        dp: Dataset locations; `features.csv` must already exist.
        models_dir: Where to save the `*.joblib` models.
        reports_dir: Where to write `training_report.json` and the out-of-fold predictions.

    Returns:
        The training report: CV scores of every candidate and what was chosen.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    gt = load_ground_truth(dp)
    singles = training_rows(pd.read_csv(dp.features, keep_default_na=False, na_values=[""]))
    thresholds = size_thresholds(gt)
    oof = oof_predictions(singles, singles, thresholds)
    oof.to_csv(reports_dir / "oof_singles.csv", index=False)
    report: dict = {"synthetic_data": dp.synthetic_flag.exists(), "n_rows": len(singles), "n_items": len(set(singles["veg_id"]))}

    # (a) crop classifier
    crop_cv = {k: round(float((oof[f"crop_pred_{k}"] == oof["crop"]).mean()), 4) for k in CROP_KINDS}
    crop_kind = max(CROP_KINDS, key=lambda k: (crop_cv[k], k == "logreg"))
    model = crop_classifier.CANDIDATES[crop_kind]().fit(singles[CROP_FEATURES].to_numpy(float), singles["crop"])
    joblib.dump({"model": model, "kind": crop_kind, "cv_accuracy": crop_cv}, models_dir / "crop_classifier.joblib")
    report["crop_classifier"] = {"chosen": crop_kind, "cv_accuracy": crop_cv}
    print(f"\nCrop classifier (LOO accuracy): {crop_cv} -> {crop_kind}")

    # (b) weight regressors and (c) Small/Big classifiers, per crop
    report["weight"], report["size"] = {}, {}
    print(f"\n{'crop':11s} {'model':16s} {'MAE g':>8s} {'MAPE %':>8s} {'R2':>7s}")
    for crop_key, df in singles.groupby("crop"):
        crop = str(crop_key)
        o = oof.loc[oof["crop"] == crop]
        y = np.asarray(o["weight_g"], dtype=float)
        per_item = {w: item_mae(o, f"weight_{w}") for w in WEIGHT_NAMES if f"weight_{w}" in o}
        if not per_item:
            print(f"{crop}: fewer than 3 items, skipped")
            continue
        chosen = select_simplest(per_item, WEIGHT_NAMES)
        scores = {w: weight_scores(y, np.asarray(o[f"weight_{w}"], dtype=float)) for w in per_item}
        for w, s in scores.items():
            mark = "  <- chosen" if w == chosen else ""
            print(f"{crop:11s} {w:16s} {s['mae_g']:8.1f} {s['mape_pct']:8.1f} {s['r2']:7.3f}{mark}")
        wm, feats = fit_weight(df, crop, chosen)
        joblib.dump(
            {
                "crop": crop,
                "model_name": chosen,
                "features": feats,
                "proxy": weight_model.proxy_for(crop),
                "model": wm,
                "cv_mae_g": scores[chosen]["mae_g"],
                "cv": scores,
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            weight_model.model_path(models_dir, crop),
        )
        report["weight"][crop] = {
            "chosen": chosen,
            "proxy": weight_model.proxy_for(crop),
            "cv": scores,
            "n_items": len(set(df["veg_id"])),
        }

        truth = y >= thresholds[crop]
        size_cv = {
            s: classification_scores(truth, np.asarray(o[f"size_{s}_{chosen}"], dtype=bool))
            for s in SIZE_NAMES
            if f"size_{s}_{chosen}" in o
        }
        rule = size_cv["weight_threshold"]["accuracy"]
        size_choice = max(size_cv, key=lambda s: (size_cv[s]["accuracy"] > rule, size_cv[s]["accuracy"], s == "weight_threshold"))
        if size_cv[size_choice]["accuracy"] <= rule:
            size_choice = "weight_threshold"
        est = wm.predict(df[feats].to_numpy(float))
        sm = fit_size(df, size_choice, thresholds[crop], est)
        joblib.dump(
            {"crop": crop, "model_name": size_choice, "model": sm, "threshold_g": thresholds[crop], "cv": size_cv},
            size_classifier.model_path(models_dir, crop),
        )
        report["size"][crop] = {"chosen": size_choice, "threshold_g": thresholds[crop], "cv": size_cv}
        accs = ", ".join(f"{s} {v['accuracy']:.2f}" for s, v in size_cv.items())
        print(f"{crop:11s} Small/Big (BIG >= {thresholds[crop]:.0f} g): {accs} -> {size_choice}")

    (reports_dir / "training_report.json").write_text(json.dumps(report, indent=2))
    if report["synthetic_data"]:
        print("\nNOTE: trained on SYNTHETIC data; these numbers validate the code, not real accuracy.")
    return report


if __name__ == "__main__":
    main()
