"""Stage 7: every metric from Section 7 of the design, all leave-one-item-out.

Every prediction for a vegetable comes from models fitted without that vegetable
(train/cv.py), whichever photo it appears in.

  calibration      mm/px, and length error (mm) vs the ruler per crop
  segmentation     mean IoU on the hand-traced photos; detection-count accuracy per photo set
  crop / grouping  per-item accuracy + confusion matrix; on mixed photos, % of items in the
                   right crop group and % of photos with every item grouped correctly
  weight           MAE, MAPE, R^2 per crop: volume-proxy vs multi-feature model
  pairwise         accuracy vs the true weight order within same-crop groups (pairs + mixed
                   photos), overall and binned by true weight difference
  Small/Big        accuracy, F1, confusion: weight threshold vs logistic regression vs SVM,
                   plus the raw-image baselines (HOG+SVM, CNN) if they have been run
  robustness       the same per stress case (touching, tilt, dim)
  efficiency       CPU latency of the deployed pipeline per photo

    python -m eval.evaluate [--data data] [--reports reports]

Run train.build_features and train.train_models first (and optionally the baselines).
Writes reports/metrics.json, reports/results_table.md, eval_rows.csv, eval_pairs.csv.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.calibrate import Calibration
from src.compare import pair_verdict, relative_difference
from src.config import REPORTS_DIR, Config, DataPaths
from src.data import imread, load_ground_truth, load_images, match_masks, polygon_mask, read_polygons
from src.segment import segment, white_balance
from train.cv import SIZE_NAMES, WEIGHT_NAMES, oof_predictions, size_thresholds
from train.train_models import classification_scores, training_rows, weight_scores

BINS = [(0, 5), (5, 10), (10, 20), (20, float("inf"))]


def bin_label(lo: float, hi: float) -> str:
    """Label for a true-difference bin, e.g. "5-10%" or ">20%".

    Args:
        lo: Lower edge of the bin, in percent.
        hi: Upper edge, or infinity for the last bin.

    Returns:
        The label.
    """
    return f">{lo:g}%" if hi == float("inf") else f"{lo:g}-{hi:g}%"


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    """A column as a float array.

    Args:
        df: The table.
        name: Column name.

    Returns:
        The column's values as floats.
    """
    return np.asarray(df[name], dtype=float)


def _rate(s) -> float | None:
    """The share of True values, rounded for the report.

    Args:
        s: Booleans (or 0/1 values).

    Returns:
        The mean to four decimals, or None if there are no values.
    """
    s = pd.Series(s)
    return round(float(s.mean()), 4) if len(s) else None


# ---------------------------------------------------------------------------
# Pairwise
# ---------------------------------------------------------------------------


def pair_rows(rows: pd.DataFrame, weight_col: str, delta: float) -> pd.DataFrame:
    """Score every same-crop pair of items within each photo.

    A pair only counts as decided if the system would actually compare it: both items
    landed in their true crop group (right crop, probability ≥ τ) and neither is cut
    off by the frame. Undecided pairs are kept so the decided rate can be reported.

    Args:
        rows: Out-of-fold rows from the photos to score.
        weight_col: Column holding the weight estimate to compare.
        delta: SIMILAR margin, for the three-way verdicts.

    Returns:
        One row per pair, with the true difference, whether it was decided, whether the
        heavier item was picked, and the true and predicted verdicts.
    """
    out = []
    for image, grp in rows.groupby("image"):
        for a, b in combinations(grp.to_dict("records"), 2):
            if a["crop"] != b["crop"] or a["weight_g"] == b["weight_g"]:
                continue
            decided = bool(a["grouped_ok"] and b["grouped_ok"] and pd.notna(a[weight_col]) and pd.notna(b[weight_col]))
            ta, tb = a["weight_g"], b["weight_g"]
            pa, pb = (a[weight_col], b[weight_col]) if decided else (np.nan, np.nan)
            out.append(
                {
                    "image": image,
                    "set": a["set"],
                    "case": a["case"],
                    "crop": a["crop"],
                    "a": a["veg_id"],
                    "b": b["veg_id"],
                    "true_diff_pct": 100 * relative_difference(ta, tb),
                    "decided": decided,
                    "order_correct": bool((pa > pb) == (ta > tb)) if decided else None,
                    "verdict_true": pair_verdict(ta, tb, delta),
                    "verdict_pred": pair_verdict(pa, pb, delta) if decided else None,
                }
            )
    return pd.DataFrame(out)


def item_pairs(singles: pd.DataFrame, delta: float) -> pd.DataFrame:
    """Score every same-crop pair of items, not just those photographed together.

    Each item is represented by the mean of its out-of-fold estimates across its
    singles. Random pairings rarely put near-identical items in one photo, so this
    fills in the small-difference bins of the pairwise chart.

    Args:
        singles: Out-of-fold rows from the singles.
        delta: SIMILAR margin, for the three-way verdicts.

    Returns:
        One row per item pair, in the same format as `pair_rows` (always decided).
    """
    per_item = singles.groupby("veg_id").agg(crop=("crop", "first"), weight_g=("weight_g", "first"), est=("w_chosen", "mean"))
    out = []
    for crop, g in per_item.groupby("crop"):
        for ra, rb in combinations(g.reset_index().to_dict("records"), 2):
            if ra["weight_g"] == rb["weight_g"] or pd.isna(ra["est"]) or pd.isna(rb["est"]):
                continue
            out.append(
                {
                    "crop": crop,
                    "a": ra["veg_id"],
                    "b": rb["veg_id"],
                    "true_diff_pct": 100 * relative_difference(ra["weight_g"], rb["weight_g"]),
                    "decided": True,
                    "order_correct": bool((ra["est"] > rb["est"]) == (ra["weight_g"] > rb["weight_g"])),
                    "verdict_true": pair_verdict(ra["weight_g"], rb["weight_g"], delta),
                    "verdict_pred": pair_verdict(ra["est"], rb["est"], delta),
                }
            )
    return pd.DataFrame(out)


def pairwise_summary(pairs: pd.DataFrame) -> dict:
    """Summarise scored pairs: overall accuracy and accuracy per true-difference bin.

    Args:
        pairs: Output of `pair_rows` or `item_pairs`.

    Returns:
        A dict with the pair count, decided rate, order and verdict accuracy, the number
        of distinct item pairs, and one entry per bin.
    """
    if pairs.empty:
        return {"n_pairs": 0}
    d = pairs[pairs["decided"]]

    def distinct(x):
        """Count distinct item pairs, since repeated photos of the same two items are correlated.

        Args:
            x: Scored pairs with "a" and "b" item IDs.

        Returns:
            The number of distinct unordered pairs.
        """
        return len({tuple(sorted(ab)) for ab in zip(x["a"], x["b"])}) if len(x) else 0

    out = {
        "n_pairs": len(pairs),
        "decided_rate": _rate(pairs["decided"]),
        "order_accuracy": _rate(d["order_correct"].astype(bool)) if len(d) else None,
        "verdict_accuracy": _rate(d["verdict_true"] == d["verdict_pred"]) if len(d) else None,
        "n_item_pairs": distinct(d),
        "bins": [],
    }
    for lo, hi in BINS:
        b = d[(d["true_diff_pct"] >= lo) & (d["true_diff_pct"] < hi)]
        out["bins"].append(
            {
                "bin": bin_label(lo, hi),
                "n": len(b),
                "n_item_pairs": distinct(b),
                "order_accuracy": _rate(b["order_correct"].astype(bool)) if len(b) else None,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def evaluate(dp: DataPaths, cfg: Config, reports_dir: Path) -> dict:
    """Compute every metric in Section 7 of the design, all leave-one-item-out.

    Args:
        dp: Dataset locations; `features.csv` must already exist.
        cfg: Settings (τ, δ, calibration and models folder).
        reports_dir: Folder with `training_report.json`, where results are also written.

    Returns:
        The metrics, also saved as `metrics.json`. `results_table.md`, `eval_rows.csv`
        and `eval_pairs.csv` are written next to it.
    """
    gt = load_ground_truth(dp)
    feats = pd.read_csv(dp.features, keep_default_na=False, na_values=[""])
    train_report = json.loads((reports_dir / "training_report.json").read_text())
    singles = training_rows(feats)
    thresholds = size_thresholds(gt)
    crop_kind = train_report["crop_classifier"]["chosen"]
    chosen_w = {c: v["chosen"] for c, v in train_report["weight"].items()}
    proxy_w = {c: (w if w.startswith("proxy") else "proxy") for c, w in chosen_w.items()}

    targets = feats.loc[feats["matched"].astype(bool) & (feats["veg_id"].fillna("") != "")].reset_index(drop=True)
    oof = oof_predictions(singles, targets, thresholds)
    oof["crop_pred"] = oof[f"crop_pred_{crop_kind}"]
    oof["crop_prob"] = oof[f"crop_prob_{crop_kind}"]
    oof["grouped_ok"] = (oof["crop_pred"] == oof["crop"]) & (oof["crop_prob"] >= cfg.tau) & ~oof["truncated"].astype(bool)

    def pick(row, mapping, prefix):
        """Look up the prediction from the model chosen for a row's crop.

        Args:
            row: One out-of-fold row.
            mapping: Chosen model name for each crop.
            prefix: Column prefix, e.g. "weight_".

        Returns:
            The prediction, or NaN if no model was chosen for the crop.
        """
        name = mapping.get(row["crop"])
        return row.get(f"{prefix}{name}") if name else np.nan

    oof["w_chosen"] = oof.apply(lambda r: pick(r, chosen_w, "weight_"), axis=1)
    oof["w_proxy"] = oof.apply(lambda r: pick(r, proxy_w, "weight_"), axis=1)
    oof["w_multi"] = oof.get("weight_ridge_log")
    oof["big_true"] = [w >= thresholds[c] for w, c in zip(oof["weight_g"], oof["crop"])]
    reports_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(reports_dir / "eval_rows.csv", index=False)

    is_single = oof["set"] == "singles"
    m: dict = {"data": {"synthetic": dp.synthetic_flag.exists(), "n_items": int(gt.shape[0]), "n_photos": len(set(feats["image"]))}}
    m["config"] = {"tau": cfg.tau, "delta": cfg.delta, "crop_classifier": crop_kind, "weight_models": chosen_w}

    # Calibration
    calib = Calibration.load(dp.calibration)
    cal: dict[str, Any] = {"mm_per_px": calib.mm_per_px}
    if dp.synthetic_flag.exists():
        cal["true_mm_per_px"] = json.loads(dp.synthetic_flag.read_text()).get("true_mm_per_px")
    s = oof.loc[is_single]
    cal["length_error"] = {}
    for c, g in s.groupby("crop"):
        measured, ruler = _col(g, "length_mm"), _col(g, "true_length_mm")
        cal["length_error"][c] = {
            "mae_mm": round(float(np.mean(np.abs(measured - ruler))), 2),
            "mean_error_mm": round(float(np.mean(measured - ruler)), 2),
            "mape_pct": round(float(100 * np.mean(np.abs(measured - ruler) / ruler)), 2),
        }
    m["calibration"] = cal

    # Segmentation
    photos = feats.drop_duplicates("image")
    seg = {"detection_count_accuracy": {k: _rate(g["matched"].astype(bool)) for k, g in photos.groupby("set")}}
    seg["reference_iou"] = reference_iou(dp, cfg, calib)
    m["segmentation"] = seg

    # Crop classifier and grouping
    crops = sorted(oof["crop"].unique())
    conf = pd.crosstab(s["crop"], s["crop_pred"]).reindex(index=crops, columns=crops, fill_value=0)
    mixed = oof[oof["set"] == "mixed"]
    mixed_photos = photos[photos["set"] == "mixed"]
    all_ok = mixed.groupby("image")["grouped_ok"].all().reindex(mixed_photos["image"]).fillna(False)  # unmatched photo = failure
    m["crop_classifier"] = {
        "singles_accuracy": {k: _rate(s[f"crop_pred_{k}"] == s["crop"]) for k in ("logreg", "knn")},
        "all_sets_accuracy": _rate(oof["crop_pred"] == oof["crop"]),
        "confusion_singles": {r: conf.loc[r].to_dict() for r in crops},
        "mixed_items_grouped_correctly": _rate(mixed["grouped_ok"]),
        "mixed_photos_all_grouped_correctly": _rate(all_ok.astype(bool)),
        "uncertain_rate": _rate(oof["crop_prob"] < cfg.tau),
    }

    # Weight (singles, per crop, every candidate)
    m["weight"] = {
        c: {w: weight_scores(_col(g, "weight_g"), _col(g, f"weight_{w}")) for w in WEIGHT_NAMES if f"weight_{w}" in g}
        for c, g in s.groupby("crop")
    }

    # Pairwise (pairs + mixed photos), for the deployed, volume-proxy and multi-feature regressors
    compare_rows = oof.loc[oof["set"].isin(["pairs", "mixed"])]
    pairs = {k: pair_rows(compare_rows, col, cfg.delta) for k, col in (("chosen", "w_chosen"), ("proxy", "w_proxy"), ("multi", "w_multi"))}
    pairs["chosen"].to_csv(reports_dir / "eval_pairs.csv", index=False)
    # Random pairings rarely put near-identical items in one photo, so also score every
    # same-crop pair of items using each item's mean leave-one-out estimate from its singles.
    pairs["all_item_pairs"] = item_pairs(s, cfg.delta)
    m["pairwise"] = {k: pairwise_summary(v) for k, v in pairs.items()}
    m["pairwise"]["chosen_per_crop"] = {c: pairwise_summary(g) for c, g in pairs["chosen"].groupby("crop")}

    # Small/Big (singles): every size model x weight model, plus baselines
    sb = {}
    for sname in SIZE_NAMES:
        for w in ("proxy", "multi", "chosen"):
            wn = {"proxy": proxy_w, "multi": dict.fromkeys(chosen_w, "ridge_log"), "chosen": chosen_w}[w]
            col = s.apply(lambda r, sname=sname, wn=wn: r.get(f"size_{sname}_{wn.get(r['crop'])}"), axis=1)
            ok = np.asarray(col.notna(), dtype=bool)
            if ok.any():
                truth = np.asarray(s["big_true"], dtype=bool)[ok]
                sb[f"{sname}|{w}"] = classification_scores(truth, np.asarray(col, dtype=object)[ok].astype(bool))
    m["small_big"] = {"thresholds_g": thresholds, "models": sb}
    for name in ("hog_svm", "cnn"):
        p = reports_dir / f"baseline_{name}.json"
        if p.exists():
            m["small_big"][f"baseline_{name}"] = json.loads(p.read_text())

    # Robustness: stress set by case
    stress_photos = photos[photos["set"] == "stress"]
    rob = {}
    for case, ph in stress_photos.groupby("case"):
        rows = oof.loc[(oof["set"] == "stress") & (oof["case"] == case)]
        pr = pair_rows(rows, "w_chosen", cfg.delta)
        rob[case] = {
            "n_photos": len(ph),
            "detection_count_accuracy": _rate(ph["matched"].astype(bool)),
            "crop_accuracy": _rate(rows["crop_pred"] == rows["crop"]) if len(rows) else None,
            "weight_mape_pct": round(
                float(100 * np.mean(np.abs(_col(rows, "w_chosen") - _col(rows, "weight_g")) / _col(rows, "weight_g"))), 2
            )
            if len(rows)
            else None,
            "pair_order_accuracy": pairwise_summary(pr).get("order_accuracy"),
            "pairs_decided_rate": pairwise_summary(pr).get("decided_rate"),
        }
    m["robustness"] = rob

    m["efficiency"] = latency(dp, cfg)
    m["headline"] = headline(m, s, proxy_w)
    if m["data"]["synthetic"]:
        m["data"]["warning"] = "SYNTHETIC DATA: these numbers validate the code, not real-world accuracy."
    (reports_dir / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
    (reports_dir / "results_table.md").write_text(results_table(m), encoding="utf-8")
    return m


def reference_iou(dp: DataPaths, cfg: Config, calib: Calibration) -> dict:
    """Compare the segmentation with the hand-traced outlines.

    Args:
        dp: Dataset locations.
        cfg: Segmentation settings.
        calib: The rig's calibration.

    Returns:
        A dict with the number of traced photos and items and their mean IoU (None if
        there are no traced photos).
    """
    files = sorted(dp.reference_masks.glob("*.txt")) if dp.reference_masks.exists() else []
    ious = []
    for f in files:
        matches = list(dp.raw.glob(f.stem + ".*"))
        if not matches:
            continue
        img = imread(matches[0])
        h, w = img.shape[:2]
        refs = [polygon_mask(p, h, w) for p in read_polygons(f, w, h)]
        mmpp, _ = calib.for_image(img.shape)
        pred = [d.mask for d in segment(white_balance(img), cfg, mmpp)]
        ious += list(match_masks(pred, refs).values())
    return {"n_photos": len(files), "n_items": len(ious), "mean_iou": round(float(np.mean(ious)), 4) if ious else None}


def latency(dp: DataPaths, cfg: Config, n: int = 30) -> dict:
    """Time the deployed pipeline on photos from the pairs and mixed sets.

    Args:
        dp: Dataset locations.
        cfg: Settings used to build the pipeline.
        n: Maximum number of photos to time.

    Returns:
        Mean and 95th-percentile latency in milliseconds, the photo size and the size of
        each model file, or an error message if the pipeline could not be loaded.
    """
    from src.pipeline import Pipeline

    try:
        pipe = Pipeline(cfg)
    except FileNotFoundError as e:
        return {"error": str(e)}
    images = load_images(dp)
    names = list(images[images["set"].isin(["pairs", "mixed"])]["image"])[:n]
    ms = []
    for name in names:
        img = imread(dp.raw / name)
        t0 = perf_counter()
        pipe.run(img)
        ms.append(1000 * (perf_counter() - t0))
    shape = imread(dp.raw / names[0]).shape if names else (0, 0)
    return {
        "photos": len(ms),
        "image_size": [shape[1], shape[0]],
        "latency_ms_mean": round(float(np.mean(ms)), 1) if ms else None,
        "latency_ms_p95": round(float(np.percentile(ms, 95)), 1) if ms else None,
        "model_files_kb": {p.name: round(p.stat().st_size / 1e3, 1) for p in sorted(Path(cfg.models_dir).glob("*.joblib"))},
    }


def headline(m: dict, singles: pd.DataFrame, proxy_w: dict) -> list[dict]:
    """Build the design's headline results table.

    Columns are weight MAPE, pairwise accuracy and Small/Big accuracy. Rows are the two
    raw-image baselines and the two feature-based approaches.

    Args:
        m: The metrics computed so far.
        singles: Out-of-fold rows from the singles.
        proxy_w: Volume-proxy model name for each crop.

    Returns:
        One dict per approach. Entries that don't apply (e.g. weight for a baseline) are None.
    """

    def mape(col):
        """Weight MAPE for one prediction column, over the singles.

        Args:
            col: Column holding the weight estimate.

        Returns:
            MAPE in percent, or None if the column has no values.
        """
        g = singles.dropna(subset=[col])
        return round(float(100 * ((g[col] - g["weight_g"]).abs() / g["weight_g"]).mean()), 1) if len(g) else None

    sb = m["small_big"]
    rows = []
    for name, label in (("hog_svm", "Raw-image HOG + SVM (baseline)"), ("cnn", "Raw-image CNN (baseline)")):
        b = sb.get(f"baseline_{name}")
        rows.append(
            {"approach": label, "weight_mape_pct": None, "pairwise_acc": None, "small_big_acc": b["overall"]["accuracy"] if b else None}
        )
    rows.append(
        {
            "approach": "Features + volume proxy (this work)",
            "weight_mape_pct": mape("w_proxy"),
            "pairwise_acc": m["pairwise"]["proxy"].get("order_accuracy"),
            "small_big_acc": sb["models"].get("weight_threshold|proxy", {}).get("accuracy"),
        }
    )
    rows.append(
        {
            "approach": "Features + SVM (this work)",
            "weight_mape_pct": mape("w_multi"),
            "pairwise_acc": m["pairwise"]["multi"].get("order_accuracy"),
            "small_big_acc": sb["models"].get("svm|multi", {}).get("accuracy"),
        }
    )
    return rows


def results_table(m: dict) -> str:
    """Render the headline table as Markdown, with a note on how to read it.

    Args:
        m: The full metrics, including "headline".

    Returns:
        The Markdown text (with the synthetic-data warning first, when it applies).
    """

    def f(v, pct=False):
        """Format one table cell.

        Args:
            v: The value, or None.
            pct: True if `v` is already a percentage, False if it is a fraction.

        Returns:
            The value as a percentage string, or an em dash for None.
        """
        if v is None:
            return "—"
        return f"{v:.1f}%" if pct else f"{100 * v:.1f}%"

    lines = []
    if m["data"].get("warning"):
        lines += [f"> **{m['data']['warning']}**", ""]
    lines += ["| Approach | Weight MAPE | Pairwise acc. | Small/Big acc. |", "|---|---|---|---|"]
    for r in m["headline"]:
        lines.append(f"| {r['approach']} | {f(r['weight_mape_pct'], pct=True)} | {f(r['pairwise_acc'])} | {f(r['small_big_acc'])} |")
    lines += [
        "",
        "All numbers are leave-one-item-out. Weight MAPE and Small/Big use the singles; pairwise uses",
        "same-crop pairs in the pairs and mixed photos. The volume-proxy row uses `weight = a * proxy`",
        "and a threshold on that estimate; the SVM row uses the multi-feature ridge regressor and an",
        "RBF SVM on the mm features plus its weight estimate.",
    ]
    return "\n".join(lines) + "\n"


def print_summary(m: dict) -> None:
    """Print a readable summary of the metrics to the console.

    Args:
        m: The metrics returned by `evaluate`.
    """
    if m["data"].get("warning"):
        print(f"\n*** {m['data']['warning']} ***")
    c = m["calibration"]
    print(f"\nCalibration: {c['mm_per_px']:.4f} mm/px" + (f" (true {c['true_mm_per_px']})" if c.get("true_mm_per_px") else ""))
    for crop, v in c["length_error"].items():
        print(f"  {crop:11s} length MAE {v['mae_mm']:.1f} mm ({v['mape_pct']:.1f}%), bias {v['mean_error_mm']:+.1f} mm")
    s = m["segmentation"]
    ref = s["reference_iou"]
    print(f"\nSegmentation: reference IoU {ref['mean_iou']} on {ref['n_items']} items")
    print(f"  detection count correct: {s['detection_count_accuracy']}")
    cc = m["crop_classifier"]
    print(f"\nCrop classifier: singles {cc['singles_accuracy']}, all sets {cc['all_sets_accuracy']}")
    items_ok, photos_ok = cc["mixed_items_grouped_correctly"], cc["mixed_photos_all_grouped_correctly"]
    print(f"  mixed photos: items grouped correctly {items_ok}, photos all correct {photos_ok}")
    print("\nWeight (singles, LOO)")
    for crop, per in m["weight"].items():
        print("  " + crop.ljust(11) + "  ".join(f"{w} {v['mape_pct']:.1f}%" for w, v in per.items()))
    p = m["pairwise"]["chosen"]
    if p.get("n_pairs"):
        bins = "  ".join(
            f"{b['bin']}: {'-' if b['order_accuracy'] is None else b['order_accuracy']} ({b['n']}/{b['n_item_pairs']})" for b in p["bins"]
        )
        print(f"\nPairwise (deployed regressor): {p['order_accuracy']}  decided {p['decided_rate']:.0%}  | {bins}")
    print(
        "\nSmall/Big (singles, LOO): "
        + ", ".join(f"{k} {v['accuracy']:.3f}" for k, v in m["small_big"]["models"].items() if k.endswith("|chosen"))
    )
    print("\nStress set")
    for case, v in m["robustness"].items():
        det, acc = v["detection_count_accuracy"], v["crop_accuracy"]
        print(f"  {case:9s} detections {det}  crop acc {acc}  weight MAPE {v['weight_mape_pct']}%  pair acc {v['pair_order_accuracy']}")
    e = m["efficiency"]
    print(f"\nLatency: {e.get('latency_ms_mean')} ms mean, {e.get('latency_ms_p95')} ms p95 on {e.get('image_size')} photos")
    print("\n" + results_table(m).replace("—", "-"))  # Windows consoles may not print an em dash


def main() -> None:
    """Command-line entry point: run the evaluation and print a summary."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DataPaths().root)
    p.add_argument("--reports", type=Path, default=REPORTS_DIR)
    args = p.parse_args()
    dp = DataPaths(args.data)
    cfg = Config(calibration_file=dp.calibration)
    m = evaluate(dp, cfg, args.reports)
    print_summary(m)
    print(f"wrote {args.reports / 'metrics.json'} and {args.reports / 'results_table.md'}")


if __name__ == "__main__":
    main()
