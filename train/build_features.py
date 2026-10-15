"""Stage 4: run segmentation + feature extraction on every photo -> data/features.csv.

Detections are matched to weighed items by left-to-right order (data/images.csv lists the
item IDs that way). No annotation is involved. Photos whose detection count does not match
are kept with `matched = False`, so the evaluation can report them as segmentation failures.

    python -m train.build_features [--data data] [--overlays]

`--overlays` saves every mask overlay to reports/overlays/ for the visual quality check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.calibrate import Calibration
from src.config import REPORTS_DIR, Config, DataPaths
from src.data import imread, load_ground_truth, load_images, match_by_order
from src.features import extract_features
from src.segment import segment, white_balance


def build(dp: DataPaths, cfg: Config, overlays: Path | None = None) -> pd.DataFrame:
    """Segment every photo in the dataset and measure every item found.

    Detections are matched to weighed items by left-to-right order. A photo where the
    number of detections does not match `images.csv` is still recorded, with
    `matched = False`, so the evaluation can count it as a segmentation failure.

    Args:
        dp: Dataset locations.
        cfg: Segmentation settings.
        overlays: Folder to save a mask overlay of every photo into, or None to skip.

    Returns:
        One row per detected item (or one placeholder row for a photo with no
        detections), with its photo, matched item ID, ground truth and features.
    """
    gt = load_ground_truth(dp)
    images = load_images(dp)
    calib = Calibration.load(dp.calibration)
    crop_of = gt["crop"].astype(str).to_dict()
    weight_of = gt["weight_g"].astype(float).to_dict()
    length_of = gt["length_mm"].astype(float).to_dict()
    rows = []
    for r in images.to_dict("records"):
        img = imread(dp.raw / str(r["image"]))
        mmpp, _ = calib.for_image(img.shape)
        balanced = white_balance(img)
        dets = segment(balanced, cfg, mmpp)
        ids = match_by_order(len(dets), r["veg_ids"])
        for k, d in enumerate(dets):
            vid = ids[k] if ids else ""
            rows.append(
                {
                    "image": r["image"],
                    "set": r["set"],
                    "case": r["case"],
                    "session": r.get("session", ""),
                    "det": k,
                    "n_detected": len(dets),
                    "n_expected": len(r["veg_ids"]),
                    "matched": ids is not None,
                    "veg_id": vid,
                    "crop": crop_of[vid] if vid else "",
                    "weight_g": weight_of[vid] if vid else np.nan,
                    "true_length_mm": length_of[vid] if vid else np.nan,
                    "truncated": "touches_border" in d.flags,
                    "cx": d.centroid[0],
                    "cy": d.centroid[1],
                    **extract_features(d.mask, balanced, mmpp),
                }
            )
        if not dets:
            rows.append(
                {
                    "image": r["image"],
                    "set": r["set"],
                    "case": r["case"],
                    "det": -1,
                    "n_detected": 0,
                    "n_expected": len(r["veg_ids"]),
                    "matched": False,
                }
            )
        if overlays is not None:
            ov = img.copy()
            for k, d in enumerate(dets):
                contours, _ = cv2.findContours(d.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(ov, contours, -1, (214, 120, 42), 3)
                cx, cy = d.centroid
                cv2.putText(ov, f"{k}:{ids[k] if ids else '?'}", (int(cx) - 30, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            cv2.imwrite(str(overlays / r["image"]), ov, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return pd.DataFrame(rows)


def main() -> None:
    """Command-line entry point: build `data/features.csv`."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DataPaths().root)
    p.add_argument("--overlays", action="store_true")
    args = p.parse_args()
    dp = DataPaths(args.data)
    ov = None
    if args.overlays:
        ov = REPORTS_DIR / "overlays"
        ov.mkdir(parents=True, exist_ok=True)
    df = build(dp, Config(), ov)
    df.to_csv(dp.features, index=False)
    photos = df.drop_duplicates("image")
    print(f"wrote {len(df)} rows from {len(photos)} photos to {dp.features}")
    for s, g in photos.groupby("set"):
        print(f"  {s:8s} detection count correct in {g['matched'].mean():.0%} of {len(g)} photos")


if __name__ == "__main__":
    main()
