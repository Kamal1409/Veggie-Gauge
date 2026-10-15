"""Reading the collected dataset. No annotation is needed.

data/ground_truth.csv     veg_id,crop,weight_g,length_mm,notes
data/images.csv           image,veg_ids,set,case
                          `veg_ids`: items in the photo separated by ';', listed LEFT TO RIGHT
                          by their centre. Detections are sorted the same way, which is how
                          each detection is matched to its weighed item.
                          `set`: singles | pairs | mixed | stress;  `case` (stress only):
                          touching | tilt | dim
data/calibration.json     {"mm_per_px": ..., "image_size": [w, h]}  (src/calibrate.py)
data/reference_masks/     optional hand-traced polygons (YOLO-seg .txt) for ~10 photos
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import DataPaths


def imread(path: Path | str) -> np.ndarray:
    """Read an image, failing loudly instead of returning None like `cv2.imread` does.

    Args:
        path: Image file to read.

    Returns:
        The image as a BGR array.

    Raises:
        FileNotFoundError: If the file is missing or cannot be decoded.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return img


def load_ground_truth(dp: DataPaths) -> pd.DataFrame:
    """Load the weighed items.

    Args:
        dp: Dataset locations.

    Returns:
        The ground-truth table, indexed by `veg_id`.
    """
    return pd.read_csv(dp.ground_truth, dtype={"veg_id": str}).set_index("veg_id")


def load_images(dp: DataPaths) -> pd.DataFrame:
    """Load the photo index, with each photo's item IDs split into a list.

    Missing `set` or `case` columns are added as empty strings, so older index files
    still load.

    Args:
        dp: Dataset locations.

    Returns:
        One row per photo. `veg_ids` holds the photo's item IDs in left-to-right order.
    """
    df = pd.read_csv(dp.images, dtype=str, keep_default_na=False)
    df["veg_ids"] = df["veg_ids"].map(lambda s: [v for v in str(s).split(";") if v])
    for col in ("set", "case"):
        if col not in df:
            df[col] = ""
    return df


def match_by_order(n_detections: int, veg_ids: list[str]) -> list[str] | None:
    """Pair detections with weighed items by their left-to-right order.

    Both the detections and the IDs in `images.csv` run left to right, so they line
    up one to one, but only if segmentation found exactly the expected number of items.

    Args:
        n_detections: Number of items the segmentation found.
        veg_ids: Item IDs listed for the photo, left to right.

    Returns:
        The IDs in detection order, or None if the counts disagree.
    """
    return list(veg_ids) if n_detections == len(veg_ids) else None


def read_polygons(path: Path, width: int, height: int) -> list[np.ndarray]:
    """Read hand-traced outlines from a YOLO-seg style text file.

    Each line holds a class id followed by normalised `x y` pairs; the class id is ignored.

    Args:
        path: The polygon file. A missing file simply gives no polygons.
        width: Width of the photo, to convert back to pixels.
        height: Height of the photo, to convert back to pixels.

    Returns:
        One `(N, 2)` array of pixel coordinates per outline.
    """
    polys = []
    if path.exists():
        for line in path.read_text().splitlines():
            vals = line.split()
            if len(vals) >= 7:
                polys.append(np.array(vals[1:], dtype=float).reshape(-1, 2) * [width, height])
    return polys


def polygon_mask(poly: np.ndarray, height: int, width: int) -> np.ndarray:
    """Fill a polygon into a binary mask.

    Args:
        poly: `(N, 2)` pixel coordinates of the outline.
        height: Height of the mask.
        width: Width of the mask.

    Returns:
        A boolean mask that is True inside the polygon.
    """
    m = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(m, [poly.round().astype(np.int32)], 1)
    return m.astype(bool)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two binary masks.

    Args:
        a: First mask.
        b: Second mask, the same shape as `a`.

    Returns:
        IoU between 0 and 1 (0 if both masks are empty).
    """
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def match_masks(pred: list[np.ndarray], gt: list[np.ndarray]) -> dict[int, float]:
    """Match predicted masks to reference masks one to one, best overlaps first.

    Args:
        pred: Predicted masks.
        gt: Reference (hand-traced) masks.

    Returns:
        The IoU of each reference mask with the prediction it was matched to, keyed by
        the reference mask's index. A reference mask left unmatched gets 0.
    """
    pairs = sorted(((mask_iou(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt)), reverse=True)
    used_p, out = set(), {j: 0.0 for j in range(len(gt))}
    for iou, i, j in pairs:
        if iou <= 0 or i in used_p or out[j] > 0:
            continue
        used_p.add(i)
        out[j] = iou
    return out
