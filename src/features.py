"""Stage B: per-contour features, converted to millimetres with the rig's mm/px."""

from __future__ import annotations

import cv2
import numpy as np

# Colour + shape features the crop classifier uses. Hue is circular, so it enters as cos/sin.
CROP_FEATURES = ["hue_cos", "hue_sin", "hue_std", "sat_mean", "sat_std", "val_mean", "val_std", "elongation", "solidity"]

# Size features (log-transformed) for the multi-feature weight model.
SIZE_FEATURES = ["area_mm2", "perimeter_mm", "major_mm", "minor_mm", "length_mm", "width_mm"]


def extract_features(mask: np.ndarray, image: np.ndarray, mm_per_px: float) -> dict:
    """Measure one item: its size and shape in millimetres, and its colour.

    Length and width come from the minimum-area rectangle, so they read like a ruler.
    The major and minor axes come from the mask's second moments, which copes better
    with bumpy outlines. Both volume proxies are always computed; the weight model
    decides which one each crop uses.

    Args:
        mask: Boolean mask of the item.
        image: The (white-balanced) BGR photo the mask belongs to, for colour features.
        mm_per_px: Scale of the rig.

    Returns:
        A dict of features: area, perimeter, axes, length and width (mm), elongation,
        solidity, the two volume proxies, and hue / saturation / value statistics. Hue
        is circular, so it is given as a unit vector (`hue_cos`, `hue_sin`) plus a
        circular standard deviation.

    Raises:
        ValueError: If the mask is empty.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")
    x0, y0 = xs.min(), ys.min()
    crop = np.ascontiguousarray(mask[y0 : ys.max() + 1, x0 : xs.max() + 1]).astype(np.uint8)
    crop = cv2.copyMakeBorder(crop, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    s = float(mm_per_px)

    area = float(crop.sum())
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.vstack(contours)
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))

    # Hull area rasterised the same way as the mask, so solidity <= 1.
    hull_img = np.zeros_like(crop)
    cv2.fillConvexPoly(hull_img, cv2.convexHull(pts), 1)
    hull_area = float(np.maximum(hull_img, crop).sum())

    # Ellipse with the mask's second moments (robust for bumpy outlines)...
    m = cv2.moments(crop, binaryImage=True)
    cov = np.array([[m["mu20"], m["mu11"]], [m["mu11"], m["mu02"]]]) / m["m00"]
    lam = np.sort(np.linalg.eigvalsh(cov))[::-1]
    major, minor = 4 * np.sqrt(max(lam[0], 0)), 4 * np.sqrt(max(lam[1], 0))
    # ...and the minimum-area rectangle, whose sides read like a ruler (length x width).
    (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
    length, width = max(rw, rh) + 1, min(rw, rh) + 1

    area_mm2 = area * s * s
    minor_mm = float(minor) * s
    f = {
        "area_mm2": area_mm2,
        "perimeter_mm": perimeter * s,
        "major_mm": float(major) * s,
        "minor_mm": minor_mm,
        "length_mm": float(length) * s,
        "width_mm": float(width) * s,
        "elongation": float(major / minor) if minor > 0 else 1.0,
        "solidity": area / hull_area,
        "proxy_sphere": area_mm2**1.5,
        "proxy_cylinder": area_mm2 * minor_mm,
    }

    pix = image[mask.astype(bool)]
    hsv = cv2.cvtColor(pix.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    ang = hsv[:, 0] * (2 * np.pi / 180)  # OpenCV hue 0-179 -> radians
    c, sn = np.cos(ang).mean(), np.sin(ang).mean()
    r = float(np.hypot(c, sn))
    f.update(
        {
            "hue_cos": float(c / r) if r > 0 else 0.0,
            "hue_sin": float(sn / r) if r > 0 else 0.0,
            "hue_mean": float(np.degrees(np.arctan2(sn, c)) % 360 / 2),
            "hue_std": float(np.sqrt(-2 * np.log(max(r, 1e-9)))),  # circular std (radians)
            "sat_mean": float(hsv[:, 1].mean()),
            "sat_std": float(hsv[:, 1].std()),
            "val_mean": float(hsv[:, 2].mean()),
            "val_std": float(hsv[:, 2].std()),
        }
    )
    return f
