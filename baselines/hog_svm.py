"""Baseline: HOG features + SVM on raw image patches -> SMALL / BIG.

python -m baselines.hog_svm [--data data] [--px 128]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from baselines.common import leave_one_item_out, load_patches, write_report
from src.config import DataPaths


def hog(gray: np.ndarray, cell: int = 16, bins: int = 9) -> np.ndarray:
    """Compute a Dalal-Triggs HOG descriptor.

    It uses unsigned gradient orientations, 2×2-cell blocks and L2-Hys normalisation.
    It is written out here because OpenCV 5 no longer ships `cv2.HOGDescriptor` in the
    main package.

    Args:
        gray: Grayscale image.
        cell: Cell size in pixels.
        bins: Number of orientation bins over 0–180°.

    Returns:
        The descriptor as a flat array.
    """
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=1)
    mag = np.hypot(gx, gy)
    ang = np.degrees(np.arctan2(gy, gx)) % 180
    ny, nx = g.shape[0] // cell, g.shape[1] // cell
    idx = np.minimum((ang / (180 / bins)).astype(int), bins - 1)
    hist = np.zeros((ny, nx, bins), np.float32)
    for b in range(bins):
        w = (mag * (idx == b))[: ny * cell, : nx * cell]
        hist[..., b] = w.reshape(ny, cell, nx, cell).sum(axis=(1, 3))
    blocks = []
    for y in range(ny - 1):
        for x in range(nx - 1):
            v = hist[y : y + 2, x : x + 2].ravel()
            v = v / (np.linalg.norm(v) + 1e-6)
            v = np.minimum(v, 0.2)
            blocks.append(v / (np.linalg.norm(v) + 1e-6))
    return np.concatenate(blocks)


def hog_features(patches: np.ndarray) -> np.ndarray:
    """HOG descriptors for a stack of BGR patches.

    Args:
        patches: Patches of shape `(N, px, px, 3)`.

    Returns:
        An `(N, D)` array of descriptors.
    """
    return np.stack([hog(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)) for p in patches])


def fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Train an RBF SVM on HOG features and predict SMALL / BIG.

    Args:
        X_train: Training descriptors.
        y_train: Training labels (True = BIG).
        X_test: Descriptors to classify.

    Returns:
        Predicted BIG flags for `X_test`.
    """
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
    return model.fit(X_train, y_train).predict(X_test).astype(bool)


def main() -> None:
    """Command-line entry point: run the HOG + SVM baseline."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DataPaths().root)
    p.add_argument("--px", type=int, default=128, help="patch size after the uniform downscale")
    args = p.parse_args()
    dp = DataPaths(args.data)
    rows, patches, meta = load_patches(dp, args.px)
    preds = leave_one_item_out(rows, hog_features(patches), meta["thresholds"], fit_predict)
    rep = write_report("hog_svm", preds, {"patch_px": args.px, "side_mm": meta["side_mm"]}, dp.synthetic_flag.exists())
    print(f"HOG+SVM Small/Big (leave-one-item-out): accuracy {rep['overall']['accuracy']:.3f}, F1 {rep['overall']['f1_big']:.3f}")
    for c, v in rep["per_crop"].items():
        print(f"  {c:11s} accuracy {v['accuracy']:.3f}  F1 {v['f1_big']:.3f}")


if __name__ == "__main__":
    main()
