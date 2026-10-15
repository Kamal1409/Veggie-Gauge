"""Stage C: which crop is each item (label + probability), so items can be grouped by crop.

The classifier works on colour + shape features (no raw pixels). Its labels are free:
each training photo contains a single crop, so the photo's crop is the label.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import CROP_FEATURES

CANDIDATES = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
    "knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
}


def feature_matrix(rows: list[dict]) -> np.ndarray:
    """Stack the crop-classifier features of several items into one matrix.

    Args:
        rows: Feature dicts, as returned by `extract_features`.

    Returns:
        An array of shape `(len(rows), len(CROP_FEATURES))`.
    """
    return np.array([[r[k] for k in CROP_FEATURES] for r in rows], dtype=float)


def predict(bundle: dict, rows: list[dict]) -> tuple[list[str], list[float]]:
    """Name the crop of each item, with the classifier's confidence.

    Args:
        bundle: The loaded `crop_classifier.joblib` bundle.
        rows: Feature dicts, as returned by `extract_features`.

    Returns:
        Two lists in the same order as `rows`: the predicted crop names and the
        probability the classifier gave each one.
    """
    model = bundle["model"]
    proba = model.predict_proba(feature_matrix(rows))
    classes = list(model.classes_)
    best = proba.argmax(axis=1)
    return [classes[i] for i in best], [float(proba[k, i]) for k, i in enumerate(best)]
