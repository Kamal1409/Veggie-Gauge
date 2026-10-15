"""Stage E, absolute mode: is this item SMALL or BIG for its crop?

BIG means the measured weight is at or above the crop's threshold. The threshold is the
median weight of the bought items, a documented constant that can be swapped for a market
or ARL grading standard.

Thresholding the *estimated* weight is itself a strong classifier, so it is the default.
An SVM or logistic regression replaces it only if it beats it in leave-one-item-out CV.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.features import SIZE_FEATURES

SMALL, BIG = "SMALL", "BIG"
ML_FEATURES = SIZE_FEATURES + ["est_weight_g"]

CANDIDATES = {
    "weight_threshold": None,  # rule, no fitting
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
    "svm": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")),
}


def model_path(models_dir: Path, crop: str) -> Path:
    """Where the Small/Big model for a crop is saved.

    Args:
        models_dir: Folder holding the trained models.
        crop: Crop name.

    Returns:
        Path to `size_<crop>.joblib`.
    """
    return Path(models_dir) / f"size_{crop}.joblib"


def predict_rows(name: str, model, threshold_g: float, X: np.ndarray, est_weight: np.ndarray) -> np.ndarray:
    """Say which items are BIG, for one Small/Big candidate model.

    Args:
        name: Candidate name: "weight_threshold", "logreg" or "svm".
        model: The fitted model, or None for the threshold rule.
        threshold_g: The crop's BIG threshold in grams (used by the rule).
        X: Feature matrix in `ML_FEATURES` order (used by the learned models).
        est_weight: Estimated weight of each item in grams (used by the rule).

    Returns:
        A boolean array: True where the item is BIG.
    """
    if name == "weight_threshold":
        return est_weight >= threshold_g
    return model.predict(X).astype(bool)


class SizeClassifiers:
    """Loads each crop's Small/Big model on first use and grades items with it.

    Attributes:
        models_dir: Folder holding the `size_<crop>.joblib` files.
    """

    def __init__(self, models_dir: Path):
        """Set up the loader; nothing is read from disk until a crop is first needed.

        Args:
            models_dir: Folder holding the `size_<crop>.joblib` files.
        """
        self.models_dir = Path(models_dir)
        self._cache: dict[str, dict | None] = {}

    def bundle(self, crop: str) -> dict | None:
        """The saved Small/Big bundle for a crop, loaded once and then cached.

        Args:
            crop: Crop name.

        Returns:
            The bundle (model name, model, threshold, CV scores), or None if no model was
            trained for this crop.
        """
        if crop not in self._cache:
            p = model_path(self.models_dir, crop)
            self._cache[crop] = joblib.load(p) if p.exists() else None
        return self._cache[crop]

    def predict(self, crop: str, f: dict, est_weight_g: float | None) -> str | None:
        """Grade one item SMALL or BIG for its crop.

        Args:
            crop: The item's crop.
            f: The item's features, as returned by `extract_features`.
            est_weight_g: The item's estimated weight in grams.

        Returns:
            "SMALL" or "BIG", or None if there is no model for the crop or no weight estimate.
        """
        b = self.bundle(crop)
        if b is None or est_weight_g is None:
            return None
        x = np.array([[{**f, "est_weight_g": est_weight_g}[k] for k in ML_FEATURES]], dtype=float)
        big = predict_rows(b["model_name"], b["model"], b["threshold_g"], x, np.array([est_weight_g]))[0]
        return BIG if big else SMALL
