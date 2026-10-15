"""Stage D: per-crop regressors from top-down measurements to fresh weight (grams).

The primary model is physical: weight = a * volume_proxy (plus an intercept if it helps),
with a shape-aware proxy per crop (sphere for tomato, flat cylinder / ellipsoid for kheera
and chamagadda). A ridge regression on log size features is the alternative it competes with.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import DEFAULT_PROXY, VOLUME_PROXY
from src.features import SIZE_FEATURES


class ProxyRegressor(BaseEstimator, RegressorMixin):
    """Weight as a straight multiple of a volume proxy: `weight = a * proxy (+ b)`.

    One or two parameters, so it is easy to interpret and very hard to overfit with
    10–20 items per crop.

    Attributes:
        fit_intercept: Whether to fit the offset `b` as well as the slope `a`.
        lr_: The underlying fitted linear regression (set by `fit`).
    """

    def __init__(self, fit_intercept: bool = False):
        """Choose between a line through the origin and one with an offset.

        Args:
            fit_intercept: Whether to fit an offset as well as the slope.
        """
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """Fit the slope (and offset) from proxy values and measured weights.

        Args:
            X: Feature matrix whose first column is the volume proxy.
            y: Measured weights in grams.

        Returns:
            The fitted regressor itself, as scikit-learn expects.
        """
        self.lr_ = LinearRegression(fit_intercept=self.fit_intercept).fit(np.asarray(X, float)[:, :1], y)
        return self

    def predict(self, X):
        """Estimate weights from proxy values.

        Args:
            X: Feature matrix whose first column is the volume proxy.

        Returns:
            Estimated weights in grams.
        """
        return self.lr_.predict(np.asarray(X, float)[:, :1])


def log_ridge():
    """Build the multi-feature alternative: ridge regression on log features.

    It predicts log weight from log sizes, because sizes multiply into volume, so
    their logs add up.

    Returns:
        An unfitted scikit-learn regressor that takes raw (positive) features and
        predicts grams.
    """
    inner = make_pipeline(FunctionTransformer(np.log), StandardScaler(), Ridge(alpha=1.0))
    return TransformedTargetRegressor(regressor=inner, func=np.log, inverse_func=np.exp)


def proxy_for(crop: str) -> str:
    """Which volume proxy suits a crop's shape.

    Args:
        crop: Crop name.

    Returns:
        The feature name: "proxy_sphere" (area^1.5) for round crops and
        "proxy_cylinder" (area × minor axis) for elongated or flattened ones. Unknown
        crops get the sphere proxy.
    """
    return VOLUME_PROXY.get(crop, DEFAULT_PROXY)


def candidates(crop: str) -> dict:
    """The weight models to try for a crop, simplest first.

    Model selection keeps the simplest model within one standard error of the best,
    so the order matters.

    Args:
        crop: Crop name, which decides the volume proxy.

    Returns:
        An ordered dict mapping each model name to `(feature names, factory)`.
    """
    proxy = proxy_for(crop)
    return {
        "proxy": ([proxy], lambda: ProxyRegressor(fit_intercept=False)),
        "proxy+intercept": ([proxy], lambda: ProxyRegressor(fit_intercept=True)),
        "ridge_log": (SIZE_FEATURES, log_ridge),
    }


def model_path(models_dir: Path, crop: str) -> Path:
    """Where the weight model for a crop is saved.

    Args:
        models_dir: Folder holding the trained models.
        crop: Crop name.

    Returns:
        Path to `weight_<crop>.joblib`.
    """
    return Path(models_dir) / f"weight_{crop}.joblib"


class WeightModels:
    """Loads each crop's weight regressor on first use and estimates weights with it.

    The bundles are written by `train/train_models.py`.

    Attributes:
        models_dir: Folder holding the `weight_<crop>.joblib` files.
    """

    def __init__(self, models_dir: Path):
        """Set up the loader; nothing is read from disk until a crop is first needed.

        Args:
            models_dir: Folder holding the `weight_<crop>.joblib` files.
        """
        self.models_dir = Path(models_dir)
        self._cache: dict[str, dict | None] = {}

    def bundle(self, crop: str) -> dict | None:
        """The saved weight bundle for a crop, loaded once and then cached.

        Args:
            crop: Crop name.

        Returns:
            The bundle (model name, features, model, CV error), or None if no model was
            trained for this crop.
        """
        if crop not in self._cache:
            p = model_path(self.models_dir, crop)
            self._cache[crop] = joblib.load(p) if p.exists() else None
        return self._cache[crop]

    def predict(self, crop: str, f: dict) -> tuple[float, float] | None:
        """Estimate one item's fresh weight.

        Args:
            crop: The item's crop.
            f: The item's features, as returned by `extract_features`.

        Returns:
            `(weight_g, error_g)`, where the error is the model's cross-validated MAE, or
            None if there is no model for the crop. Negative estimates are clipped to 0.
        """
        b = self.bundle(crop)
        if b is None:
            return None
        x = np.array([[f[k] for k in b["features"]]], dtype=float)
        return max(float(b["model"].predict(x)[0]), 0.0), float(b["cv_mae_g"])
