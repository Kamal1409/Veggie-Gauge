"""Paths and tunable parameters shared by every stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

CROPS = ["tomato", "kheera", "chamagadda"]

# Top-down volume proxy per crop (weight ~ density x volume). See docs/PIPELINE.md, Stage 5b.
VOLUME_PROXY = {
    "tomato": "proxy_sphere",  # ~sphere: area^1.5
    "kheera": "proxy_cylinder",  # ~cylinder lying flat: area * minor axis (~ length x width^2)
    "chamagadda": "proxy_cylinder",  # ~ellipsoid: area * minor axis
}
DEFAULT_PROXY = "proxy_sphere"


@dataclass(frozen=True)
class DataPaths:
    """Where each part of the collected dataset lives.

    The layout is described in `src/data.py`. Point `root` somewhere else to work on a
    different dataset, e.g. a temporary one in the tests.

    Attributes:
        root: The dataset folder (defaults to `data/`).
    """

    root: Path = DATA_DIR

    @property
    def raw(self) -> Path:
        """The folder holding every photo, including the ruler photo."""
        return self.root / "raw"

    @property
    def ground_truth(self) -> Path:
        """The CSV of weighed items (`veg_id, crop, weight_g, length_mm, notes`)."""
        return self.root / "ground_truth.csv"

    @property
    def images(self) -> Path:
        """The CSV listing each photo, its item IDs (left to right), photo set and stress case."""
        return self.root / "images.csv"

    @property
    def calibration(self) -> Path:
        """The JSON file holding the rig's mm/px."""
        return self.root / "calibration.json"

    @property
    def features(self) -> Path:
        """The generated CSV with one row of features per detected item."""
        return self.root / "features.csv"

    @property
    def reference_masks(self) -> Path:
        """The folder of hand-traced outlines (YOLO-seg .txt) used for the segmentation IoU check."""
        return self.root / "reference_masks"

    @property
    def synthetic_flag(self) -> Path:
        """A marker file that exists only when the dataset was made by `tools/synth_data.py`."""
        return self.root / "SYNTHETIC"


@dataclass
class Config:
    """Tunable settings for the online pipeline.

    The defaults match the design document; the CLI, API and demo can override τ and δ.

    Attributes:
        calibration_file: Where the rig's mm/px is stored.
        min_sat: Lowest saturation threshold allowed, in case Otsu finds no real split
            (e.g. a photo with no produce in it).
        dark_frac: Pixels darker than this fraction of the sheet's lightness count as
            produce, which is how the brown chamagadda is found.
        open_mm: Size of the opening that removes specks and chamagadda hairs.
        close_mm: Size of the closing that fills small gaps in a mask.
        min_area_mm2: Smaller blobs are treated as noise.
        split_touching: Whether to split blobs that contain several touching items.
        split_depth: How deep the neck between two items must be, as a fraction of an
            item's peak inner distance, for them to be split apart.
        border_px: Items within this many pixels of the frame edge are flagged as cut off.
        tau: Lowest crop-classifier probability for an item to be used.
        exclude_truncated: Leave items cut off by the frame out of every comparison.
        delta: Relative difference at or below which two items are SIMILAR.
        models_dir: Folder holding the trained `*.joblib` models.
    """

    # Calibration (fixed camera): mm/px comes from data/calibration.json
    calibration_file: Path = DATA_DIR / "calibration.json"

    # A. Segmentation
    min_sat: int = 40  # floor under Otsu's saturation threshold (a sheet with no produce has no real split)
    dark_frac: float = 0.6  # "dark" = lightness below this fraction of the sheet's lightness (brown chamagadda)
    open_mm: float = 2.0  # opening removes specks and chamagadda hairs thinner than ~this
    close_mm: float = 3.0
    min_area_mm2: float = 300.0  # smaller blobs are noise
    split_touching: bool = True  # distance transform + watershed
    split_depth: float = 0.25  # a seed counts as a separate item if its neck is this deep (fraction of its peak)
    border_px: int = 2  # contours within this many px of the edge are truncated

    # C. Gate
    tau: float = 0.6  # minimum crop-classifier probability
    exclude_truncated: bool = True

    # E. Decision
    delta: float = 0.08  # relative difference at or below which items are SIMILAR

    models_dir: Path = MODELS_DIR
