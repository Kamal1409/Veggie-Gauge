from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import Config, DataPaths
from tools.synth_data import MMPP, Scene, make_items


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    """A small synthetic dataset with features built and models trained, shared by the tests.

    Args:
        tmp_path_factory: pytest's factory for temporary folders.

    Returns:
        The dataset locations, the models folder and the reports folder.
    """
    from tools.synth_data import generate
    from train.build_features import build
    from train.train_models import run

    root = tmp_path_factory.mktemp("data")
    generate(root, seed=1, n_items={"tomato": 4, "kheera": 4, "chamagadda": 4}, scale=0.34)
    dp = DataPaths(root)
    build(dp, Config()).to_csv(dp.features, index=False)
    models, reports = tmp_path_factory.mktemp("models"), tmp_path_factory.mktemp("reports")
    run(dp, models, reports)
    return dp, models, reports


@pytest.fixture(scope="session")
def items():
    """A dozen synthetic items (four per crop) for rendering test photos.

    Returns:
        The ground-truth table indexed by `veg_id` and the items' shape parameters.
    """
    gt, shapes = make_items(seed=3, n_items={"tomato": 4, "kheera": 4, "chamagadda": 4})
    return gt.set_index("veg_id", drop=False), shapes


def render(items, ids: list[str], seed: int = 0, session: str = "daylight"):
    """Render a test photo of the given items.

    Args:
        items: The `items` fixture.
        ids: Item IDs to put in the photo.
        seed: Random seed for placement and lighting.
        session: Lighting session.

    Returns:
        The photo and the true visible mask of each item.

    Raises:
        RuntimeError: If the items cannot be placed.
    """
    gt, shapes = items
    rng = np.random.default_rng(seed)
    rows = [gt.loc[v] for v in ids]
    for _ in range(30):
        scene = Scene(rng, session)
        poses = scene.place(rows, shapes)
        if poses is not None:
            masks = scene.render(rows, shapes, poses)
            return scene.finish(), masks
    raise RuntimeError("placement failed")


def config_for(dp: DataPaths, models: Path) -> Config:
    """Settings pointing at a test dataset's calibration and a test models folder.

    Args:
        dp: The test dataset.
        models: Folder holding the test models.

    Returns:
        The settings.
    """
    return Config(calibration_file=dp.calibration, models_dir=models)


TRUE_MMPP = MMPP
