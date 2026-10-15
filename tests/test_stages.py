"""Unit tests for each stage."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
import pytest

from src.calibrate import Calibration, from_points, ruler_length_px
from src.compare import BIGGER, SIMILAR, SMALLER, rank_group, relative_difference
from src.config import Config
from src.data import mask_iou, match_by_order
from src.features import extract_features
from src.grouping import NO_ITEMS, OK, group_items
from src.segment import segment, split_touching, white_balance
from src.size_classifier import predict_rows
from src.weight_model import ProxyRegressor, log_ridge
from tests.conftest import TRUE_MMPP, render
from tools.synth_data import RULER_MM, ruler_photo
from train.cv import select_simplest

# --- calibration -----------------------------------------------------------------


def test_from_points():
    """Two points 500 px apart spanning 100 mm give 0.2 mm/px."""
    assert from_points((0, 0), (300, 400), 100) == pytest.approx(0.2)


def test_auto_ruler_calibration_recovers_scale():
    """Automatic ruler detection recovers the simulated rig's scale within 1%."""
    img = ruler_photo(np.random.default_rng(0))
    assert RULER_MM / ruler_length_px(img) == pytest.approx(TRUE_MMPP, rel=0.01)


def test_calibration_rescales_for_resolution_and_warns_on_mismatch(tmp_path):
    """Same framing at another resolution is rescaled; a different framing triggers a warning."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"mm_per_px": 0.25, "image_size": [1600, 1200]}))
    c = Calibration.load(p)
    assert c.for_image((1200, 1600, 3)) == (0.25, None)
    mmpp, warn = c.for_image((600, 800, 3))  # same framing, half resolution
    assert mmpp == pytest.approx(0.5) and warn is not None and "rescaled" in warn
    _, warn = c.for_image((1000, 1000, 3))
    assert warn is not None and "re-calibrate" in warn


# --- segmentation ---------------------------------------------------------------


def test_segments_every_crop_and_matches_true_outlines(items):
    """All three crops are found, with masks matching their true outlines (IoU > 0.95)."""
    ids = ["T01", "K01", "C01"]
    img, masks = render(items, ids)
    dets = segment(white_balance(img), Config(), TRUE_MMPP)
    assert len(dets) == 3
    for m in masks:  # chamagadda hairs and soft shadows must not leak into the masks
        assert max(mask_iou(d.mask, m.astype(bool)) for d in dets) > 0.95


def test_detections_are_left_to_right(items):
    """Detections come back ordered left to right."""
    img, _ = render(items, ["T01", "T02", "C02"], seed=4)
    xs = [d.centroid[0] for d in segment(white_balance(img), Config(), TRUE_MMPP)]
    assert xs == sorted(xs)


def test_split_touching_discs_at_the_contact_point():
    """Two touching discs of different sizes are split near their true areas."""
    m = np.zeros((300, 500), np.uint8)
    cv2.circle(m, (150, 150), 100, 1, -1)
    cv2.circle(m, (315, 150), 70, 1, -1)  # overlaps the first by ~5 px: touching
    parts = split_touching(m.astype(bool), 0.25, 10)
    assert len(parts) == 2
    big, small = sorted(parts, key=lambda p: -p.sum())
    assert big.sum() == pytest.approx(np.pi * 100**2, rel=0.08)
    assert small.sum() == pytest.approx(np.pi * 70**2, rel=0.12)


def test_split_keeps_a_kheera_whole():
    """A long, thin item is not split into pieces."""
    m = np.zeros((200, 800), np.uint8)
    cv2.ellipse(m, (400, 100), (350, 40), 0, 0, 360, 1, -1)
    assert len(split_touching(m.astype(bool), 0.25, 10)) == 1


def test_white_balance_neutralises_dim_coloured_light():
    """A dim, tinted sheet comes out neutral at the target grey."""
    img = np.full((100, 100, 3), (130, 120, 110), np.uint8)  # dim, bluish sheet
    out = white_balance(img)
    assert np.allclose(out[0, 0], 235, atol=1)


# --- features -------------------------------------------------------------------


def test_features_of_a_disc():
    """A red disc gives the expected area, length, width, elongation, solidity, proxies and hue."""
    m = np.zeros((400, 400), np.uint8)
    cv2.circle(m, (200, 200), 100, 1, -1)
    img = np.full((400, 400, 3), (40, 40, 200), np.uint8)
    f = extract_features(m.astype(bool), img, 0.5)
    assert f["area_mm2"] == pytest.approx(np.pi * 50**2, rel=0.01)
    assert f["length_mm"] == pytest.approx(100, rel=0.02) and f["width_mm"] == pytest.approx(100, rel=0.02)
    assert f["elongation"] == pytest.approx(1.0, abs=0.02) and f["solidity"] == pytest.approx(1.0, abs=0.01)
    assert f["proxy_sphere"] == pytest.approx(f["area_mm2"] ** 1.5)
    assert f["proxy_cylinder"] == pytest.approx(f["area_mm2"] * f["minor_mm"])
    assert f["hue_mean"] < 5 or f["hue_mean"] > 175  # red


def test_elongated_shape():
    """A long rectangle measures 100 × 10 mm and reads as strongly elongated."""
    m = np.zeros((200, 600), np.uint8)
    cv2.rectangle(m, (100, 80), (499, 119), 1, -1)  # 400 x 40 px
    f = extract_features(m.astype(bool), np.zeros((200, 600, 3), np.uint8), 0.25)
    assert f["length_mm"] == pytest.approx(100, rel=0.02) and f["width_mm"] == pytest.approx(10, rel=0.05)
    assert f["elongation"] > 8


def test_match_by_order():
    """Items are matched only when the detection count agrees with the listed IDs."""
    assert match_by_order(2, ["A", "B"]) == ["A", "B"]
    assert match_by_order(1, ["A", "B"]) is None


# --- grouping -------------------------------------------------------------------


def test_grouping_by_crop_with_uncertain_and_truncated_items():
    """Items are grouped by crop, with low-confidence and cut-off items set aside."""
    crops = ["tomato", "chamagadda", "tomato", "chamagadda", "kheera", "tomato"]
    probs = [0.99, 0.95, 0.97, 0.4, 0.9, 0.99]
    flags = [[], [], [], [], [], ["touches_border"]]
    g = group_items(crops, probs, flags, tau=0.6)
    assert g.status == OK
    assert g.groups == {"tomato": [0, 2], "chamagadda": [1], "kheera": [4]}
    assert g.uncertain == [3] and 5 in g.excluded


def test_everything_uncertain():
    """A photo with no confident items reports no_items."""
    assert group_items(["tomato"], [0.3], [[]]).status == NO_ITEMS


# --- pairwise decision ----------------------------------------------------------------


def test_two_items():
    """Two items reproduce the design document's BIGGER / SMALLER example (59%)."""
    d = rank_group([221, 139], 0.08)
    assert d["pairwise"] == [BIGGER, SMALLER] and d["ranks"] == [1, 2]
    assert d["difference_pct"] == pytest.approx(59.0, abs=0.1)  # the design doc's example


def test_similar_within_margin():
    """Items within the margin are both SIMILAR."""
    assert rank_group([100, 105], 0.08)["pairwise"] == [SIMILAR, SIMILAR]


def test_three_items_adjacent_comparisons():
    """Three items reproduce the design's chamagadda ladder (SIMILAR, then BIGGER)."""
    d = rank_group([52, 31, 49], 0.08)  # the design doc's chamagadda example
    assert d["ranks"] == [1, 3, 2] and d["pairwise"] is None
    assert [(a["higher"], a["lower"], a["verdict"]) for a in d["adjacent"]] == [(0, 2, SIMILAR), (2, 1, BIGGER)]


def test_relative_difference():
    """Relative difference is measured against the smaller item, and is infinite for zero."""
    assert relative_difference(0, 10) == float("inf") and relative_difference(110, 100) == pytest.approx(0.1)


# --- models ----------------------------------------------------------------------


def test_proxy_regressor_through_origin():
    """The proxy model recovers a known slope through the origin."""
    x = np.array([[1.0], [2.0], [4.0]])
    m = ProxyRegressor().fit(x, 3 * x[:, 0])
    assert m.predict(np.array([[10.0]]))[0] == pytest.approx(30.0)


def test_log_ridge_fits_a_power_law():
    """The log-ridge model fits a power law to within 10%."""
    rng = np.random.default_rng(0)
    X = rng.uniform(10, 100, (60, 6))
    y = 0.01 * X[:, 0] ** 1.5
    m = log_ridge().fit(X, y)
    assert np.mean(np.abs(m.predict(X) - y) / y) < 0.1


def test_threshold_rule():
    """The threshold rule calls an item BIG at or above the threshold."""
    big = predict_rows("weight_threshold", None, 100.0, np.zeros((3, 1)), np.array([50.0, 100.0, 150.0]))
    assert big.tolist() == [False, True, True]


def test_select_simplest_within_one_se():
    """The simplest model wins when within one SE of the best, and loses when clearly worse."""
    per_item = {"proxy": pd.Series([5.0, 6.0, 7.0, 6.0]), "ridge_log": pd.Series([4.5, 6.5, 4.5, 6.5])}
    assert select_simplest(per_item, ["proxy", "ridge_log"]) == "proxy"
    per_item["proxy"] = pd.Series([20.0, 21.0, 22.0, 21.0])
    assert select_simplest(per_item, ["proxy", "ridge_log"]) == "ridge_log"
