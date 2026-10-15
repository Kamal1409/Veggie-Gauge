"""Integration tests on a small synthetic dataset: features, training, pipeline, evaluation, API."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.pipeline import Pipeline
from tests.conftest import config_for, render


def test_features_match_every_photo(tiny):
    """Every photo's detections line up with its listed items, for all three crops."""
    dp, _, _ = tiny
    f = pd.read_csv(dp.features)
    assert np.asarray(f.drop_duplicates("image")["matched"], dtype=bool).all()
    assert set(f["crop"]) == {"tomato", "kheera", "chamagadda"}


def test_training_writes_all_models(tiny):
    """Training saves a crop classifier plus a weight and a Small/Big model per crop."""
    _, models, reports = tiny
    names = {p.name for p in models.glob("*.joblib")}
    assert "crop_classifier.joblib" in names
    for crop in ("tomato", "kheera", "chamagadda"):
        assert f"weight_{crop}.joblib" in names and f"size_{crop}.joblib" in names
    r = json.loads((reports / "training_report.json").read_text())
    assert r["synthetic_data"] and r["crop_classifier"]["chosen"] in {"logreg", "knn"}


def test_mixed_photo_is_grouped_by_crop(tiny, items):
    """A tomato-and-chamagadda photo gives two separate groups, never a cross-crop comparison."""
    dp, models, _ = tiny
    img, _ = render(items, ["T01", "T02", "C01", "C02", "C03"], seed=2)
    r = Pipeline(config_for(dp, models)).run(img).result
    assert r["status"] == "ok"
    groups = {g["crop"]: g for g in r["groups"]}
    assert set(groups) == {"tomato", "chamagadda"}  # never a tomato-vs-chamagadda comparison
    assert len(groups["tomato"]["items"]) == 2 and len(groups["chamagadda"]["items"]) == 3
    assert [it["rank"] for it in groups["chamagadda"]["items"]] == [1, 2, 3]
    for it in groups["tomato"]["items"]:
        assert it["pairwise"] in {"BIGGER", "SMALLER", "SIMILAR"} and it["absolute"] in {"SMALL", "BIG"}
    json.dumps(r)


def test_heavier_item_ranks_first(tiny, items):
    """Of two kheera, the heavier one is ranked first and called BIGGER."""
    dp, models, _ = tiny
    gt, _ = items
    k = gt[gt["crop"] == "kheera"].sort_values("weight_g")
    light, heavy = k.index[0], k.index[-1]
    out = Pipeline(config_for(dp, models)).run(render(items, [light, heavy], seed=5)[0])
    (g,) = out.result["groups"]
    assert g["items"][0]["weight_g"] > g["items"][1]["weight_g"] and g["items"][0]["pairwise"] == "BIGGER"
    assert out.overlay().shape == out.image.shape


def test_single_item_gets_absolute_grade_only(tiny, items):
    """A lone item gets SMALL / BIG but no pairwise verdict."""
    dp, models, _ = tiny
    r = Pipeline(config_for(dp, models)).run(render(items, ["T03"])[0]).result
    (it,) = r["groups"][0]["items"]
    assert "pairwise" not in it and it["absolute"] in {"SMALL", "BIG"} and "partner" in it["note"]


def test_evaluate_produces_every_metric_group(tiny):
    """The evaluation reports every metric group and flags synthetic data."""
    from eval.evaluate import evaluate

    dp, models, reports = tiny
    m = evaluate(dp, config_for(dp, models), reports)
    for k in ("calibration", "segmentation", "crop_classifier", "weight", "pairwise", "small_big", "robustness", "efficiency", "headline"):
        assert k in m
    assert m["data"]["synthetic"] and m["segmentation"]["reference_iou"]["mean_iou"] > 0.9
    assert (reports / "results_table.md").read_text(encoding="utf-8").startswith("> **SYNTHETIC")


def test_hog_baseline_runs(tiny):
    """The HOG + SVM baseline runs its leave-one-item-out folds end to end."""
    from baselines.common import leave_one_item_out, load_patches
    from baselines.hog_svm import fit_predict, hog_features

    dp, _, _ = tiny
    rows, patches, meta = load_patches(dp, 64)
    preds = leave_one_item_out(rows, hog_features(patches), meta["thresholds"], fit_predict)
    assert len(preds) == len(rows) and preds["big_pred"].dtype == bool


def test_api(tiny, items, monkeypatch):
    """The API reports its models, groups an uploaded photo by crop and rejects a non-image."""
    from app import api

    dp, models, _ = tiny
    pipe = Pipeline(config_for(dp, models))
    monkeypatch.setattr(api, "get_pipeline", lambda: pipe)
    client = TestClient(api.app)
    h = client.get("/health").json()
    assert h["status"] == "ok" and set(h["crops"]) == {"tomato", "kheera", "chamagadda"}

    ok, buf = cv2.imencode(".jpg", render(items, ["T01", "K01"], seed=7)[0])
    resp = client.post("/compare", files={"file": ("p.jpg", buf.tobytes(), "image/jpeg")}, params={"overlay": True, "delta": 0.1})
    assert resp.status_code == 200
    body = resp.json()
    assert {g["crop"] for g in body["groups"]} == {"tomato", "kheera"} and body["overlay_jpeg_base64"]
    assert client.post("/compare", files={"file": ("x.jpg", b"nope", "image/jpeg")}).status_code == 400
