"""HTTP API for the sorting station: POST a photo, get per-crop groups with verdicts.

uvicorn app.api:app --host 0.0.0.0 --port 8000

curl -F "file=@photo.jpg" "http://localhost:8000/compare"
"""

from __future__ import annotations

import base64
from functools import lru_cache

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from src import __version__
from src.config import Config
from src.pipeline import Pipeline

MAX_UPLOAD_BYTES = 25 * 1024 * 1024

app = FastAPI(title="Veggie Gauge", version=__version__)


@lru_cache(maxsize=1)
def get_pipeline() -> Pipeline:
    """The shared pipeline, created on the first request and reused after that.

    Returns:
        The loaded pipeline.
    """
    return Pipeline(Config())


@app.get("/health")
def health() -> dict:
    """Report that the service is up and which models it has loaded.

    Returns:
        The status, version, rig scale, known crops, and the crops that have weight and
        Small/Big models.
    """
    pipe = get_pipeline()
    crops = [str(c) for c in pipe.crop_model["model"].classes_]
    return {
        "status": "ok",
        "version": __version__,
        "mm_per_px": pipe.calibration.mm_per_px,
        "crops": crops,
        "weight_models": [c for c in crops if pipe.weights.bundle(c) is not None],
        "size_models": [c for c in crops if pipe.sizes.bundle(c) is not None],
    }


@app.post("/compare")
async def compare(
    file: UploadFile = File(..., description="top-down photo from the fixed rig"),
    overlay: bool = Query(False, description="include the annotated image as base64 JPEG"),
    delta: float | None = Query(None, ge=0, le=1, description="SIMILAR margin as a fraction"),
    tau: float | None = Query(None, ge=0, le=1, description="minimum crop probability"),
) -> dict:
    """Analyse an uploaded photo and return the per-crop groups.

    Args:
        file: Top-down photo from the fixed rig.
        overlay: Also return the annotated photo as a base64 JPEG.
        delta: SIMILAR margin for this request, overriding the default.
        tau: Minimum crop probability for this request, overriding the default.

    Returns:
        The pipeline's JSON result.

    Raises:
        HTTPException: 413 if the upload is over 25 MB, 400 if it isn't a readable image.
    """
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image larger than 25 MB")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "could not decode the upload as an image")

    pipe = get_pipeline()
    overrides = {k: v for k, v in (("delta", delta), ("tau", tau)) if v is not None}
    if overrides:
        pipe = pipe.with_config(**overrides)
    out = pipe.run(image)

    result = dict(out.result)
    if overlay:
        ok, buf = cv2.imencode(".jpg", out.overlay(), [cv2.IMWRITE_JPEG_QUALITY, 85])
        result["overlay_jpeg_base64"] = base64.b64encode(buf.tobytes()).decode() if ok else None
    return result
