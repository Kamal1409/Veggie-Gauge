"""End-to-end inference at the sorting station.

image -> A. segment -> B. features (mm) -> C. classify + group by crop -> D. weight -> E. decide per group -> F. JSON + overlay

    python -m src.pipeline photo.jpg [more.jpg ...] [--out-dir reports/demo] [--delta 0.08] [--tau 0.6]
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

import cv2
import joblib
import numpy as np

from src import crop_classifier
from src.calibrate import Calibration
from src.compare import SIMILAR, rank_group
from src.config import Config
from src.features import extract_features
from src.grouping import Grouping, group_items
from src.segment import Detection, segment, white_balance
from src.size_classifier import SizeClassifiers
from src.weight_model import WeightModels

# Categorical palette (fixed order) as BGR; one colour per crop group.
_PALETTE_HEX = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PALETTE_BGR = [tuple(int(h[i : i + 2], 16) for i in (5, 3, 1)) for h in _PALETTE_HEX]
GREY = (150, 150, 150)


@dataclass
class PipelineOutput:
    """Everything one run of the pipeline produced.

    Attributes:
        result: The JSON-ready answer (groups, uncertain and excluded items, notes).
        image: The photo that was analysed.
        detections: Segmented items, left to right.
        features: Measured features of each detection.
        crops: Predicted crop of each detection.
        probs: Classifier probability of each prediction.
        grouping: How the detections were grouped.
    """

    result: dict
    image: np.ndarray
    detections: list[Detection]
    features: list[dict]
    crops: list[str]
    probs: list[float]
    grouping: Grouping

    def overlay(self) -> np.ndarray:
        """Draw the result on top of the photo.

        Returns:
            A copy of the photo with outlines, labels and a summary line.
        """
        return draw_overlay(self)

    def item(self, det_index: int) -> dict | None:
        """Find a detection's record in the output, wherever it ended up.

        Args:
            det_index: Index of the detection.

        Returns:
            The item's record from a group, the uncertain list or the excluded list, or
            None if it is not in the output.
        """
        r = self.result
        for rec in [it for g in r["groups"] for it in g["items"]] + r["uncertain"] + r["excluded"]:
            if rec["id"] == det_index:
                return rec
        return None


class Pipeline:
    """The sorting-station pipeline: a photo in, per-crop verdicts out.

    The calibration and every model are loaded once, when the pipeline is created,
    so each call to `run` only does the per-photo work.

    Attributes:
        cfg: Settings in use.
        calibration: The rig's mm/px.
        crop_model: The loaded crop classifier bundle.
        weights: Per-crop weight regressors.
        sizes: Per-crop Small/Big classifiers.
    """

    def __init__(self, cfg: Config | None = None):
        """Load the calibration and the trained models.

        Args:
            cfg: Settings to use. Defaults to `Config()`.

        Raises:
            FileNotFoundError: If the calibration file or the crop classifier is missing;
                the message says which command creates it.
        """
        self.cfg = cfg or Config()
        if not Path(self.cfg.calibration_file).exists():
            raise FileNotFoundError(f"{self.cfg.calibration_file} missing: run `python -m src.calibrate` on a ruler photo first")
        crop_path = Path(self.cfg.models_dir) / "crop_classifier.joblib"
        if not crop_path.exists():
            raise FileNotFoundError(f"{crop_path} missing: run `python -m train.train_models` first")
        self.calibration = Calibration.load(self.cfg.calibration_file)
        self.crop_model = joblib.load(crop_path)
        self.weights = WeightModels(self.cfg.models_dir)
        self.sizes = SizeClassifiers(self.cfg.models_dir)

    def with_config(self, **changes) -> Pipeline:
        """Make a copy of the pipeline with some settings changed.

        The loaded models are shared rather than reloaded, so this is cheap enough to do
        per request, e.g. when an API call passes its own τ or δ.

        Args:
            **changes: Config fields to override, such as `tau=0.7` or `delta=0.05`.

        Returns:
            A new pipeline using the changed settings.
        """
        new = copy.copy(self)
        new.cfg = replace(self.cfg, **changes)
        return new

    def run(self, image: np.ndarray) -> PipelineOutput:
        """Analyse one photo from the rig.

        Steps: white-balance to the sheet, segment, measure, classify and group by crop,
        estimate weights, then rank each group and grade every item SMALL or BIG.

        Args:
            image: BGR photo taken with the fixed rig.

        Returns:
            The JSON-ready result plus the intermediate data (masks, features, grouping)
            needed to draw an overlay or evaluate the run.
        """
        cfg = self.cfg
        t0 = perf_counter()
        mmpp, warning = self.calibration.for_image(image.shape)
        notes = [warning] if warning else []

        balanced = white_balance(image)  # the white sheet is the colour reference
        dets = segment(balanced, cfg, mmpp)  # A
        feats = [extract_features(d.mask, balanced, mmpp) for d in dets]  # B
        crops, probs = crop_classifier.predict(self.crop_model, feats) if feats else ([], [])  # C
        grouping = group_items(crops, probs, [d.flags for d in dets], cfg.tau, cfg.exclude_truncated)

        def record(i: int) -> dict:
            """Build the basic output record for one detection.

            Args:
                i: Index of the detection.

            Returns:
                A dict with the item's id, crop, crop probability, size in mm, bounding box and flags.
            """
            f = feats[i]
            return {
                "id": i,
                "crop": crops[i],
                "crop_prob": round(probs[i], 3),
                "length_mm": round(f["length_mm"], 1),
                "width_mm": round(f["width_mm"], 1),
                "area_mm2": round(f["area_mm2"], 1),
                "bbox": list(dets[i].box),
                "flags": dets[i].flags,
            }

        groups = []
        for crop, idx in grouping.groups.items():
            items = []
            for i in idx:  # D. weight, E. absolute grade
                rec = record(i)
                w = self.weights.predict(crop, feats[i])
                if w is not None:
                    rec["weight_g"], rec["weight_err_g"] = round(w[0], 1), round(w[1], 1)
                rec["absolute"] = self.sizes.predict(crop, feats[i], w[0] if w else None)
                items.append(rec)
            group = {"crop": crop, "basis": None, "difference_pct": None, "items": items}

            if len(items) >= 2:  # E. pairwise within the group
                basis = "weight_g" if all("weight_g" in it for it in items) else "area_mm2"
                if basis == "area_mm2":
                    notes.append(f"no weight model for {crop!r}: compared by projected area")
                d = rank_group([it[basis] for it in items], cfg.delta)
                for it, r in zip(items, d["ranks"]):
                    it["rank"] = r
                if d["pairwise"]:
                    for it, v in zip(items, d["pairwise"]):
                        it["pairwise"] = v
                for adj in d["adjacent"]:
                    if adj["verdict"] == SIMILAR and len(items) > 2:
                        hi, lo = items[adj["higher"]], items[adj["lower"]]
                        lo["note"] = f"SIMILAR to rank {hi['rank']}"
                group.update(
                    basis=basis,
                    difference_pct=d["difference_pct"],
                    adjacent=[{**a, "higher": items[a["higher"]]["id"], "lower": items[a["lower"]]["id"]} for a in d["adjacent"]],
                )
                items.sort(key=lambda it: it["rank"])
            else:
                items[0]["rank"] = 1
                items[0]["note"] = "no same-crop partner in this photo"
            groups.append(group)

        result = {
            "status": grouping.status,
            "message": grouping.message,
            "mm_per_px": round(mmpp, 5),
            "groups": groups,
            "uncertain": [record(i) for i in grouping.uncertain],
            "excluded": [{**record(i), "reason": r} for i, r in grouping.excluded.items()],
            "notes": notes,
            "latency_ms": round(1000 * (perf_counter() - t0), 1),
        }
        return PipelineOutput(result, image, dets, feats, crops, probs, grouping)


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------


def draw_overlay(out: PipelineOutput) -> np.ndarray:
    """Draw outlines and labels for every item, coloured by crop group.

    Uncertain and excluded items are drawn in grey and labelled as not compared.

    Args:
        out: The output of a pipeline run.

    Returns:
        A copy of the photo with the overlay drawn on it.
    """
    img = out.image.copy()
    h, w = img.shape[:2]
    scale = max(h, w) / 1600
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, max(1, round(2 * scale))
    group_color = {g["crop"]: PALETTE_BGR[k % len(PALETTE_BGR)] for k, g in enumerate(out.result["groups"])}

    for i, d in enumerate(out.detections):
        rec = out.item(i) or {}
        in_group = i not in out.grouping.excluded and i not in out.grouping.uncertain
        color = group_color.get(rec.get("crop"), GREY) if in_group else GREY
        contours, _ = cv2.findContours(d.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color, max(2, th + 1))
        lines = [
            f"#{i} {rec.get('crop', '?')} ({rec.get('crop_prob', 0):.2f})",
            f"{rec.get('length_mm', 0):.0f} x {rec.get('width_mm', 0):.0f} mm",
        ]
        if "weight_g" in rec:
            lines.append(f"{rec['weight_g']:.0f} g +/- {rec['weight_err_g']:.0f}")
        verdict = " ".join(
            str(v) for v in (f"rank {rec['rank']}" if "rank" in rec else None, rec.get("pairwise"), rec.get("absolute")) if v
        )
        if verdict:
            lines.append(verdict)
        if i in out.grouping.uncertain:
            lines.append("uncertain: not compared")
        if i in out.grouping.excluded:
            lines.append("truncated: not compared")
        x1, y1, x2, y2 = d.box
        _label(img, lines, ((x1 + x2) // 2, y2 + 10), font, fs, th)

    r = out.result
    if r["status"] != "ok":
        head = r["message"]
    else:
        head = " | ".join(
            f"{g['crop']}: {len(g['items'])}" + (f", diff {g['difference_pct']:.0f}%" if g["difference_pct"] is not None else "")
            for g in r["groups"]
        )
    _label(img, [head], (w // 2, int(20 * scale)), font, fs * 1.15, th)
    return img


def _label(img, lines, anchor, font, fs, th):
    """Draw a block of text on a dark, semi-transparent box.

    The box is placed just below `anchor`, or above it if it would run off the bottom
    of the image, and is kept inside the frame.

    Args:
        img: Image to draw on, modified in place.
        lines: Lines of text, top to bottom.
        anchor: `(x, y)` point the box is centred under.
        font: OpenCV font face.
        fs: Font scale.
        th: Text thickness.
    """
    sizes = [cv2.getTextSize(t, font, fs, th)[0] for t in lines]
    lh = max(s[1] for s in sizes) + int(10 * fs)
    bw = max(s[0] for s in sizes) + int(16 * fs)
    bh = lh * len(lines) + int(8 * fs)
    x0 = int(np.clip(anchor[0] - bw // 2, 0, img.shape[1] - bw))
    y0 = anchor[1] if anchor[1] + bh <= img.shape[0] else anchor[1] - bh - 40
    y0 = int(np.clip(y0, 0, img.shape[0] - bh))
    sub = img[y0 : y0 + bh, x0 : x0 + bw]
    img[y0 : y0 + bh, x0 : x0 + bw] = (sub * 0.25 + np.array([25, 25, 25]) * 0.75).astype(np.uint8)
    for k, t in enumerate(lines):
        cv2.putText(img, t, (x0 + int(8 * fs), y0 + lh * (k + 1)), font, fs, (255, 255, 255), th, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_config_args(p: argparse.ArgumentParser) -> None:
    """Add the shared pipeline options (calibration, models, τ, δ) to a CLI parser.

    Args:
        p: The parser to extend.
    """
    p.add_argument("--calibration", type=Path, help="calibration.json (default data/calibration.json)")
    p.add_argument("--models", type=Path, help="models directory (default models/)")
    p.add_argument("--tau", type=float, help="minimum crop probability (default 0.6)")
    p.add_argument("--delta", type=float, help="SIMILAR margin as a fraction (default 0.08)")


def build_config(args: argparse.Namespace) -> Config:
    """Turn parsed CLI options into a `Config`, keeping defaults for anything not given.

    Args:
        args: Parsed arguments from a parser set up with `add_config_args`.

    Returns:
        The resulting settings.
    """
    cfg = Config()
    if getattr(args, "calibration", None):
        cfg.calibration_file = args.calibration
    if getattr(args, "models", None):
        cfg.models_dir = args.models
    for k in ("tau", "delta"):
        if getattr(args, k, None) is not None:
            setattr(cfg, k, getattr(args, k))
    return cfg


def main() -> None:
    """Command-line entry point: analyse photos and print (optionally save) the results."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+", type=Path)
    p.add_argument("--out-dir", type=Path, help="write <name>.json and <name>_overlay.jpg here")
    add_config_args(p)
    args = p.parse_args()

    pipe = Pipeline(build_config(args))
    for path in args.images:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"cannot read {path}")
        out = pipe.run(image)
        print(json.dumps({"image": path.name, **out.result}, indent=2))
        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            (args.out_dir / f"{path.stem}.json").write_text(json.dumps(out.result, indent=2))
            cv2.imwrite(str(args.out_dir / f"{path.stem}_overlay.jpg"), out.overlay())


if __name__ == "__main__":
    main()
