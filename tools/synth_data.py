"""Generate a SYNTHETIC dataset in exactly the format of the real one.

It exists so every stage (calibration, segmentation, training, evaluation, demo, tests)
can be run end to end before produce has been bought and photographed. Numbers measured
on it show the code works; they say nothing about real accuracy. The dataset is stamped
with a `SYNTHETIC` file and every report produced from it carries a warning.

What it simulates (docs/PIPELINE.md, Stages 1-3):
  * the fixed rig: phone overhead at a fixed height over a white sheet, so mm/px is
    constant, plus one ruler photo for calibration;
  * tomato (~sphere), kheera (elongated, sometimes curved or with lifted ends) and
    chamagadda (brown, irregular, hairy). Weight comes from volume x density with hidden
    height and density variation, so a top-down photo cannot recover it exactly;
  * the photo sets: singles, same-crop pairs/groups, mixed-crop photos (e.g. 2 tomatoes +
    3 chamagadda) and a stress set (touching items, a tilted camera, dim light);
  * exact item outlines for 10 photos in data/reference_masks/, standing in for the
    hand-traced masks used for the segmentation IoU check.

    python -m tools.synth_data [--out data] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.calibrate import ruler_length_px
from src.config import DataPaths

W, H = 1600, 1200
MMPP = 0.25  # true mm per pixel of the simulated rig (field of view 400 x 300 mm)
RULER_MM = 320.0  # physical body length of the simulated steel ruler
N_ITEMS = {"tomato": 15, "kheera": 10, "chamagadda": 18}
PREFIX = {"tomato": "T", "kheera": "K", "chamagadda": "C"}
SESSIONS = {
    # gain, BGR colour cast, noise
    "daylight": dict(gain=1.0, cast=(1.0, 1.0, 1.0), noise=2.5),
    "room": dict(gain=0.93, cast=(0.92, 0.98, 1.05), noise=3.0),
    "dim": dict(gain=0.55, cast=(1.06, 1.0, 0.95), noise=6.0),
}


# ---------------------------------------------------------------------------
# Items and ground truth
# ---------------------------------------------------------------------------


def make_items(seed: int, n_items: dict[str, int] = N_ITEMS) -> tuple[pd.DataFrame, dict]:
    """Invent the vegetables: their ground truth and how each one looks.

    Weight comes from volume times density, with hidden height and density variation,
    so a top-down photo cannot recover it exactly, just as with real produce.

    Args:
        seed: Random seed.
        n_items: How many items of each crop to make.

    Returns:
        The ground-truth table (one row per item) and the shape parameters each item
        is drawn with, keyed by `veg_id`.
    """
    rng = np.random.default_rng(seed)
    rows, shapes = [], {}
    for i in range(n_items.get("tomato", 0)):
        d = rng.uniform(40, 85)
        ecc = rng.uniform(0.88, 1.0)
        h = d * rng.uniform(0.82, 0.95)
        w = rng.uniform(0.93, 1.0) * math.pi / 6 * d * (d * ecc) * h / 1000
        vid = f"T{i + 1:02d}"
        rows.append(dict(veg_id=vid, crop="tomato", weight_g=round(w), length_mm=round(d), notes="slightly oval" if ecc < 0.92 else ""))
        shapes[vid] = dict(d=d, ecc=ecc, base=(rng.uniform(35, 55), rng.uniform(40, 70), rng.uniform(190, 225)))
    for i in range(n_items.get("kheera", 0)):
        length = rng.uniform(120, 240)
        diam = float(np.clip(0.21 * length * rng.uniform(0.8, 1.25), 26, 50))
        bend = math.radians(rng.choice([0, rng.uniform(5, 30)]))  # some are curved
        lift = rng.uniform(0.94, 1.0)  # ends lifted off the sheet shorten the top-down view
        w = rng.uniform(0.92, 0.99) * math.pi / 4 * diam**2 * length * 0.9 / 1000
        chord = length if bend == 0 else 2 * (length / bend) * math.sin(bend / 2)
        vid = f"K{i + 1:02d}"
        note = "slightly curved" if bend > math.radians(12) else ""
        rows.append(dict(veg_id=vid, crop="kheera", weight_g=round(w), length_mm=round(chord), notes=note))
        shapes[vid] = dict(
            length=length, diam=diam, bend=bend, lift=lift, base=(rng.uniform(40, 60), rng.uniform(95, 125), rng.uniform(50, 75))
        )
    for i in range(n_items.get("chamagadda", 0)):
        a = rng.uniform(35, 80)
        b = a * rng.uniform(0.6, 0.85)
        c = b * rng.uniform(0.75, 1.0)  # hidden height
        w = rng.uniform(1.05, 1.15) * rng.uniform(0.88, 1.0) * math.pi / 6 * a * b * c / 1000
        vid = f"C{i + 1:02d}"
        rows.append(dict(veg_id=vid, crop="chamagadda", weight_g=round(w), length_mm=round(a), notes=""))
        shapes[vid] = dict(
            a=a,
            b=b,
            bumps=[(float(rng.uniform(0.01, 0.05)), k, float(rng.uniform(0, 2 * np.pi))) for k in (2, 3, 4, 5)],
            base=(rng.uniform(35, 55), rng.uniform(65, 85), rng.uniform(100, 130)),
        )
    return pd.DataFrame(rows), shapes


# ---------------------------------------------------------------------------
# Drawing (all geometry in pixels)
# ---------------------------------------------------------------------------


def _pt(c) -> tuple[int, int]:
    """Round a point to integer pixel coordinates for OpenCV drawing calls.

    Args:
        c: A point `(x, y)`.

    Returns:
        The point as a pair of ints.
    """
    return int(round(float(c[0]))), int(round(float(c[1])))


def _rot(pts: np.ndarray, angle: float) -> np.ndarray:
    """Rotate points around the origin.

    Args:
        pts: `(N, 2)` points.
        angle: Rotation in radians.

    Returns:
        The rotated points.
    """
    ca, sa = math.cos(angle), math.sin(angle)
    return pts @ np.array([[ca, sa], [-sa, ca]])


def _shade(canvas: np.ndarray, layer: np.ndarray, mask: np.ndarray, radius_px: float, lo: float, rng) -> None:
    """Paint an item onto the canvas, darker towards its outline and with some texture.

    Args:
        canvas: Image being drawn, modified in place.
        layer: Copy of the canvas with the item's flat colours drawn on.
        mask: The item's mask; only pixels inside it are painted.
        radius_px: Distance from the outline at which shading reaches full brightness.
        lo: Brightness right at the outline, as a fraction.
        rng: Random generator for the texture.
    """
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    shade = lo + (1 - lo) * np.clip(dist / max(radius_px, 1.0), 0, 1) ** 0.6
    tex = cv2.GaussianBlur(rng.normal(0, 1, mask.shape).astype(np.float32), (0, 0), 2.0)
    tex = 1 + 0.05 * tex / (tex.std() + 1e-6)
    out = layer.astype(np.float32) * (shade * tex)[..., None]
    canvas[mask > 0] = np.clip(out[mask > 0], 0, 255)


def tomato_outline(s: dict, angle: float) -> np.ndarray:
    """Outline of a tomato seen from above: a slightly oval ellipse.

    Args:
        s: The tomato's shape parameters.
        angle: Rotation in radians.

    Returns:
        `(N, 2)` outline points in pixels, centred on the origin.
    """
    a, b = s["d"] / 2 / MMPP, s["d"] / 2 * s["ecc"] / MMPP
    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    return _rot(np.stack([a * np.cos(t), b * np.sin(t)], 1), angle)


def draw_tomato(canvas, c, s, angle, rng) -> np.ndarray:
    """Draw a shaded tomato with a highlight and a green calyx.

    Args:
        canvas: Image being drawn, modified in place.
        c: Centre in pixels.
        s: The tomato's shape and colour parameters.
        angle: Rotation in radians.
        rng: Random generator.

    Returns:
        The tomato's mask.
    """
    hw = canvas.shape[:2]
    a, b = s["d"] / 2 / MMPP, s["d"] / 2 * s["ecc"] / MMPP
    mask = np.zeros(hw, np.uint8)
    cv2.fillPoly(mask, [np.round(tomato_outline(s, angle) + c).astype(np.int32)], 1)
    ys, xs = np.nonzero(mask)
    dx, dy = xs - c[0], ys - c[1]
    u = (dx * np.cos(angle) + dy * np.sin(angle)) / a
    v = (-dx * np.sin(angle) + dy * np.cos(angle)) / b
    shade = np.clip(1 - 0.38 * ((u + 0.3) ** 2 + (v + 0.3) ** 2), 0.45, 1.0)
    canvas[ys, xs] = np.clip(np.array(s["base"])[None] * shade[:, None], 0, 255)
    hl = np.zeros(hw, np.float32)
    hc = c + np.array([-0.35 * a, -0.35 * b])
    cv2.ellipse(hl, ((float(hc[0]), float(hc[1])), (0.35 * a, 0.22 * b), -30), 1.0, -1)
    hl = cv2.GaussianBlur(hl, (0, 0), max(1.0, a / 12)) * 0.6 * mask
    canvas[:] = np.clip(canvas * (1 - hl[..., None]) + 245 * hl[..., None], 0, 255)
    rot = rng.uniform(0, 2 * np.pi)
    star = [c + (0.28 if k % 2 == 0 else 0.1) * a * np.array([np.cos(rot + k * np.pi / 5), np.sin(rot + k * np.pi / 5)]) for k in range(10)]
    cv2.fillPoly(canvas, [np.round(np.array(star)).astype(np.int32)], (40, 115, 55), cv2.LINE_AA)
    return mask


def kheera_geometry(s: dict, angle: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Geometry of a kheera seen from above: a capsule, possibly curved.

    Lifted ends shorten the top-down view, so the drawn length can be slightly less
    than the true length.

    Args:
        s: The kheera's shape parameters.
        angle: Rotation in radians.

    Returns:
        The outline polygon, the centreline, the unit normals along it and the half-width
        at each centreline point, all in pixels, centred and rotated.
    """
    lp = s["length"] * s["lift"] / MMPP
    t = np.linspace(-0.5, 0.5, 200)
    if s["bend"] < 1e-3:
        centre = np.stack([t * lp, np.zeros_like(t)], 1)
        normal = np.tile([0.0, 1.0], (len(t), 1))
    else:
        r = lp / s["bend"]
        phi = t * s["bend"]
        centre = np.stack([r * np.sin(phi), r * (1 - np.cos(phi))], 1)
        centre[:, 1] -= centre[:, 1].mean()
        normal = np.stack([-np.sin(phi), np.cos(phi)], 1)
    half = s["diam"] / 2 / MMPP * np.sqrt(np.clip(1 - np.abs(2 * t) ** 6, 0, 1))
    outline = np.vstack([centre + half[:, None] * normal, (centre - half[:, None] * normal)[::-1]])
    return _rot(outline, angle), _rot(centre, angle), _rot(normal, angle), half


def draw_kheera(canvas, c, s, angle, rng) -> np.ndarray:
    """Draw a shaded kheera with pale stripes and small bumps.

    Args:
        canvas: Image being drawn, modified in place.
        c: Centre in pixels.
        s: The kheera's shape and colour parameters.
        angle: Rotation in radians.
        rng: Random generator.

    Returns:
        The kheera's mask.
    """
    outline, centre, normal, half = kheera_geometry(s, angle)
    mask = np.zeros(canvas.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(outline + c).astype(np.int32)], 1)
    layer = canvas.copy()
    base = np.array(s["base"])
    layer[mask > 0] = base
    stripe = tuple(float(v) for v in base * 1.45)
    for off in (-0.55, 0.0, 0.55):
        line = centre + c + (off * half)[:, None] * normal
        cv2.polylines(layer, [np.round(line[15:-15]).astype(np.int32)], False, stripe, 3, cv2.LINE_AA)
    for _ in range(int(s["length"] / 4)):  # small bumps
        k = rng.integers(10, len(centre) - 10)
        p = centre[k] + c + rng.uniform(-0.8, 0.8) * half[k] * normal[k]
        cv2.circle(layer, _pt(p), 2, tuple(float(v) for v in base * 0.6), -1)
    _shade(canvas, layer, mask, s["diam"] / 2 / MMPP, 0.55, rng)
    return mask


def chamagadda_outline(s: dict, angle: float) -> np.ndarray:
    """Outline of a chamagadda seen from above: an ellipse with gentle bumps.

    Args:
        s: The corm's shape parameters.
        angle: Rotation in radians.

    Returns:
        `(N, 2)` outline points in pixels, centred on the origin.
    """
    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    a, b = s["a"] / 2 / MMPP, s["b"] / 2 / MMPP
    r = 1 + sum(amp * np.sin(k * t + ph) for amp, k, ph in s["bumps"])
    return _rot(np.stack([a * r * np.cos(t), b * r * np.sin(t)], 1), angle)


def draw_chamagadda(canvas, c, s, angle, rng) -> np.ndarray:
    """Draw a brown chamagadda with growth rings and fine hairs.

    The hairs stick out past the outline but are not part of the mask; the
    segmentation's opening is expected to remove them.

    Args:
        canvas: Image being drawn, modified in place.
        c: Centre in pixels.
        s: The corm's shape and colour parameters.
        angle: Rotation in radians.
        rng: Random generator.

    Returns:
        The corm's mask, without the hairs.
    """
    poly = chamagadda_outline(s, angle) + c
    mask = np.zeros(canvas.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    layer = canvas.copy()
    base = np.array(s["base"])
    layer[mask > 0] = base
    a, b = s["a"] / 2 / MMPP, s["b"] / 2 / MMPP
    ring = tuple(float(v) for v in base * 0.62)
    u = np.array([math.cos(angle), math.sin(angle)])
    for f in (-0.55, -0.2, 0.15, 0.5):  # growth rings across the corm
        rc = c + f * a * u
        cv2.ellipse(layer, ((float(rc[0]), float(rc[1])), (0.35 * b, 2.2 * b), math.degrees(angle)), ring, 3, cv2.LINE_AA)
    _shade(canvas, layer, mask, b, 0.6, rng)
    # Fine hairs sticking out of the outline (segmentation's opening should remove them).
    hair = tuple(float(v) for v in base * 0.55)
    for k in rng.choice(len(poly), 60, replace=False):
        p = poly[k]
        out = (p - c) / (np.linalg.norm(p - c) + 1e-9)
        q = p + out * rng.uniform(1.5, 4.0) / MMPP
        cv2.line(canvas, _pt(p), _pt(q), hair, 1, cv2.LINE_AA)
    return mask


DRAW = {"tomato": draw_tomato, "kheera": draw_kheera, "chamagadda": draw_chamagadda}


def outline(crop: str, s: dict, angle: float) -> np.ndarray:
    """Outline of any crop, for placing items without overlap.

    Args:
        crop: Crop name.
        s: The item's shape parameters.
        angle: Rotation in radians.

    Returns:
        `(N, 2)` outline points in pixels, centred on the origin.
    """
    if crop == "tomato":
        return tomato_outline(s, angle)
    if crop == "kheera":
        return kheera_geometry(s, angle)[0]
    return chamagadda_outline(s, angle)


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------


class Scene:
    """One simulated photo from the rig: a white sheet, items, shadows and lighting.

    Attributes:
        rng: Random generator.
        session: Lighting session ("daylight", "room" or "dim").
        img: The image being built, as floats.
    """

    def __init__(self, rng, session: str):
        """Start a scene with an empty white sheet with a little paper texture.

        Args:
            rng: Random generator.
            session: Lighting session ("daylight", "room" or "dim").
        """
        self.rng = rng
        self.session = session
        sheet = np.array([rng.uniform(236, 246), rng.uniform(238, 247), rng.uniform(238, 248)])
        self.img = np.ones((H, W, 3), np.float32) * sheet
        self.img += cv2.GaussianBlur(rng.normal(0, 3, (H, W)).astype(np.float32), (0, 0), 2)[..., None]

    def place(self, rows, shapes, gap_mm: float = 12.0, tries: int = 800):
        """Find random positions and rotations that keep items apart and inside the frame.

        Args:
            rows: Ground-truth rows of the items to place.
            shapes: Shape parameters, keyed by `veg_id`.
            gap_mm: Minimum gap between items.
            tries: Attempts per item before giving up.

        Returns:
            One `(centre, angle)` pose per item, or None if they could not all be placed.
        """
        poses, polys = [], []
        for row in rows:
            for _ in range(tries):
                ang = float(self.rng.uniform(0, 2 * np.pi))
                poly = outline(row["crop"], shapes[row["veg_id"]], ang)
                lo, hi = poly.min(0), poly.max(0)
                if np.any(hi - lo > [W - 30, H - 30]):
                    return None
                c = self.rng.uniform(15 - lo, [W - 15, H - 15] - hi)
                if all(_poly_distance(poly + c, q) > gap_mm / MMPP for q in polys):
                    poses.append((c, ang))
                    polys.append(poly + c)
                    break
            else:
                return None
        return poses

    def render(self, rows, shapes, poses) -> list[np.ndarray]:
        """Draw soft shadows, then the items in order.

        Later items cover earlier ones, which is how touching items overlap.

        Args:
            rows: Ground-truth rows of the items.
            shapes: Shape parameters, keyed by `veg_id`.
            poses: One `(centre, angle)` per item.

        Returns:
            The visible part of each item's mask, in the same order as `rows`.
        """
        shadow = np.zeros((H, W), np.float32)
        for row, (c, ang) in zip(rows, poses):
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [np.round(outline(row["crop"], shapes[row["veg_id"]], ang) + c + [6, 8]).astype(np.int32)], 1)
            shadow = np.maximum(shadow, cv2.GaussianBlur(m.astype(np.float32), (0, 0), 10))
        self.img *= (1 - 0.18 * shadow)[..., None]  # diffuse light: faint, grey shadows
        masks: list[np.ndarray] = []
        for row, (c, ang) in zip(rows, poses):
            m = DRAW[row["crop"]](self.img, np.asarray(c, float), shapes[row["veg_id"]], ang, self.rng)
            masks = [prev & (1 - m) for prev in masks]
            masks.append(m)
        return masks

    def finish(self) -> np.ndarray:
        """Apply the session's lighting, sensor noise and a slight blur.

        Returns:
            The finished photo as 8-bit BGR.
        """
        s = SESSIONS[self.session]
        gx, gy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
        a = self.rng.uniform(0, 2 * np.pi)
        light = (1 + 0.04 * (np.cos(a) * gx + np.sin(a) * gy)) * (1 - 0.04 * (gx**2 + gy**2))
        img = self.img * (s["gain"] * light)[..., None] * np.array(s["cast"])
        img += self.rng.normal(0, s["noise"], img.shape)
        return cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (0, 0), 0.6)


def _poly_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Rough gap between two outlines.

    Args:
        p: First outline, `(N, 2)` pixels.
        q: Second outline, `(M, 2)` pixels.

    Returns:
        The smallest distance between sampled outline points, or -1 if they overlap.
    """
    if cv2.pointPolygonTest(q.astype(np.float32), _pt(p.mean(0)), False) >= 0:
        return -1.0
    d = np.sqrt(((p[::4, None, :] - q[None, ::4, :]) ** 2).sum(-1)).min()
    inside = any(cv2.pointPolygonTest(q.astype(np.float32), (float(x), float(y)), False) >= 0 for x, y in p[::8])
    return -1.0 if inside else float(d)


def make_touching(scene: Scene, rows, shapes, rng):
    """Place two items so that their outlines just touch.

    The second item is slid towards the first, using a binary search on the gap.

    Args:
        scene: The scene to place them in.
        rows: Ground-truth rows of the two items.
        shapes: Shape parameters, keyed by `veg_id`.
        rng: Random generator.

    Returns:
        Two `(centre, angle)` poses, or None if the items could not be placed.
    """
    poses = scene.place(rows, shapes, gap_mm=40)
    if poses is None:
        return None
    (c0, a0), (c1, a1) = poses
    p0 = outline(rows[0]["crop"], shapes[rows[0]["veg_id"]], a0) + c0
    p1 = outline(rows[1]["crop"], shapes[rows[1]["veg_id"]], a1)
    direction = (c1 - c0) / np.linalg.norm(c1 - c0)
    lo, hi = 0.0, float(np.linalg.norm(c1 - c0))
    for _ in range(25):
        mid = (lo + hi) / 2
        m0 = np.zeros((H, W), np.uint8)
        m1 = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m0, [np.round(p0).astype(np.int32)], 1)
        cv2.fillPoly(m1, [np.round(p1 + c0 + direction * mid).astype(np.int32)], 1)
        if (m0 & m1).sum() > 0:
            lo = mid
        else:
            hi = mid
    return [(c0, a0), (c0 + direction * (hi - 1.5), a1)]  # nudge in so they touch


def tilt(img: np.ndarray, masks: list[np.ndarray], rng):
    """Simulate a camera pitched a few degrees: the far edge of the sheet shrinks.

    Args:
        img: The finished photo.
        masks: Item masks, warped the same way.
        rng: Random generator.

    Returns:
        The tilted photo and the tilted masks.
    """
    t = rng.uniform(0.05, 0.08) * W
    src = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
    far = [[t, 0], [W - t, 0], [W, H], [0, H]] if rng.random() < 0.5 else [[0, 0], [W, 0], [W - t, H], [t, H]]
    dst = np.array(far, dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    return out, [cv2.warpPerspective(m, M, (W, H), flags=cv2.INTER_NEAREST) for m in masks]


def ruler_photo(rng) -> np.ndarray:
    """Simulate the one-off calibration photo: a steel ruler with mm ticks on the sheet.

    Args:
        rng: Random generator.

    Returns:
        The ruler photo as 8-bit BGR.
    """
    scene = Scene(rng, "daylight")
    L, Wd = RULER_MM / MMPP, 28 / MMPP
    ang = math.radians(rng.uniform(-6, 6))
    c = np.array([W / 2, H / 2 + rng.uniform(-100, 100)])
    body = _rot(np.array([[-L / 2, -Wd / 2], [L / 2, -Wd / 2], [L / 2, Wd / 2], [-L / 2, Wd / 2]]), ang) + c
    cv2.fillPoly(scene.img, [np.round(body).astype(np.int32)], (118, 121, 125), cv2.LINE_AA)
    for mm in range(0, 301):  # ticks every mm from 10 mm in
        x = -L / 2 + (10 + mm) / MMPP
        ln = (0.45 if mm % 10 == 0 else 0.3 if mm % 5 == 0 else 0.18) * Wd
        p, q = _rot(np.array([[x, -Wd / 2], [x, -Wd / 2 + ln]]), ang) + c
        cv2.line(scene.img, _pt(p), _pt(q), (40, 40, 40), 1, cv2.LINE_AA)
    return scene.finish()


def mask_to_polygon_line(mask: np.ndarray) -> str | None:
    """Write a mask's outline as one line of a YOLO-seg style polygon file.

    Args:
        mask: Binary mask.

    Returns:
        The line ("0" then normalised `x y` pairs), or None if the mask has no usable outline.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1.0, True).reshape(-1, 2)
    h, w = mask.shape
    return "0 " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in c)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate(out: Path, seed: int = 0, n_items: dict[str, int] = N_ITEMS, scale: float = 1.0) -> None:
    """Generate the whole synthetic dataset in the real data format.

    Writes the ground truth, the photos of all four sets, `images.csv`, ten reference
    outlines, a ruler photo, the resulting `calibration.json` and the SYNTHETIC flag.

    Args:
        out: Dataset folder to write into; existing photos there are replaced.
        seed: Random seed.
        n_items: How many items of each crop to make.
        scale: Multiplier on the number of photos (the tests use a small dataset).
    """
    dp = DataPaths(out)
    rng = np.random.default_rng(seed)
    for d in (dp.raw, dp.reference_masks):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    gt, shapes = make_items(seed, n_items)
    dp.ground_truth.write_text(gt.to_csv(index=False, lineterminator="\n"))
    by_crop = {c: list(gt[gt["crop"] == c]["veg_id"]) for c in n_items if n_items[c]}
    gt_idx = gt.set_index("veg_id", drop=False)
    records, ref_written = [], 0

    def compose(ids, session, gap_mm=12.0):
        """Start a scene and place the given items in it.

        Args:
            ids: Item IDs to place.
            session: Lighting session.
            gap_mm: Minimum gap between items.

        Returns:
            The scene, the items' ground-truth rows and their poses.

        Raises:
            RuntimeError: If the items cannot be placed after many attempts.
        """
        rows = [gt_idx.loc[v] for v in ids]
        for _ in range(40):
            scene = Scene(rng, session)
            poses = scene.place(rows, shapes, gap_mm)
            if poses is not None:
                return scene, rows, poses
        raise RuntimeError(f"could not place {ids}")

    def emit(name, rows, masks, img, set_, case="", session="daylight", reference=False):
        """Save one photo and record it in the index (and optionally as a reference outline).

        Item IDs are recorded left to right, as `images.csv` requires.

        Args:
            name: File name of the photo.
            rows: Ground-truth rows of the items in it.
            masks: Visible mask of each item.
            img: The photo.
            set_: Photo set: singles, pairs, mixed or stress.
            case: Stress case, if any.
            session: Lighting session.
            reference: Also save the exact outlines as a stand-in for hand tracing.
        """
        nonlocal ref_written
        keep = [(r, m) for r, m in zip(rows, masks) if m.sum() > 50]
        keep.sort(key=lambda rm: np.nonzero(rm[1])[1].mean())  # left to right, as images.csv requires
        cv2.imwrite(str(dp.raw / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append(dict(image=name, veg_ids=";".join(r["veg_id"] for r, _ in keep), set=set_, case=case, session=session))
        if reference:
            lines = [mask_to_polygon_line(m) for _, m in keep]
            (dp.reference_masks / f"{Path(name).stem}.txt").write_text("\n".join(x for x in lines if x) + "\n")
            ref_written += 1

    def session():
        """Pick a normal lighting session at random.

        Returns:
            "daylight" or "room".
        """
        return ["daylight", "room"][int(rng.integers(2))]

    k = 0
    # Singles: every item alone, 3 photos in different positions and rotations.
    for vid in gt["veg_id"]:
        for _ in range(max(1, round(3 * scale))):
            s = session()
            scene, rows, poses = compose([vid], s)
            k += 1
            emit(f"single_{k:04d}.jpg", rows, scene.render(rows, shapes, poses), scene.finish(), "singles", session=s)

    # Same-crop pairs and groups of 3.
    for j in range(round(60 * scale)):
        crop = list(by_crop)[j % len(by_crop)]
        n = min(len(by_crop[crop]), 3 if rng.random() < 0.35 else 2)
        if n < 2:
            continue
        s = session()
        scene, rows, poses = compose(list(rng.choice(by_crop[crop], n, replace=False)), s)
        emit(f"pair_{j:04d}.jpg", rows, scene.render(rows, shapes, poses), scene.finish(), "pairs", session=s, reference=ref_written < 6)

    # Mixed-crop photos: several items of 2-3 crops, e.g. 2 tomatoes + 3 chamagadda.
    for j in range(round(20 * scale)):
        crops = list(rng.choice(list(by_crop), min(len(by_crop), int(rng.integers(2, 4))), replace=False))
        ids = []
        for c in crops:
            ids += list(rng.choice(by_crop[c], min(len(by_crop[c]), int(rng.integers(1, 4))), replace=False))
        ids = ids[:5]
        s = session()
        scene, rows, poses = compose(ids, s, gap_mm=10)
        emit(f"mixed_{j:04d}.jpg", rows, scene.render(rows, shapes, poses), scene.finish(), "mixed", session=s, reference=ref_written < 10)

    # Stress set: touching items, a tilted camera, dim light.
    for j in range(round(24 * scale)):
        case = ["touching", "tilt", "dim"][j % 3]
        crop = list(by_crop)[int(rng.integers(len(by_crop)))]
        ids = list(rng.choice(by_crop[crop], min(2, len(by_crop[crop])), replace=False))
        if case == "touching":
            scene = Scene(rng, "daylight")
            rows = [gt_idx.loc[v] for v in ids]
            poses = None
            for _ in range(40):
                poses = make_touching(scene, rows, shapes, rng)
                if poses is not None:
                    break
            masks = scene.render(rows, shapes, poses)
            img, s = scene.finish(), "daylight"
        else:
            s = "dim" if case == "dim" else "daylight"
            scene, rows, poses = compose(ids, s)
            masks = scene.render(rows, shapes, poses)
            img = scene.finish()
            if case == "tilt":
                img, masks = tilt(img, masks, rng)
        emit(f"stress_{case}_{j:03d}.jpg", rows, masks, img, "stress", case, s)

    pd.DataFrame(records).to_csv(dp.images, index=False, lineterminator="\n")

    # One ruler photo, calibrated exactly as `python -m src.calibrate ruler.jpg --auto --mm 320` would.
    ruler = ruler_photo(rng)
    cv2.imwrite(str(dp.raw / "ruler.jpg"), ruler, [cv2.IMWRITE_JPEG_QUALITY, 95])
    mmpp = RULER_MM / ruler_length_px(ruler)
    dp.calibration.write_text(
        json.dumps(
            {"mm_per_px": round(mmpp, 6), "image_size": [W, H], "method": "auto", "span_mm": RULER_MM, "source": "ruler.jpg"},
            indent=2,
        )
    )
    dp.synthetic_flag.write_text(
        json.dumps(
            {
                "warning": "SYNTHETIC DATA. Metrics computed on it validate the code, not real-world accuracy.",
                "true_mm_per_px": MMPP,
                "calibrated_mm_per_px": round(mmpp, 6),
                "reference_masks": "exact outlines (stand-in for hand tracing)",
                "seed": seed,
            },
            indent=2,
        )
    )
    sets = pd.DataFrame(records)["set"].value_counts().to_dict()
    print(f"wrote {len(gt)} items, {len(records)} photos {sets}, ruler -> {mmpp:.4f} mm/px (true {MMPP}), to {out}")


def main() -> None:
    """Command-line entry point: generate the dataset, refusing to overwrite real data."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DataPaths().root)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dp = DataPaths(args.out)
    if dp.ground_truth.exists() and not dp.synthetic_flag.exists():
        raise SystemExit(f"{args.out} holds real data (no SYNTHETIC flag); refusing to overwrite it")
    generate(args.out, args.seed)


if __name__ == "__main__":
    main()
