"""Stage A: classical segmentation of vegetables on a plain white sheet.

No training, no annotation:
  1. HSV saturation and Lab lightness. Produce is saturated (red tomato, green kheera)
     or dark (brown chamagadda); the sheet is bright and unsaturated, and shadows are
     grey (low saturation, only slightly darker), so they drop out.
  2. Otsu threshold on saturation, OR lightness below a fraction of the sheet's.
  3. Morphology: open (specks, chamagadda hairs), close (gaps), fill holes.
  4. Connected blobs above a minimum area.
  5. Touching items: distance transform + watershed splits them.
  6. Blobs touching the image border are flagged as truncated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.config import Config


@dataclass
class Detection:
    """One segmented item.

    Attributes:
        mask: Boolean mask of the item, the same size as the photo.
        box: Bounding box as `(x1, y1, x2, y2)`, with x2 and y2 exclusive.
        flags: Notes from segmentation, e.g. "touches_border" or "split_from_touching".
    """

    mask: np.ndarray  # bool, H x W
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    flags: list[str] = field(default_factory=list)

    @property
    def area_px(self) -> int:
        """The item's area in pixels."""
        return int(self.mask.sum())

    @property
    def centroid(self) -> tuple[float, float]:
        """The item's centre `(x, y)` in pixels; detections are ordered left to right by it."""
        ys, xs = np.nonzero(self.mask)
        return float(xs.mean()), float(ys.mean())


def mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box of the True pixels in a mask.

    Args:
        mask: A non-empty boolean mask.

    Returns:
        `(x1, y1, x2, y2)`, with x2 and y2 exclusive.
    """
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill any holes inside a mask's outer outlines.

    This closes gaps such as a tomato's specular highlight or the spaces between leaves.

    Args:
        mask: Binary mask.

    Returns:
        A boolean mask with every enclosed hole filled.
    """
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    cv2.drawContours(out, contours, -1, 1, thickness=cv2.FILLED)
    return out.astype(bool)


def white_balance(image: np.ndarray, target: float = 235.0) -> np.ndarray:
    """Scale each colour channel so the white sheet comes out neutral grey.

    The sheet fills most of the frame, so the per-channel median is the sheet's colour.
    Dividing it out removes the light's brightness and colour cast (dim light, warm room
    light) before any colour is measured. The sheet is a free white reference.

    Args:
        image: BGR photo from the rig.
        target: Grey level the sheet should end up at.

    Returns:
        The white-balanced photo.
    """
    med = np.median(image[::4, ::4].reshape(-1, 3), axis=0).astype(np.float32)  # subsample: speed
    gain = target / np.maximum(med, 1.0)
    return np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def _kernel(mm: float, mm_per_px: float) -> np.ndarray:
    """Elliptical morphology kernel of a given physical size.

    Args:
        mm: Kernel diameter in millimetres.
        mm_per_px: Scale of the rig.

    Returns:
        The kernel, at least 3 pixels across and always an odd size.
    """
    k = max(3, int(round(mm / mm_per_px)) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def foreground(image: np.ndarray, cfg: Config, mm_per_px: float) -> np.ndarray:
    """Find every pixel that belongs to produce rather than the sheet.

    A pixel counts as produce if it is clearly saturated (Otsu threshold on HSV
    saturation, red tomatoes and green kheera) or clearly darker than the sheet (Lab
    lightness, brown chamagadda). Shadows are grey and only a little darker, so they
    fail both tests. An opening then removes specks and chamagadda hairs, a closing
    fills small gaps, and enclosed holes are filled.

    Args:
        image: White-balanced BGR photo.
        cfg: Segmentation settings.
        mm_per_px: Scale of the rig, so the morphology works in millimetres.

    Returns:
        A boolean produce mask.
    """
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    sat = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)[..., 1]
    light = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)[..., 0]

    t_sat, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sheet = float(np.median(light[::4, ::4]))  # the sheet fills most of the frame
    fg = (sat > max(t_sat, cfg.min_sat)) | (light < cfg.dark_frac * sheet)

    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, _kernel(cfg.open_mm, mm_per_px))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _kernel(cfg.close_mm, mm_per_px))
    return fill_holes(fg)


def split_touching(blob: np.ndarray, depth: float, min_peak_px: float) -> list[np.ndarray]:
    """Split a blob into separate items if several vegetables are touching.

    Convex blobs are returned as they are: touching items always leave a concave waist.
    Otherwise:

    1. Seeds (h-maxima style). The distance map is swept from high to low. Each new
       region above the current level starts a seed at its peak. When two regions
       merge, the lower seed records the merge level, its "saddle" (the neck between
       the items). A seed is kept only if that neck is deep relative to its own peak:
       `(peak - saddle) >= depth * peak`. Two touching items have a narrow neck and both
       survive; a bump on a chamagadda or a ripple along a kheera's ridge does not.
       Unlike a single global threshold, a small item touching a much bigger one still
       gets its own seed.
    2. Cores. Each kept seed's region above one common level, chosen between the deepest
       kept saddle and the lowest kept peak. Because every core then sits the same depth
       below its own outline, the boundary between two items passes through their
       contact point.
    3. Every blob pixel is assigned to its nearest core, which amounts to a watershed
       with equal flooding speed.

    `cv2.watershed` is not used: it floods by colour difference between neighbouring
    pixels, so on a smooth distance map one marker can race around another.

    Args:
        blob: Boolean mask of one connected blob.
        depth: How deep a neck must be, as a fraction of the seed's peak, to split there.
        min_peak_px: Seeds thinner than this (inner radius in pixels) are ignored.

    Returns:
        One mask per item. A blob holding a single item comes back unchanged.
    """
    # Touching items always leave a concave waist; a convex blob is a single item (saves time).
    b8 = blob.astype(np.uint8)
    contours, _ = cv2.findContours(b8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_area = cv2.contourArea(cv2.convexHull(np.vstack(contours)))
    if hull_area > 0 and cv2.contourArea(max(contours, key=cv2.contourArea)) / hull_area >= 0.93:
        return [blob]
    dist = cv2.distanceTransform(b8, cv2.DIST_L2, 5)
    dmax = float(dist.max())
    seeds: list[dict] = []  # {"pt": (y, x), "peak": float, "saddle": float | None}
    for t in np.linspace(0.97 * dmax, 0.03 * dmax, 24):
        n, lab = cv2.connectedComponents((dist > t).astype(np.uint8))
        comp = [lab[sd["pt"]] for sd in seeds]
        for c in range(1, n):
            members = [k for k, cc in enumerate(comp) if cc == c]
            if not members:
                region = lab == c
                flat = np.where(region, dist, -1).argmax()
                pt = np.unravel_index(flat, dist.shape)
                seeds.append({"pt": pt, "peak": float(dist[pt]), "saddle": None})
            elif len(members) > 1:
                top = max(members, key=lambda k: seeds[k]["peak"])
                for k in members:
                    if k != top and seeds[k]["saddle"] is None:
                        seeds[k]["saddle"] = float(t)
    kept = [sd for sd in seeds if sd["peak"] >= min_peak_px and (sd["saddle"] is None or sd["peak"] - sd["saddle"] >= depth * sd["peak"])]
    if len(kept) < 2:
        return [blob]
    saddles = [sd["saddle"] for sd in kept if sd["saddle"] is not None]
    low_peak = min(sd["peak"] for sd in kept)
    level = (max(saddles) + low_peak) / 2 if saddles else low_peak / 2  # no saddle: parts never joined
    n, cores = cv2.connectedComponents((dist > level).astype(np.uint8))
    core_ids = sorted({int(cores[sd["pt"]]) for sd in kept} - {0})
    if len(core_ids) < 2:
        return [blob]
    not_core = np.ones(blob.shape, np.uint8)
    for i in core_ids:
        not_core[cores == i] = 0
    # Each pixel gets the label of the nearest zero-valued (core) component.
    _, nearest = cv2.distanceTransformWithLabels(not_core, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_CCOMP)
    parts = [fill_holes(blob & (nearest == nearest[cores == i][0])) for i in core_ids]
    return [p for p in parts if p.any()]


def segment(image: np.ndarray, cfg: Config, mm_per_px: float) -> list[Detection]:
    """Find every vegetable in a photo.

    Args:
        image: White-balanced BGR photo from the rig.
        cfg: Segmentation settings.
        mm_per_px: Scale of the rig.

    Returns:
        The detected items, ordered left to right. Items touching the frame edge are
        flagged "touches_border"; items separated from a touching neighbour are flagged
        "split_from_touching".
    """
    h, w = image.shape[:2]
    fg = foreground(image, cfg, mm_per_px)
    min_area = cfg.min_area_mm2 / mm_per_px**2
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8))
    dets = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        # Work inside the blob's bounding box (plus a 1 px margin) for speed.
        x0, y0, x1, y1 = max(x - 1, 0), max(y - 1, 0), min(x + bw + 1, w), min(y + bh + 1, h)
        blob = labels[y0:y1, x0:x1] == i
        min_peak = np.sqrt(min_area / np.pi)  # inner radius of the smallest allowed item
        sub_parts = split_touching(blob, cfg.split_depth, min_peak) if cfg.split_touching else [blob]
        parts = []
        for sp in sub_parts:
            full = np.zeros((h, w), bool)
            full[y0:y1, x0:x1] = sp
            parts.append(full)
        for part in parts:
            if part.sum() < min_area:
                continue
            x1, y1, x2, y2 = mask_box(part)
            b = cfg.border_px
            flags = ["touches_border"] if (x1 < b or y1 < b or x2 > w - b or y2 > h - b) else []
            if len(parts) > 1:
                flags.append("split_from_touching")
            dets.append(Detection(part, (x1, y1, x2, y2), flags))
    dets.sort(key=lambda d: d.centroid[0])
    return dets
