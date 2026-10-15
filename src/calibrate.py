"""One-time calibration for the fixed rig: a ruler photo -> mm per pixel.

The camera height never changes, so one photo of a ruler lying on the sheet fixes the
scale for every later photo. Two ways to measure the known span:

    # Manual (recommended): two points on the ruler's scale, read off any image viewer
    python -m src.calibrate data/raw/ruler.jpg --points 212 604 1388 610 --mm 300

    # Automatic: the ruler is the largest dark object; give its physical body length
    python -m src.calibrate data/raw/ruler.jpg --auto --mm 320

Writes data/calibration.json:  {"mm_per_px": 0.25, "image_size": [1600, 1200], ...}

Re-calibrate only when the camera height, phone or zoom changes. `--check` measures a
known item in a photo and reports the error against its ruler length.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Calibration:
    """Scale of the fixed rig, as measured from one ruler photo.

    Attributes:
        mm_per_px: Millimetres covered by one pixel at the rig's camera height.
        image_size: Width and height of the calibration photo, used to spot photos taken
            at a different resolution. None if it was not recorded.
    """

    mm_per_px: float
    image_size: tuple[int, int] | None = None  # (width, height) of the calibration photo

    @classmethod
    def load(cls, path: Path) -> Calibration:
        """Read a calibration from the JSON file written by `python -m src.calibrate`.

        Args:
            path: Path to `calibration.json`.

        Returns:
            The calibration stored in the file.
        """
        d = json.loads(Path(path).read_text())
        size = tuple(d["image_size"]) if d.get("image_size") else None
        return cls(float(d["mm_per_px"]), size)

    def for_image(self, shape: tuple[int, ...]) -> tuple[float, str | None]:
        """Work out mm/px for a particular photo and check it came from the same rig.

        A photo with the same framing but a different resolution (say the phone saved a
        smaller copy) is handled by rescaling. A photo with a different aspect ratio
        probably came from a different setup, so it gets a warning.

        Args:
            shape: The photo's array shape, `(height, width[, channels])`.

        Returns:
            A `(mm_per_px, warning)` pair. The warning is None when the photo matches the
            calibration photo exactly.
        """
        if self.image_size is None:
            return self.mm_per_px, None
        w, h = shape[1], shape[0]
        cw, ch = self.image_size
        if (w, h) == (cw, ch):
            return self.mm_per_px, None
        if abs(w / h - cw / ch) < 0.01:
            # Same framing at a different resolution (e.g. the phone saved a smaller copy).
            return self.mm_per_px * cw / w, f"photo is {w}x{h}, calibration {cw}x{ch}: mm/px rescaled"
        return self.mm_per_px, f"photo is {w}x{h} but calibration was {cw}x{ch}: sizes may be wrong, re-calibrate"


def from_points(p1: tuple[float, float], p2: tuple[float, float], span_mm: float) -> float:
    """Scale from two points a known distance apart, e.g. two ruler ticks.

    Args:
        p1: First point `(x, y)` in pixels.
        p2: Second point `(x, y)` in pixels.
        span_mm: Real distance between the two points, in millimetres.

    Returns:
        Millimetres per pixel.
    """
    return span_mm / float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))


def ruler_length_px(image: np.ndarray) -> float:
    """Measure the ruler's length in pixels.

    The ruler is taken to be the largest dark object on the sheet; its length is the
    long side of that object's minimum-area rectangle.

    Args:
        image: BGR photo of the ruler lying on the white sheet.

    Returns:
        Length of the ruler in pixels.

    Raises:
        ValueError: If nothing dark enough to be a ruler is found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("no ruler found")
    (_, _), (w, h), _ = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    return float(max(w, h))


def main() -> None:
    """Command-line entry point: calibrate from a ruler photo, or check an existing calibration."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("photo", type=Path)
    p.add_argument("--mm", type=float, required=True, help="known span (manual) or ruler body length (auto) in mm")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--points", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    g.add_argument("--auto", action="store_true")
    g.add_argument("--check", type=Path, help="existing calibration.json: measure a known item instead")
    p.add_argument("--out", type=Path, default=Path("data/calibration.json"))
    args = p.parse_args()

    image = cv2.imread(str(args.photo))
    if image is None:
        raise SystemExit(f"cannot read {args.photo}")

    if args.check:
        mmpp, warn = Calibration.load(args.check).for_image(image.shape)
        measured = ruler_length_px(image) * mmpp
        print(f"measured {measured:.1f} mm vs ruler {args.mm:.1f} mm: error {100 * (measured - args.mm) / args.mm:+.1f}%")
        if warn:
            print("WARNING:", warn)
        return

    if args.points:
        x1, y1, x2, y2 = args.points
        mmpp, method = from_points((x1, y1), (x2, y2), args.mm), "points"
    else:
        mmpp, method = args.mm / ruler_length_px(image), "auto"
    out = {
        "mm_per_px": round(mmpp, 6),
        "image_size": [image.shape[1], image.shape[0]],
        "method": method,
        "span_mm": args.mm,
        "source": args.photo.name,
        "date": date.today().isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
