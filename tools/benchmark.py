"""CPU latency benchmark for the full pipeline, e.g. on the Raspberry Pi at the packing table.

python -m tools.benchmark photo.jpg [--runs 30] [--warmup 3]
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from src.pipeline import Pipeline, add_config_args, build_config
from src.segment import segment, white_balance


def main() -> None:
    """Command-line entry point: time segmentation and the full pipeline on one photo."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", type=Path)
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--warmup", type=int, default=3)
    add_config_args(p)
    args = p.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"cannot read {args.image}")
    pipe = Pipeline(build_config(args))
    for _ in range(args.warmup):
        pipe.run(image)

    seg_ms, total_ms = [], []
    for _ in range(args.runs):
        t0 = perf_counter()
        segment(white_balance(image), pipe.cfg, pipe.calibration.mm_per_px)
        seg_ms.append(1000 * (perf_counter() - t0))
        t0 = perf_counter()
        pipe.run(image)
        total_ms.append(1000 * (perf_counter() - t0))

    def stats(v):
        """Summarise a list of timings.

        Args:
            v: Timings in milliseconds.

        Returns:
            Mean, median and 95th percentile, rounded to 0.1 ms.
        """
        return {"mean": round(float(np.mean(v)), 1), "p50": round(float(np.median(v)), 1), "p95": round(float(np.percentile(v, 95)), 1)}

    print(
        json.dumps(
            {
                "machine": f"{platform.system()} {platform.machine()} ({platform.processor() or 'unknown cpu'})",
                "image": f"{image.shape[1]}x{image.shape[0]}",
                "segmentation_ms": stats(seg_ms),
                "pipeline_ms": stats(total_ms),
                "runs": args.runs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
