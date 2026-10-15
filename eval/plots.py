"""Charts for the results write-up, drawn from the files eval/evaluate.py writes.

  python -m eval.plots [--reports reports]

pairwise_vs_difference.png   the key chart: pairwise accuracy against true weight difference
weight_pred_vs_true.png      predicted vs measured weight per crop (leave-one-item-out)
small_big_comparison.png     Small/Big accuracy: raw-image baselines vs feature-based models
stress_cases.png             pairwise accuracy and weight error per stress case
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import REPORTS_DIR  # noqa: E402

# Validated categorical palette (fixed slot order) and chart chrome.
SERIES = ["#2a78d6", "#eb6834"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def style() -> None:
    """Apply the shared chart style: palette, fonts, recessive grid and axes."""
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update(
        {
            "font.family": ["Segoe UI", "DejaVu Sans"],
            "font.size": 10.5,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": MUTED,
            "ytick.left": False,
            "legend.frameon": False,
            "legend.labelcolor": INK2,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.facecolor": SURFACE,
        }
    )


def synthetic_note(fig, metrics: dict) -> None:
    """Add a footnote saying the chart comes from synthetic data, when it does.

    Args:
        fig: The figure to annotate.
        metrics: The metrics, whose "data" section says whether the data was synthetic.
    """
    if metrics.get("data", {}).get("synthetic"):
        fig.text(
            0.01, -0.02, "Synthetic data: validates the code path, not real-world accuracy.", color=MUTED, fontsize=9, ha="left", va="top"
        )


def plot_pairwise(m: dict, out: Path) -> None:
    """Draw the key chart: pairwise accuracy against the true weight difference.

    Args:
        m: The metrics from `evaluate`.
        out: Folder to save `pairwise_vs_difference.png` into.
    """
    sources = [
        (k, lbl)
        for k, lbl in (("chosen", "Pairs in the same photo"), ("all_item_pairs", "All same-crop item pairs"))
        if m["pairwise"].get(k, {}).get("n_pairs")
    ]
    if not sources:
        return
    bins = [b["bin"] for b in m["pairwise"][sources[0][0]]["bins"]]
    x = np.arange(len(bins))
    width = 0.11
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for s, (key, label) in enumerate(sources):
        vals = m["pairwise"][key]["bins"]
        offs = x + (s - (len(sources) - 1) / 2) * 0.3
        acc = [100 * v["order_accuracy"] if v["order_accuracy"] is not None else 0 for v in vals]
        ax.bar(offs, acc, width, color=SERIES[s], label=label, zorder=3)
        for xo, a, v in zip(offs, acc, vals):
            txt = f"{a:.0f}%\n{v['n']} / {v['n_item_pairs']}" if v["order_accuracy"] is not None else "none"
            ax.text(xo, a + 1.5, txt, ha="center", va="bottom", fontsize=8.5, color=INK2, linespacing=1.1)
    ax.axhline(50, color=MUTED, linewidth=1, zorder=2)
    ax.text(0.5, 51, "chance", color=MUTED, fontsize=8.5, ha="center", va="bottom")
    ax.set_xticks(x, bins)
    ax.set_xlim(-0.5, len(bins) - 0.5)
    ax.set_xlabel("True weight difference   (bar labels: accuracy, then pairs / distinct item pairs)")
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Picks the heavier item: accuracy by true weight difference", pad=30)
    ax.legend(loc="lower left", ncols=len(sources), fontsize=9, bbox_to_anchor=(0, 1.0), handlelength=1, borderaxespad=0.3)
    synthetic_note(fig, m)
    fig.savefig(out / "pairwise_vs_difference.png")
    plt.close(fig)


def plot_weights(rows: pd.DataFrame, m: dict, out: Path) -> None:
    """Scatter predicted against measured weight, one panel per crop.

    Args:
        rows: The out-of-fold rows from `eval_rows.csv`.
        m: The metrics from `evaluate`, for the chosen models and MAPE.
        out: Folder to save `weight_pred_vs_true.png` into.
    """
    d = rows.loc[rows["set"] == "singles"].dropna(subset=["w_chosen"])
    crops = sorted(d["crop"].unique())
    if not crops:
        return
    fig, axes = plt.subplots(1, len(crops), figsize=(4.0 * len(crops), 4.2), squeeze=False)
    for ax, crop in zip(axes[0], crops):
        g = d[d["crop"] == crop]
        lo = 0.85 * min(g["weight_g"].min(), g["w_chosen"].min())
        hi = 1.08 * max(g["weight_g"].max(), g["w_chosen"].max())
        ax.plot([lo, hi], [lo, hi], color=AXIS, linewidth=1.2, zorder=2)
        ax.scatter(g["weight_g"], g["w_chosen"], s=42, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(axis="both")
        model = m["config"]["weight_models"].get(crop, "")
        mape = m["weight"].get(crop, {}).get(model, {}).get("mape_pct", float("nan"))
        ax.set_title(f"{crop.capitalize()}: {model}, MAPE {mape:.1f}%", fontsize=11)
        ax.set_xlabel("Measured weight (g)")
        ax.set_ylabel("Predicted weight (g)")
        ax.text(hi, hi, "perfect ", color=MUTED, fontsize=8.5, ha="right", va="bottom")
    fig.suptitle(
        "Predicted vs measured weight (leave-one-item-out, singles)", x=0.01, ha="left", fontsize=12, fontweight="semibold", color=INK
    )
    fig.tight_layout()
    synthetic_note(fig, m)
    fig.savefig(out / "weight_pred_vs_true.png")
    plt.close(fig)


def plot_small_big(m: dict, out: Path) -> None:
    """Compare Small/Big accuracy of the raw-image baselines and the feature-based models.

    Args:
        m: The metrics from `evaluate`.
        out: Folder to save `small_big_comparison.png` into.
    """
    sb = m["small_big"]
    entries = []
    for name, label in (("baseline_hog_svm", "HOG + SVM, raw patches"), ("baseline_cnn", "CNN, raw patches")):
        if name in sb:
            entries.append((label, sb[name]["overall"]["accuracy"]))
    for key, label in (
        ("weight_threshold|chosen", "Threshold on estimated weight"),
        ("logreg|chosen", "Logistic regression, features"),
        ("svm|chosen", "SVM, features"),
    ):
        if key in sb["models"]:
            entries.append((label, sb["models"][key]["accuracy"]))
    if not entries:
        return
    labels, accs = zip(*entries)
    fig, ax = plt.subplots(figsize=(7.6, 0.55 * len(entries) + 1.3))
    y = np.arange(len(entries))[::-1]
    ax.barh(y, [100 * a for a in accs], 0.3, color=SERIES[0], zorder=3)
    for yi, a in zip(y, accs):
        ax.text(100 * a + 1.5, yi, f"{100 * a:.0f}%", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_title("Small / Big accuracy (leave-one-item-out)")
    synthetic_note(fig, m)
    fig.savefig(out / "small_big_comparison.png")
    plt.close(fig)


def plot_stress(m: dict, out: Path) -> None:
    """Show pairwise accuracy and weight error for each stress case.

    Args:
        m: The metrics from `evaluate`.
        out: Folder to save `stress_cases.png` into.
    """
    rob = m["robustness"]
    if not rob:
        return
    names = list(rob)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (key, title, mult) in zip(
        axes, (("pair_order_accuracy", "Pairwise accuracy", 100), ("weight_mape_pct", "Weight error (MAPE)", 1))
    ):
        vals = [(rob[n][key] or 0) * mult for n in names]
        ax.bar(names, vals, 0.15, color=SERIES[0], zorder=3)
        for i, v in enumerate(vals):
            label = f"{v:.0f}%" if mult == 100 else f"{v:.1f}%"
            ax.text(i, v + max(vals + [1]) * 0.02, label, ha="center", va="bottom", fontsize=9, color=INK2)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(vals + [1]) * 1.2)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    fig.suptitle("Stress set: touching items, tilted camera, dim light", x=0.01, ha="left", fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout()
    synthetic_note(fig, m)
    fig.savefig(out / "stress_cases.png")
    plt.close(fig)


def main() -> None:
    """Command-line entry point: draw every chart from the saved reports."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reports", type=Path, default=REPORTS_DIR)
    args = p.parse_args()
    style()
    m = json.loads((args.reports / "metrics.json").read_text())
    rows = pd.read_csv(args.reports / "eval_rows.csv")
    plot_pairwise(m, args.reports)
    plot_weights(rows, m, args.reports)
    plot_small_big(m, args.reports)
    plot_stress(m, args.reports)
    print(f"charts written to {args.reports}")


if __name__ == "__main__":
    main()
