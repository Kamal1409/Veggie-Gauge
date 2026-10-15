"""Demo UI: upload a photo, see the contours, the per-crop results table and the raw JSON.

python -m app.gradio_app [--port 7860] [--share]
"""

from __future__ import annotations

import argparse

import cv2
import gradio as gr
import numpy as np

from src.config import Config, DataPaths
from src.pipeline import Pipeline

COLUMNS = ["crop group", "#", "crop prob.", "length (mm)", "width (mm)", "weight (g)", "± (g)", "rank", "pairwise", "absolute", "note"]


def make_analyse(base: Pipeline):
    """Wrap a loaded pipeline in the callback the demo's controls call.

    Args:
        base: The loaded pipeline; each call reuses its models.

    Returns:
        The callback.
    """

    def analyse(image: np.ndarray | None, delta_pct: float, tau: float):
        """Run the pipeline on the uploaded photo and format the results for the UI.

        Args:
            image: The uploaded photo (RGB), or None if nothing has been uploaded yet.
            delta_pct: SIMILAR margin in percent, from the slider.
            tau: Minimum crop probability, from the slider.

        Returns:
            The annotated photo, a Markdown headline, the table rows and the raw JSON result.
        """
        if image is None:
            return None, "Upload a photo first.", [], {}
        out = base.with_config(delta=delta_pct / 100, tau=tau).run(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        r = out.result

        rows = []
        for g in r["groups"]:
            for it in g["items"]:
                rows.append(
                    [
                        g["crop"],
                        it["id"],
                        it["crop_prob"],
                        it["length_mm"],
                        it["width_mm"],
                        it.get("weight_g"),
                        it.get("weight_err_g"),
                        it.get("rank"),
                        it.get("pairwise", ""),
                        it.get("absolute", ""),
                        it.get("note", ""),
                    ]
                )
        for it in r["uncertain"]:
            rows.append(
                ["(uncertain)", it["id"], it["crop_prob"], it["length_mm"], it["width_mm"], None, None, None, "", "", "not compared"]
            )
        for it in r["excluded"]:
            rows.append(["(excluded)", it["id"], it["crop_prob"], it["length_mm"], it["width_mm"], None, None, None, "", "", it["reason"]])

        if r["status"] != "ok":
            head = f"### {r['message'].capitalize()}"
        else:
            parts = []
            for g in r["groups"]:
                s = f"**{g['crop']}** × {len(g['items'])}"
                if g["difference_pct"] is not None:
                    s += f" (largest vs smallest {g['difference_pct']:.0f}%)"
                parts.append(s)
            head = "### " + " · ".join(parts)
        head += f"\n\n{r['mm_per_px']:.3f} mm/px · {r['latency_ms']:.0f} ms · items of different crops are never compared with each other"
        for n in r["notes"]:
            head += f"\n\n> {n}"
        return cv2.cvtColor(out.overlay(), cv2.COLOR_BGR2RGB), head, rows, r

    return analyse


def example_images(dp: DataPaths) -> list[str]:
    """Pick one example photo from each photo set, if a dataset is present.

    Args:
        dp: Dataset locations.

    Returns:
        Paths to the example photos (empty if there is no dataset).
    """
    if not dp.images.exists():
        return []
    import pandas as pd

    df = pd.read_csv(dp.images, dtype=str, keep_default_na=False)
    if "set" not in df.columns:
        return []
    picks = []
    for s in ("pairs", "mixed", "stress", "singles"):
        sel = df[df["set"] == s]
        picks += [str(dp.raw / n) for n in list(sel["image"])[:1] if (dp.raw / n).exists()]
    return picks


def build(pipe: Pipeline) -> gr.Blocks:
    """Lay out the demo: photo in, annotated photo, table and JSON out.

    Args:
        pipe: The loaded pipeline.

    Returns:
        The Gradio app, ready to launch.
    """
    analyse = make_analyse(pipe)
    with gr.Blocks(title="Vegetable size comparison") as demo:
        gr.Markdown(
            "# Vegetable size comparison\n"
            "Upload a **top-down** photo from the fixed rig (phone overhead, white sheet). Every item is "
            "segmented, measured in millimetres, identified as tomato / kheera / chamagadda, and its fresh "
            "weight estimated. Items are **grouped by crop**: each group is ranked (**BIGGER / SMALLER / "
            "SIMILAR**) and every item is graded **SMALL / BIG** for its crop."
        )
        with gr.Row():
            inp = gr.Image(type="numpy", label="Photo")
            out_img = gr.Image(type="numpy", label="Result")
        with gr.Row():
            delta = gr.Slider(0, 30, value=pipe.cfg.delta * 100, step=1, label="SIMILAR margin δ (%)")
            tau = gr.Slider(0, 1, value=pipe.cfg.tau, step=0.05, label="Min. crop probability τ")
        btn = gr.Button("Analyse", variant="primary")
        head = gr.Markdown()
        table = gr.Dataframe(headers=COLUMNS, label="Items", interactive=False)
        raw = gr.JSON(label="API response")
        examples = example_images(DataPaths())
        if examples:
            gr.Examples(examples=examples, inputs=inp)
        btn.click(analyse, [inp, delta, tau], [out_img, head, table, raw])
        inp.upload(analyse, [inp, delta, tau], [out_img, head, table, raw])
    return demo


def main() -> None:
    """Command-line entry point: launch the demo."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--share", action="store_true")
    args = p.parse_args()
    build(Pipeline(Config())).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
