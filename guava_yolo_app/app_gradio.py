"""
Gradio Web Application for Guava Defect Detection & Morphological Analysis
Run via:
    python app_gradio.py
Then open http://127.0.0.1:7860 in your browser.
"""

import os
from functools import lru_cache

import cv2
import numpy as np
import gradio as gr
from pipeline import GuavaDefectPipeline

THEME = gr.themes.Soft()

CSS = """
.gradio-container {max-width: 1500px !important; margin: auto;}
#run-btn {font-size: 1.1rem; padding: 14px;}
#report {border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 12px 18px;}
"""


@lru_cache(maxsize=4)
def get_pipeline(conf: float, polar: float, lenticel: float) -> GuavaDefectPipeline:
    """Build the pipeline once per parameter set so models are not reloaded on every click."""
    return GuavaDefectPipeline(
        conf_threshold=conf,
        polar_exclusion_ratio=polar,
        max_lenticel_area_pct=lenticel,
    )


def run_guava_analysis(
    image_rgb: np.ndarray,
    conf_threshold: float,
    polar_exclusion_ratio: float,
    max_lenticel_area_pct: float,
):
    if image_rgb is None:
        return None, None, None, None, "⚠️ Please upload or select a guava image first."

    try:
        pipe = get_pipeline(
            float(conf_threshold),
            float(polar_exclusion_ratio),
            float(max_lenticel_area_pct),
        )
        out = pipe.analyze(image_rgb)
        mask_rgb = cv2.cvtColor(out.yolov8_mask, cv2.COLOR_GRAY2RGB)
        return (
            out.annotated_yolov5_rgb,
            out.segmented_guava_rgb,
            mask_rgb,
            out.chart_figure,
            out.diagnostic_markdown,
        )
    except Exception as e:  # show the error in the UI instead of a silent failure
        return None, None, None, None, f"❌ **Analysis failed:** `{type(e).__name__}: {e}`"


def build_gradio_interface() -> gr.Blocks:
    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
    candidate_examples = [
        [os.path.join(sample_dir, "sample_1_anthracnose.jpg"), 0.45, 0.22, 0.30],
        [os.path.join(sample_dir, "sample_2_canker.jpg"), 0.45, 0.22, 0.30],
        [os.path.join(sample_dir, "sample_3_healthy.jpg"), 0.45, 0.22, 0.30],
    ]
    examples = [e for e in candidate_examples if os.path.exists(e[0])]

    with gr.Blocks(title="GuavaVision AI — YOLOv8-Seg + YOLOv5 Defect Analyzer") as demo:
        gr.Markdown(
            """
            # 🍐 GuavaVision AI: Guava Segmentation & Defect Detection
            **Stage 1 ** separates the fruit from the background  ·
            **Stage 2 ** detects true defects (**red**) and ignores stem, blossom-end and lenticels (**cyan**)  ·
            **Stage 3:** colour & shape analysis with a pathology report
            """
        )

        with gr.Row():
            # ---------- Left: inputs ----------
            with gr.Column(scale=1, min_width=340):
                input_img = gr.Image(type="numpy", label="Input Guava Image", height=360)
                with gr.Accordion("⚙️ Detection settings", open=True):
                    conf_slider = gr.Slider(
                        0.25, 0.90, value=0.45, step=0.05,
                        label="Defect confidence threshold",
                        info="Higher = fewer, more certain detections.",
                    )
                    polar_slider = gr.Slider(
                        0.10, 0.32, value=0.22, step=0.02,
                        label="Stem & blossom polar exclusion ratio",
                        info="Share of fruit length at each end treated as stem/calyx zone.",
                    )
                    lenticel_slider = gr.Slider(
                        0.05, 0.80, value=0.30, step=0.05,
                        label="Natural lenticel / russet max area (%)",
                        info="Marks smaller than this are treated as natural.",
                    )
                analyze_btn = gr.Button(
                    "🚀 Run Analysis", variant="primary", elem_id="run-btn"
                )

            # ---------- Right: outputs ----------
            with gr.Column(scale=2):
                with gr.Row():
                    out_det = gr.Image(
                        type="numpy",
                        label="Stage 2: Defects (red) & ignored anatomy (cyan)",
                        height=420,
                    )
                    out_seg = gr.Image(
                        type="numpy",
                        label="Stage 1: Background removed",
                        height=420,
                    )
                with gr.Accordion("Stage 1 binary mask", open=False):
                    out_mask = gr.Image(type="numpy", label="YOLOv8 binary mask", height=300)

                out_chart = gr.Plot(label="Stage 3: Colour & shape analysis")
                out_report = gr.Markdown(
                    value="*Run an analysis to see the pathology report here.*",
                    elem_id="report",
                )

        outputs = [out_det, out_seg, out_mask, out_chart, out_report]
        inputs = [input_img, conf_slider, polar_slider, lenticel_slider]

        if examples:
            gr.Examples(
                examples=examples,
                inputs=inputs,
                outputs=outputs,
                fn=run_guava_analysis,
                cache_examples=False,
            )

        analyze_btn.click(fn=run_guava_analysis, inputs=inputs, outputs=outputs)

    return demo


if __name__ == "__main__":
    app = build_gradio_interface()
    # share=False avoids the "Could not create share link" message.
    # Set share=True only if you want a public *.gradio.live link.
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=THEME,
        css=CSS,
    )