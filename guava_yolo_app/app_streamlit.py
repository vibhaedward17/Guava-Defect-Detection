"""
Streamlit Web Application for Guava Defect Detection & Morphological Analysis
Run via:
    streamlit run app_streamlit.py
"""

import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from pipeline import GuavaDefectPipeline

st.set_page_config(
    page_title="GuavaVision AI — YOLOv8-Seg + YOLOv5 Defect Analyzer",
    page_icon="🍐",
    layout="wide",
)

st.title("🍐 GuavaVision AI: Two-Stage YOLOv8 Segmentation & YOLOv5 Defect Detection")
st.markdown(
    """
    **Zero-Confusion Guava Inspection Pipeline:**
    1. **Stage 1 (YOLOv8-Seg):** Isolates the guava fruit from complex backgrounds and computes contour geometry.
    2. **Stage 2 (YOLOv5 + Anatomical Suppression):** Detects true pathological defects (**Anthracnose, Fruit Canker/Scab, Stylar Rot, Insect Punctures**) while strictly **excluding Stem/Pedicel, Blossom-End Calyx, and Natural Lenticel Marks**.
    3. **Stage 3 (Morphological & Color Analytics):** Generates dual Pie Charts for HSV/LAB Surface Color and Shape/Morphology metrics plus a full pathology guide.
    """
)

with st.sidebar:
    st.header("⚙️ Model & Suppression Settings")
    yolov8_weights = st.text_input("YOLOv8-Seg Weights Path", value="yolov8n-seg.pt")
    yolov5_weights = st.text_input("YOLOv5 Defect Weights Path", value="guava_yolov5_defect.pt")
    conf_thresh = st.slider("YOLOv5 Defect Confidence Threshold", 0.25, 0.90, 0.45, 0.05)
    polar_ratio = st.slider(
        "Stem & Blossom Polar Exclusion Zone",
        0.10,
        0.32,
        0.22,
        0.02,
        help="Suppresses apical stem/pedicel and basal blossom-end calyx regions from being marked as defects.",
    )
    lenticel_thresh = st.slider(
        "Natural Lenticel / Russeting Max Area (%)",
        0.05,
        0.80,
        0.30,
        0.05,
        help="Harmless natural skin speckles smaller than this percentage of fruit area are ignored.",
    )

    sample_dir = os.path.join(os.path.dirname(__file__), "samples")
    sample_options = {
        "Upload Custom Guava Image": None,
        "Sample 1: Anthracnose Lesion + Stem & Blossom Calyx": os.path.join(sample_dir, "sample_1_anthracnose.jpg"),
        "Sample 2: Fruit Canker / Scab + Natural Lenticels": os.path.join(sample_dir, "sample_2_canker.jpg"),
        "Sample 3: Healthy Guava (Stem, Blossom & Lenticels Ignored)": os.path.join(sample_dir, "sample_3_healthy.jpg"),
    }
    selected_sample = st.selectbox("Select Test Sample or Upload", list(sample_options.keys()), index=1)

uploaded_file = st.file_uploader("Upload a Guava Image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])

image_rgb = None
if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(pil_img)
elif sample_options[selected_sample] and os.path.exists(sample_options[selected_sample]):
    bgr = cv2.imread(sample_options[selected_sample])
    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

if image_rgb is not None:
    pipeline = GuavaDefectPipeline(
        yolov8_seg_weights=yolov8_weights,
        yolov5_defect_weights=yolov5_weights,
        conf_threshold=conf_thresh,
        polar_exclusion_ratio=polar_ratio,
        max_lenticel_area_pct=lenticel_thresh,
    )
    result = pipeline.analyze(image_rgb)

    # Top KPI Metrics Row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Overall Fruit Status", "Healthy" if not result.defects else f"{len(result.defects)} Defect(s)")
    k2.metric("Suppressed Anatomy (Ignored)", f"{len(result.suppressed_regions)} Region(s)")
    k3.metric("Dominant Shape Profile", result.shape_classification.split("(")[0].strip())
    k4.metric("Surface Integrity Score", f"{result.shape_metrics['Epidermal Surface Integrity']:.1f}%")

    # Visual Pipeline Row
    st.subheader("🔍 Stage-by-Stage YOLOv8 Segmentation & YOLOv5 Defect Localization")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.image(result.original_rgb, caption="1. Input Guava Image", use_container_width=True)
    with c2:
        st.image(result.yolov8_mask, caption="2. YOLOv8-Seg Binary Mask", use_container_width=True)
    with c3:
        st.image(result.segmented_guava_rgb, caption="3. YOLOv8 Background Removed", use_container_width=True)
    with c4:
        st.image(
            result.annotated_yolov5_rgb,
            caption="4. YOLOv5 Defects (Red) vs. Ignored Anatomy (Cyan)",
            use_container_width=True,
        )

    # Pie Charts + Diagnostic Text
    st.subheader("📊 Surface Color & Morphological Shape Pie Charts")
    st.pyplot(result.chart_figure)

    st.subheader("🧬 Pathology Description & Anatomical Suppression Breakdown")
    st.markdown(result.diagnostic_markdown)
else:
    st.info("Please upload a guava image or select one of the built-in test samples from the sidebar.")
