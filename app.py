"""
Egg vs Cracked Egg Detector – Streamlit App (Image + Video)
-----------------------------------------------------------

Run with:
    conda activate crack_egg
    cd D:\ML\crack_egg_detector
    streamlit run app.py
"""

import os
import io
import tempfile
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
from ultralytics import YOLO


# ========= CONFIG =========
MODEL_PATH = "egg_eggDetector_best.pt"   # trained model
VIDEO_RUNS_DIR = "runs_video"            # where annotated videos will be saved
PAGE_TITLE = "Egg vs Cracked Egg Detection"
STANDARD_SIZE = (640, 640)               # width, height for image processing


# ========= STYLES =========
CUSTOM_CSS = """
<style>
    /* Main page width */
    .main > div {
        max-width: 1100px;
        margin: 0 auto;
    }

    /* Top header card */
    .app-header {
        padding: 1.2rem 1.5rem;
        border-radius: 0.75rem;
        background: linear-gradient(90deg, #f7d9ff, #ffe8d6);
        border: 1px solid #f0c6ff;
        margin-bottom: 1.2rem;
    }
    .app-header h1 {
        margin: 0;
        font-size: 1.8rem;
    }
    .app-header p {
        margin: 0.2rem 0 0;
        font-size: 0.95rem;
        color: #444;
    }

    /* Footer */
    .footer {
        margin-top: 3rem;
        padding-top: 0.5rem;
        border-top: 1px solid #e0e0e0;
        text-align: center;
        font-size: 0.85rem;
        color: #666;
    }
</style>
"""


# ========= HELPERS =========
@st.cache_resource
def load_model(path: str):
    """Load YOLO model once and cache it."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Put your trained model (egg_eggDetector_best.pt) next to app.py."
        )
    return YOLO(path)


def resize_to_standard(img: Image.Image) -> Image.Image:
    """Resize input image to STANDARD_SIZE (640x640)."""
    return img.resize(STANDARD_SIZE)


def run_inference_image(model: YOLO, image: Image.Image, conf: float, max_det: int):
    """
    Run YOLO on a PIL image (already resized to STANDARD_SIZE) and return:
    - annotated RGB numpy array
    - result object (ultralytics.engine.results.Results)
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        image.save(tmp_path)

    results = model.predict(
        source=tmp_path,
        conf=conf,
        max_det=max_det,
        verbose=False,
    )
    os.remove(tmp_path)

    r = results[0]
    annotated_bgr = r.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]  # BGR -> RGB
    return annotated_rgb, r


def summarize_detections(result, names_dict):
    """Return (Counter, DataFrame) for one YOLO result."""
    if len(result.boxes) == 0:
        return Counter(), pd.DataFrame()

    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    xyxy = result.boxes.xyxy.cpu().numpy()

    records = []
    for cid, conf, box in zip(cls_ids, confs, xyxy):
        label = names_dict.get(int(cid), f"class_{cid}")
        x1, y1, x2, y2 = box.tolist()
        records.append(
            {
                "class_id": int(cid),
                "label": label,
                "confidence": float(conf),
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
            }
        )

    df = pd.DataFrame.from_records(records)
    counts = Counter(df["label"].tolist())
    return counts, df


def image_to_download_bytes(rgb_array):
    """Convert RGB numpy array to PNG bytes for download."""
    pil_img = Image.fromarray(rgb_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def run_inference_video(model: YOLO, video_bytes: bytes, conf: float, max_det: int):
    """
    Run YOLO on an uploaded video.

    Returns:
        annotated_video_path (str): path to saved annotated video
        total_counts (Counter): aggregated per-class counts over all frames
    """
    # Save uploaded video to a temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        video_path = tmp.name

    # Run prediction and save annotated video
    results = model.predict(
        source=video_path,
        conf=conf,
        max_det=max_det,
        save=True,                       # save annotated video
        project=VIDEO_RUNS_DIR,
        name="pred",                     # VIDEO_RUNS_DIR/pred/
        exist_ok=True,
        verbose=False,
    )

    # YOLO will create VIDEO_RUNS_DIR/pred/... with annotated video
    out_dir = os.path.join(VIDEO_RUNS_DIR, "pred")
    annotated_path = None
    if os.path.isdir(out_dir):
        vids = [
            f for f in os.listdir(out_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
        if vids:
            annotated_path = os.path.join(out_dir, vids[0])

    # Aggregate counts over all frames
    names_dict = model.names
    total_counts = Counter()
    for r in results:
        frame_counts, _ = summarize_detections(r, names_dict)
        total_counts.update(frame_counts)

    # Clean up temp source video (keep annotated)
    os.remove(video_path)

    return annotated_path, total_counts


def render_footer():
    st.markdown(
        '<div class="footer">Copyright reserved:&nbsp;&nbsp; Md.Maksudul Haque</div>',
        unsafe_allow_html=True,
    )


# ========= STREAMLIT LAYOUT =========
st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Load YOLO model once ---
with st.spinner("Loading YOLO model..."):
    model = load_model(MODEL_PATH)
names = model.names  # id -> class name

# ---------- HEADER ----------
st.markdown(
    """
<div class="app-header">
  <h1>🥚 Egg vs Cracked Egg Detection</h1>
  <p>Upload an image or video of eggs. The model will detect <b>cracked eggs</b> and <b>whole eggs</b>, draw bounding boxes, and summarize counts.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Detection Settings")

conf_threshold = st.sidebar.slider(
    "Confidence threshold",
    0.05,
    0.9,
    0.50,
    0.05,
    help="Minimum confidence for a detection.",
)

max_detections = st.sidebar.slider(
    "Maximum detections per frame",
    5,
    100,
    50,
    5,
    help="Upper limit on number of boxes per frame.",
)

show_table = st.sidebar.checkbox("Show detailed table (images)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Model info")
st.sidebar.write(f"**Weights:** `{MODEL_PATH}`")
st.sidebar.write("**Classes:**")
for cid, label in names.items():
    st.sidebar.write(f"- {cid}: `{label}`")

# ---------- MAIN TABS ----------
tab_image, tab_video = st.tabs(["🖼 Image Detection", "🎬 Video Detection"])

# ========== IMAGE TAB ==========
with tab_image:
    st.subheader("Image detection")

    uploaded_file = st.file_uploader(
        "Upload a JPG/PNG image",
        type=["jpg", "jpeg", "png"],
        help="The image will be resized to 640×640 for detection.",
    )

    if uploaded_file is None:
        st.info("👆 Upload an image to start.")
    else:
        # Load & resize
        orig_img = Image.open(uploaded_file).convert("RGB")
        img = resize_to_standard(orig_img)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Input image (resized to 640×640)**")
            st.image(img, width=640)

        with st.spinner("Running egg detector on image..."):
            annotated_rgb, result = run_inference_image(
                model=model,
                image=img,
                conf=conf_threshold,
                max_det=max_detections,
            )
            counts, df_dets = summarize_detections(result, names)

        with col2:
            st.markdown("**Detection result**")
            st.image(
                annotated_rgb,
                caption="Annotated image (640×640)",
                width=640,
            )
            download_buf = image_to_download_bytes(annotated_rgb)
            st.download_button(
                label="⬇️ Download annotated image (PNG)",
                data=download_buf,
                file_name=f"annotated_{os.path.splitext(uploaded_file.name)[0]}.png",
                mime="image/png",
            )

        st.markdown("### Summary")

        cracked_count = counts.get("cracked_egg", 0)
        whole_count = counts.get("whole_egg", 0)
        total_count = sum(counts.values())

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Cracked eggs", cracked_count)
        mc2.metric("Whole eggs", whole_count)
        mc3.metric("Total detections", total_count)

        if counts:
            chart_df = pd.DataFrame(
                {"class": list(counts.keys()), "count": list(counts.values())}
            ).set_index("class")
            st.bar_chart(chart_df)
        else:
            st.info("No detections at this confidence threshold.")

        if show_table:
            st.markdown("### Detailed detections")
            if df_dets.empty:
                st.write("No boxes to display.")
            else:
                df_view = df_dets.copy()
                df_view["confidence"] = df_view["confidence"].map(lambda x: f"{x:.2f}")
                st.dataframe(df_view, use_container_width=True)

# ========== VIDEO TAB ==========
with tab_video:
    st.subheader("Video detection")

    video_file = st.file_uploader(
        "Upload a short video (mp4 / avi / mov / mkv)",
        type=["mp4", "avi", "mov", "mkv"],
        help="Processing long or high-resolution videos may be slow on CPU.",
    )

    if video_file is None:
        st.info("👆 Upload a video to start.")
    else:
        st.markdown("**Original video preview**")
        st.video(video_file)

        with st.spinner("Running egg detector on video (this may take a while)..."):
            annotated_path, total_counts = run_inference_video(
                model=model,
                video_bytes=video_file.read(),
                conf=conf_threshold,
                max_det=max_detections,
            )

        st.markdown("### Detection result (video)")

        if annotated_path is None or not os.path.exists(annotated_path):
            st.error("Annotated video file not found.")
        else:
            st.write("**Annotated video**")
            with open(annotated_path, "rb") as f:
                st.video(f.read())

            with open(annotated_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download annotated video",
                    data=f,
                    file_name=os.path.basename(annotated_path),
                    mime="video/mp4",
                )

        cracked_count = total_counts.get("cracked_egg", 0)
        whole_count = total_counts.get("whole_egg", 0)
        total_count = sum(total_counts.values())

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Cracked eggs (all frames)", cracked_count)
        mc2.metric("Whole eggs (all frames)", whole_count)
        mc3.metric("Total detections (all frames)", total_count)

        if total_counts:
            chart_df = pd.DataFrame(
                {"class": list(total_counts.keys()), "count": list(total_counts.values())}
            ).set_index("class")
            st.bar_chart(chart_df)
        else:
            st.info("No detections in this video at the chosen confidence threshold.")

# ---------- FOOTER ----------
render_footer()
