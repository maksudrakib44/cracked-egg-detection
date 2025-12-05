"""
Egg vs Cracked Egg Detector – Streamlit App (Image / Video / Webcam)
------------------------------------------------------------------

Features added / ensured for assignment:
 - Input Source selection: Webcam capture (camera snapshot) OR Image upload OR Video upload
 - Confidence slider, Max detections slider
 - Clean, commented, and organized code
 - Counters (per-class) + summary charts (image & video)
 - Annotated image / video download
 - FPS display for webcam snapshots
 - Model info in sidebar and ability to override model path
 - Safe error handling & helpful messages
 - Caching of model load for performance

Run with:

    streamlit run app.py
"""

import os
import io
import time
import tempfile
import sys
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
from ultralytics import YOLO

# ========= STREAMLIT CLOUD DETECTION =========
# Detect if we're on Streamlit Cloud
if 'STREAMLIT_SHARING' in os.environ or 'STREAMLIT_DEPLOY' in os.environ:
    IS_CLOUD = True
    VIDEO_RUNS_DIR = "/tmp/runs_video"  # Use /tmp for cloud
else:
    IS_CLOUD = False
    VIDEO_RUNS_DIR = "runs_video"  # Use local folder for local

# ========= CONFIG =========
DEFAULT_MODEL_PATH = "egg_eggDetector_best.pt"   
PAGE_TITLE = "🥚 Egg vs Cracked Egg Detection"
STANDARD_SIZE = (640, 640)                       # standard size used for inference display (width, height)

# ========= STYLES =========
CUSTOM_CSS = """
<style>
    /* Constrain main area width */
    .main > div {
        max-width: 1100px;
        margin: 0 auto;
    }

    /* Header card */
    .app-header {
        padding: 1.2rem 1.5rem;
        border-radius: 0.75rem;
        background: linear-gradient(90deg, #f7d9ff, #ffe8d6);
        border: 1px solid #f0c6ff;
        margin-bottom: 1.2rem;
    }
    .app-header h1 { margin: 0; font-size: 1.8rem; }
    .app-header p { margin: 0.2rem 0 0; font-size: 0.95rem; color: #444; }

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
    """Load the YOLO model with cloud compatibility."""
    try:
        # Debug info
        st.sidebar.write(f"🔍 Looking for model at: {os.path.abspath(path)}")
        
        # Check if file exists
        if not os.path.exists(path):
            # Try alternative paths (for cloud deployment)
            alt_paths = [
                path,  # Original path
                f"./{path}",  # Current directory
                f"cracked-egg-detection/{path}",  # GitHub folder structure
                f"/app/cracked-egg-detection/{path}"  # Streamlit Cloud path
            ]
            
            found = False
            for alt in alt_paths:
                if os.path.exists(alt):
                    path = alt
                    found = True
                    st.sidebar.success(f"✅ Found model at: {path}")
                    break
            
            if not found:
                st.error(f"❌ Model not found. Checked locations: {alt_paths}")
                # Show directory for debugging
                st.sidebar.write("📁 Current directory files:")
                for f in os.listdir('.'):
                    st.sidebar.write(f"   - {f}")
                return None
        
        # Show loading progress
        file_size = os.path.getsize(path) // 1024 // 1024  # Convert to MB
        with st.spinner(f"📦 Loading model ({file_size}MB)... This may take a moment"):
            model = YOLO(path)
        
        st.sidebar.success(f"✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"❌ Model load error: {str(e)}")
        
        # Fallback option for demo
        st.info("🔄 Trying pretrained YOLOv8n as fallback...")
        try:
            model = YOLO('yolov8n.pt')
            st.success("✅ Using pretrained YOLOv8n for demo")
            return model
        except:
            st.error("❌ Could not load any model")
            return None


def resize_to_standard(img: Image.Image) -> Image.Image:
    """Resize input image to STANDARD_SIZE (maintains aspect by forcing resize for simplicity)."""
    return img.resize(STANDARD_SIZE)


def run_inference_image(model: YOLO, image: Image.Image, conf: float, max_det: int):
    """
    Run YOLO on a PIL image and return:
      - annotated_rgb: annotated image as an RGB numpy array
      - results_obj: the YOLO result object for further analysis
    Uses a temporary file since ultralytics' predict expects a path or array.
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
    # cleanup
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    r = results[0]
    annotated_bgr = r.plot()               # returns BGR numpy array
    annotated_rgb = annotated_bgr[:, :, ::-1]  # BGR -> RGB
    return annotated_rgb, r


def summarize_detections(result, names_dict):
    """
    Given a single YOLO result, return:
      - counts: Counter mapping label -> count
      - df: DataFrame with per-box rows (label, confidence, bbox coords)
    """
    if not hasattr(result, "boxes") or len(result.boxes) == 0:
        return Counter(), pd.DataFrame()

    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    xyxy = result.boxes.xyxy.cpu().numpy()

    records = []
    for cid, conf, box in zip(cls_ids, confs, xyxy):
        label = names_dict.get(int(cid), f"class_{cid}")
        x1, y1, x2, y2 = box.tolist()
        records.append({
            "class_id": int(cid),
            "label": label,
            "confidence": float(conf),
            "x1": round(x1, 1),
            "y1": round(y1, 1),
            "x2": round(x2, 1),
            "y2": round(y2, 1),
        })

    df = pd.DataFrame.from_records(records)
    counts = Counter(df["label"].tolist())
    return counts, df


def image_to_download_bytes(rgb_array):
    """Convert RGB numpy array to PNG bytes for download buttons."""
    pil_img = Image.fromarray(rgb_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def run_inference_video(model: YOLO, video_bytes: bytes, conf: float, max_det: int):
    """
    Run YOLO on an uploaded video with cloud compatibility.
    Returns:
      - annotated_path (str) : path to saved annotated video (or None)
      - total_counts (Counter) : aggregated counts across all frames
    """
    # CLOUD COMPATIBILITY - Use appropriate directory
    if IS_CLOUD:
        # On Streamlit Cloud, use temp directory
        video_runs_dir = tempfile.mkdtemp()
    else:
        # Local development
        video_runs_dir = VIDEO_RUNS_DIR
    
    # Ensure directory exists
    os.makedirs(video_runs_dir, exist_ok=True)
    
    # Save source video to a temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=video_runs_dir) as tmp:
        tmp.write(video_bytes)
        video_path = tmp.name

    try:
        # Run prediction, set save=True so ULTRALYTICS saves annotated outputs
        results = model.predict(
            source=video_path,
            conf=conf,
            max_det=max_det,
            save=True,
            project=video_runs_dir,  # Use the dynamic directory
            name="pred",
            exist_ok=True,
            verbose=False,
        )

        # Locate annotated video file created by YOLO
        out_dir = os.path.join(video_runs_dir, "pred")
        annotated_path = None
        if os.path.isdir(out_dir):
            vids = [f for f in os.listdir(out_dir) if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov"))]
            if vids:
                # pick first annotated video
                annotated_path = os.path.join(out_dir, vids[0])

        # Aggregate frame-by-frame detection counts
        names_dict = model.names
        total_counts = Counter()
        for r in results:
            frame_counts, _ = summarize_detections(r, names_dict)
            total_counts.update(frame_counts)

        return annotated_path, total_counts
        
    except Exception as e:
        st.error(f"Video processing error: {str(e)}")
        return None, Counter()
    finally:
        # Cleanup temp files
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except:
            pass


def check_environment():
    """Check environment and show debug info."""
    with st.expander("🛠️ Environment Info (Debug)"):
        st.write(f"**Python Version:** {sys.version}")
        st.write(f"**Current Directory:** {os.getcwd()}")
        st.write(f"**Is Cloud Deployment:** {IS_CLOUD}")
        st.write(f"**Model Path:** {DEFAULT_MODEL_PATH}")
        st.write(f"**Model Exists:** {os.path.exists(DEFAULT_MODEL_PATH)}")
        
        if os.path.exists(DEFAULT_MODEL_PATH):
            size_mb = os.path.getsize(DEFAULT_MODEL_PATH) // 1024 // 1024
            st.write(f"**Model Size:** {size_mb}MB")
        
        st.write("**Files in current directory:**")
        for file in os.listdir('.'):
            st.write(f"- {file}")


def render_footer():
    st.markdown(
        '<div class="footer">Copyright reserved:&nbsp;&nbsp; Md.Maksudul Haque</div>',
        unsafe_allow_html=True,
    )


# ========== STREAMLIT APP LAYOUT ==========

st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Sidebar: model path override, settings, metadata ---
st.sidebar.header("⚙️ App Settings & Model")
model_path_input = st.sidebar.text_input("Model path (relative to app.py)", DEFAULT_MODEL_PATH)
st.sidebar.caption("Put your trained weights next to app.py or provide a path here.")

# Confidence & max detections - assignment required controls
conf_threshold = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.50, 0.01,
                                   help="Minimum confidence to show detection")
max_detections = st.sidebar.slider("Maximum detections per frame", 1, 200, 50, 1,
                                   help="Upper cap on number of boxes per frame")

st.sidebar.markdown("---")
st.sidebar.subheader("Model info & metadata")
st.sidebar.info("Model will be loaded from the path above (cached).")
user_map = st.sidebar.text_input("Optional: mAP / Notes (paste training mAP here)", "")

# Debug button
if st.sidebar.button("🛠️ Show Debug Info"):
    check_environment()

st.sidebar.markdown("##")

# ---------- Load model (safe) ----------
with st.spinner("Loading YOLO model..."):
    try:
        model = load_model(model_path_input)
        if model is None:
            st.error("""
            ### ⚠️ Model could not be loaded!
            
            **Possible solutions:**
            1. Make sure `egg_eggDetector_best.pt` in  GitHub repository
            2. Check the file size (should be <200MB for Streamlit Cloud)
            3. Verify the filename is correct
            4. For local testing, place the model file in same folder as app.py
            """)
            st.stop()
        
        names = model.names  # id -> label mapping
        st.sidebar.success("Model loaded")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        st.stop()  # stop app until model is available

# Show detected class names & allow remapping if desired
st.sidebar.write("**Detected classes**")
for cid, label in names.items():
    st.sidebar.write(f"- {cid}: `{label}`")

# ---------- Header ----------
st.markdown(
    """
<div class="app-header">
  <h1>🥚 Egg vs Cracked Egg Detection</h1>
  <p>Detect <b>cracked eggs</b> vs <b>normal eggs</b>. Use Webcam / Upload Image / Upload Video. Adjust confidence and view counts.</p>
</div>
""", unsafe_allow_html=True
)

# ---------- Input Source (Image / Video / Webcam) ----------
st.subheader("Input Source")
source = st.radio("Choose input source", ["Webcam (capture)", "Image upload", "Video upload"], index=0)

# Provide placeholders for outputs
output_col1, output_col2 = st.columns([0.6, 0.4])

# ---------- WEBCAM (capture mode) ----------
if source == "Webcam (capture)":
    st.info("Use your webcam to take a snapshot. Click 'Capture' to grab a frame for detection.")
    cam_file = st.camera_input("Take a picture (camera)")

    if cam_file is not None:
        # Read image from camera input
        orig_img = Image.open(cam_file).convert("RGB")
        img = resize_to_standard(orig_img)

        # Display input and run inference
        with output_col1:
            st.markdown("**Webcam snapshot (resized to 640×640)**")
            st.image(img, width=640)

        start_time = time.time()
        with st.spinner("Running detection on webcam snapshot..."):
            annotated_rgb, result = run_inference_image(model=model, image=img,
                                                        conf=conf_threshold, max_det=max_detections)
            counts, df_dets = summarize_detections(result, names)
        end_time = time.time()
        elapsed = end_time - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        with output_col2:
            st.markdown("**Detection result**")
            st.image(annotated_rgb, caption="Annotated snapshot (640×640)", width=640)
            st.download_button("⬇️ Download annotated image (PNG)",
                               data=image_to_download_bytes(annotated_rgb),
                               file_name="annotated_webcam.png",
                               mime="image/png")
            st.metric("FPS (snapshot)", f"{fps:.2f}")

        # Summary metrics
        cracked_count = counts.get("cracked_egg", 0) + counts.get("cracked", 0)
        whole_count = counts.get("whole_egg", 0) + counts.get("whole", 0)
        total_count = sum(counts.values())

        st.markdown("### Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Cracked eggs", cracked_count)
        c2.metric("Whole eggs", whole_count)
        c3.metric("Total detections", total_count)

        # Chart & detailed table
        if counts:
            chart_df = pd.DataFrame({"class": list(counts.keys()), "count": list(counts.values())}).set_index("class")
            st.bar_chart(chart_df)
        if not df_dets.empty:
            df_view = df_dets.copy()
            df_view["confidence"] = df_view["confidence"].map(lambda x: f"{x:.2f}")
            st.markdown("#### Detailed detections")
            st.dataframe(df_view, use_container_width=True)
    else:
        st.info("No webcam snapshot yet. Use the camera widget above to take a picture.")

# ---------- IMAGE UPLOAD ----------
elif source == "Image upload":
    st.info("Upload one or more images (JPG/PNG). Each will be processed and results shown.")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

    if uploaded_file is None:
        st.info("👆 Upload an image to begin detection.")
    else:
        orig_img = Image.open(uploaded_file).convert("RGB")
        img = resize_to_standard(orig_img)

        # Display side-by-side
        col_img, col_res = st.columns(2)
        with col_img:
            st.markdown("**Input image (resized to 640×640)**")
            st.image(img, width=640)

        with st.spinner("Running egg detector on image..."):
            annotated_rgb, result = run_inference_image(model=model, image=img,
                                                        conf=conf_threshold, max_det=max_detections)
            counts, df_dets = summarize_detections(result, names)

        with col_res:
            st.markdown("**Detection result**")
            st.image(annotated_rgb, caption="Annotated image (640×640)", width=640)
            st.download_button("⬇️ Download annotated image (PNG)",
                               data=image_to_download_bytes(annotated_rgb),
                               file_name=f"annotated_{os.path.splitext(uploaded_file.name)[0]}.png",
                               mime="image/png")

        # Summaries
        cracked_count = counts.get("cracked_egg", 0) + counts.get("cracked", 0)
        whole_count = counts.get("whole_egg", 0) + counts.get("whole", 0)
        total_count = sum(counts.values())

        st.markdown("### Summary")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Cracked eggs", cracked_count)
        mc2.metric("Whole eggs", whole_count)
        mc3.metric("Total detections", total_count)

        if counts:
            chart_df = pd.DataFrame({"class": list(counts.keys()), "count": list(counts.values())}).set_index("class")
            st.bar_chart(chart_df)
        else:
            st.info("No detections at chosen confidence threshold.")

        st.markdown("### Detailed detections")
        if df_dets.empty:
            st.write("No boxes to display.")
        else:
            df_view = df_dets.copy()
            df_view["confidence"] = df_view["confidence"].map(lambda x: f"{x:.2f}")
            st.dataframe(df_view, use_container_width=True)

# ---------- VIDEO UPLOAD ----------
elif source == "Video upload":
    st.info("Upload a short video (mp4 / avi / mov / mkv). The app will process and return an annotated video.")
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if video_file is None:
        st.info("👆 Upload a video to start.")
    else:
        # Preview original video
        st.markdown("**Original video preview**")
        st.video(video_file)

        # Run inference on the uploaded video (this may take some time depending on CPU/GPU)
        with st.spinner("Running egg detector on video (this may take a while)..."):
            annotated_path, total_counts = run_inference_video(model=model,
                                                              video_bytes=video_file.read(),
                                                              conf=conf_threshold,
                                                              max_det=max_detections)

        st.markdown("### Detection result (video)")

        if annotated_path is None or not os.path.exists(annotated_path):
            st.error("Annotated video file not found. Processing may have failed or video format is unsupported.")
        else:
            st.write("**Annotated video**")
            with open(annotated_path, "rb") as f:
                st.video(f.read())

            # Download annotated video
            with open(annotated_path, "rb") as f:
                st.download_button("⬇️ Download annotated video", data=f,
                                   file_name=os.path.basename(annotated_path), mime="video/mp4")

        cracked_count = total_counts.get("cracked_egg", 0) + total_counts.get("cracked", 0)
        whole_count = total_counts.get("whole_egg", 0) + total_counts.get("whole", 0)
        total_count = sum(total_counts.values())

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Cracked eggs (all frames)", cracked_count)
        mc2.metric("Whole eggs (all frames)", whole_count)
        mc3.metric("Total detections (all frames)", total_count)

        if total_counts:
            chart_df = pd.DataFrame({"class": list(total_counts.keys()), "count": list(total_counts.values())}).set_index("class")
            st.bar_chart(chart_df)
        else:
            st.info("No detections in this video at the chosen confidence threshold.")

# ---------- Footer and supplementary information ----------
st.markdown("---")
st.markdown("### Model & Training Notes")
st.write("Model path:", f"`{model_path_input}`")
if user_map:
    st.write("Training metric (user-provided):", user_map)
else:
    st.write("Training metric (mAP): not provided — you can paste the mAP value in the sidebar 'mAP / Notes' field.")

st.markdown("### Helpful tips for video demos")
st.markdown("""
- For best accuracy, show the egg near the camera and move it slowly.
- Use a plain background for clearer detection during the demo.
- When showing the webcam demo, take multiple snapshots to show consistent detection.
""")

render_footer()