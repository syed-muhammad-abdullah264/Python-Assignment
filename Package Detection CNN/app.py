import os
from pathlib import Path
import time

import streamlit as st
from PIL import Image
import pandas as pd

# Optional dependency: install with `pip install ultralytics`
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

st.set_page_config(
    page_title="PackageVision AI | Package Detection",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --ink:#eaf0ff; --muted:#98a7c7; --panel:#111b31; --line:rgba(159,181,225,.16); --cyan:#66e3d2; --blue:#7798ff; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: radial-gradient(ellipse at 80% 0%, #1c3156 0%, #0b1222 38%, #080d18 100%); color:var(--ink); }
.block-container { padding-top:1.5rem; padding-bottom:3rem; max-width:1500px; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:#f4f7ff !important; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#101a30,#0b1221); border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:#eaf0ff; }
.hero { padding:30px 32px; border:1px solid var(--line); border-radius:24px; background:linear-gradient(120deg,rgba(31,52,88,.94),rgba(16,27,48,.83)); box-shadow:0 18px 55px rgba(0,0,0,.18); position:relative; overflow:hidden; }
.hero:after { content:'📦'; position:absolute; right:35px; top:14px; font-size:82px; opacity:.13; transform:rotate(-12deg); }
.eyebrow { color:var(--cyan); text-transform:uppercase; letter-spacing:2px; font-size:11px; font-weight:700; }
.hero h1 { font-size:clamp(30px,4vw,48px); line-height:1.08; margin:10px 0 12px; letter-spacing:-1.7px; }
.hero p { color:#b6c4e0; max-width:720px; font-size:15px; line-height:1.7; margin:0; }
.pill { display:inline-block; padding:6px 10px; border-radius:999px; border:1px solid rgba(102,227,210,.3); background:rgba(102,227,210,.08); color:#9df4e8; font-size:12px; margin:18px 6px 0 0; }
.metric { background:linear-gradient(145deg,rgba(25,39,67,.95),rgba(15,24,42,.95)); border:1px solid var(--line); border-radius:17px; padding:18px 20px; min-height:112px; }
.metric-label { color:#9eacc9; font-size:12px; font-weight:600; letter-spacing:.5px; text-transform:uppercase; }
.metric-value { color:#f5f7ff; font-family:'Space Grotesk',sans-serif; font-size:29px; font-weight:700; margin-top:7px; }
.metric-note { color:#72e2d1; font-size:11px; margin-top:2px; }
.section-head { display:flex; align-items:center; gap:10px; margin:28px 0 13px; font-weight:700; font-family:'Space Grotesk',sans-serif; font-size:20px; }
.section-head span { width:7px; height:24px; border-radius:5px; background:linear-gradient(180deg,var(--cyan),var(--blue)); display:inline-block; }
.panel { border:1px solid var(--line); border-radius:18px; padding:20px; background:rgba(16,26,46,.78); }
.small-muted { color:#9eacc9; font-size:13px; line-height:1.65; }
div.stButton > button, div.stDownloadButton > button { border-radius:12px; border:1px solid rgba(102,227,210,.32); background:linear-gradient(100deg,#3dd3c0,#6c94ff); color:#071321; font-weight:800; padding:.65rem 1.1rem; }
div.stButton > button:hover, div.stDownloadButton > button:hover { border-color:#a3fff4; color:#071321; filter:brightness(1.07); }
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] { background:rgba(22,35,59,.7); border-radius:10px; padding:10px 16px; color:#cbd7ef; }
.stTabs [aria-selected="true"] { background:rgba(102,227,210,.12) !important; color:#9df4e8 !important; }
[data-testid="stFileUploader"] { background:rgba(17,28,49,.72); border:1px dashed rgba(125,155,218,.4); border-radius:16px; padding:14px; }
hr { border-color:var(--line); }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

CLASSES = ["Box", "Box_broken", "Open_package", "Package"]

# Always load best.pt from the same root folder as this app.py file.
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "best.pt"

def find_default_weights():
    return MODEL_PATH if MODEL_PATH.is_file() else None

@st.cache_resource(show_spinner=False)
def load_model(model_source):
    if YOLO is None:
        raise ImportError("Ultralytics is not installed. Run: pip install ultralytics")
    return YOLO(model_source)

def metric_card(label, value, note=""):
    st.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown("## 📦 PackageVision")
    st.caption("AI-powered package condition inspection")
    st.markdown("---")
    st.markdown("### ⚙️ Detection settings")
    default_weights = find_default_weights()
    model_path = str(default_weights) if default_weights else None
    if model_path:
        st.success("Model ready: best.pt")
        st.caption("Loaded automatically from the project root folder.")
    else:
        st.error("best.pt was not found in the project root folder.")
        st.caption("Place best.pt in the same folder as app.py, then restart the app.")

    confidence = st.slider("Confidence threshold", 0.10, 0.95, 0.35, 0.05,
                           help="Higher confidence filters out weaker detections.")
    iou_threshold = st.slider("IoU threshold", 0.10, 0.90, 0.45, 0.05)
    image_size = st.select_slider("Inference image size", options=[320, 416, 512, 640, 768, 960], value=640)
    st.markdown("---")
    st.markdown("### 🧠 Model snapshot")
    st.markdown('<div class="small-muted">YOLO11n · 4 package classes · trained for 30 epochs</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted" style="margin-top:10px">Reported notebook metrics are reference results; your uploaded image predictions may vary.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">Computer vision · Smart inspection</div>
  <h1>Package Detection<br>Studio</h1>
  <p>Inspect parcel images with a trained YOLO model. Detect package conditions, review confidence scores, and export an annotated image for your workflow.</p>
  <span class="pill">✦ YOLO11n</span><span class="pill">4 detection classes</span><span class="pill">Instant inference</span>
</div>
""", unsafe_allow_html=True)

st.markdown("")
metric_cols = st.columns(4)
with metric_cols[0]: metric_card("mAP @ 0.5", "94.7%", "Reported notebook result")
with metric_cols[1]: metric_card("mAP @ 0.5:0.95", "85.1%", "Reported notebook result")
with metric_cols[2]: metric_card("Precision", "94.1%", "Reported notebook result")
with metric_cols[3]: metric_card("Recall", "89.3%", "Reported notebook result")

tab_detect, tab_classes, tab_about = st.tabs(["🔍  Detect packages", "🏷️  Class guide", "ℹ️  About project"])

with tab_detect:
    st.markdown('<div class="section-head"><span></span>Run a new inspection</div>', unsafe_allow_html=True)
    left, right = st.columns([0.92, 1.08], gap="large")
    with left:
        st.markdown('<div class="panel"><b>1. Upload an image</b><p class="small-muted">Choose a JPG, JPEG, PNG or WEBP image showing one or more packages.</p></div>', unsafe_allow_html=True)
        uploaded_image = st.file_uploader("Choose package image", type=["jpg", "jpeg", "png", "webp"], key="package_image")
        if uploaded_image:
            original = Image.open(uploaded_image).convert("RGB")
            st.image(original, caption="Input image", use_container_width=True)
        else:
            original = None
            st.markdown('<div class="small-muted" style="padding:22px 4px">Your uploaded image will appear here before detection.</div>', unsafe_allow_html=True)
        run_detection = st.button("✨  Analyze image", type="primary", use_container_width=True, disabled=(original is None))
    with right:
        st.markdown('<div class="panel"><b>2. Detection results</b><p class="small-muted">Bounding boxes and class labels will be shown here after analysis.</p></div>', unsafe_allow_html=True)
        if run_detection and original is not None:
            if not model_path or not Path(str(model_path)).is_file():
                st.error("best.pt was not found in the project root folder. Put best.pt beside app.py and restart Streamlit.")
            elif YOLO is None:
                st.error("Ultralytics is missing. Install it using `pip install ultralytics`.")
            else:
                try:
                    with st.spinner("Loading model and inspecting image…"):
                        model = load_model(str(model_path))
                        start_time = time.time()
                        results = model.predict(
                            source=original,
                            conf=confidence,
                            iou=iou_threshold,
                            imgsz=image_size,
                            verbose=False,
                        )
                        elapsed = time.time() - start_time
                        result = results[0]
                        plotted = result.plot()
                        # Ultralytics returns a BGR numpy array; convert for Streamlit.
                        import cv2
                        annotated = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
                    st.image(annotated, caption="Annotated detection output", use_container_width=True)
                    boxes = result.boxes
                    count = len(boxes) if boxes is not None else 0
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Objects detected", count)
                    with c2:
                        st.metric("Inference time", f"{elapsed:.2f}s")
                    if count:
                        names = result.names
                        rows = []
                        for box in boxes:
                            cls_id = int(box.cls[0].item())
                            score = float(box.conf[0].item())
                            xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
                            rows.append({
                                "Detected class": names.get(cls_id, str(cls_id)),
                                "Confidence": f"{score * 100:.1f}%",
                                "X1": xyxy[0], "Y1": xyxy[1], "X2": xyxy[2], "Y2": xyxy[3],
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    else:
                        st.info("No objects passed the selected confidence threshold. Try lowering the confidence slider.")
                    from io import BytesIO
                    output = Image.fromarray(annotated)
                    buffer = BytesIO()
                    output.save(buffer, format="PNG")
                    st.download_button(
                        "⬇️ Download annotated image",
                        data=buffer.getvalue(),
                        file_name="package_detection_result.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    st.caption("Check that best.pt is a compatible Ultralytics YOLO model and that all dependencies are installed.")
        elif not uploaded_image:
            st.markdown('<div class="small-muted" style="padding:22px 4px">Waiting for an image. Upload one and press Analyze image.</div>', unsafe_allow_html=True)

with tab_classes:
    st.markdown('<div class="section-head"><span></span>What the model can identify</div>', unsafe_allow_html=True)
    class_cols = st.columns(4)
    class_info = [
        ("📦", "Box", "A package or box in its regular state."),
        ("⚠️", "Box_broken", "A box showing visible damage or breakage."),
        ("📬", "Open_package", "A package that appears open."),
        ("🧰", "Package", "A package identified by the trained dataset."),
    ]
    for col, (emoji, title, desc) in zip(class_cols, class_info):
        with col:
            st.markdown(f'<div class="panel" style="height:185px"><div style="font-size:30px">{emoji}</div><h3 style="font-size:18px;margin:12px 0 8px">{title}</h3><div class="small-muted">{desc}</div></div>', unsafe_allow_html=True)
    st.info("Class labels and detections depend on the examples used during training. Treat results as decision support and verify important or ambiguous cases manually.")

with tab_about:
    st.markdown('<div class="section-head"><span></span>Project overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel">
      <h3>Package Detection using YOLO11n</h3>
      <p class="small-muted">This interface is based on the supplied project notebook, which describes fine-tuning a YOLO11n model on a custom package dataset for 30 epochs. The original notebook reports four classes: Box, Box_broken, Open_package and Package.</p>
      <hr>
      <b>Workflow</b>
      <p class="small-muted">Upload an image → choose detection thresholds → run inference → inspect labels and confidence → download the annotated image.</p>
      <b>Important</b>
      <p class="small-muted">The reported evaluation numbers are displayed as project-reported metrics, not recalculated by this interface. Live predictions require the trained `best.pt` weights file.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div style="text-align:center;color:#7f90b2;font-size:12px">PACKAGEVISION AI · Streamlit interface for package condition detection</div>', unsafe_allow_html=True)
