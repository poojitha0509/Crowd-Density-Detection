
"""
Crowd Density Monitoring System
Tech: OpenCV (HOG person detector) + Streamlit
Run with:  streamlit run app.py
"""

import time
import cv2
import numpy as np
import streamlit as st
import pandas as pd
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="Crowd Density Monitor", layout="wide")
st.title("👥 Crowd Density Monitoring System")
st.caption("YOLOv8 (Ultralytics) person detector · Streamlit dashboard")

# ----------------------------------------------------------------------
# Cached detector
# ----------------------------------------------------------------------
@st.cache_resource
def load_detector():
    # yolov8n.pt = small/fast "nano" model, auto-downloads on first run
    model = YOLO("yolov8n.pt")
    return model

model = load_detector()
PERSON_CLASS_ID = 0  # COCO class id for "person"


def detect_people(frame, scale=0.6, conf_threshold=0.3):
    """Run YOLOv8 on a frame, return boxes + annotated frame.
    'scale' resizes the frame before inference (lower = faster, less accurate).
    'conf_threshold' is the minimum detection confidence (0-1)."""
    small = cv2.resize(frame, None, fx=scale, fy=scale)

    results = model(small, classes=[PERSON_CLASS_ID], conf=conf_threshold, verbose=False)[0]

    boxes = []
    annotated = frame.copy()
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        # scale coordinates back up to original frame size
        x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        boxes.append([x, y, w, h])
        conf = float(box.conf[0])
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, f"{conf:.2f}", (x, max(y - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return np.array(boxes), annotated


def density_level(count, frame_area, low_thr, high_thr):
    """Classify density based on people count per unit area (people / 100k px)."""
    density_score = count / (frame_area / 100000.0)
    if density_score < low_thr:
        level, color = "LOW", "🟢"
    elif density_score < high_thr:
        level, color = "MEDIUM", "🟡"
    else:
        level, color = "HIGH", "🔴"
    return level, color, density_score


# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.header("Settings")
source_type = st.sidebar.radio("Input source", ["Image", "Video file", "Webcam"])
scale = st.sidebar.slider("Detection scale (lower = faster)", 0.3, 1.0, 1.0, 0.05)
conf_thr = st.sidebar.slider("Detection confidence threshold", 0.1, 0.9, 0.3, 0.05)
low_thr = st.sidebar.number_input("Low/Medium density threshold", value=2.0, step=0.5)
high_thr = st.sidebar.number_input("Medium/High density threshold", value=5.0, step=0.5)
alert_count = st.sidebar.number_input("Alert if people count exceeds", value=10, step=1)
frame_skip = st.sidebar.slider("Process every Nth frame (video/webcam)", 1, 10, 3)

# Session log
if "log" not in st.session_state:
    st.session_state.log = []

placeholder_frame = st.empty()
col1, col2, col3 = st.columns(3)
metric_count = col1.empty()
metric_level = col2.empty()
metric_score = col3.empty()
alert_box = st.empty()
chart_box = st.empty()


def update_dashboard(count, frame_area):
    level, color, score = density_level(count, frame_area, low_thr, high_thr)
    metric_count.metric("People Detected", count)
    metric_level.metric("Density Level", f"{color} {level}")
    metric_score.metric("Density Score", f"{score:.2f}")

    if count > alert_count:
        alert_box.error(f"⚠️ ALERT: Crowd count ({count}) exceeds threshold ({alert_count})!")
    else:
        alert_box.empty()

    st.session_state.log.append({"time": time.strftime("%H:%M:%S"), "count": count, "level": level})
    if len(st.session_state.log) > 200:
        st.session_state.log.pop(0)

    df = pd.DataFrame(st.session_state.log)
    if not df.empty:
        chart_box.line_chart(df.set_index("time")["count"])


# ----------------------------------------------------------------------
# IMAGE MODE
# ----------------------------------------------------------------------
if source_type == "Image":
    uploaded = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        boxes, annotated = detect_people(frame, scale, conf_thr)
        placeholder_frame.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                 caption="Detected people", use_container_width=True)
        update_dashboard(len(boxes), frame.shape[0] * frame.shape[1])
    else:
        st.info("Upload an image from the sidebar to begin.")

# ----------------------------------------------------------------------
# VIDEO FILE MODE
# ----------------------------------------------------------------------
elif source_type == "Video file":
    uploaded = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    run = st.sidebar.button("Start processing")
    if uploaded is not None and run:
        tmp_path = f"/tmp/{uploaded.name}"
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())

        cap = cv2.VideoCapture(tmp_path)
        stop_btn = st.sidebar.button("Stop")
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue
            boxes, annotated = detect_people(frame, scale, conf_thr)
            placeholder_frame.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                     caption=f"Frame {frame_idx}", use_container_width=True)
            update_dashboard(len(boxes), frame.shape[0] * frame.shape[1])
            if stop_btn:
                break
        cap.release()
    else:
        st.info("Upload a video and click 'Start processing'.")

# ----------------------------------------------------------------------
# WEBCAM MODE (runs locally in PyCharm / local Streamlit server only)
# ----------------------------------------------------------------------
elif source_type == "Webcam":
    st.warning("Webcam mode requires running Streamlit locally (not in a sandboxed browser).")
    run = st.sidebar.checkbox("Start webcam")
    if run:
        cap = cv2.VideoCapture(0)
        frame_idx = 0
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Could not access webcam.")
                break
            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue
            boxes, annotated = detect_people(frame, scale, conf_thr)
            placeholder_frame.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                     caption="Live feed", use_container_width=True)
            update_dashboard(len(boxes), frame.shape[0] * frame.shape[1])
            run = st.sidebar.checkbox("Start webcam", value=True, key=f"chk_{frame_idx}")
        cap.release()

# ----------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Density score = people count / (frame area / 100,000 px)")
