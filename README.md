# Crowd Density Monitoring System

Real-time crowd density monitoring built with **OpenCV + YOLOv8 (Ultralytics)** person detector and **Streamlit** (dashboard UI). Designed to be opened and run from **PyCharm**.

## Features
- Detects people in images, uploaded videos, or a live webcam feed using OpenCV's HOG + SVM pedestrian detector
- Computes a density score (people per unit frame area) and classifies it as LOW / MEDIUM / HIGH
- Live dashboard: people count, density level, density score, and a running count chart
- Configurable alert threshold that flashes a warning when crowd count is exceeded
- Adjustable detection scale / sensitivity / frame-skip for performance tuning

## Project structure
```
crowd_density_monitor/
├── app.py              # Streamlit application (entry point)
├── requirements.txt    # Python dependencies
└── README.md
```

## Setup in PyCharm

1. **Open the project**: File → Open → select the `crowd_density_monitor` folder.
2. **Create a virtual environment** (PyCharm will usually prompt you):
   - File → Settings → Project → Python Interpreter → Add Interpreter → Virtualenv Environment
3. **Install dependencies** — open the PyCharm Terminal (bottom panel) and run:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the app** — since Streamlit apps aren't launched with the normal ▶ Run button, use the Terminal:
   ```bash
   streamlit run app.py
   ```
   This opens the dashboard at `http://localhost:8501` in your browser.

   (Optional) To run it via PyCharm's Run button instead: create a Run Configuration of type "Python", set the script to your venv's `streamlit` executable, with parameters `run app.py`.

## How to use
1. Pick an input source in the sidebar: **Image**, **Video file**, or **Webcam**.
2. Upload a file (or enable the webcam — webcam mode only works when Streamlit runs locally, not in a hosted/sandboxed browser).
3. Tune sliders:
   - **Detection scale** — lower values speed up detection on large frames.
   - **Detector sensitivity** — lower hit threshold detects more (possibly false) people.
   - **Density thresholds** — set where LOW/MEDIUM/HIGH boundaries fall, based on your camera's typical frame area and expected crowd sizes.
   - **Alert count** — triggers a red warning banner when exceeded.
4. Watch the live count, density level, score, and trend chart update.

## Notes & next steps
- The detector now uses **YOLOv8n** (Ultralytics), which handles overlapping people, partial bodies, and varied poses far better than the older HOG method. The first run will auto-download the small `yolov8n.pt` model weights (~6MB) — make sure you have an internet connection the first time you launch the app.
- For even denser crowds (hundreds of people, heavy overlap), consider a dedicated crowd-counting model like CSRNet, or a larger YOLOv8 variant (`yolov8s.pt` / `yolov8m.pt`) for higher accuracy at the cost of speed — just change the model name in `load_detector()`.
- For multi-camera deployments, you can refactor the source selector to loop over an RTSP URL list.
- All density thresholds are heuristic; calibrate `low_thr`/`high_thr` against your real camera footage.
