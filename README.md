# ARUGA: AI-Assisted Recognition and Understanding for Guided Contextual Risk Assessment

A real-time computer vision system for fall detection and prolonged inactivity monitoring. Designed for prototyping and demonstrations **without requiring custom dataset collection or model training**.

---

## 🌟 Key Features

- **Zero-Dataset Biomechanical Rules**: Uses pre-trained 33-point body pose estimation (**MediaPipe Pose**) combined with deterministic kinematics:
  - **Spine Inclination Angle ($\theta$)**: Measures torso vector relative to vertical ($0^\circ-30^\circ$ upright, $\ge 58^\circ$ fallen).
  - **Centroid Vertical Velocity ($V_y$)**: Detects rapid downward acceleration spikes.
  - **Aspect Ratio**: Bounding box width vs height validation.
  - **Joint Mobility / Stillness Energy**: Differentiates normal resting from unresponsive motionless states.
- **Inactivity Stopwatch**: Tracks prolonged stillness following a fall or sudden collapse, triggering an emergency **`INACTIVE ALERT`** if motionless beyond a configurable timeout (e.g. 5-30 seconds).
- **Interactive Web Dashboard**: Streamlit interface with live webcam feed, real-time skeleton overlay, kinematics trajectory chart, sensitivity sliders, audio alarm warnings, snapshot captures, and CSV incident report export.

---

## 🚀 Quick Start (For Non-Tech Savvy Users)

If you downloaded or cloned this repository:

### 🌟 Just Double-Click `run.bat`!

Double-click the **`run.bat`** file in the project folder for the web dashboard,
**`run_desktop.bat`** for the native desktop app (same features, better
frame rate — no browser roundtrip), or **`run_hallway.bat`** for the
multi-person hallway monitor (see below).

#### What `run.bat` does automatically:
1. **Checks for Python**: Checks if Python is installed on your computer. If not, it displays a clear download link and instructions.
2. **First-Time Auto-Setup**: If the virtual environment (`.venv`) is missing, it automatically creates it and installs all dependencies (`requirements.txt`).
3. **Launches the App**: Starts the dashboard and automatically opens your default browser at `http://localhost:8501`.
4. **Subsequent Runs**: On future runs, it skips installation and launches immediately in 1-2 seconds!

---

## 💻 Recipient System Requirements

For another user to run this repository, their computer needs:

| Requirement | Details |
| :--- | :--- |
| **Operating System** | Windows 10 or Windows 11 (64-bit) |
| **Python** | **Python 3.10, 3.11, or 3.12** installed. *(Crucial: Check **"Add python.exe to PATH"** during Python setup).* |
| **Webcam** | Built-in laptop webcam or plug-and-play USB camera. |
| **Internet (First run only)** | Required once during setup for `run.bat` to download packages via pip. |

---

## 🛠️ Manual Terminal Run (For Developers)

If you prefer using PowerShell / Command Prompt manually:

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit dashboard
streamlit run app.py
```

Or run the lightweight OpenCV desktop pipeline directly:

```powershell
python main.py --source 0
```

---

## 📁 Project Structure

```
fall detection and inactivity system/
├── run.bat                    # 1-click launcher: Streamlit web dashboard
├── run_desktop.bat            # 1-click launcher: native desktop app (faster)
├── run_hallway.bat            # 1-click launcher: hallway multi-person monitor
├── app.py                     # Interactive Streamlit Web Dashboard
├── desktop_app.py             # Native desktop client (same UX, threaded pipeline)
├── hallway_app.py             # Hallway/waiting-area monitor (multi-person, zone-aware)
├── main.py                    # Lightweight standalone OpenCV runner
├── requirements.txt           # Dependency specifications
├── core/
│   ├── pose_estimator.py      # 33-landmark pose extractor (MediaPipe Pose)
│   ├── person_detector.py     # Multi-person box proposals (tiled pose + HOG, optional YOLO drop-in)
│   ├── person_tracker.py      # Centroid track-ID assignment
│   ├── multi_person_manager.py# Per-person pose + fall + inactivity state
│   ├── feature_extractor.py   # Spine angle, aspect ratio, vertical speed, motion energy
│   ├── fall_detector.py       # Finite State Machine (NORMAL, PRE_FALL, FALLEN, RECOVERED)
│   └── inactivity_monitor.py # Stillness detector & emergency timeout monitor
├── utils/
│   ├── visualizer.py          # Real-time HUD, skeleton lines, status tags
│   ├── logger.py              # Event logging and snapshot manager (CSV download)
│   └── synthetic_generator.py # Procedural mannequin fall simulator for testing
└── assets/                    # Project demo assets
```

---

## ⚙️ Configuration & Sensitivity Tuning

You can adjust these parameters live in the dashboard sidebar:

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Spine Angle Threshold** | `58.0°` | Angle relative to vertical where posture is considered horizontal/fallen. |
| **Velocity Threshold ($V_y$)** | `0.32` | Downward speed threshold triggering falling transition. |
| **Inactivity Timeout** | `6.0s` | Seconds of motionless horizontal posture before triggering the emergency alert. |
| **Audio Warning** | Enabled | Plays an audible alarm warning upon fall or prolonged inactivity. |

---

## 🏥 Hallway Monitor (`hallway_app.py` — clinic hallways / waiting areas)

Self-contained multi-person version. One YOLOv8n-pose ONNX inference per frame
(all bodies at once) via ONNX Runtime; the fastest execution provider is
benchmarked at startup (**CUDA → DirectML (AMD/Intel iGPU) → CPU**), so the same
code runs on a Ryzen 3 laptop and scales to a gaming rig — the active backend
shows in the status bar. Models ship in `assets/models/`; after setup the app
runs **fully offline**.

Extra risk states for seating areas: **RESTING** (reclined + still on a bench,
no alarm) and **SLUMP** (sudden collapse into seating — check on person).
Draw bench/floor/ignore polygons via **Calibrate Zones** (needs a captured
empty-hallway background); with no zones file the whole frame is floor.
Each saved camera keeps its own calibration file.

```powershell
.\.venv\Scripts\python.exe hallway_app.py
# or: python tools/eval_clips.py   (batch-score recorded clips, headless)
#     python tests/simulate.py     (logic suite, no footage needed)
```

Camera sources: USB webcams (auto-discovered with a ⟳ rescan), IP/CCTV
cameras over **RTSP**/RTMP/HTTP (with connection test + auto-reconnect), and
video files. Save named cameras under 💾 Saved Cameras to switch bodies
clearly — each keeps its own zone calibration. All three apps
(`hallway_app`, `desktop_app`, `app.py`) share the same camera and frame
drivers (`core/camera_sources.py`, `core/frame_pump.py`) — no torch needed
anywhere; the .onnx model ships in assets/models/, so the app runs fully
offline after setup.

Scale-up path (no code changes): on an NVIDIA machine replace
`onnxruntime-directml` with `onnxruntime-gpu` in `requirements.txt`, optionally
drop in a larger `*-pose.onnx` as `assets/models/yolov8n-pose.onnx`.

> **Licensing note:** YOLO pose weights are AGPL-3.0 (Ultralytics export) and the
> UR/Le2i eval datasets are non-commercial/research — fine for internal
> evaluation; review licensing before any commercial deployment.
