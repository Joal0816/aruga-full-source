import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import cv2
import numpy as np
import re
import time
import os
import tempfile
import pandas as pd
from PIL import Image
from urllib.parse import urlparse

from core.pose_estimator import PoseEstimator
from core.feature_extractor import FeatureExtractor
from core.fall_detector import FallDetector, SystemState
from core.inactivity_monitor import InactivityMonitor
from core.multi_person_manager import MultiPersonManager
from core.camera_sources import (
    CameraProfiles, validate_network_url, probe_usb_cameras, describe_camera,
)
from core.frame_pump import FramePump, SourceLost
from utils.visualizer import Visualizer
from utils.logger import EventLogger
from utils.beep import AlarmAck
from utils.risk_palette import RISK_HEX


def _rgb(hex_color):
    return f"{int(hex_color[1:3], 16)}, {int(hex_color[3:5], 16)}, {int(hex_color[5:7], 16)}"
from utils.synthetic_generator import generate_synthetic_fall_video

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ARUGA | AI-Assisted Recognition and Understanding for Guided Contextual Risk Assessment",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS & THEME ---
# Risk pill colors come from utils/risk_palette via __C__/__E__ tokens so the
# web UI can never drift from the desktop/hallway palette.
_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }
    
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .subtitle {
        color: #9ca3af;
        font-size: 0.95rem;
        margin-top: 5px;
    }
    
    /* Telemetry Metric Cards */
    .metric-card {
        background: #1e2430;
        border-radius: 10px;
        padding: 14px 18px;
        border: 1px solid #2d3748;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }
    
    /* Alert Status Pills */
    .status-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .status-normal   { background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid #34d399; }
    .status-unusual   { background: rgba(251, 191, 36, 0.18); color: #fbbf24; border: 1px solid #fbbf24; }
    .status-concerning{ background: rgba(__C_RGB__, 0.22); color: __C__; border: 1px solid __C__; animation: pulse 1.2s infinite; }
    .status-cardiac   { background: rgba(244, 63, 94, 0.22); color: #fb7185; border: 1px solid #f43f5e; animation: pulse 1.0s infinite; }
    .status-emergency { background: rgba(__E_RGB__, 0.28); color: __E__; border: 1px solid __E__; animation: pulse 0.8s infinite; }
    
    @keyframes pulse {
        0% { opacity: 0.8; transform: scale(0.99); }
        50% { opacity: 1; transform: scale(1.02); }
        100% { opacity: 0.8; transform: scale(0.99); }
    }
</style>
"""
st.markdown(_CSS
            .replace("__C__", RISK_HEX["CONCERNING"])
            .replace("__C_RGB__", _rgb(RISK_HEX["CONCERNING"]))
            .replace("__E__", RISK_HEX["EMERGENCY"])
            .replace("__E_RGB__", _rgb(RISK_HEX["EMERGENCY"])),
            unsafe_allow_html=True)

# --- BROWSER AUDIO ALARM SYNTHESIZER ---
AUDIO_ALARM_HTML = """
<audio autoplay>
  <source src="data:audio/wav;base64,UklGRl9vT19XQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YU5vT18Ae3h3eHd4d3h3eHd4d3h3eHd4d3h3eHd4d3h3eHd4d3h3eHd4d3h3eHd4d3h3eHd4d3h3eHd4d3h3eHd4d3h3eHd4d3h3eHd4d3h3eHd4d3h3eHd4d3h3eHd4d3h3eHd4d3h3eHd4d3h3eHd4d3h3eA==" type="audio/wav">
</audio>
"""

# --- HEADER SECTION ---
st.markdown("""
<div class="main-header">
    <div class="main-title">🛡️ ARUGA</div>
    <div class="subtitle">AI-Assisted Recognition and Understanding for Guided Contextual Risk Assessment</div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR: CONFIGURATION & CONTROLS ---
st.sidebar.header("⚙️ System Configuration")

# Sensitivity Presets
sensitivity_mode = st.sidebar.selectbox(
    "Sensitivity Preset",
    ["Standard (Balanced)", "High Sensitivity (Elderly Care)", "Low Sensitivity (Active / Visitors)"],
    index=0
)

if sensitivity_mode == "High Sensitivity (Elderly Care)":
    default_angle = 50.0
    default_vel = 0.22
    default_timeout = 4.0
elif sensitivity_mode == "Low Sensitivity (Active / Visitors)":
    default_angle = 68.0
    default_vel = 0.45
    default_timeout = 10.0
else:
    default_angle = 58.0
    default_vel = 0.32
    default_timeout = 6.0

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Threshold Adjustments")

angle_thresh = st.sidebar.slider(
    "Spine Angle Threshold (degrees)",
    min_value=40.0,
    max_value=80.0,
    value=default_angle,
    step=1.0,
    help="Spine tilt angle relative to vertical. > 55°-65° indicates horizontal posture."
)

vel_thresh = st.sidebar.slider(
    "Downward Velocity Threshold",
    min_value=0.15,
    max_value=0.80,
    value=default_vel,
    step=0.02,
    help="Normalized downward centroid speed per second."
)

inactivity_timeout = st.sidebar.slider(
    "Inactivity Timeout (seconds)",
    min_value=3.0,
    max_value=30.0,
    value=default_timeout,
    step=1.0,
    help="Duration of motionless horizontal posture before emergency alert is triggered."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔔 Alerts & Overlays")
enable_audio = st.sidebar.checkbox("Enable Audio Warning", value=True)
show_skeleton = st.sidebar.checkbox("Render Skeleton Overlay", value=True,
                                    help="Toggling restarts the stream with the new overlay setting.")
show_telemetry_box = st.sidebar.checkbox("Render Telemetry Bounding Box", value=True,
                                         help="Toggling restarts the stream with the new overlay setting.")

# --- INPUT SOURCE SELECTION (shared core/camera_sources backends) ---
source_type = st.sidebar.radio(
    "Input Video Source",
    ["📹 USB Camera", "🌐 IP Camera (RTSP)", "📁 Upload Video File", "🎯 Synthetic Fall Simulation Demo"],
    key="src_type",
)

with st.sidebar.expander("💾 Saved Cameras"):
    _prof = CameraProfiles()
    _names = _prof.names()
    _sel = st.selectbox("Switch camera", ["—"] + _names, key="prof_sel")
    if _sel != "—" and _sel != st.session_state.get("prof_applied"):
        _p = _prof.get(_sel)
        if _p:
            _kind = _p.get("kind", "usb")
            st.session_state["src_type"] = {
                "usb": "📹 USB Camera", "rtsp": "🌐 IP Camera (RTSP)",
                "file": "📁 Upload Video File"}[_kind]
            if _kind == "usb":
                st.session_state["usb_sel"] = f"Camera {_p.get('index', 0)} (saved)"
            elif _kind == "rtsp":
                st.session_state["rtsp_url"] = _p.get("url", "")
            st.session_state["prof_applied"] = _sel
            _prof.active = _sel
            _prof.save()
            st.rerun()
    _new_name = st.text_input("Name current setup", key="prof_name")
    _cols = st.columns([1, 1])
    with _cols[0]:
        if st.button("Save current", use_container_width=True):
            _d = _current_source_dict((_new_name or "Camera").strip())
            try:
                _prof.upsert(_d)
                _prof.active = _d["name"]
                _prof.save()
                st.session_state["prof_applied"] = _d["name"]
                st.success(f"Saved '{_d['name']}'.")
                st.rerun()
            except ValueError as e:
                st.error(str(e))
    with _cols[1]:
        if st.button("Delete selected", use_container_width=True) and _sel != "—":
            _prof.delete(_sel)
            st.session_state["prof_applied"] = None
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("👥 People Detection")
detection_mode = st.sidebar.radio(
    "Detection Mode",
    ["Single person (original)", "Multi-person (2-3)"],
    index=0,
    help="Single = original full-frame MediaPipe (fastest). Multi = tiled-pose + HOG person boxes with per-person tracking (2-3 people, no extra downloads).",
)
multi_mode = detection_mode.startswith("Multi")
if multi_mode:
    max_persons = st.sidebar.slider("Max persons", min_value=2, max_value=3, value=3, step=1)
    detect_interval = st.sidebar.slider(
        "Detector refresh (frames)", min_value=2, max_value=10, value=5, step=1,
        help="Run the person detector every N frames; tracking carries boxes in between. Higher = faster, slightly stickier boxes."
    )
else:
    max_persons = 1
    detect_interval = 5

camera_idx = 0
rtsp_url = st.session_state.get("rtsp_url", "")
if source_type == "📹 USB Camera":
    if st.sidebar.button("🔍 Rescan USB cameras"):
        with st.spinner("Probing cameras…"):
            st.session_state.usb_cams = probe_usb_cameras()
        st.rerun()
    _opts = [describe_camera(c) for c in st.session_state.get("usb_cams", [])]
    _cur = st.session_state.get("usb_sel")
    if _cur and _cur not in _opts:
        _opts = _opts + [_cur]
    if not _opts:
        _opts = ["Camera 0 (default)"]
    _sel_usb = st.sidebar.selectbox(
        "USB camera (index shown first — use ⟳ if empty)",
        _opts, key="usb_sel",
        help="Plug the camera in, then Rescan. Saved profiles remember the index.")
    _m = re.match(r"Camera (\d+)", _sel_usb or "")
    camera_idx = int(_m.group(1)) if _m else 0
elif source_type == "🌐 IP Camera (RTSP)":
    rtsp_url = st.sidebar.text_input(
        "Stream URL", value="rtsp://", key="rtsp_url",
        help="CCTV/IP camera, e.g. rtsp://user:pass@192.168.1.50:554/stream")
    if st.sidebar.button("Test stream"):
        _ok, _msg = validate_network_url(rtsp_url)
        if not _ok:
            st.sidebar.error(_msg)
        else:
            with st.spinner("Connecting…"):
                try:
                    from core.camera_sources import open_capture
                    _cap = open_capture(rtsp_url, "rtsp")
                    _ok2 = _cap.isOpened()
                    _fr = _cap.read()[1] if _ok2 else None
                    _cap.release()
                    if _ok2 and _fr is not None:
                        st.sidebar.success(f"Stream OK — {_fr.shape[1]}x{_fr.shape[0]}")
                    else:
                        st.sidebar.error("Connected but got no frames.")
                except Exception as e:
                    st.sidebar.error(str(e))

# --- INITIALIZE SESSION STATE ---
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "event_logger" not in st.session_state:
    st.session_state.event_logger = EventLogger()
if "ack" not in st.session_state:
    st.session_state.ack = AlarmAck()
if "loop" not in st.session_state:
    # Persists pose estimator + rising-edge log dedup across Streamlit reruns
    st.session_state.loop = {}
if "fall_detector" not in st.session_state:
    st.session_state.fall_detector = FallDetector(
        angle_threshold=angle_thresh,
        velocity_threshold=vel_thresh
    )
if "inactivity_monitor" not in st.session_state:
    st.session_state.inactivity_monitor = InactivityMonitor(
        inactivity_timeout=inactivity_timeout
    )
if "feature_extractor" not in st.session_state:
    st.session_state.feature_extractor = FeatureExtractor()
if "multi_manager" not in st.session_state:
    st.session_state.multi_manager = MultiPersonManager(
        max_persons=max_persons,
        angle_threshold=angle_thresh,
        velocity_threshold=vel_thresh,
        inactivity_timeout=inactivity_timeout,
        detect_interval=detect_interval,
    )

# Update dynamic parameters
st.session_state.fall_detector.update_parameters(
    angle_threshold=angle_thresh,
    velocity_threshold=vel_thresh
)
st.session_state.inactivity_monitor.update_parameters(
    inactivity_timeout=inactivity_timeout
)
st.session_state.multi_manager.update_parameters(
    angle_threshold=angle_thresh,
    velocity_threshold=vel_thresh,
    inactivity_timeout=inactivity_timeout,
    max_persons=max_persons,
    detect_interval=detect_interval,
)

# --- MAIN LAYOUT ---
col_video, col_telemetry = st.columns([3, 2])

with col_video:
    st.subheader("📺 Video Stream & Real-time HUD")
    video_placeholder = st.empty()
    run_button_col1, run_button_col2, run_button_col3 = st.columns([1, 1, 1])
    with run_button_col1:
        if st.button("▶️ Start Monitoring", use_container_width=True, type="primary"):
            st.session_state.is_running = True
    with run_button_col2:
        if st.button("⏹️ Stop / Reset", use_container_width=True,
                     help="Full stop: releases the camera and clears detection state. "
                          "Press Start Monitoring to resume (reopens the source)."):
            st.session_state.is_running = False
            st.session_state.fall_detector.reset()
            st.session_state.inactivity_monitor.reset()
            st.session_state.feature_extractor.reset()
            st.session_state.multi_manager.reset()
            for _k in ("sp_logged_falls", "sp_logged_inactive",
                       "mp_logged_falls", "mp_logged_inactive"):
                st.session_state.loop.pop(_k, None)
            st.session_state.telemetry_history = {"Spine Angle (°)": [], "Downward Speed (Vy)": []}
            st.rerun()
    with run_button_col3:
        if st.button("🔕 Acknowledge (10 min)", use_container_width=True,
                     help="Silences the repeating alarm sound for 10 minutes. "
                          "Visual alerts and logs keep running; re-arms on the next incident."):
            st.session_state.ack.ack(time.time())

with col_telemetry:
    st.subheader("📊 Live Telemetry & Metrics")
    stream_status_ph = st.empty()
    stream_status_ph.markdown(
        '<span style="color:#94a3b8;">○ Idle — press Start Monitoring</span>',
        unsafe_allow_html=True)
    metric_cols1, metric_cols2 = st.columns(2)
    with metric_cols1:
        metric_state_ph = st.empty()
        metric_angle_ph = st.empty()
    with metric_cols2:
        metric_vel_ph = st.empty()
        metric_timer_ph = st.empty()
    metric_pattern_ph = st.empty()
        
    st.markdown("---")
    st.markdown("**📈 Real-Time Kinematic Trajectory**")
    chart_placeholder = st.empty()

# --- VIDEO SOURCE RESOLUTION ---
video_source = None
video_kind = "webcam"
source_label = ""
loop_files = False
temp_file_path = None

if source_type == "🎯 Synthetic Fall Simulation Demo":
    demo_path = os.path.join("assets", "synthetic_fall_demo.mp4")
    if not os.path.exists(demo_path):
        with st.spinner("Rendering synthetic demo video..."):
            generate_synthetic_fall_video(demo_path)
    video_source = demo_path
    video_kind = "file"
    source_label = "Synthetic Demo"
    loop_files = True

elif source_type == "📹 USB Camera":
    video_source = camera_idx
    video_kind = "webcam"
    source_label = f"USB Camera {camera_idx}"

elif source_type == "🌐 IP Camera (RTSP)":
    _ok, _msg = validate_network_url(rtsp_url)
    if not _ok:
        st.sidebar.error(f"Bad stream URL: {_msg}")
    else:
        _host = urlparse(rtsp_url).hostname or rtsp_url[:32]
        video_source = rtsp_url
        video_kind = "rtsp"
        source_label = f"RTSP {_host}"

elif source_type == "📁 Upload Video File":
    uploaded_file = st.sidebar.file_uploader("Upload MP4 / AVI / MOV", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        temp_file_path = tfile.name
        video_source = temp_file_path
        video_kind = "file"
        source_label = uploaded_file.name
        loop_files = False  # uploaded evidence clips end instead of looping
    else:
        st.info("Please upload a video file in the sidebar to start monitoring.")

# Risk-level display helper
def render_risk_pill(risk_level: str, inactive_dur: float = 0.0, behavior: str = "NORMAL") -> str:
    if risk_level == "EMERGENCY":
        return f'<span class="status-pill status-emergency">🚨 EMERGENCY — {inactive_dur:.1f}s INACTIVE</span>'
    elif behavior == "LEVINE_SIGN_DISTRESS" or risk_level == "CORONARY_DISTRESS":
        return '<span class="status-pill status-cardiac">💔 CARDIAC DISTRESS — LEVINE SIGN</span>'
    elif risk_level == "CONCERNING":
        return '<span class="status-pill status-concerning">🛑 CONCERNING — FALL DETECTED</span>'
    elif risk_level == "UNUSUAL":
        return '<span class="status-pill status-unusual">⚠️ UNUSUAL — UNSTABLE POSTURE</span>'
    return '<span class="status-pill status-normal">🟢 NORMAL — ROUTINE MOBILITY</span>'

# Default metric placeholders
metric_state_ph.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Risk Assessment</div>
    <div style="margin-top: 8px;">{render_risk_pill("NORMAL", 0.0)}</div>
</div>
""", unsafe_allow_html=True)

metric_pattern_ph.markdown("""
<div class="metric-card" style="margin-top: 10px;">
    <div class="metric-label">Behavioral Pattern Recognition (2 Active Patterns)</div>
    <div style="margin-top: 8px; font-size: 0.88rem; line-height: 1.6;">
        <div style="margin-bottom: 4px;"><b>1. Fall &amp; Inactivity:</b> <span style="color: #34d399;">🟢 Upright &amp; Active</span></div>
        <div><b>2. Levine's Sign (Cardiac Distress):</b> <span style="color: #94a3b8;">⚪ Clear / No Chest Clutch</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

metric_angle_ph.markdown("""
<div class="metric-card">
    <div class="metric-label">Spine Inclination</div>
    <div class="metric-value">0.0°</div>
</div>
""", unsafe_allow_html=True)

metric_vel_ph.markdown("""
<div class="metric-card">
    <div class="metric-label">Downward Speed (Vy)</div>
    <div class="metric-value">0.00</div>
</div>
""", unsafe_allow_html=True)

metric_timer_ph.markdown("""
<div class="metric-card">
    <div class="metric-label">Inactivity Timer</div>
    <div class="metric-value">0.0s</div>
</div>
""", unsafe_allow_html=True)

# Telemetry data buffer for live chart (persists across reruns)
if "telemetry_history" not in st.session_state:
    st.session_state.telemetry_history = {
        "Spine Angle (°)": [],
        "Downward Speed (Vy)": []
    }
telemetry_history = st.session_state.telemetry_history
multi_table_ph = st.empty()

_RISK_ORDER = {"NORMAL": 0, "UNUSUAL": 1, "CONCERNING": 2, "EMERGENCY": 3}

def _worst_person(persons):
    """Return person dict with highest risk, or None."""
    if not persons:
        return None
    def _risk(p):
        r = p.get("fall_status", {}).get("risk_level", "NORMAL")
        if p.get("inactivity_status", {}).get("is_inactive_alert", False):
            return "EMERGENCY"
        return r
    return max(persons, key=lambda p: _RISK_ORDER.get(_risk(p), 0))

def _current_source_dict(name):
    """Snapshot sidebar source widgets into a profile dict (for Save)."""
    stype = st.session_state.get("src_type", "📹 USB Camera")
    if stype == "🌐 IP Camera (RTSP)":
        return {"name": name, "kind": "rtsp", "url": st.session_state.get("rtsp_url", "")}
    if stype == "📁 Upload Video File":
        return {"name": name, "kind": "file", "path": ""}
    m = re.match(r"Camera (\d+)", st.session_state.get("usb_sel", "") or "")
    return {"name": name, "kind": "usb", "index": int(m.group(1)) if m else 0}

def _drain_pump(pump, alive_cb):
    """Yield pump frames; a dead source is stashed for display instead of
    raising through Streamlit (which would traceback every rerun)."""
    try:
        for frame in pump.frames(alive=alive_cb):
            yield frame
    except SourceLost as e:
        st.session_state["_pump_error"] = str(e)

def _take_pump_error():
    return st.session_state.pop("_pump_error", None)

# --- VIDEO PROCESSING LOOP (sources opened via shared FramePump) ---
if st.session_state.is_running and video_source is not None:
    try:
        pump = FramePump(video_source, kind=video_kind, loop_files=loop_files,
                         abort=lambda: not st.session_state.is_running)
    except SourceLost as e:
        st.error(f"Unable to open video stream ({source_label}): {e}")
        st.session_state.is_running = False
        pump = None
    if pump is None:
        pass
    elif multi_mode:
        # ================= MULTI-PERSON LOOP (2-3) =================
        # NO reset() on entry — tracker/FSM state persists across reruns
        # (a widget change mid-incident used to wipe all per-person state).
        visualizer = Visualizer()

        frame_idx = 0
        audio_trigger_placeholder = st.empty()
        st.session_state.loop.setdefault("mp_logged_falls", {})
        st.session_state.loop.setdefault("mp_logged_inactive", set())
        _last_t, _fps = time.time(), 0.0
        st.session_state.pop("_pump_error", None)

        for frame in _drain_pump(pump, lambda: st.session_state.is_running):
            frame_idx += 1
            now = time.time()
            _dt = max(1e-3, now - _last_t)
            _last_t = now
            _fps = 0.9 * _fps + 0.1 * (1.0 / _dt) if _fps else 1.0 / _dt
            stream_status_ph.markdown(
                f'<span style="color:#34d399; font-weight:700;">● LIVE</span> '
                f'<span style="color:#94a3b8;">— {_fps:.1f} fps analysis</span>',
                unsafe_allow_html=True)
            persons = st.session_state.multi_manager.process(frame, current_time=now)

            # Log per-person events: one log per incident (rising edge).
            # A FALL log fires when a new fall is counted; an INACTIVITY log
            # fires when stillness first trips (independent ifs: the old
            # if/elif chain starved INACTIVITY while still FALLEN).
            logged_falls = st.session_state.loop["mp_logged_falls"]
            logged_inactive = st.session_state.loop["mp_logged_inactive"]
            for p in persons:
                tid = p["track_id"]
                fs, ins, feats = p["fall_status"], p["inactivity_status"], p["features"]
                if feats is None:
                    continue
                if fs.get("state") == "FALLEN" and fs.get("total_falls", 0) > logged_falls.get(tid, 0):
                    logged_falls[tid] = fs["total_falls"]
                    st.session_state.event_logger.log_event(
                        "FALL_DETECTED", feats, fs.get("fall_confidence", 0.8), frame, person_id=tid)
                if ins.get("is_inactive_alert") and tid not in logged_inactive:
                    logged_inactive.add(tid)
                    st.session_state.event_logger.log_event(
                        "INACTIVITY_EMERGENCY", feats, 1.0, frame,
                        extra_note=f"P{tid} stationary for {ins.get('inactive_duration'):.1f}s",
                        person_id=tid)
                elif not ins.get("is_inactive_alert") and tid in logged_inactive:
                    logged_inactive.discard(tid)
            _live = {p["track_id"] for p in persons}
            for _tid in [t for t in list(logged_falls) if t not in _live]:
                del logged_falls[_tid]
            logged_inactive &= _live

            display_frame = visualizer.draw_hud_multi(
                frame, persons, show_skeleton=show_skeleton, show_bbox=show_telemetry_box)
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            worst = _worst_person(persons)
            worst_behavior = "NORMAL"
            if worst is not None and worst.get("features"):
                curr_risk = worst["fall_status"].get("risk_level", "NORMAL")
                worst_behavior = worst["fall_status"].get("behavior", "NORMAL")
                if worst["inactivity_status"].get("is_inactive_alert", False):
                    curr_risk = "EMERGENCY"
                elif worst_behavior == "LEVINE_SIGN_DISTRESS" or worst["fall_status"].get("state") == "CORONARY_DISTRESS":
                    if curr_risk not in ("EMERGENCY", "CONCERNING"):
                        curr_risk = "CONCERNING"
                inact_dur = worst["inactivity_status"].get("inactive_duration", 0.0)
                is_inact = worst["inactivity_status"].get("is_inactive_alert", False)
                angle = worst["features"].get("spine_angle", 0.0)
                vy = worst["features"].get("vertical_velocity", 0.0)
            else:
                curr_risk, inact_dur, is_inact, angle, vy = "NORMAL", 0.0, False, 0.0, 0.0

            metric_state_ph.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Risk Assessment (worst of {len(persons)})</div>
                <div style="margin-top: 8px;">{render_risk_pill(curr_risk, inact_dur, worst_behavior)}</div>
            </div>
            """, unsafe_allow_html=True)

            metric_angle_ph.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Spine Inclination (worst)</div>
                <div class="metric-value">{angle:.1f}°</div>
            </div>
            """, unsafe_allow_html=True)

            metric_vel_ph.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Downward Speed (Vy, worst)</div>
                <div class="metric-value">{vy:+.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            metric_timer_ph.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Inactivity Duration (worst)</div>
                <div class="metric-value" style="color: {'#dc2626' if is_inact else '#f8fafc'};">{inact_dur:.1f}s</div>
            </div>
            """, unsafe_allow_html=True)

            # Display 2 clinical behavioral patterns (Fall & Inactivity vs Levine's Sign Cardiac Distress)
            if any(p.get("inactivity_status", {}).get("is_inactive_alert", False) for p in persons):
                p1_desc = f'<span style="color: {RISK_HEX["EMERGENCY"]}; font-weight: 700;">🚨 INACTIVITY EMERGENCY</span>'
            elif any(p.get("fall_status", {}).get("state") == "FALLEN" for p in persons):
                p1_desc = f'<span style="color: {RISK_HEX["CONCERNING"]}; font-weight: 700;">⚠️ FALL DETECTED</span>'
            elif any(p.get("fall_status", {}).get("state") == "PRE_FALL" for p in persons):
                p1_desc = '<span style="color: #fbbf24; font-weight: 700;">⚠️ UNSTABLE / PRE-FALL DESCENT</span>'
            else:
                p1_desc = '<span style="color: #34d399; font-weight: 600;">🟢 Upright &amp; Routine Mobility</span>'

            has_levine_distress = any(
                (p.get("fall_status", {}).get("behavior") == "LEVINE_SIGN_DISTRESS" or
                 p.get("fall_status", {}).get("state") == "CORONARY_DISTRESS" or
                 (p.get("features") and p["features"].get("is_levine_gesture", False)))
                for p in persons
            )
            if has_levine_distress:
                p2_desc = '<span style="color: #fb7185; font-weight: 700;">💔 ACTIVE — chest-clutch gesture detected (possible cardiac distress — check on person)</span>'
            else:
                p2_desc = '<span style="color: #94a3b8;">🟢 Clear / No Chest Clutching</span>'

            metric_pattern_ph.markdown(f"""
            <div class="metric-card" style="margin-top: 10px;">
                <div class="metric-label">Behavioral Pattern Recognition (2 Active Patterns)</div>
                <div style="margin-top: 8px; font-size: 0.88rem; line-height: 1.6;">
                    <div style="margin-bottom: 4px;"><b>1. Fall &amp; Inactivity:</b> {p1_desc}</div>
                    <div><b>2. Levine's Sign (Cardiac Distress):</b> {p2_desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            alarm_active = (curr_risk in ("CONCERNING", "EMERGENCY")
                            or worst_behavior == "LEVINE_SIGN_DISTRESS")
            st.session_state.ack.rearm_if_cleared(alarm_active, time.time())
            if enable_audio and alarm_active and not st.session_state.ack.muted(time.time()):
                audio_trigger_placeholder.markdown(AUDIO_ALARM_HTML, unsafe_allow_html=True)
            else:
                audio_trigger_placeholder.empty()

            # Per-person table
            if persons:
                rows = []
                for p in persons:
                    fs, ins, feats = p["fall_status"], p["inactivity_status"], p["features"]
                    r = fs.get("risk_level", "NORMAL")
                    if ins.get("is_inactive_alert", False):
                        r = "EMERGENCY"
                    rows.append({
                        "ID": f"P{p['track_id']}",
                        "Risk": r,
                        "Behavior": fs.get("behavior", "NORMAL"),
                        "Angle": f"{feats.get('spine_angle', 0.0):.1f}°" if feats else "-",
                        "Vy": f"{feats.get('vertical_velocity', 0.0):+.2f}" if feats else "-",
                        "Inactive": f"{ins.get('inactive_duration', 0.0):.1f}s",
                        "Pose": "yes" if p["has_pose"] else "no",
                    })
                multi_table_ph.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                multi_table_ph.info("No persons detected — detector runs every "
                                    f"{detect_interval} frames (HOG fallback sees upright persons only).")

            if frame_idx % 2 == 0:
                telemetry_history["Spine Angle (°)"].append(angle)
                telemetry_history["Downward Speed (Vy)"].append(vy * 100.0)
                if len(telemetry_history["Spine Angle (°)"]) > 50:
                    telemetry_history["Spine Angle (°)"] = telemetry_history["Spine Angle (°)"][-50:]
                    telemetry_history["Downward Speed (Vy)"] = telemetry_history["Downward Speed (Vy)"][-50:]
                chart_df = pd.DataFrame(telemetry_history)
                chart_placeholder.line_chart(chart_df, height=180)

        pump.release()
        _err = _take_pump_error()
        if _err:
            st.error(f"Source lost ({source_label}): {_err}")
            st.session_state.is_running = False
            stream_status_ph.markdown(
                f'<span style="color:{RISK_HEX["CONCERNING"]}; font-weight:700;">○ SOURCE LOST</span> '
                f'<span style="color:#94a3b8;">— {_err}</span>', unsafe_allow_html=True)
        else:
            stream_status_ph.markdown(
                '<span style="color:#94a3b8;">○ Stream stopped</span>', unsafe_allow_html=True)
    else:
        # ================= SINGLE-PERSON LOOP (original) =================
        # Detectors/pose/log-edge state persist in session_state across
        # Streamlit reruns: widget changes mid-incident used to re-enter this
        # loop and reset() the FSM, losing fall timestamps / inactivity timers.
        if st.session_state.loop.get("pose") is None:
            st.session_state.loop["pose"] = PoseEstimator()
        pose_estimator = st.session_state.loop["pose"]
        visualizer = Visualizer()

        frame_idx = 0
        audio_trigger_placeholder = st.empty()
        st.session_state.loop.setdefault("sp_logged_falls", 0)
        st.session_state.loop.setdefault("sp_logged_inactive", False)
        _last_t, _fps = time.time(), 0.0
        st.session_state.pop("_pump_error", None)

        for frame in _drain_pump(pump, lambda: st.session_state.is_running):
            frame_idx += 1
            now = time.time()
            _dt = max(1e-3, now - _last_t)
            _last_t = now
            _fps = 0.9 * _fps + 0.1 * (1.0 / _dt) if _fps else 1.0 / _dt
            stream_status_ph.markdown(
                f'<span style="color:#34d399; font-weight:700;">● LIVE</span> '
                f'<span style="color:#94a3b8;">— {_fps:.1f} fps analysis</span>',
                unsafe_allow_html=True)

            # 1. Pose estimation
            has_pose, raw_landmarks, keypoints = pose_estimator.process_frame(frame)
            
            features = None
            fall_status = {
                "state": "NORMAL",
                "risk_level": "NORMAL",
                "fall_confidence": 0.0,
                "total_falls": st.session_state.fall_detector.total_falls_detected,
                "is_horizontal": False
            }
            inactivity_status = {
                "is_inactive_alert": False,
                "inactive_duration": 0.0,
                "is_still": False
            }

            if has_pose and keypoints:
                # 2. Extract features
                features = st.session_state.feature_extractor.extract_features(keypoints, current_time=now)

                # 3. Inactivity pre-check (needed before fall detector for risk escalation)
                temp_fall = {"state": st.session_state.fall_detector.current_state.value, "is_horizontal": (features.get("spine_angle", 0) >= angle_thresh)}
                inactivity_status = st.session_state.inactivity_monitor.process(features, temp_fall)

                # 4. Fall detection with inactivity context
                fall_status = st.session_state.fall_detector.process(
                    features, is_inactive=inactivity_status.get("is_inactive_alert", False)
                )

                # Log events: one log per incident (rising edge)
                if fall_status.get("state") == "FALLEN" \
                        and fall_status.get("total_falls", 0) > st.session_state.loop["sp_logged_falls"]:
                    st.session_state.loop["sp_logged_falls"] = fall_status["total_falls"]
                    st.session_state.event_logger.log_event(
                        "FALL_DETECTED", features, fall_status.get("fall_confidence", 0.8), frame
                    )
                if inactivity_status.get("is_inactive_alert") and not st.session_state.loop["sp_logged_inactive"]:
                    st.session_state.loop["sp_logged_inactive"] = True
                    st.session_state.event_logger.log_event(
                        "INACTIVITY_EMERGENCY", features, 1.0, frame,
                        extra_note=f"Stationary for {inactivity_status.get('inactive_duration'):.1f}s"
                    )
                elif not inactivity_status.get("is_inactive_alert"):
                    st.session_state.loop["sp_logged_inactive"] = False

            # 5. Visualizer HUD
            display_frame = visualizer.draw_hud(
                frame, keypoints, features, fall_status, inactivity_status,
                show_skeleton=show_skeleton, show_bbox=show_telemetry_box)
            
            # Streamlit image output (convert BGR to RGB)
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # 6. Update Telemetry metrics
            curr_risk = fall_status.get("risk_level", "NORMAL")
            behavior = fall_status.get("behavior", "NORMAL")
            is_inact = inactivity_status.get("is_inactive_alert", False)
            inact_dur = inactivity_status.get("inactive_duration", 0.0)
            angle = features.get("spine_angle", 0.0) if features else 0.0
            vy = features.get("vertical_velocity", 0.0) if features else 0.0

            metric_state_ph.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Risk Assessment</div>
                <div style="margin-top: 8px;">{render_risk_pill(curr_risk, inact_dur, behavior)}</div>
            </div>
            """, unsafe_allow_html=True)

            metric_angle_ph.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Spine Inclination</div>
                <div class="metric-value">{angle:.1f}°</div>
            </div>
            """, unsafe_allow_html=True)

            metric_vel_ph.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Downward Speed (Vy)</div>
                <div class="metric-value">{vy:+.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            metric_timer_ph.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Inactivity Duration</div>
                <div class="metric-value" style="color: {'#dc2626' if is_inact else '#f8fafc'};">{inact_dur:.1f}s</div>
            </div>
            """, unsafe_allow_html=True)

            # Display 2 clinical behavioral patterns (Fall & Inactivity vs Levine's Sign Cardiac Distress)
            if is_inact:
                p1_desc = f'<span style="color: {RISK_HEX["EMERGENCY"]}; font-weight: 700;">🚨 INACTIVITY EMERGENCY ({inact_dur:.1f}s)</span>'
            elif fall_status.get("state") == "FALLEN":
                p1_desc = f'<span style="color: {RISK_HEX["CONCERNING"]}; font-weight: 700;">⚠️ FALL DETECTED ({int(fall_status.get("fall_confidence", 0.8)*100)}%)</span>'
            elif fall_status.get("state") == "PRE_FALL":
                p1_desc = '<span style="color: #fbbf24; font-weight: 700;">⚠️ UNSTABLE / PRE-FALL DESCENT</span>'
            else:
                p1_desc = '<span style="color: #34d399; font-weight: 600;">🟢 Upright &amp; Routine Mobility</span>'

            is_levine_active = (
                behavior == "LEVINE_SIGN_DISTRESS" or
                fall_status.get("is_levine_sign", False) or
                (features and features.get("is_levine_gesture", False))
            )
            if is_levine_active:
                p2_desc = '<span style="color: #fb7185; font-weight: 700;">💔 ACTIVE — chest-clutch gesture detected (possible cardiac distress — check on person)</span>'
            else:
                p2_desc = '<span style="color: #94a3b8;">🟢 Clear / No Chest Clutching</span>'

            metric_pattern_ph.markdown(f"""
            <div class="metric-card" style="margin-top: 10px;">
                <div class="metric-label">Behavioral Pattern Recognition (2 Active Patterns)</div>
                <div style="margin-top: 8px; font-size: 0.88rem; line-height: 1.6;">
                    <div style="margin-bottom: 4px;"><b>1. Fall &amp; Inactivity:</b> {p1_desc}</div>
                    <div><b>2. Levine's Sign (Cardiac Distress):</b> {p2_desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Audio alert on CONCERNING, EMERGENCY, or LEVINE_SIGN_DISTRESS
            alarm_active = (curr_risk in ("CONCERNING", "EMERGENCY")
                            or behavior == "LEVINE_SIGN_DISTRESS")
            st.session_state.ack.rearm_if_cleared(alarm_active, time.time())
            if enable_audio and alarm_active and not st.session_state.ack.muted(time.time()):
                audio_trigger_placeholder.markdown(AUDIO_ALARM_HTML, unsafe_allow_html=True)
            else:
                audio_trigger_placeholder.empty()

            # Record telemetry for live chart (every 2nd frame)
            if frame_idx % 2 == 0:
                telemetry_history["Spine Angle (°)"].append(angle)
                telemetry_history["Downward Speed (Vy)"].append(vy * 100.0) # scaled for visibility
                
                # Keep last 50 data points
                if len(telemetry_history["Spine Angle (°)"]) > 50:
                    telemetry_history["Spine Angle (°)"] = telemetry_history["Spine Angle (°)"][-50:]
                    telemetry_history["Downward Speed (Vy)"] = telemetry_history["Downward Speed (Vy)"][-50:]
                    
                chart_df = pd.DataFrame(telemetry_history)
                chart_placeholder.line_chart(chart_df, height=180)

            # No sleep — let Streamlit refresh as fast as possible for lower latency

        pump.release()
        _err = _take_pump_error()
        if _err:
            st.error(f"Source lost ({source_label}): {_err}")
            st.session_state.is_running = False
            stream_status_ph.markdown(
                f'<span style="color:{RISK_HEX["CONCERNING"]}; font-weight:700;">○ SOURCE LOST</span> '
                f'<span style="color:#94a3b8;">— {_err}</span>', unsafe_allow_html=True)
        else:
            stream_status_ph.markdown(
                '<span style="color:#94a3b8;">○ Stream stopped</span>', unsafe_allow_html=True)

# Cleanup temp video file if uploaded
if temp_file_path and os.path.exists(temp_file_path):
    try:
        os.remove(temp_file_path)
    except Exception:
        pass

# --- INCIDENT LOG & SNAPSHOT GALLERY ---
st.markdown("---")
st.subheader("📋 Incident History & Snapshot Log")

log_df = st.session_state.event_logger.get_dataframe()
_LOG_COLS = {"id": "#", "timestamp": "Time", "person_id": "Person", "event_type": "Event",
             "confidence": "Confidence", "spine_angle": "Angle°", "vertical_velocity": "Speed",
             "aspect_ratio": "BBox AR", "note": "Note"}

col_table, col_actions = st.columns([4, 1])
with col_table:
    if not log_df.empty:
        st.dataframe(log_df.rename(columns=_LOG_COLS), use_container_width=True)
    else:
        st.write("No fall or inactivity incidents recorded yet.")

with col_actions:
    if not log_df.empty:
        csv_data = log_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV Log",
            data=csv_data,
            file_name="fall_inactivity_incidents.csv",
            mime="text/csv",
            use_container_width=True
        )
    if st.button("🗑️ Clear Log", use_container_width=True):
        st.session_state.event_logger.clear()
        st.rerun()

# Display recent snapshot captures
if os.path.exists("alerts") and len(os.listdir("alerts")) > 0:
    st.markdown("##### 📸 Recent Alert Snapshots")
    alert_files = sorted(
        [f for f in os.listdir("alerts") if f.endswith(".jpg")],
        key=lambda x: os.path.getmtime(os.path.join("alerts", x)),
        reverse=True
    )[:6]
    
    if alert_files:
        snap_cols = st.columns(len(alert_files))
        for i, file_name in enumerate(alert_files):
            file_path = os.path.join("alerts", file_name)
            with snap_cols[i]:
                st.image(file_path, caption=file_name, use_container_width=True)
