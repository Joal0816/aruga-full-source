"""
ARUGA Desktop — native desktop client mirroring the Streamlit dashboard UX.

Why desktop over web here:
  - Pipeline runs in a worker thread; Tk only displays. No per-frame
    JPEG-over-websocket roundtrip like Streamlit's st.image().
  - Bounded display queue (drop-oldest) so the UI never lags behind inference.
  - Native file dialogs, winsound beeps (stdlib), cheaper live chart (Canvas).

Single vs multi, thresholds, overlays and logging behave like app.py.
Run:  .\\.venv\\Scripts\\python.exe desktop_app.py   (or double-click run_desktop.bat)
"""

import os
import queue
import re
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from urllib.parse import urlparse
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import customtkinter as ctk
from PIL import Image, ImageTk

from core.pose_estimator import PoseEstimator
from core.feature_extractor import FeatureExtractor
from core.fall_detector import FallDetector
from core.inactivity_monitor import InactivityMonitor
from core.multi_person_manager import MultiPersonManager
from core.camera_sources import (
    CameraProfiles, validate_network_url, probe_usb_cameras, describe_camera, open_capture,
)
from core.frame_pump import FramePump, SourceLost
from utils.visualizer import Visualizer
from utils.logger import EventLogger
from utils.synthetic_generator import generate_synthetic_fall_video
from utils.beep import beep as _beep, AlarmAck
from utils.risk_palette import RISK_HEX as RISK_COLORS, ICONS


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Risk colors: shared palette (utils/risk_palette) — EMERGENCY red, CONCERNING deep orange.
RISK_ORDER = {"NORMAL": 0, "UNUSUAL": 1, "CONCERNING": 2, "EMERGENCY": 3}
RISK_TEXT = {
    "NORMAL": "NORMAL — ROUTINE MOBILITY",
    "UNUSUAL": "UNUSUAL — UNSTABLE POSTURE",
    "CONCERNING": "CONCERNING — FALL DETECTED",
    "EMERGENCY": "EMERGENCY — INACTIVE",
}

PRESETS = {
    "Standard (Balanced)": (58.0, 0.32, 6.0),
    "High Sensitivity (Elderly Care)": (50.0, 0.22, 4.0),
    "Low Sensitivity (Active Gym/Sports)": (68.0, 0.45, 10.0),
}


class ArugaDesktopApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ARUGA — Fall Detection & Inactivity Monitor (Desktop)")
        self.geometry(f"{self.WIN_W}x{self.WIN_H}")
        self.minsize(1100, 700)

        self.logger = EventLogger()
        self.profiles = CameraProfiles()
        self.profile_name = self.profiles.active
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ack = AlarmAck()
        self._packets: queue.Queue = queue.Queue(maxsize=2)
        self._running = False
        self._last_event_count = 0
        self._chart_angle: list[float] = []
        self._chart_vy: list[float] = []
        self._photo: ImageTk.PhotoImage | None = None
        self._thumbs: list[ImageTk.PhotoImage] = []
        self._video_path: str | None = None

        # Live-tunable params (main thread writes, worker reads each frame)
        self.params = {
            "angle": 58.0, "vel": 0.32, "timeout": 6.0,
            "audio": True, "skeleton": True, "bbox": True,
        }

        self._build_layout()
        self._apply_preset("Standard (Balanced)")
        self._refresh_profile_menu()
        self._refresh_source_rows()
        self._rescan()  # background USB discovery on boot
        self.after(33, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ================= layout =================

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- header ---
        header = ctk.CTkFrame(self, corner_radius=10)
        header.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="ew")
        ctk.CTkLabel(header, text="🛡️ ARUGA",
                     font=ctk.CTkFont(size=22, weight="bold")).pack(side="left", padx=16, pady=10)
        ctk.CTkLabel(header,
                     text="AI-Assisted Recognition and Understanding for Guided Contextual Risk Assessment  •  Desktop",
                     text_color="gray").pack(side="left", padx=8)
        self.status_label = ctk.CTkLabel(header, text="Idle", text_color="gray")
        self.status_label.pack(side="right", padx=16)
        ctk.CTkButton(header, text="🔕 Acknowledge (10 min)", width=170,
                      command=self._do_ack).pack(side="right", padx=(0, 4))

        # --- draggable splitters (nested PanedWindows) ---
        # Outer vertical split: top area (sidebar|video|telemetry) / bottom area (log+snapshots).
        # Inner horizontal split: sidebar | video | telemetry.
        # Drag any divider to freely resize; positions persist until the window closes.
        self.outer_pane = tk.PanedWindow(self, orient="vertical", background="#212121",
                                         sashwidth=8, sashrelief="flat", borderwidth=0)
        self.outer_pane.grid(row=1, column=0, padx=12, pady=(6, 12), sticky="nsew")
        self.top_pane = tk.PanedWindow(self.outer_pane, orient="horizontal", background="#212121",
                                       sashwidth=8, sashrelief="flat", borderwidth=0)

        # --- sidebar ---
        # Note: CTkScrollableFrame embeds itself inside an internal canvas, so it
        # cannot be added to a PanedWindow directly — it lives in a plain holder.
        # propagate(False) stops tall content requests from inflating the window;
        # the pane allocation (sash-driven) is what the user actually sees.
        side_holder = ctk.CTkFrame(self.top_pane, fg_color="transparent")
        side_holder.pack_propagate(False)
        self.top_pane.add(side_holder, minsize=250, sticky="nsew", stretch="never")
        side = ctk.CTkScrollableFrame(side_holder, width=270, corner_radius=10, label_text="⚙️ System Configuration")
        side.pack(fill="both", expand=True)

        ctk.CTkLabel(side, text="Sensitivity Preset").pack(anchor="w", padx=8, pady=(8, 0))
        self.preset_menu = ctk.CTkOptionMenu(side, values=list(PRESETS.keys()), command=self._apply_preset)
        self.preset_menu.pack(fill="x", padx=8, pady=4)

        self.angle_slider, self.angle_val = self._slider_row(side, "Spine Angle Threshold (°)", 40.0, 80.0, 58.0, "angle")
        self.vel_slider, self.vel_val = self._slider_row(side, "Downward Velocity Threshold", 0.15, 0.80, 0.32, "vel")
        self.timeout_slider, self.timeout_val = self._slider_row(side, "Inactivity Timeout (s)", 3.0, 30.0, 6.0, "timeout")

        ctk.CTkLabel(side, text="🔔 Alerts & Overlays").pack(anchor="w", padx=8, pady=(10, 0))
        self.audio_var = ctk.BooleanVar(value=True)
        self.skel_var = ctk.BooleanVar(value=True)
        self.bbox_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(side, text="Enable Audio Warning", variable=self.audio_var).pack(anchor="w", padx=8, pady=2)
        ctk.CTkCheckBox(side, text="Render Skeleton Overlay", variable=self.skel_var).pack(anchor="w", padx=8, pady=2)
        ctk.CTkCheckBox(side, text="Render Telemetry Bounding Box", variable=self.bbox_var).pack(anchor="w", padx=8, pady=2)

        ctk.CTkLabel(side, text="👥 People Detection").pack(anchor="w", padx=8, pady=(10, 0))
        self.mode_menu = ctk.CTkOptionMenu(side, values=["Single person", "Multi-person (2-3)"])
        self.mode_menu.pack(fill="x", padx=8, pady=4)
        self.maxp_menu = ctk.CTkOptionMenu(side, values=["2", "3"])
        self.maxp_menu.set("3")
        ctk.CTkLabel(side, text="Max persons (applies on Start)").pack(anchor="w", padx=8)
        self.maxp_menu.pack(fill="x", padx=8, pady=(0, 4))
        self.det_slider, self.det_val = self._slider_row(side, "Detector refresh (frames)", 2, 10, 5, None, is_int=True)
        self.pose_menu = ctk.CTkOptionMenu(side, values=["Balanced", "Fast"])
        ctk.CTkLabel(side, text="Pose detail: Fast = higher FPS (applies on Start)").pack(anchor="w", padx=8, pady=(6, 0))
        self.pose_menu.pack(fill="x", padx=8, pady=4)

        ctk.CTkLabel(side, text="📷 Camera Source (applies on Start)").pack(anchor="w", padx=8, pady=(10, 0))
        self.source_menu = ctk.CTkOptionMenu(
            side, values=["USB Camera", "IP Camera (RTSP)", "Video File", "Synthetic Demo"],
            command=self._on_source_type_changed)
        self.source_menu.pack(fill="x", padx=8, pady=4)

        # Grid container: show/hide never disturbs sidebar order (see hallway_app).
        self.source_box = ctk.CTkFrame(side, fg_color="transparent")
        self.source_box.pack(fill="x", padx=0, pady=0)
        self.source_box.grid_columnconfigure(0, weight=1)

        self.usb_row = ctk.CTkFrame(self.source_box, fg_color="transparent")
        self.usb_row.grid(row=0, column=0, sticky="ew", padx=8, pady=2)
        self.cam_menu = ctk.CTkOptionMenu(self.usb_row, values=["Scanning for USB cameras…"])
        self.cam_menu.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkButton(self.usb_row, text="⟳", width=44, command=self._rescan).pack(side="right")
        self._usb_cams: list = []

        self.rtsp_row = ctk.CTkFrame(self.source_box, fg_color="transparent")
        self.rtsp_row.grid(row=1, column=0, sticky="ew", padx=8, pady=2)
        self.rtsp_entry = ctk.CTkEntry(self.rtsp_row, placeholder_text="rtsp://user:pass@192.168.1.50:554/stream")
        self.rtsp_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkButton(self.rtsp_row, text="Test", width=60, command=self._test_rtsp).pack(side="right")
        self.rtsp_status = ctk.CTkLabel(self.source_box, text="", text_color="gray",
                                        wraplength=240, justify="left")
        self.rtsp_status.grid(row=2, column=0, sticky="w", padx=8)

        self.file_row = ctk.CTkFrame(self.source_box, fg_color="transparent")
        self.file_row.grid(row=3, column=0, sticky="ew", padx=8, pady=2)
        ctk.CTkButton(self.file_row, text="Browse…", width=90, command=self._browse).pack(side="left")
        self.file_label = ctk.CTkLabel(self.source_box, text="No file chosen", text_color="gray",
                                       wraplength=240, justify="left")
        self.file_label.grid(row=4, column=0, sticky="w", padx=8, pady=2)
        self.synth_label = ctk.CTkLabel(self.source_box, text="Procedural mannequin clip (pose test only).",
                                        text_color="gray", wraplength=240, justify="left")
        self.synth_label.grid(row=5, column=0, sticky="w", padx=8, pady=2)

        ctk.CTkLabel(side, text="💾 Saved Cameras").pack(anchor="w", padx=8, pady=(10, 0))
        self.profile_menu = ctk.CTkOptionMenu(side, values=["No saved cameras"],
                                              command=self._apply_profile)
        self.profile_menu.pack(fill="x", padx=8, pady=4)
        prof_row = ctk.CTkFrame(side, fg_color="transparent")
        prof_row.pack(fill="x", padx=8, pady=2)
        self.profile_entry = ctk.CTkEntry(prof_row, placeholder_text="Name this camera…")
        self.profile_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkButton(prof_row, text="Save", width=60, command=self._save_profile).pack(side="right")
        ctk.CTkButton(side, text="Delete Selected Camera", width=180,
                      command=self._delete_profile).pack(anchor="w", padx=8, pady=(2, 0))
        self.profile_hint = ctk.CTkLabel(side, text="", text_color="gray",
                                         wraplength=240, justify="left")
        self.profile_hint.pack(anchor="w", padx=8)

        self.start_btn = ctk.CTkButton(side, text="▶️ Start Monitoring", command=self.start)
        self.start_btn.pack(fill="x", padx=8, pady=(12, 4))
        self.stop_btn = ctk.CTkButton(side, text="⏹️ Stop / Reset", command=self.stop, state="disabled")
        self.stop_btn.pack(fill="x", padx=8, pady=(0, 12))

        # --- video ---
        video_frame = ctk.CTkFrame(self.top_pane, corner_radius=10)
        self.top_pane.add(video_frame, minsize=360, sticky="nsew", stretch="always")
        ctk.CTkLabel(video_frame, text="📺 Video Stream & Real-time HUD").pack(anchor="w", padx=12, pady=(8, 0))
        self.video_label = ctk.CTkLabel(video_frame, text="Press Start Monitoring")
        self.video_label.pack(expand=True, fill="both", padx=12, pady=12)

        # --- telemetry (scrollable holder, same pattern as sidebar) ---
        tele_holder = ctk.CTkFrame(self.top_pane, fg_color="transparent")
        tele_holder.pack_propagate(False)
        self.top_pane.add(tele_holder, minsize=250, sticky="nsew", stretch="never")
        tele = ctk.CTkScrollableFrame(tele_holder, width=270, corner_radius=10, label_text="📊 Live Telemetry")
        tele.pack(fill="both", expand=True)
        self.card_risk = self._metric_card(tele, "Risk Assessment")
        self.card_angle = self._metric_card(tele, "Spine Inclination")
        self.card_vy = self._metric_card(tele, "Downward Speed (Vy)")
        self.card_timer = self._metric_card(tele, "Inactivity Timer")
        ctk.CTkLabel(tele, text="📈 Kinematic Trajectory").pack(anchor="w", padx=8, pady=(8, 0))
        self.chart = tk.Canvas(tele, width=250, height=150, bg="#1e2430", highlightthickness=0)
        self.chart.pack(fill="x", padx=8, pady=4)
        ctk.CTkLabel(tele, text="Per-person (multi)").pack(anchor="w", padx=8, pady=(4, 0))
        self.person_tree = self._tree(tele, [("ID", 40), ("Risk", 80), ("Angle", 56), ("Vy", 56), ("Inactive", 62)], height=3)
        self.person_tree.pack(fill="x", padx=8, pady=(0, 8))

        # --- bottom: incidents + snapshots ---
        bottom = ctk.CTkFrame(self.outer_pane, corner_radius=10)
        self.outer_pane.add(self.top_pane, minsize=300, sticky="nsew", stretch="always")
        self.outer_pane.add(bottom, minsize=140, sticky="nsew", stretch="never")
        self.after(150, self._init_sashes)
        bottom.grid_columnconfigure(0, weight=3)
        bottom.grid_columnconfigure(1, weight=2)
        ctk.CTkLabel(bottom, text="📋 Incident History").grid(row=0, column=0, sticky="w", padx=12, pady=(6, 0))
        ctk.CTkLabel(bottom, text="📸 Recent Alert Snapshots").grid(row=0, column=1, sticky="w", padx=12, pady=(6, 0))
        self.log_tree = self._tree(
            bottom, ["#", "Time", "Person", "Event", "Confidence",
                     "Angle", "Speed", "BBox AR", "Note"], height=5)
        self.log_tree.grid(row=1, column=0, padx=12, pady=6, sticky="ew")
        btn_row = ctk.CTkFrame(bottom, fg_color="transparent")
        btn_row.grid(row=2, column=0, padx=12, pady=(0, 8), sticky="w")
        ctk.CTkButton(btn_row, text="📥 Export CSV", width=130, command=self._export_csv).pack(side="left", padx=(0, 8))
        ctk.CTkButton(btn_row, text="🗑️ Clear Log", width=130, command=self._clear_log).pack(side="left")
        self.snap_frame = ctk.CTkScrollableFrame(bottom, height=120, orientation="horizontal")
        self.snap_frame.grid(row=1, column=1, rowspan=2, padx=12, pady=6, sticky="ew")

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Treeview", background="#1e2430", fieldbackground="#1e2430",
                        foreground="#f8fafc", rowheight=22, borderwidth=0)
        style.configure("Treeview.Heading", background="#2d3748", foreground="#f8fafc", relief="flat")

    WIN_W, WIN_H = 1320, 820  # must match geometry() in __init__

    def _init_sashes(self, tries: int = 0):
        """Place dividers at sensible proportional defaults (user can drag after).

        Proportional (not absolute-pixel) targets so placement is correct under
        any OS display scaling. Every target is clamped to the *current*
        allocation: placing a sash beyond it makes Tk grow the whole window.
        Retries until layout is settled.
        """
        try:
            self.update_idletasks()
            aw = self.top_pane.winfo_width()
            ah = self.outer_pane.winfo_height()
            if (aw < 800 or ah < 500) and tries < 10:
                self.after(250, lambda: self._init_sashes(tries + 1))
                return
            if aw >= 800:
                s0 = max(220, min(int(aw * 0.24), aw - 360 - 258))   # sidebar | video
                s1 = max(s0 + 228, min(int(aw * 0.77), aw - 258))    # video | telemetry
                self.top_pane.sash_place(0, s0, 0)
                self.top_pane.sash_place(1, s1, 0)
            if ah >= 500:
                vy = max(300, min(int(ah * 0.72), ah - 148))         # top | bottom log
                self.outer_pane.sash_place(0, 0, vy)
            self.geometry(f"{self.WIN_W}x{self.WIN_H}")
        except Exception:
            pass

    def _slider_row(self, parent, title, lo, hi, default, param_key, is_int=False):
        ctk.CTkLabel(parent, text=title).pack(anchor="w", padx=8, pady=(8, 0))
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8)
        val_label = ctk.CTkLabel(row, text="", width=52)
        val_label.pack(side="right")

        def _fmt(v):
            return str(int(round(v))) if is_int else f"{v:.2f}"

        def _on_change(v):
            v = float(v)
            val_label.configure(text=_fmt(v))
            if param_key:
                self.params[param_key] = int(round(v)) if is_int else v

        slider = ctk.CTkSlider(parent, from_=lo, to=hi, command=_on_change)
        slider.set(default)
        slider.pack(fill="x", padx=8)
        val_label.configure(text=_fmt(default))
        if param_key:
            self.params[param_key] = int(round(default)) if is_int else default
        slider.is_int_slider = is_int
        return slider, val_label

    def _metric_card(self, parent, title):
        card = ctk.CTkFrame(parent, corner_radius=8)
        card.pack(fill="x", padx=12, pady=4)
        ctk.CTkLabel(card, text=title.upper(), text_color="gray",
                     font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=10, pady=(6, 0))
        val = ctk.CTkLabel(card, text="—", font=ctk.CTkFont(size=18, weight="bold"))
        val.pack(anchor="w", padx=10, pady=(0, 6))
        return val

    @staticmethod
    def _tree(parent, columns, height=5, default_width=90):
        # columns: ["name", ...] or [("name", width), ...]
        names, widths = [], []
        for c in columns:
            if isinstance(c, (tuple, list)):
                names.append(c[0])
                widths.append(c[1])
            else:
                names.append(c)
                widths.append(default_width)
        tree = ttk.Treeview(parent, columns=names, show="headings", height=height)
        for c, w in zip(names, widths):
            tree.heading(c, text=c)
            tree.column(c, width=w, minwidth=w, anchor="center")
        return tree

    # ================= sidebar actions =================

    def _apply_preset(self, name):
        angle, vel, timeout = PRESETS.get(name, PRESETS["Standard (Balanced)"])
        self.angle_slider.set(angle)
        self.vel_slider.set(vel)
        self.timeout_slider.set(timeout)
        self.angle_val.configure(text=f"{angle:.2f}")
        self.vel_val.configure(text=f"{vel:.2f}")
        self.timeout_val.configure(text=f"{timeout:.2f}")
        self.params.update({"angle": angle, "vel": vel, "timeout": timeout})

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Choose video file",
            filetypes=[("Video", "*.mp4 *.avi *.mov"), ("All files", "*.*")])
        if path:
            self._video_path = path
            self.file_label.configure(text=os.path.basename(path))
            self.source_menu.set("Video File")
            self._mark_manual()
            self._refresh_source_rows()

    def _export_csv(self):
        if not self.logger.events:
            messagebox.showinfo("Export CSV", "No incidents recorded yet.")
            return
        path = filedialog.asksaveasfilename(title="Save incident log",
                                            defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv")])
        if path:
            df = pd.DataFrame(list(self.logger.events))
            df.to_csv(path, index=False)
            messagebox.showinfo("Export CSV", f"Saved {len(df)} incidents to:\n{path}")

    def _clear_log(self):
        self.logger.clear()
        self._last_event_count = 0
        for i in self.log_tree.get_children():
            self.log_tree.delete(i)
        for w in self.snap_frame.winfo_children():
            w.destroy()
        self._thumbs.clear()

    # ================= camera sources =================

    def _refresh_source_rows(self):
        kind = self.source_menu.get()
        self._show_row(self.usb_row, kind == "USB Camera")
        self._show_row(self.rtsp_row, kind == "IP Camera (RTSP)")
        self._show_row(self.rtsp_status, kind == "IP Camera (RTSP)")
        self._show_row(self.file_row, kind == "Video File")
        self._show_row(self.file_label, kind == "Video File")
        self._show_row(self.synth_label, kind == "Synthetic Demo")

    @staticmethod
    def _show_row(widget, show):
        if show:
            widget.grid()
        else:
            widget.grid_remove()

    def _on_source_type_changed(self, _=None):
        self._mark_manual()
        self._refresh_source_rows()

    def _mark_manual(self):
        if self.profile_name is not None:
            self.profile_name = None
            self.profile_hint.configure(text="Manual setup (unsaved).")

    def _rescan(self):
        self.cam_menu.configure(values=["Scanning for USB cameras…"])
        self.cam_menu.set("Scanning for USB cameras…")
        threading.Thread(target=self._probe_worker, daemon=True).start()

    def _probe_worker(self):
        try:
            found = probe_usb_cameras()
        except Exception:
            found = []
        try:
            self.after(0, lambda: self._probe_done(found))
        except RuntimeError:
            pass

    def _probe_done(self, found):
        self._usb_cams = [c for c in found if c.get("ok")]
        if self._usb_cams:
            self.cam_menu.configure(values=[describe_camera(c) for c in self._usb_cams])
            self.cam_menu.set(describe_camera(self._usb_cams[0]))
        else:
            self.cam_menu.configure(values=["No USB cameras found"])
            self.cam_menu.set("No USB cameras found")

    def _selected_usb_index(self):
        m = re.match(r"Camera (\d+)", self.cam_menu.get() or "")
        if not m:
            return None
        idx = int(m.group(1))
        return idx if any(c["index"] == idx for c in self._usb_cams) else None

    def _test_rtsp(self):
        url = self.rtsp_entry.get().strip()
        ok, msg = validate_network_url(url)
        if not ok:
            self.rtsp_status.configure(text=f"✗ {msg}", text_color="#ef4444")
            return
        self._mark_manual()
        self.rtsp_status.configure(text="Testing… (up to ~10s)", text_color="gray")
        threading.Thread(target=self._rtsp_worker, args=(url,), daemon=True).start()

    def _rtsp_worker(self, url):
        cap = None
        try:
            cap = open_capture(url, "rtsp")
            if not cap.isOpened():
                raise RuntimeError("open failed")
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("connected but got no frames")
            h, w = frame.shape[:2]
            msg, good = f"✓ Stream OK — {w}x{h}", True
        except Exception as e:
            msg, good = f"✗ {e} — check URL / credentials / network", False
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
        try:
            self.after(0, lambda: self._rtsp_done(msg, good))
        except RuntimeError:
            pass

    def _rtsp_done(self, msg, good):
        self.rtsp_status.configure(text=msg, text_color="#34d399" if good else "#ef4444")

    def _current_source_dict(self, name):
        choice = self.source_menu.get()
        kind = {"USB Camera": "usb", "IP Camera (RTSP)": "rtsp"}.get(choice, "file")
        d: dict = {"name": name, "kind": kind}
        if kind == "usb":
            idx = self._selected_usb_index()
            d["index"] = idx if idx is not None else 0
        elif kind == "rtsp":
            d["url"] = self.rtsp_entry.get().strip()
        elif choice == "Synthetic Demo":
            d["path"] = os.path.join("assets", "synthetic_fall_demo.mp4")
        else:
            d["path"] = self._video_path or ""
        return d

    def _refresh_profile_menu(self):
        names = self.profiles.names()
        if not names:
            self.profile_menu.configure(values=["No saved cameras"])
            self.profile_menu.set("No saved cameras")
            return
        self.profile_menu.configure(values=names)
        self.profile_menu.set(self.profile_name if self.profile_name in names else names[0])

    def _save_profile(self):
        name = self.profile_entry.get().strip() or "Camera"
        existing = set(self.profiles.names())
        i, base = 2, name
        while name in existing:
            name, i = f"{base} {i}", i + 1
        try:
            self.profiles.upsert(self._current_source_dict(name))
        except ValueError as e:
            messagebox.showwarning("Save Camera", str(e))
            return
        self.profiles.active = name
        self.profiles.save()
        self.profile_name = name
        self.profile_entry.delete(0, "end")
        self._refresh_profile_menu()
        self.profile_hint.configure(text=f"Saved '{name}'.")

    def _delete_profile(self):
        name = self.profile_menu.get()
        if name in self.profiles.names():
            self.profiles.delete(name)
            if self.profile_name == name:
                self.profile_name = None
            self._refresh_profile_menu()
            self.profile_hint.configure(text="Profile deleted.")

    def _apply_profile(self, name):
        prof = self.profiles.get(name)
        if prof is None:
            return
        kind = prof.get("kind", "usb")
        self.source_menu.set({"usb": "USB Camera", "rtsp": "IP Camera (RTSP)"}.get(kind, "Video File"))
        if kind == "usb":
            self._rescan()
        elif kind == "rtsp":
            self.rtsp_entry.delete(0, "end")
            self.rtsp_entry.insert(0, prof.get("url", ""))
        else:
            self._video_path = prof.get("path", "") or None
            self.file_label.configure(
                text=os.path.basename(self._video_path) if self._video_path else "No file chosen")
        self.profile_name = name
        self.profiles.active = name
        self.profiles.save()
        self._refresh_source_rows()
        self.profile_hint.configure(text=f"Camera: {name} ({self.profiles.describe(prof)}).")

    # ================= run control =================

    def _resolve_source(self):
        """Returns (source, kind, label) or None. kind in webcam|rtsp|file."""
        choice = self.source_menu.get()
        if choice == "USB Camera":
            idx = self._selected_usb_index()
            if idx is None:
                messagebox.showwarning("Camera", "No working USB camera selected — press ⟳ to rescan.")
                return None
            return idx, "webcam", f"USB Camera {idx}"
        if choice == "IP Camera (RTSP)":
            url = self.rtsp_entry.get().strip()
            ok, msg = validate_network_url(url)
            if not ok:
                messagebox.showwarning("Camera", f"Bad stream URL: {msg}")
                return None
            host = urlparse(url).hostname or url[:32]
            return url, "rtsp", f"RTSP {host}"
        if choice == "Synthetic Demo":
            demo = os.path.join("assets", "synthetic_fall_demo.mp4")
            if not os.path.exists(demo):
                self.status_label.configure(text="Rendering synthetic demo…")
                self.update_idletasks()
                generate_synthetic_fall_video(demo)
            return demo, "file", "Synthetic Demo"
        if not self._video_path or not os.path.exists(self._video_path):
            messagebox.showwarning("Source", "Pick a video file first (Browse…).")
            return None
        return self._video_path, "file", os.path.basename(self._video_path)

    def _do_ack(self):
        """Silence repeating alarm beeps for 10 min; visuals/logs keep running.
        Re-arms automatically when the alarm clears (next incident sounds)."""
        self._ack.ack(time.time())
        self.status_label.configure(text="Alarm acknowledged — beeps silenced for 10 min (visual alert active)")

    def start(self):
        if self._running:
            return
        resolved = self._resolve_source()
        if resolved is None:
            return
        source, kind, label = resolved
        if self.profile_name:
            label = f"{self.profile_name} • {label}"
        multi = self.mode_menu.get().startswith("Multi")
        try:
            max_persons = int(self.maxp_menu.get())
        except ValueError:
            max_persons = 3
        cfg = {
            "source": source,
            "kind": kind,
            "label": label,
            "multi": multi,
            "max_persons": max(1, min(3, max_persons)),
            "detect_interval": int(round(float(self.det_slider.get()))),
            "complexity": 0 if self.pose_menu.get() == "Fast" else 1,
        }
        # Snapshot live params
        self.params.update({
            "angle": float(self.angle_slider.get()),
            "vel": float(self.vel_slider.get()),
            "timeout": float(self.timeout_slider.get()),
            "audio": bool(self.audio_var.get()),
            "skeleton": bool(self.skel_var.get()),
            "bbox": bool(self.bbox_var.get()),
        })
        self._stop_event.clear()
        while not self._packets.empty():
            try:
                self._packets.get_nowait()
            except queue.Empty:
                break
        self._worker = threading.Thread(target=self._pipeline_loop, args=(cfg,), daemon=True)
        self._running = True
        self._worker.start()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(text=f"Running ({'multi' if multi else 'single'})…")

    def stop(self):
        if not self._running:
            return
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=5)
            self._worker = None
        self._running = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="Idle")

    def _on_close(self):
        try:
            self._stop_event.set()
            if self._worker is not None:
                self._worker.join(timeout=5)
        finally:
            self.destroy()

    # ================= worker =================

    def _push(self, packet):
        try:
            self._packets.put_nowait(packet)
        except queue.Full:
            try:
                self._packets.get_nowait()
            except queue.Empty:
                pass
            try:
                self._packets.put_nowait(packet)
            except queue.Full:
                pass

    def _pipeline_loop(self, cfg):
        try:
            pump = FramePump(cfg["source"], kind=cfg.get("kind", "auto"),
                             on_status=lambda m: self._push({"status": m}),
                             abort=self._stop_event.is_set)
        except SourceLost as e:
            self._push({"error": str(e)})
            self._running = False
            return

        visualizer = Visualizer()
        last_beep = 0.0
        fps, t_prev = 0.0, time.time()
        logged_fall_count = 0
        logged_inactive = False
        logged_falls: dict = {}
        logged_inactive_ids: set = set()

        if cfg["multi"]:
            mgr = MultiPersonManager(
                max_persons=cfg["max_persons"],
                angle_threshold=self.params["angle"],
                velocity_threshold=self.params["vel"],
                inactivity_timeout=self.params["timeout"],
                detect_interval=cfg["detect_interval"],
                model_complexity=cfg["complexity"])
            single = None
        else:
            mgr = None
            single = {
                "pose": PoseEstimator(model_complexity=cfg["complexity"]),
                "ext": FeatureExtractor(),
                "fall": FallDetector(angle_threshold=self.params["angle"],
                                     velocity_threshold=self.params["vel"]),
                "inact": InactivityMonitor(inactivity_timeout=self.params["timeout"]),
            }

        try:
            for frame in pump.frames(alive=lambda: not self._stop_event.is_set()):
                now = time.time()
                dt = max(1e-3, now - t_prev)
                t_prev = now
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

                # Live-tunable params (thresholds + overlays + audio)
                angle_t, vel_t, timeout_t = self.params["angle"], self.params["vel"], self.params["timeout"]
                show_skel, show_bbox = self.params["skeleton"], self.params["bbox"]

                if cfg["multi"]:
                    mgr.update_parameters(angle_threshold=angle_t, velocity_threshold=vel_t,
                                          inactivity_timeout=timeout_t)
                    persons = mgr.process(frame, current_time=now)
                    for p in persons:
                        feats, fs, ins = p["features"], p["fall_status"], p["inactivity_status"]
                        if feats is None:
                            continue
                        tid = p["track_id"]
                        if fs.get("state") == "FALLEN" and fs.get("total_falls", 0) > logged_falls.get(tid, 0):
                            logged_falls[tid] = fs["total_falls"]
                            self.logger.log_event("FALL_DETECTED", feats,
                                                  fs.get("fall_confidence", 0.8), frame,
                                                  person_id=tid)
                        if ins.get("is_inactive_alert") and tid not in logged_inactive_ids:
                            logged_inactive_ids.add(tid)
                            self.logger.log_event(
                                "INACTIVITY_EMERGENCY", feats, 1.0, frame,
                                extra_note=f"P{tid} stationary for {ins.get('inactive_duration'):.1f}s",
                                person_id=tid)
                        elif not ins.get("is_inactive_alert") and tid in logged_inactive_ids:
                            logged_inactive_ids.discard(tid)
                    tids = {p["track_id"] for p in persons}
                    logged_falls = {k: v for k, v in logged_falls.items() if k in tids}
                    logged_inactive_ids &= tids
                    display = visualizer.draw_hud_multi(frame, persons,
                                                        show_skeleton=show_skel, show_bbox=show_bbox)
                    worst = self._worst(persons)
                    if worst is not None and worst.get("features"):
                        risk = worst["fall_status"].get("risk_level", "NORMAL")
                        if worst["inactivity_status"].get("is_inactive_alert"):
                            risk = "EMERGENCY"
                        pkt = {"risk": risk,
                               "angle": worst["features"].get("spine_angle", 0.0),
                               "vy": worst["features"].get("vertical_velocity", 0.0),
                               "inact": worst["inactivity_status"].get("inactive_duration", 0.0),
                               "is_inact": worst["inactivity_status"].get("is_inactive_alert", False),
                               "persons": persons}
                    else:
                        pkt = {"risk": "NORMAL", "angle": 0.0, "vy": 0.0,
                               "inact": 0.0, "is_inact": False, "persons": persons}
                    pkt["backend"] = mgr.detector.backend
                else:
                    single["fall"].update_parameters(angle_threshold=angle_t, velocity_threshold=vel_t)
                    single["inact"].update_parameters(inactivity_timeout=timeout_t)
                    has_pose, _, kp = single["pose"].process_frame(frame)
                    feats, fs = None, {"state": "NORMAL", "risk_level": "NORMAL",
                                       "fall_confidence": 0.0,
                                       "total_falls": single["fall"].total_falls_detected}
                    ins = {"is_inactive_alert": False, "inactive_duration": 0.0}
                    if has_pose and kp:
                        feats = single["ext"].extract_features(kp, current_time=now)
                        tmp = {"state": single["fall"].current_state.value,
                               "is_horizontal": feats.get("spine_angle", 0) >= angle_t}
                        ins = single["inact"].process(feats, tmp)
                        fs = single["fall"].process(
                            feats, is_inactive=ins.get("is_inactive_alert", False))
                        if fs.get("state") == "FALLEN" and fs.get("total_falls", 0) > logged_fall_count:
                            logged_fall_count = fs["total_falls"]
                            self.logger.log_event("FALL_DETECTED", feats,
                                                  fs.get("fall_confidence", 0.8), frame)
                        if ins.get("is_inactive_alert") and not logged_inactive:
                            logged_inactive = True
                            self.logger.log_event(
                                "INACTIVITY_EMERGENCY", feats, 1.0, frame,
                                extra_note=f"Stationary for {ins.get('inactive_duration'):.1f}s")
                        elif not ins.get("is_inactive_alert"):
                            logged_inactive = False
                    display = visualizer.draw_hud(frame, kp if has_pose else None, feats, fs, ins,
                                                  show_skeleton=show_skel, show_bbox=show_bbox)
                    pkt = {"risk": fs.get("risk_level", "NORMAL"),
                           "angle": feats.get("spine_angle", 0.0) if feats else 0.0,
                           "vy": feats.get("vertical_velocity", 0.0) if feats else 0.0,
                           "inact": ins.get("inactive_duration", 0.0),
                           "is_inact": ins.get("is_inactive_alert", False),
                           "persons": [], "backend": "mediapipe"}

                alarm_active = pkt["risk"] in ("CONCERNING", "EMERGENCY")
                self._ack.rearm_if_cleared(alarm_active, now)
                if self.params["audio"] and alarm_active and not self._ack.muted(now):
                    if now - last_beep > 1.5:
                        _beep()
                        last_beep = now

                # Display-size frame (keeps queue light)
                disp_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                h, w = disp_rgb.shape[:2]
                scale = min(1.0, 640.0 / max(1, w))
                if scale < 1.0:
                    disp_rgb = cv2.resize(disp_rgb, (int(w * scale), int(h * scale)))
                pkt.update({"frame": disp_rgb, "fps": fps,
                            "events": len(self.logger.events),
                            "source_label": cfg.get("label", "")})
                self._push(pkt)
        except SourceLost as e:
            if not self._stop_event.is_set():
                self._push({"error": str(e)})
        except Exception as e:  # never kill silently; surface in status bar
            self._push({"error": str(e)})
        finally:
            pump.release()
            try:
                if mgr is not None:
                    mgr.close()
                else:
                    single["pose"].close()
            except Exception:
                pass
            self._running = False

    @staticmethod
    def _worst(persons):
        if not persons:
            return None

        def _risk(p):
            r = p.get("fall_status", {}).get("risk_level", "NORMAL")
            if p.get("inactivity_status", {}).get("is_inactive_alert", False):
                return "EMERGENCY"
            return r

        return max(persons, key=lambda p: RISK_ORDER.get(_risk(p), 0))

    # ================= UI poll =================

    def _poll(self):
        # The after() chain is re-armed in `finally`: no UI/render error may
        # ever freeze the display loop (that failure looks like "monitoring stopped").
        try:
            # Sync live controls -> shared params (worker picks these up per frame)
            self.params.update({
                "angle": float(self.angle_slider.get()),
                "vel": float(self.vel_slider.get()),
                "timeout": float(self.timeout_slider.get()),
                "audio": bool(self.audio_var.get()),
                "skeleton": bool(self.skel_var.get()),
                "bbox": bool(self.bbox_var.get()),
            })
            pkt = None
            try:
                while True:  # drain, keep newest
                    pkt = self._packets.get_nowait()
            except queue.Empty:
                pass
            if pkt is not None:
                if "error" in pkt:
                    self.status_label.configure(text=f"Error: {pkt['error']}")
                    self.stop()
                elif "status" in pkt and "frame" not in pkt:
                    self.status_label.configure(text=pkt["status"])
                else:
                    try:
                        self._render_packet(pkt)
                    except Exception as e:
                        self.status_label.configure(text=f"Display error (monitoring continues): {e}")
            # If the worker died on its own (source lost), restore the buttons
            if (self._worker is not None and not self._worker.is_alive()
                    and self.start_btn.cget("state") == "disabled"):
                self._worker = None
                self._running = False
                self.start_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")
                if "Error" not in self.status_label.cget("text"):
                    self.status_label.configure(text="Stopped (source ended)")
        except Exception as e:
            try:
                self.status_label.configure(text=f"UI error (monitoring continues): {e}")
            except Exception:
                pass
        finally:
            self.after(33, self._poll)

    def _render_packet(self, pkt):
        img = Image.fromarray(pkt["frame"])
        self._photo = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.video_label.configure(image=self._photo, text="")

        risk = pkt.get("risk", "NORMAL")
        color = RISK_COLORS.get(risk, "#f8fafc")
        inact = pkt.get("inact", 0.0)
        self.card_risk.configure(
            text=f"{ICONS.get(risk, '')} "
                 f"{RISK_TEXT.get(risk, risk)}" + (f"  {inact:.1f}s" if risk == "EMERGENCY" else ""),
            text_color=color)
        self.card_angle.configure(text=f"{pkt.get('angle', 0.0):.1f}°")
        self.card_vy.configure(text=f"{pkt.get('vy', 0.0):+.2f}")
        self.card_timer.configure(text=f"{inact:.1f}s",
                                  text_color=RISK_COLORS["EMERGENCY"] if pkt.get("is_inact") else "#f8fafc")

        n_p = len(pkt.get("persons", []))
        extra = f"  •  {n_p} person(s) [{pkt.get('backend', '')}]" if n_p or pkt.get("backend") == "builtin" else ""
        src_txt = f"  •  {pkt['source_label']}" if pkt.get("source_label") else ""
        self.status_label.configure(text=f"{pkt.get('fps', 0.0):.1f} FPS{extra}{src_txt}")

        # chart (every packet is fine; canvas redraw is cheap)
        self._chart_angle.append(pkt.get("angle", 0.0))
        self._chart_vy.append(pkt.get("vy", 0.0) * 100.0)
        if len(self._chart_angle) > 60:
            self._chart_angle = self._chart_angle[-60:]
            self._chart_vy = self._chart_vy[-60:]
        self._draw_chart()

        # per-person rows
        for i in self.person_tree.get_children():
            self.person_tree.delete(i)
        for p in pkt.get("persons", []):
            fs, ins, feats = p["fall_status"], p["inactivity_status"], p["features"]
            r = fs.get("risk_level", "NORMAL")
            if ins.get("is_inactive_alert"):
                r = "EMERGENCY"
            self.person_tree.insert("", "end", values=(
                f"P{p['track_id']}", r,
                f"{feats.get('spine_angle', 0.0):.1f}°" if feats else "-",
                f"{feats.get('vertical_velocity', 0.0):+.2f}" if feats else "-",
                f"{ins.get('inactive_duration', 0.0):.1f}s"))

        # incident log + snapshots on new events
        if pkt.get("events", 0) != self._last_event_count:
            self._last_event_count = pkt["events"]
            self._refresh_log()
            self._refresh_snaps()

    def _draw_chart(self):
        c = self.chart
        c.delete("all")
        W = c.winfo_width() or 300
        H = c.winfo_height() or 170
        mid = H // 2
        c.create_line(0, mid, W, mid, fill="#374151")
        # angle threshold line
        thr = float(self.angle_slider.get())
        ty = (H - 8) - (min(90.0, max(0.0, thr)) / 90.0) * (H - 16)
        c.create_line(0, ty, W, ty, fill="#ef4444", dash=(4, 3))
        n = len(self._chart_angle)
        if n < 2:
            return

        def _xs(i):
            return 4 + i * (W - 8) / 59

        pts_a = [(_xs(i + max(0, 60 - n)), (H - 8) - (min(90.0, max(0.0, a)) / 90.0) * (H - 16))
                 for i, a in enumerate(self._chart_angle[-60:])]
        pts_v = [(_xs(i + max(0, 60 - n)), mid - (min(100.0, max(-100.0, v)) / 100.0) * (mid - 8))
                 for i, v in enumerate(self._chart_vy[-60:])]
        for pts, col in ((pts_a, "#60a5fa"), (pts_v, "#fbbf24")):
            for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
                c.create_line(x1, y1, x2, y2, fill=col, width=2)
        c.create_text(8, 10, text="— angle (°)", fill="#60a5fa", anchor="w")
        c.create_text(8, 24, text="— Vy ×100", fill="#fbbf24", anchor="w")

    def _refresh_log(self):
        for i in self.log_tree.get_children():
            self.log_tree.delete(i)
        for ev in list(self.logger.events)[-200:]:
            self.log_tree.insert("", "end", values=(
                ev.get("id"), ev.get("timestamp"), ev.get("person_id"), ev.get("event_type"),
                ev.get("confidence"), ev.get("spine_angle"), ev.get("vertical_velocity"),
                ev.get("aspect_ratio"), ev.get("note", "")))

    def _refresh_snaps(self):
        for w in self.snap_frame.winfo_children():
            w.destroy()
        self._thumbs.clear()
        if not os.path.isdir("alerts"):
            return
        files = sorted((f for f in os.listdir("alerts") if f.endswith(".jpg")),
                       key=lambda x: os.path.getmtime(os.path.join("alerts", x)),
                       reverse=True)[:6]
        for fn in files:
            try:
                im = Image.open(os.path.join("alerts", fn))
                im.thumbnail((150, 95))
                ph = ctk.CTkImage(light_image=im, dark_image=im, size=im.size)
                self._thumbs.append(ph)
                lbl = ctk.CTkLabel(self.snap_frame, image=ph, text=fn,
                                   compound="top", wraplength=150,
                                   font=ctk.CTkFont(size=10))
                lbl.pack(side="left", padx=6, pady=4)
            except Exception:
                continue


if __name__ == "__main__":
    ArugaDesktopApp().mainloop()
