"""
ARUGA Hallway — self-contained multi-person fall monitoring for clinic
hallways / waiting areas. Runs 100% on one laptop after setup (then airgappable).

Engine: ONE YOLOv8n-pose ONNX inference per frame (all bodies at once) via
ONNX Runtime. EP auto-selected at startup: CUDA > DirectML (AMD/Intel iGPU)
> CPU, so the same code scales from a Ryzen 3 laptop to a gaming rig —
see the backend label in the status bar.

New vs the prototype apps: zone-aware risk (floor = alarm, bench = RESTING
unless a sudden SLUMP, ignore = capped), N-person tracking (default 6),
FPS-first 480p-friendly pipeline.

Run:  .\\.venv\\Scripts\\python.exe hallway_app.py   (or run_hallway.bat)
Model + zones ship with the repo: assets/models/yolov8n-pose.onnx (AGPL-3.0
Ultralytics weights — fine for internal eval; review licensing before any
commercial deployment), assets/zones.json (created via Calibrate Zones).
"""

import os
import queue
import re
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import customtkinter as ctk
from PIL import Image

from core.hallway_manager import HallwayManager
from core.zones import ZoneManager, ZONE_COLORS
from core.camera_sources import (
    CameraProfiles, zones_path_for, validate_network_url,
    probe_usb_cameras, describe_camera,
)
from core.frame_pump import FramePump, SourceLost
from utils.hallway_overlay import draw_hallway_hud
from utils.logger import EventLogger
from utils.beep import beep as _beep, AlarmAck
from utils.risk_palette import RISK_HEX, ICONS


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Risk colors: shared palette (utils/risk_palette) — EMERGENCY red, CONCERNING deep orange.
RISK_TEXT = {"NORMAL": "NORMAL — ROUTINE MOBILITY", "RESTING": "RESTING IN SEATING",
             "UNUSUAL": "UNUSUAL — UNSTABLE POSTURE", "SLUMP": "SLUMP — CHECK ON PERSON",
             "CONCERNING": "CONCERNING — FALL DETECTED", "EMERGENCY": "EMERGENCY — INACTIVE"}
PRESETS = {"Standard (Balanced)": (58.0, 0.32, 6.0),
           "High Sensitivity (Elderly Care)": (50.0, 0.22, 4.0),
           "Low Sensitivity (Active / Visitors)": (68.0, 0.45, 10.0)}

# tkinter Canvas can't show CTkImage reliably across versions; store PhotoImage refs.
from PIL import ImageTk as _ImageTk


class ZoneEditor(ctk.CTkToplevel):
    """Click-to-draw polygon calibration on a captured background frame."""

    def __init__(self, master, background_bgr, zones: ZoneManager,
                 save_path: str, profile_name: str, on_apply):
        super().__init__(master)
        self.title(f"Calibrate Zones [{profile_name}] — click points, then Add Zone")
        self.geometry("1000x700")
        self._zones = zones
        self._save_path = save_path
        self._on_apply = on_apply
        self._pts: list = []
        self._photo = None

        h, w = background_bgr.shape[:2]
        scale = min(940 / w, 600 / h, 1.0)
        dw, dh = int(w * scale), int(h * scale)
        rgb = cv2.cvtColor(cv2.resize(background_bgr, (dw, dh)), cv2.COLOR_BGR2RGB)
        self._dw, self._dh = dw, dh
        self._photo = _ImageTk.PhotoImage(Image.fromarray(rgb))

        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=10, pady=6)
        self.type_menu = ctk.CTkOptionMenu(top, values=["bench", "floor", "ignore"])
        self.type_menu.pack(side="left", padx=4)
        self.name_entry = ctk.CTkEntry(top, placeholder_text="zone name (auto if empty)", width=220)
        self.name_entry.pack(side="left", padx=4)
        ctk.CTkButton(top, text="Add Zone", width=100, command=self._add).pack(side="left", padx=4)
        ctk.CTkButton(top, text="Undo Point", width=100, command=self._undo).pack(side="left", padx=4)
        ctk.CTkButton(top, text="Capture New BG", width=130, command=self._recapture).pack(side="left", padx=4)
        self._master_ref = master

        self.canvas = tk.Canvas(self, width=dw, height=dh, bg="black", highlightthickness=0)
        self.canvas.pack(padx=10, pady=4)
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo, tags="bg")
        self.canvas.bind("<Button-1>", self._click)

        bot = ctk.CTkFrame(self, fg_color="transparent")
        bot.pack(fill="x", padx=10, pady=6)
        self.zone_list = tk.Listbox(bot, height=5)
        self.zone_list.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(bot, text="Delete Selected", width=130, command=self._delete).pack(side="left", padx=4)
        ctk.CTkButton(bot, text="Save + Apply", width=130, command=self._save).pack(side="left", padx=4)
        self._refresh_list()
        self._redraw()

    def _click(self, ev):
        self._pts.append((ev.x / self._dw, ev.y / self._dh))
        self._redraw()

    def _undo(self):
        self._pts = self._pts[:-1]
        self._redraw()

    def _redraw(self):
        self.canvas.delete("poly")
        for z in self._zones.zones:
            if z.get("name") == "default_floor" or len(z.get("polygon", [])) < 3:
                continue
            pts = [(x * self._dw, y * self._dh) for x, y in z["polygon"]]
            flat = [c for p in pts for c in p]
            col = "#%02x%02x%02x" % tuple(reversed(ZONE_COLORS.get(z["type"], (200, 200, 200))))
            self.canvas.create_polygon(flat, outline=col, fill=col, stipple="gray25",
                                       width=2, tags="poly")
        if self._pts:
            flat = [c for p in self._pts for c in (p[0] * self._dw, p[1] * self._dh)]
            if len(self._pts) > 1:
                self.canvas.create_line(flat, fill="yellow", width=2, tags="poly")
            for (x, y) in self._pts:
                self.canvas.create_oval(x * self._dw - 4, y * self._dh - 4,
                                        x * self._dw + 4, y * self._dh + 4,
                                        fill="yellow", tags="poly")

    def _add(self):
        if len(self._pts) < 3:
            messagebox.showwarning("Add Zone", "Click at least 3 points first.")
            return
        ztype = self.type_menu.get()
        name = self.name_entry.get().strip() or f"{ztype}_{len(self._zones.zones) + 1}"
        try:
            self._zones.add_zone(name, ztype, list(self._pts))
        except ValueError as e:
            messagebox.showwarning("Add Zone", str(e))
            return
        self._pts = []
        self.name_entry.delete(0, "end")
        self._refresh_list()
        self._redraw()

    def _delete(self):
        sel = list(self.zone_list.curselection())
        names = [z.get("name", "") for z in self._zones.zones]
        for i in sorted(sel, reverse=True):
            if i < len(names):
                self._zones.remove_zone(names[i])
        self._refresh_list()
        self._redraw()

    def _refresh_list(self):
        self.zone_list.delete(0, "end")
        for z in self._zones.zones:
            self.zone_list.insert("end", f"{z.get('name')}  [{z.get('type')}]")

    def _recapture(self):
        """Replace the background with the current live frame (keeps drawn points; normalized)."""
        app = self._master_ref
        with app._frame_lock:
            frame = None if app._last_frame is None else app._last_frame.copy()
        if frame is None:
            messagebox.showinfo("Capture", "No live frame yet — Start monitoring first.")
            return
        h, w = frame.shape[:2]
        scale = min(940 / w, 600 / h, 1.0)
        self._dw, self._dh = int(w * scale), int(h * scale)
        rgb = cv2.cvtColor(cv2.resize(frame, (self._dw, self._dh)), cv2.COLOR_BGR2RGB)
        self._photo = _ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.configure(width=self._dw, height=self._dh)
        self.canvas.delete("bg")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo, tags="bg")
        self._redraw()

    def _save(self):
        self._zones.save(self._save_path)
        self._on_apply()
        messagebox.showinfo("Zones", f"Saved {len(self._zones.zones)} zone(s) to\n{self._save_path}\nand applied.")
        self.destroy()


class HallwayApp(ctk.CTk):
    WIN_W, WIN_H = 1320, 820

    def __init__(self):
        super().__init__()
        self.title("ARUGA Hallway — Multi-Person Fall Monitor")
        self.geometry(f"{self.WIN_W}x{self.WIN_H}")
        self.minsize(1100, 700)

        self.logger = EventLogger(output_dir="alerts_hallway")
        self.profiles = CameraProfiles()
        self.profile_name = self.profiles.active  # None = unsaved/manual setup
        self.zones = ZoneManager()
        self._load_zones_for_profile()
        self._worker = None
        self._stop_event = threading.Event()
        self._ack = AlarmAck()
        self._packets: queue.Queue = queue.Queue(maxsize=2)
        self._running = False
        self._last_event_count = 0
        self._chart_angle: list = []
        self._chart_vy: list = []
        self._photo = None
        self._thumbs: list = []
        self._video_path = None
        self._frame_lock = threading.Lock()
        self._last_frame = None
        self._editor = None
        self.params = {"angle": 58.0, "vel": 0.32, "timeout": 6.0,
                       "audio": True, "skeleton": True, "bbox": True, "show_zones": True,
                       "maxp": 6, "detconf": 0.45, "lowlight": False}

        self._build_layout()
        self._apply_preset("Standard (Balanced)")
        self._refresh_profile_menu()
        self._refresh_source_rows()
        self._rescan()  # background USB discovery on boot
        self.after(33, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ================= camera sources =================

    def _zone_path(self) -> str:
        return zones_path_for(self.profile_name or "default")

    def _load_zones_for_profile(self):
        path = self._zone_path()
        if os.path.exists(path) and self.zones.load(path):
            return f"Zones loaded ({os.path.basename(path)})."
        self.zones.clear()
        return "No calibration for this camera yet — use Calibrate Zones."

    # ================= layout =================

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, corner_radius=10)
        header.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="ew")
        ctk.CTkLabel(header, text="🏥 ARUGA Hallway",
                     font=ctk.CTkFont(size=22, weight="bold")).pack(side="left", padx=16, pady=10)
        ctk.CTkLabel(header, text="Clinic hallway / waiting-area multi-person monitor  •  100% on-device",
                     text_color="gray").pack(side="left", padx=8)
        self.status_label = ctk.CTkLabel(header, text="Idle — press Start", text_color="gray")
        self.status_label.pack(side="right", padx=16)
        self.ack_btn = ctk.CTkButton(
            header, text="🔕 Acknowledge (10 min)", width=170,
            command=lambda: self._do_ack())
        self.ack_btn.pack(side="right", padx=(0, 4))

        self.outer_pane = tk.PanedWindow(self, orient="vertical", background="#212121",
                                         sashwidth=8, sashrelief="flat", borderwidth=0)
        self.outer_pane.grid(row=1, column=0, padx=12, pady=(6, 12), sticky="nsew")
        self.top_pane = tk.PanedWindow(self.outer_pane, orient="horizontal", background="#212121",
                                       sashwidth=8, sashrelief="flat", borderwidth=0)

        # sidebar
        side_holder = ctk.CTkFrame(self.top_pane, fg_color="transparent")
        side_holder.pack_propagate(False)
        self.top_pane.add(side_holder, minsize=250, sticky="nsew", stretch="never")
        side = ctk.CTkScrollableFrame(side_holder, width=270, corner_radius=10,
                                      label_text="⚙️ Configuration")
        side.pack(fill="both", expand=True)

        ctk.CTkLabel(side, text="Sensitivity Preset").pack(anchor="w", padx=8, pady=(8, 0))
        self.preset_menu = ctk.CTkOptionMenu(side, values=list(PRESETS.keys()), command=self._apply_preset)
        self.preset_menu.pack(fill="x", padx=8, pady=4)
        self.angle_slider, _ = self._slider_row(side, "Spine Angle (°)", 40.0, 80.0, 58.0, "angle")
        self.vel_slider, _ = self._slider_row(side, "Downward Velocity", 0.15, 0.80, 0.32, "vel")
        self.timeout_slider, _ = self._slider_row(side, "Inactivity Timeout (s)", 3.0, 30.0, 6.0, "timeout")
        self.det_slider, self.det_val = self._slider_row(side, "Detector confidence", 0.25, 0.70, 0.45, None)
        self.maxp_slider, self.maxp_val = self._slider_row(side, "Max persons", 1, 10, 6, None, is_int=True)

        ctk.CTkLabel(side, text="🔔 Alerts & Overlays").pack(anchor="w", padx=8, pady=(10, 0))
        self.audio_var = ctk.BooleanVar(value=True)
        self.skel_var = ctk.BooleanVar(value=True)
        self.bbox_var = ctk.BooleanVar(value=True)
        self.zone_var = ctk.BooleanVar(value=True)
        self.lowlight_var = ctk.BooleanVar(value=False)
        for text, var in (("Enable Audio Warning", self.audio_var),
                          ("Render Skeleton Overlay", self.skel_var),
                          ("Render Telemetry Bounding Box", self.bbox_var),
                          ("Show Zone Overlays", self.zone_var),
                          ("Low-light boost (dim rooms)", self.lowlight_var)):
            ctk.CTkCheckBox(side, text=text, variable=var).pack(anchor="w", padx=8, pady=2)

        ctk.CTkLabel(side, text="📷 Camera Source (applies on Start)").pack(anchor="w", padx=8, pady=(10, 0))
        self.source_menu = ctk.CTkOptionMenu(
            side, values=["USB Camera", "IP Camera (RTSP)", "Video File"],
            command=self._on_source_type_changed)
        self.source_menu.pack(fill="x", padx=8, pady=4)

        # Source rows live in a dedicated grid container: show/hide via
        # grid/grid_remove preserves each row's position. (pack_forget +
        # re-pack would drop them to the bottom of the sidebar.)
        self.source_box = ctk.CTkFrame(side, fg_color="transparent")
        self.source_box.pack(fill="x", padx=0, pady=0)
        self.source_box.grid_columnconfigure(0, weight=1)

        # USB row: discovered camera dropdown + rescan
        self.usb_row = ctk.CTkFrame(self.source_box, fg_color="transparent")
        self.usb_row.grid(row=0, column=0, sticky="ew", padx=8, pady=2)
        self.cam_menu = ctk.CTkOptionMenu(self.usb_row, values=["Scanning for USB cameras…"])
        self.cam_menu.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkButton(self.usb_row, text="⟳", width=44, command=self._rescan).pack(side="right")
        self._usb_cams: list = []

        # RTSP row: URL entry + test + status
        self.rtsp_row = ctk.CTkFrame(self.source_box, fg_color="transparent")
        self.rtsp_row.grid(row=1, column=0, sticky="ew", padx=8, pady=2)
        self.rtsp_entry = ctk.CTkEntry(self.rtsp_row, placeholder_text="rtsp://user:pass@192.168.1.50:554/stream")
        self.rtsp_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkButton(self.rtsp_row, text="Test", width=60, command=self._test_rtsp).pack(side="right")
        self.rtsp_status = ctk.CTkLabel(self.source_box, text="", text_color="gray",
                                        wraplength=240, justify="left")
        self.rtsp_status.grid(row=2, column=0, sticky="w", padx=8)

        # File row: browse + label
        self.file_row = ctk.CTkFrame(self.source_box, fg_color="transparent")
        self.file_row.grid(row=3, column=0, sticky="ew", padx=8, pady=2)
        ctk.CTkButton(self.file_row, text="Browse…", width=90, command=self._browse).pack(side="left")
        self.file_label = ctk.CTkLabel(self.source_box, text="No file chosen", text_color="gray",
                                       wraplength=240, justify="left")
        self.file_label.grid(row=4, column=0, sticky="w", padx=8, pady=2)

        # Saved camera profiles: switch bodies clearly, each keeps its own zones.
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

        ctk.CTkButton(side, text="📐 Calibrate Zones…", command=self._open_editor).pack(fill="x", padx=8, pady=(8, 4))
        ctk.CTkButton(side, text="📸 Capture Background", command=self._capture_bg).pack(fill="x", padx=8, pady=(0, 4))
        self.start_btn = ctk.CTkButton(side, text="▶️ Start Monitoring", command=self.start)
        self.start_btn.pack(fill="x", padx=8, pady=(8, 4))
        self.stop_btn = ctk.CTkButton(side, text="⏹️ Stop / Reset", command=self.stop, state="disabled")
        self.stop_btn.pack(fill="x", padx=8, pady=(0, 12))

        # video
        video_frame = ctk.CTkFrame(self.top_pane, corner_radius=10)
        self.top_pane.add(video_frame, minsize=360, sticky="nsew", stretch="always")
        ctk.CTkLabel(video_frame, text="📺 Hallway Stream — zones + per-person HUD").pack(
            anchor="w", padx=12, pady=(8, 0))
        self.video_label = ctk.CTkLabel(video_frame, text="Press Start Monitoring")
        self.video_label.pack(expand=True, fill="both", padx=12, pady=12)

        # telemetry
        tele_holder = ctk.CTkFrame(self.top_pane, fg_color="transparent")
        tele_holder.pack_propagate(False)
        self.top_pane.add(tele_holder, minsize=250, sticky="nsew", stretch="never")
        tele = ctk.CTkScrollableFrame(tele_holder, width=270, corner_radius=10,
                                      label_text="📊 Live Telemetry")
        tele.pack(fill="both", expand=True)
        self.card_risk = self._metric_card(tele, "Worst Risk")
        self.card_persons = self._metric_card(tele, "Persons Tracked")
        self.card_angle = self._metric_card(tele, "Spine Inclination (worst)")
        self.card_vy = self._metric_card(tele, "Downward Speed (worst)")
        self.card_timer = self._metric_card(tele, "Inactivity (worst)")
        ctk.CTkLabel(tele, text="📈 Kinematic Trajectory").pack(anchor="w", padx=8, pady=(8, 0))
        self.chart = tk.Canvas(tele, width=250, height=150, bg="#1e2430", highlightthickness=0)
        self.chart.pack(fill="x", padx=8, pady=4)
        ctk.CTkLabel(tele, text="Per-person").pack(anchor="w", padx=8, pady=(4, 0))
        self.person_tree = self._tree(tele, [("ID", 36), ("Risk", 78), ("Zone", 70),
                                             ("Angle", 52), ("Still", 52)], height=5)
        self.person_tree.pack(fill="x", padx=8, pady=(0, 2))
        ctk.CTkLabel(tele, text="P1* = signal lost, holding last pose   ·   … = weak pose",
                     text_color="gray", font=ctk.CTkFont(size=10)).pack(anchor="w", padx=8, pady=(0, 8))

        # bottom
        bottom = ctk.CTkFrame(self.outer_pane, corner_radius=10)
        self.outer_pane.add(self.top_pane, minsize=300, sticky="nsew", stretch="always")
        self.outer_pane.add(bottom, minsize=140, sticky="nsew", stretch="never")
        self.after(150, self._init_sashes)
        bottom.grid_columnconfigure(0, weight=3)
        bottom.grid_columnconfigure(1, weight=2)
        ctk.CTkLabel(bottom, text="📋 Incident History").grid(row=0, column=0, sticky="w", padx=12, pady=(6, 0))
        ctk.CTkLabel(bottom, text="📸 Recent Alert Snapshots").grid(row=0, column=1, sticky="w", padx=12, pady=(6, 0))
        self.log_tree = self._tree(bottom, ["#", "Time", "Person", "Event", "Confidence",
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

    def _init_sashes(self, tries: int = 0):
        try:
            self.update_idletasks()
            aw, ah = self.top_pane.winfo_width(), self.outer_pane.winfo_height()
            if (aw < 800 or ah < 500) and tries < 10:
                self.after(250, lambda: self._init_sashes(tries + 1))
                return
            if aw >= 800:
                s0 = max(220, min(int(aw * 0.24), aw - 360 - 258))
                s1 = max(s0 + 228, min(int(aw * 0.77), aw - 258))
                self.top_pane.sash_place(0, s0, 0)
                self.top_pane.sash_place(1, s1, 0)
            if ah >= 500:
                self.outer_pane.sash_place(0, 0, max(300, min(int(ah * 0.72), ah - 148)))
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
        return slider, val_label

    def _metric_card(self, parent, title):
        card = ctk.CTkFrame(parent, corner_radius=8)
        card.pack(fill="x", padx=8, pady=4)
        ctk.CTkLabel(card, text=title.upper(), text_color="gray",
                     font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=10, pady=(6, 0))
        val = ctk.CTkLabel(card, text="—", font=ctk.CTkFont(size=18, weight="bold"))
        val.pack(anchor="w", padx=10, pady=(0, 6))
        return val

    @staticmethod
    def _tree(parent, columns, height=5, default_width=90):
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

    # ================= actions =================

    def _apply_preset(self, name):
        a, v, t = PRESETS.get(name, PRESETS["Standard (Balanced)"])
        self.angle_slider.set(a)
        self.vel_slider.set(v)
        self.timeout_slider.set(t)
        self.params.update({"angle": a, "vel": v, "timeout": t})

    def _browse(self):
        path = filedialog.askopenfilename(title="Choose video file",
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
        path = filedialog.asksaveasfilename(title="Save incident log", defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv")])
        if path:
            pd.DataFrame(list(self.logger.events)).to_csv(path, index=False)
            messagebox.showinfo("Export CSV", f"Saved to:\n{path}")

    def _clear_log(self):
        self.logger.clear()
        self._last_event_count = 0
        for i in self.log_tree.get_children():
            self.log_tree.delete(i)
        for w in self.snap_frame.winfo_children():
            w.destroy()
        self._thumbs.clear()

    def _capture_bg(self):
        with self._frame_lock:
            frame = None if self._last_frame is None else self._last_frame.copy()
        if frame is None:
            messagebox.showinfo("Background", "Start monitoring first, then capture\n"
                                              "when the hallway is empty.")
            return
        cv2.imwrite(os.path.join("assets", "hallway_bg.jpg"), frame)
        messagebox.showinfo("Background", "Saved assets/hallway_bg.jpg —\nnow press Calibrate Zones.")

    def _open_editor(self):
        if self._editor is not None and self._editor.winfo_exists():
            self._editor.focus()
            return
        bg_path = os.path.join("assets", "hallway_bg.jpg")
        if os.path.exists(bg_path):
            bg = cv2.imread(bg_path)
        else:
            with self._frame_lock:
                bg = None if self._last_frame is None else self._last_frame.copy()
        if bg is None:
            messagebox.showinfo("Calibrate Zones", "No background yet: Start monitoring and\n"
                                                   "use Capture Background on an empty hallway.")
            return
        import copy
        working = ZoneManager()
        working.zones = copy.deepcopy(self.zones.zones)
        save_path = self._zone_path()
        profile_label = self.profile_name or "unsaved setup"

        def _apply():
            self.zones.zones = working.zones
            self._update_profile_hint(f"Zones applied ({os.path.basename(save_path)}).")

        self._editor = ZoneEditor(self, bg, working, save_path, profile_label, _apply)

    # ================= camera sources =================

    def _refresh_source_rows(self):
        kind = self.source_menu.get()
        self._show_row(self.usb_row, kind == "USB Camera")
        self._show_row(self.rtsp_row, kind == "IP Camera (RTSP)")
        self._show_row(self.rtsp_status, kind == "IP Camera (RTSP)")
        self._show_row(self.file_row, kind == "Video File")
        self._show_row(self.file_label, kind == "Video File")

    @staticmethod
    def _show_row(widget, show):
        # grid_remove() remembers the grid options, so bare grid()
        # restores the row to its exact original position.
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
            self._update_profile_hint("Manual setup (unsaved).")

    # ----- USB discovery (background thread; opens can block) -----

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
            pass  # app closed mid-scan; nothing to update

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

    # ----- RTSP test (background thread; connects can block) -----

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
            msg = f"✓ Stream OK — {w}x{h}"
            good = True
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
            pass  # app closed mid-test; nothing to update

    def _rtsp_done(self, msg, good):
        self.rtsp_status.configure(text=msg, text_color="#34d399" if good else "#ef4444")

    # ----- profiles -----

    def _current_source_dict(self, name):
        kind = {"USB Camera": "usb", "IP Camera (RTSP)": "rtsp"}.get(self.source_menu.get(), "file")
        d: dict = {"name": name, "kind": kind}
        if kind == "usb":
            idx = self._selected_usb_index()
            d["index"] = idx if idx is not None else 0
        elif kind == "rtsp":
            d["url"] = self.rtsp_entry.get().strip()
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
        if self.profile_name in names:
            self.profile_menu.set(self.profile_name)
        else:
            self.profile_menu.set(names[0])

    def _save_profile(self):
        name = self.profile_entry.get().strip()
        if not name:
            kind = self.source_menu.get()
            name = {"USB Camera": "USB camera", "IP Camera (RTSP)": "IP camera"}.get(kind, "Video file")
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
        self._update_profile_hint(f"Saved '{name}'. Zones: {self._zone_hint()}")

    def _delete_profile(self):
        name = self.profile_menu.get()
        if name in self.profiles.names():
            self.profiles.delete(name)
            if self.profile_name == name:
                self.profile_name = None
            self._refresh_profile_menu()
            self._update_profile_hint("Profile deleted.")

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
        self._update_profile_hint(f"Camera: {name} ({self.profiles.describe(prof)}). "
                                  f"Zones: {self._zone_hint()}.")

    def _zone_hint(self):
        path = self._zone_path()
        return f"{os.path.basename(path)} found — calibrated." if os.path.exists(path) \
            else "not calibrated yet — use Calibrate Zones."

    def _update_profile_hint(self, text):
        self.profile_hint.configure(text=text)

    # ================= run control =================

    def _resolve_source(self):
        """Returns (source, kind, label) or None. kind in webcam|rtsp|file."""
        kind = {"USB Camera": "usb", "IP Camera (RTSP)": "rtsp"}.get(self.source_menu.get(), "file")
        if kind == "usb":
            idx = self._selected_usb_index()
            if idx is None:
                messagebox.showwarning("Camera", "No working USB camera selected — press ⟳ to rescan.")
                return None
            return idx, "webcam", f"USB Camera {idx}"
        if kind == "rtsp":
            url = self.rtsp_entry.get().strip()
            ok, msg = validate_network_url(url)
            if not ok:
                messagebox.showwarning("Camera", f"Bad stream URL: {msg}")
                return None
            host = urlparse(url).hostname or url[:32]
            return url, "rtsp", f"RTSP {host}"
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
        cfg = {"source": source, "kind": kind, "label": label,
               "max_persons": int(round(float(self.maxp_slider.get()))),
               "det_conf": float(self.det_slider.get())}
        self.params.update({"maxp": cfg["max_persons"], "detconf": cfg["det_conf"]})
        self.params.update({"angle": float(self.angle_slider.get()),
                            "vel": float(self.vel_slider.get()),
                            "timeout": float(self.timeout_slider.get()),
                            "audio": bool(self.audio_var.get()),
                            "skeleton": bool(self.skel_var.get()),
                            "bbox": bool(self.bbox_var.get()),
                            "show_zones": bool(self.zone_var.get()),
                            "lowlight": bool(self.lowlight_var.get())})
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
        self.status_label.configure(text="Starting backend (EP benchmark)…")

    def stop(self):
        if not self._running:
            return
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=8)
            self._worker = None
        self._running = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="Idle")

    def _on_close(self):
        try:
            self._stop_event.set()
            if self._worker is not None:
                self._worker.join(timeout=8)
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
            mgr = HallwayManager(max_persons=cfg["max_persons"],
                                 angle_threshold=self.params["angle"],
                                 velocity_threshold=self.params["vel"],
                                 inactivity_timeout=self.params["timeout"],
                                 det_conf=cfg["det_conf"], zones=self.zones)
        except Exception as e:
            self._push({"error": f"Backend init failed: {e}"})
            self._running = False
            return
        try:
            pump = FramePump(cfg["source"], kind=cfg.get("kind", "auto"),
                             on_status=lambda m: self._push({"status": m}),
                             abort=self._stop_event.is_set)
        except SourceLost as e:
            self._push({"error": str(e)})
            self._running = False
            return
        last_beep, fps, t_prev = 0.0, 0.0, time.time()
        logged_falls: dict = {}
        logged_inactive: set = set()
        logged_lost: set = set()
        backend_name = mgr.backend.backend_name
        try:
            for frame in pump.frames(alive=lambda: not self._stop_event.is_set()):
                now = time.time()
                dt = max(1e-3, now - t_prev)
                t_prev = now
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
                with self._frame_lock:
                    self._last_frame = frame.copy()

                angle_t, vel_t, timeout_t = self.params["angle"], self.params["vel"], self.params["timeout"]
                mgr.update_parameters(angle_threshold=angle_t, velocity_threshold=vel_t,
                                      inactivity_timeout=timeout_t,
                                      max_persons=self.params["maxp"])
                mgr.backend.conf_thresh = self.params["detconf"]
                mgr.backend.low_light_boost = self.params["lowlight"]
                persons = mgr.process(frame, current_time=now)

                for p in persons:
                    feats, fs = p["features"], p["fall_status"]
                    ins, hw, tid = p["inactivity_status"], p["hallway"], p["track_id"]
                    if feats is None:
                        continue
                    note = f"P{tid} {hw['risk_level']} in {hw['zone']}"
                    if fs.get("state") == "FALLEN" and fs.get("total_falls", 0) > logged_falls.get(tid, 0):
                        logged_falls[tid] = fs["total_falls"]
                        self.logger.log_event("FALL_DETECTED", feats, fs.get("fall_confidence", 0.8),
                                              frame, extra_note=note, person_id=tid)
                    if hw["risk_level"] == "SLUMP" and tid not in logged_inactive:
                        # SLUMP has no FSM counter: latch one log per slump episode.
                        logged_inactive.add(f"slump{tid}")
                        self.logger.log_event("SLUMP_ALERT", feats, 0.75, frame,
                                              extra_note=note, person_id=tid)
                    if ins.get("is_inactive_alert") and tid not in logged_inactive:
                        logged_inactive.add(tid)
                        self.logger.log_event("INACTIVITY_EMERGENCY", feats, 1.0, frame,
                                              extra_note=f"{note} {ins.get('inactive_duration'):.1f}s",
                                              person_id=tid)
                    elif not ins.get("is_inactive_alert") and hw["risk_level"] not in ("SLUMP",) \
                            and tid in logged_inactive:
                        logged_inactive.discard(tid)
                        logged_inactive.discard(f"slump{tid}")
                tids = {p["track_id"] for p in persons}
                logged_falls = {k: v for k, v in logged_falls.items() if k in tids}

                # Stand-up handoffs: the fallen track resolved into a live upright one.
                for rec in mgr.pop_recoveries():
                    nf = next((p["features"] for p in persons
                               if p["track_id"] == rec["new_tid"] and p["features"]), None)
                    self.logger.log_event("RECOVERED", nf, 0.9, frame,
                                          extra_note=f"P{rec['old_tid']} stood back up "
                                                     f"(now P{rec['new_tid']}) in {rec['zone']}",
                                          person_id=rec["new_tid"])

                # Latched lost-alarms: log once, beep once, show until Stop/Reset.
                for a in mgr.lost_alarms:
                    if a["track_id"] not in logged_lost:
                        logged_lost.add(a["track_id"])
                        self.logger.log_event(
                            "SIGNAL_LOST", None, 1.0, frame,
                            extra_note=f"P{a['track_id']} was {a['risk_level']} in "
                                       f"{a['zone']} then left view — VERIFY IN PERSON",
                            person_id=a["track_id"])
                        if self.params["audio"]:
                            _beep()

                disp = frame.copy()
                if self.params["show_zones"]:
                    disp = self.zones.draw(disp)
                disp = draw_hallway_hud(disp, persons, show_skeleton=self.params["skeleton"],
                                        show_bbox=self.params["bbox"],
                                        extra_status=f"• {backend_name}",
                                        lost_alarms=mgr.lost_alarms)
                # worst person for telemetry cards
                order = {"NORMAL": 0, "RESTING": 1, "UNUSUAL": 2, "SLUMP": 3,
                         "CONCERNING": 4, "EMERGENCY": 5}
                scored = [(order.get(p["hallway"]["risk_level"], 0), p) for p in persons if p["features"]]
                if scored:
                    _, w = max(scored, key=lambda t: t[0])
                    risk = w["hallway"]["risk_level"]
                    pkt = {"risk": risk, "angle": w["features"].get("spine_angle", 0.0),
                           "vy": w["features"].get("vertical_velocity", 0.0),
                           "inact": w["inactivity_status"].get("inactive_duration", 0.0),
                           "is_inact": w["inactivity_status"].get("is_inactive_alert", False)}
                else:
                    pkt = {"risk": "NORMAL", "angle": 0.0, "vy": 0.0, "inact": 0.0, "is_inact": False}

                alarm_active = pkt["risk"] in ("SLUMP", "CONCERNING", "EMERGENCY")
                self._ack.rearm_if_cleared(alarm_active, now)
                if self.params["audio"] and alarm_active and not self._ack.muted(now):
                    if now - last_beep > 2.0:
                        _beep()
                        last_beep = now

                disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                h0, w0 = disp_rgb.shape[:2]
                sc = min(1.0, 640.0 / max(1, w0))
                if sc < 1.0:
                    disp_rgb = cv2.resize(disp_rgb, (int(w0 * sc), int(h0 * sc)))
                pkt.update({"frame": disp_rgb, "fps": fps, "backend": backend_name,
                            "persons": persons, "events": len(self.logger.events),
                            "source_label": cfg.get("label", ""),
                            "lost": [(a["track_id"], a["risk_level"]) for a in mgr.lost_alarms]})
                self._push(pkt)
        except SourceLost as e:
            # User-initiated stops surface here too when abort fires mid-reconnect.
            if not self._stop_event.is_set():
                self._push({"error": str(e)})
        except Exception as e:
            self._push({"error": str(e)})
        finally:
            pump.release()
            self._running = False

    # ================= UI poll =================

    def _poll(self):
        # The after() chain is re-armed in `finally`: no UI/render error may
        # ever freeze the display loop (that failure looks like "monitoring stopped").
        try:
            self.params.update({"angle": float(self.angle_slider.get()),
                                "vel": float(self.vel_slider.get()),
                                "timeout": float(self.timeout_slider.get()),
                                "audio": bool(self.audio_var.get()),
                                "skeleton": bool(self.skel_var.get()),
                            "bbox": bool(self.bbox_var.get()),
                            "show_zones": bool(self.zone_var.get()),
                            "lowlight": bool(self.lowlight_var.get()),
                            "maxp": int(round(float(self.maxp_slider.get()))),
                            "detconf": float(self.det_slider.get())})
            pkt = None
            try:
                while True:
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
            if (self._worker is not None and not self._worker.is_alive()
                    and self.start_btn.cget("state") == "disabled"):
                self._worker = None
                self._running = False
                self.start_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")
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
        color = RISK_HEX.get(risk, "#f8fafc")
        inact = pkt.get("inact", 0.0)
        icon = ICONS.get(risk, "")
        self.card_risk.configure(text=f"{icon} {RISK_TEXT.get(risk, risk)}", text_color=color)
        self.card_persons.configure(text=f"{len(pkt.get('persons', []))} tracked")
        self.card_angle.configure(text=f"{pkt.get('angle', 0.0):.1f}°")
        self.card_vy.configure(text=f"{pkt.get('vy', 0.0):+.2f}")
        self.card_timer.configure(text=f"{inact:.1f}s",
                                  text_color=RISK_HEX["EMERGENCY"] if pkt.get("is_inact") else "#f8fafc")
        lost = pkt.get("lost", [])
        lost_txt = ("  •  ⚠️ UNVERIFIED: " + ", ".join(f"P{tid} {r}" for tid, r in lost)) if lost else ""
        src_txt = f"  •  {pkt['source_label']}" if pkt.get("source_label") else ""
        self.status_label.configure(
            text=f"{pkt.get('fps', 0.0):.1f} FPS  •  {pkt.get('backend', '?')}  •  "
                 f"{len(pkt.get('persons', []))} person(s){lost_txt}{src_txt}")

        self._chart_angle.append(pkt.get("angle", 0.0))
        self._chart_vy.append(pkt.get("vy", 0.0) * 100.0)
        if len(self._chart_angle) > 60:
            self._chart_angle = self._chart_angle[-60:]
            self._chart_vy = self._chart_vy[-60:]
        self._draw_chart()

        for i in self.person_tree.get_children():
            self.person_tree.delete(i)
        for p in pkt.get("persons", []):
            fs, ins, feats, hw = (p["fall_status"], p["inactivity_status"],
                                  p["features"], p["hallway"])
            tag = f"P{p['track_id']}{'*' if p.get('stale') else ''}"
            self.person_tree.insert("", "end", values=(
                tag, hw["risk_level"], hw["zone"],
                f"{feats.get('spine_angle', 0.0):.1f}°" if feats else ("…" if p["weak_pose"] else "-"),
                f"{ins.get('inactive_duration', 0.0):.1f}s"))

        if pkt.get("events", 0) != self._last_event_count:
            self._last_event_count = pkt["events"]
            self._refresh_log()
            self._refresh_snaps()

    def _draw_chart(self):
        c = self.chart
        c.delete("all")
        W = c.winfo_width() or 250
        H = c.winfo_height() or 150
        mid = H // 2
        c.create_line(0, mid, W, mid, fill="#374151")
        thr = float(self.angle_slider.get())
        ty = (H - 8) - (min(90.0, max(0.0, thr)) / 90.0) * (H - 16)
        c.create_line(0, ty, W, ty, fill="#ef4444", dash=(4, 3))
        n = len(self._chart_angle)
        if n < 2:
            return

        def _xs(i):
            return 4 + i * (W - 8) / 59

        m = max(0, 60 - n)
        pts_a = [(_xs(i + m), (H - 8) - (min(90.0, max(0.0, a)) / 90.0) * (H - 16))
                 for i, a in enumerate(self._chart_angle[-60:])]
        pts_v = [(_xs(i + m), mid - (min(100.0, max(-100.0, v)) / 100.0) * (mid - 8))
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
        if not os.path.isdir("alerts_hallway"):
            return
        files = sorted((f for f in os.listdir("alerts_hallway") if f.endswith(".jpg")),
                       key=lambda x: os.path.getmtime(os.path.join("alerts_hallway", x)),
                       reverse=True)[:6]
        for fn in files:
            try:
                im = Image.open(os.path.join("alerts_hallway", fn))
                im.thumbnail((150, 95))
                ph = ctk.CTkImage(light_image=im, dark_image=im, size=im.size)
                self._thumbs.append(ph)
                ctk.CTkLabel(self.snap_frame, image=ph, text=fn, compound="top",
                             wraplength=150, font=ctk.CTkFont(size=10)).pack(side="left", padx=6, pady=4)
            except Exception:
                continue


if __name__ == "__main__":
    HallwayApp().mainloop()
