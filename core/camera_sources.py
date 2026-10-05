"""Camera source management: USB discovery, RTSP handling, saved profiles.

No new dependencies (plain OpenCV + stdlib). Profiles live in
assets/cameras.json; each profile keeps its own calibration at
assets/zones_<slug>.json ("default" reuses the legacy assets/zones.json).
"""

import json
import os
import re
import urllib.parse
import cv2
from typing import List, Dict, Any, Optional, Tuple

PROFILES_FILE = os.path.join("assets", "cameras.json")
LEGACY_ZONES_FILE = os.path.join("assets", "zones.json")

NETWORK_SCHEMES = ("rtsp://", "rtmp://", "http://", "https://")

# FFmpeg options for network streams: TCP transport (fewer artifacts than UDP),
# 5s open/read timeout so a dead camera fails fast instead of hanging the UI.
FFMPEG_NET_OPTS = "rtsp_transport;tcp|stimeout;5000000|reorder_queue_size;0"


def validate_network_url(url: str) -> Tuple[bool, str]:
    """Accept rtsp/rtmp/http(s) camera URLs. Returns (ok, message)."""
    u = (url or "").strip()
    if not u:
        return False, "empty URL"
    if not u.lower().startswith(NETWORK_SCHEMES):
        return False, "URL must start with rtsp://, rtmp://, http:// or https://"
    if len(u) > 512:
        return False, "URL unusually long — check for paste errors"
    return True, "URL looks valid"


def build_tapo_rtsp_url(ip: str, user: str, password: str, stream: str = "stream2") -> str:
    """Tapo C200/C230/C310 RTSP URL with properly escaped credentials.

    stream1 = high-def (1080p/2K), stream2 = low-def (360p/480p — recommended
    for high-FPS inference). Standard Tapo RTSP port 554.
    """
    esc_user = urllib.parse.quote(user, safe="")
    esc_pass = urllib.parse.quote(password, safe="")
    return f"rtsp://{esc_user}:{esc_pass}@{ip}:554/{stream}"


def probe_usb_cameras(max_index: int = 6) -> List[Dict[str, Any]]:
    """Probe indices 0..max_index-1 for USB cameras.

    Tries DirectShow first (fast fail on Windows); if it finds nothing at all,
    falls back to the default backend once (slower, but some devices only
    answer to MSMF/AVFoundation). Returns [{index, ok, width, height, fps}].
    Never raises.
    """

    def _scan(backend) -> List[Dict[str, Any]]:
        found = []
        for i in range(max(0, max_index)):
            info: Dict[str, Any] = {"index": i, "ok": False, "width": 0, "height": 0, "fps": 0.0}
            try:
                cap = cv2.VideoCapture(i, backend) if backend is not None else cv2.VideoCapture(i)
                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        info.update({"ok": True,
                                     "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                                     "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                                     "fps": round(float(cap.get(cv2.CAP_PROP_FPS) or 0.0), 1)})
                cap.release()
            except Exception:
                pass
            found.append(info)
        return found

    found = _scan(cv2.CAP_DSHOW)
    if not any(c.get("ok") for c in found):
        found = _scan(None)  # default backend fallback
    return found


def describe_camera(info: Dict[str, Any]) -> str:
    if not info.get("ok"):
        return f"Camera {info['index']} — no signal"
    res = f"{info['width']}x{info['height']}" if info.get("width") else "unknown res"
    return f"Camera {info['index']} — {res}"


def open_capture(source, kind: str = "auto"):
    """Open a capture with sane per-kind settings. Returns cv2.VideoCapture."""
    if kind == "auto":
        if isinstance(source, str) and source.lower().startswith(NETWORK_SCHEMES):
            kind = "rtsp"
        elif isinstance(source, str):
            kind = "file"
        else:
            kind = "webcam"
    if kind == "rtsp":
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = FFMPEG_NET_OPTS
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    return cv2.VideoCapture(source)


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")
    return slug or "cam"


def zones_path_for(profile_name: str) -> str:
    slug = slugify(profile_name or "default")
    if slug == "default":
        return LEGACY_ZONES_FILE
    return os.path.join("assets", f"zones_{slug}.json")


class CameraProfiles:
    """Named camera presets persisted to JSON. {name, kind, index|url|path}."""

    KINDS = ("usb", "rtsp", "file")

    def __init__(self, path: str = PROFILES_FILE):
        self.path = path
        self.profiles: List[Dict[str, Any]] = []
        self.active: Optional[str] = None
        self.load()

    # ---------- persistence ----------

    def load(self) -> bool:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            profs = [p for p in data.get("profiles", []) if self._valid(p)]
            self.profiles = profs
            active = data.get("active")
            self.active = active if any(p["name"] == active for p in profs) else None
            return True
        except (OSError, ValueError):
            self.profiles, self.active = [], None
            return False

    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"active": self.active, "profiles": self.profiles}, f, indent=2)

    @classmethod
    def _valid(cls, p: Any) -> bool:
        return (isinstance(p, dict) and p.get("name")
                and p.get("kind") in cls.KINDS)

    # ---------- CRUD ----------

    def names(self) -> List[str]:
        return [p["name"] for p in self.profiles]

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return next((p for p in self.profiles if p["name"] == name), None)

    def upsert(self, profile: Dict[str, Any]):
        if not self._valid(profile):
            raise ValueError(f"bad profile (need name + kind in {self.KINDS}): {profile}")
        self.profiles = [p for p in self.profiles if p["name"] != profile["name"]]
        self.profiles.append(dict(profile))
        self.save()

    def delete(self, name: str):
        self.profiles = [p for p in self.profiles if p["name"] != name]
        if self.active == name:
            self.active = None
        self.save()

    def describe(self, profile: Dict[str, Any]) -> str:
        kind = profile.get("kind")
        if kind == "usb":
            return f"USB camera index {profile.get('index', 0)}"
        if kind == "rtsp":
            return str(profile.get("url", ""))
        return str(profile.get("path", ""))
