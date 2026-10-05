import json
import os
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Zone types and their overlay colors (BGR)
ZONE_COLORS = {
    "floor": (113, 204, 46),    # green — monitored floor area
    "bench": (36, 191, 251),    # amber — seating / waiting area
    "ignore": (120, 120, 120),  # gray — doorways, counters, dead areas
}

# Lookup priority: first match wins
ZONE_PRIORITY = ("ignore", "bench", "floor")


def point_in_polygon(x: float, y: float, polygon: List[List[float]]) -> bool:
    """Ray-casting test. Point and polygon in normalized 0-1 coords."""
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


class ZoneManager:
    """
    Spatial context for hallway / waiting-area monitoring.

    Zones are polygons in NORMALIZED coords (resolution-independent) with types:
      floor  — monitored walking/floor area (falls here alarm)
      bench  — seating area (reclined + still = RESTING, sudden slump = SLUMP)
      ignore — doorways/counters (risk capped at UNUSUAL)

    With no zones file, the whole frame is one default floor zone, i.e. the
    system behaves like the zone-less fall detector.
    """

    def __init__(self):
        self.zones: List[Dict[str, Any]] = [self._default_floor()]

    @staticmethod
    def _default_floor() -> Dict[str, Any]:
        return {"name": "default_floor", "type": "floor",
                "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]}

    # ---------- query ----------

    def zone_at(self, x_norm: float, y_norm: float) -> Dict[str, Any]:
        for ztype in ZONE_PRIORITY:
            for z in self.zones:
                if z.get("type") == ztype and point_in_polygon(x_norm, y_norm, z.get("polygon", [])):
                    return z
        # Outside every polygon: treat as floor (conservative toward detection)
        return {"name": "outside", "type": "floor", "polygon": []}

    def zone_for_person(self, bbox: List[int], frame_w: int, frame_h: int,
                        keypoints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Feet point (bbox bottom-center); falls back to hip midpoint."""
        x1, y1, x2, y2 = bbox
        fx, fy = ((x1 + x2) / 2.0 / max(1, frame_w), y2 / max(1, frame_h))
        if keypoints is not None:
            try:
                lh, rh = keypoints["left_hip"], keypoints["right_hip"]
                if lh["visibility"] > 0.3 and rh["visibility"] > 0.3:
                    hx = (lh["x_norm"] + rh["x_norm"]) / 2.0
                    hy = (lh["y_norm"] + rh["y_norm"]) / 2.0
                    # Blend feet + hips so bench-seated people (feet on floor
                    # under bench, hips on bench) resolve to the bench.
                    fx, fy = (fx + hx) / 2.0, (fy + hy) / 2.0
            except (KeyError, TypeError):
                pass
        return self.zone_at(fx, fy)

    # ---------- editing ----------

    def add_zone(self, name: str, ztype: str, polygon: List[List[float]]):
        if ztype not in ZONE_COLORS:
            raise ValueError(f"bad zone type: {ztype}")
        if len(polygon) < 3:
            raise ValueError("polygon needs >= 3 points")
        # First user zone replaces the default full-frame floor
        if len(self.zones) == 1 and self.zones[0]["name"] == "default_floor":
            self.zones = []
        self.zones.append({"name": name, "type": ztype,
                           "polygon": [[float(x), float(y)] for x, y in polygon]})

    def remove_zone(self, name: str):
        self.zones = [z for z in self.zones if z.get("name") != name]
        if not self.zones:
            self.zones = [self._default_floor()]

    def clear(self):
        self.zones = [self._default_floor()]

    # ---------- persistence ----------

    def to_dict(self) -> Dict[str, Any]:
        return {"version": 1, "zones": self.zones}

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self, path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            zones = data.get("zones", [])
            if not zones:
                return False
            self.zones = zones
            return True
        except (OSError, ValueError):
            return False

    # ---------- overlay ----------

    def draw(self, frame: np.ndarray, alpha: float = 0.18) -> np.ndarray:
        """Translucent zone fills + outlines + name tags. Skips the default full-frame zone."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        for z in self.zones:
            poly = z.get("polygon", [])
            if len(poly) < 3:
                continue
            if z.get("name") == "default_floor":
                continue
            pts = np.array([[[int(x * w), int(y * h)] for x, y in poly]], dtype=np.int32)
            color = ZONE_COLORS.get(z.get("type"), (200, 200, 200))
            cv2.fillPoly(overlay, pts, color)
            cv2.polylines(frame, pts, True, color, 2)
            x0 = int(min(p[0] for p in poly) * w)
            y0 = int(max(0, min(p[1] for p in poly) * h - 8))
            cv2.putText(frame, f"{z.get('name', '')} [{z.get('type', '')}]", (x0 + 4, max(14, y0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        return frame
