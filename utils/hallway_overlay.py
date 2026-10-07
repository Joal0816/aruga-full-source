import cv2
import numpy as np
from typing import Dict, Any, List, Optional

from core.yolo_pose_backend import COCO_PAIRS
from utils.risk_palette import bgr as risk_bgr, ASCII_TAG

# Risk colors come from the shared palette; local extras only.
COLORS = {r: risk_bgr(r) for r in
          ("NORMAL", "RESTING", "UNUSUAL", "SLUMP", "CONCERNING", "EMERGENCY")}
COLORS.update({"WEAK": (150, 150, 150), "DARK_BG": (20, 24, 28)})
RISK_ORDER = {"NORMAL": 0, "RESTING": 1, "UNUSUAL": 2, "SLUMP": 3, "CONCERNING": 4, "EMERGENCY": 5}

# ASCII only — these badges are drawn with cv2.putText (no emoji support).
BADGES = {
    "NORMAL": "NORMAL: ROUTINE MOBILITY",
    "RESTING": "RESTING IN SEATING AREA",
    "UNUSUAL": ASCII_TAG["UNUSUAL"] + "UNUSUAL: UNSTABLE / SEVERE TILT",
    "SLUMP": ASCII_TAG["SLUMP"] + "SLUMP: CHECK ON PERSON",
    "CONCERNING": ASCII_TAG["CONCERNING"] + "CONCERNING: FALL DETECTED",
    "EMERGENCY": ASCII_TAG["EMERGENCY"] + "EMERGENCY: UNRESPONSIVE",
}


def worst_risk(persons: List[Dict[str, Any]]) -> str:
    risks = [p.get("hallway", {}).get("risk_level", "NORMAL") for p in persons] or ["NORMAL"]
    return max(risks, key=lambda r: RISK_ORDER.get(r, 0))


def draw_hallway_hud(frame: np.ndarray, persons: List[Dict[str, Any]],
                     show_skeleton: bool = True, show_bbox: bool = True,
                     extra_status: str = "",
                     lost_alarms: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
    """Header badge (worst risk) + per-person bbox / label / spine / COCO skeleton."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    worst = worst_risk(persons)
    color = COLORS.get(worst, (220, 220, 220))

    hud_h = 74 + (26 * len(lost_alarms) if lost_alarms else 0)
    cv2.rectangle(overlay, (0, 0), (w, hud_h), COLORS["DARK_BG"], -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.line(frame, (0, hud_h), (w, hud_h), color, 2)
    cv2.putText(frame, f"HALLWAY | MULTI-PERSON (n={len(persons)}) {extra_status}",
                (16, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 175, 190), 1, cv2.LINE_AA)
    cv2.putText(frame, BADGES.get(worst, worst), (16, 56),
                cv2.FONT_HERSHEY_DUPLEX, 0.70, color, 2, cv2.LINE_AA)
    if lost_alarms:
        for i, a in enumerate(lost_alarms):
            cv2.putText(frame,
                        f"!! P{a['track_id']} {a['risk_level']} LEFT VIEW ({a['zone']}) — VERIFY IN PERSON",
                        (16, 56 + 26 * (i + 1)), cv2.FONT_HERSHEY_DUPLEX, 0.60,
                        COLORS["CONCERNING"], 2, cv2.LINE_AA)

    for p in persons:
        tid = p.get("track_id", 0)
        risk = p.get("hallway", {}).get("risk_level", "NORMAL")
        zone = p.get("hallway", {}).get("zone", "?")
        pcolor = COLORS.get(risk, (220, 220, 220))
        x1, y1, x2, y2 = [int(v) for v in p.get("bbox", [0, 0, 0, 0])]
        feats, kp = p.get("features"), p.get("keypoints")
        weak = p.get("weak_pose", False)
        stale = p.get("stale", False)
        angle = feats.get("spine_angle", 0.0) if feats else 0.0

        if stale:
            # Detector-blind hold: state machine is live (held off last good
            # skeleton). Drawn in risk color with an honesty tag; the separate
            # lost-alarm banner covers the genuinely-gone case.
            feats = p.get("features") or {}
            angle = feats.get("spine_angle", 0.0)
            extra = f" | {angle:.0f}° (holding)" if feats else " (signal lost)"
            label = f"P{tid} {risk} [{zone}]{extra}"
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), pcolor, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + tw + 8, max(22, y1)), pcolor, -1)
            cv2.putText(frame, label, (x1 + 4, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA)
            continue
        if weak:
            pcolor = COLORS["WEAK"]
        if show_bbox:
            cv2.rectangle(frame, (x1, y1), (x2, y2), pcolor, 2)
            label = f"P{tid} {risk} [{zone}] | {angle:.0f}°" if not weak else f"P{tid} ..."
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + tw + 8, max(22, y1)), pcolor, -1)
            cv2.putText(frame, label, (x1 + 4, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"P{tid}", (x1 + 4, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, pcolor, 1, cv2.LINE_AA)

        if show_skeleton and feats and not weak:
            hip, sh = feats.get("mid_hip_px"), feats.get("mid_shoulder_px")
            if hip and sh:
                cv2.line(frame, hip, sh, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, hip, 4, (0, 255, 255), -1)
                cv2.circle(frame, sh, 4, (0, 255, 255), -1)
        if show_skeleton and kp and not weak:
            lms = kp.get("all_landmarks", [])
            for a, b in COCO_PAIRS:
                if a >= len(lms) or b >= len(lms):
                    continue
                la, lb = lms[a], lms[b]
                if la.visibility > 0.35 and lb.visibility > 0.35:
                    cv2.line(frame, (int(la.x * w), int(la.y * h)),
                             (int(lb.x * w), int(lb.y * h)), pcolor, 2, cv2.LINE_AA)
            for idx in (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
                if idx < len(lms) and lms[idx].visibility > 0.35:
                    pt = (int(lms[idx].x * w), int(lms[idx].y * h))
                    cv2.circle(frame, pt, 3, (255, 255, 255), -1)
                    cv2.circle(frame, pt, 2, pcolor, -1)

    if worst == "EMERGENCY":
        cv2.rectangle(frame, (0, 0), (w, h), COLORS["EMERGENCY"], 8)
    elif worst in ("CONCERNING", "SLUMP"):
        cv2.rectangle(frame, (0, 0), (w, h), COLORS[worst], 4)
    return frame
