import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional

# Color constants (BGR)
COLOR_NORMAL = (113, 204, 46)        # Vibrant Green
COLOR_UNUSUAL = (36, 191, 251)       # Amber / Gold
COLOR_CONCERNING = (60, 76, 231)     # Red / Coral
COLOR_EMERGENCY = (182, 89, 155)     # Deep Magenta / Purple Alert
COLOR_NEUTRAL = (220, 220, 220)      # Soft White
COLOR_DARK_BG = (20, 24, 28)         # Dark slate HUD bg
COLOR_LEVINE_DISTRESS = (80, 50, 240) # Rose / Amber-red (Cardiac distress)
COLOR_LEVINE_BEACON = (100, 70, 255) # Rose beacon core

# Skeleton joint connection pairs (MediaPipe pose indices)
POSE_CONNECTIONS = [
    (11, 12), # shoulders
    (11, 13), (13, 15), # left arm
    (12, 14), (14, 16), # right arm
    (11, 23), (12, 24), # torso sides
    (23, 24), # hips
    (23, 25), (25, 27), # left leg
    (24, 26), (26, 28)  # right leg
]

class Visualizer:
    """
    Renders ARUGA contextual risk assessment HUD, skeleton pose overlays,
    and multi-tier alert indicators on video frames.
    """
    def __init__(self):
        pass

    def get_risk_color(self, risk_level: str) -> Tuple[int, int, int]:
        if risk_level == "NORMAL":
            return COLOR_NORMAL
        elif risk_level == "UNUSUAL":
            return COLOR_UNUSUAL
        elif risk_level == "CONCERNING":
            return COLOR_CONCERNING
        elif risk_level == "EMERGENCY":
            return COLOR_EMERGENCY
        return COLOR_NEUTRAL

    def draw_levine_beacon(
        self,
        frame: np.ndarray,
        sternum_px: Tuple[int, int],
        wrist_pts: list,
        label: str = "LEVINE SIGN (CHEST PAIN DISTRESS)"
    ):
        """
        Renders a distinct cardiac distress beacon badge over the chest/sternum midpoint
        and highlights the wrist-to-sternum vector.
        """
        h, w = frame.shape[:2]
        sx, sy = sternum_px

        # Highlight wrist-to-sternum vector(s)
        for wpt in wrist_pts:
            cv2.line(frame, wpt, (sx, sy), (40, 30, 160), 4, cv2.LINE_AA)
            cv2.line(frame, wpt, (sx, sy), COLOR_LEVINE_BEACON, 2, cv2.LINE_AA)
            cv2.circle(frame, wpt, 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, wpt, 3, COLOR_LEVINE_DISTRESS, -1, cv2.LINE_AA)

        # Concentric beacon rings over chest/sternum midpoint
        cv2.circle(frame, (sx, sy), 15, (60, 40, 230), 2, cv2.LINE_AA)
        cv2.circle(frame, (sx, sy), 8, COLOR_LEVINE_BEACON, -1, cv2.LINE_AA)
        cv2.circle(frame, (sx, sy), 4, (255, 255, 255), -1, cv2.LINE_AA)

        # Cardiac distress badge
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        bx = max(10, min(w - tw - 12, sx - tw // 2))
        by = max(86, sy - 22)
        cv2.rectangle(frame, (bx - 5, by - th - 5), (bx + tw + 5, by + 5), (15, 15, 25), -1)
        cv2.rectangle(frame, (bx - 5, by - th - 5), (bx + tw + 5, by + 5), COLOR_LEVINE_DISTRESS, 1)
        cv2.putText(frame, label, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 140, 255), 1, cv2.LINE_AA)

    def draw_hud(
        self,
        frame: np.ndarray,
        keypoints: Optional[Dict[str, Any]],
        features: Optional[Dict[str, Any]],
        fall_status: Dict[str, Any],
        inactivity_status: Dict[str, Any],
        show_skeleton: bool = True,
        show_bbox: bool = True,
    ) -> np.ndarray:
        h, w, _ = frame.shape
        overlay = frame.copy()
        
        # Determine active 4-tier risk level
        risk_level = fall_status.get("risk_level", "NORMAL")
        is_levine = (
            (features is not None and features.get("is_levine_gesture", False)) or
            fall_status.get("behavior") == "LEVINE_SIGN_DISTRESS" or
            fall_status.get("is_levine_sign", False) or
            fall_status.get("state") == "CORONARY_DISTRESS"
        )
        if inactivity_status.get("is_inactive_alert", False):
            risk_level = "EMERGENCY"
        elif fall_status.get("state") == "FALLEN" and risk_level != "EMERGENCY":
            risk_level = "CONCERNING"

        risk_color = COLOR_LEVINE_DISTRESS if (is_levine and risk_level != "EMERGENCY" and fall_status.get("state") != "FALLEN") else self.get_risk_color(risk_level)
        
        # 1. Top HUD Header Bar
        hud_h = 74
        cv2.rectangle(overlay, (0, 0), (w, hud_h), COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
        
        # Bottom border line of HUD colored by risk level
        cv2.line(frame, (0, hud_h), (w, hud_h), risk_color, 2)
        
        # Subtitle brand
        cv2.putText(frame, "ARUGA | RISK ASSESSMENT", (16, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 175, 190), 1, cv2.LINE_AA)

        # 2. Risk Level Badge
        if risk_level == "EMERGENCY":
            badge_text = f"🚨 EMERGENCY: INACTIVE {inactivity_status.get('inactive_duration', 0):.1f}s"
        elif fall_status.get("state") == "FALLEN":
            badge_text = f"⚠️ CONCERNING: FALL DETECTED ({int(fall_status.get('fall_confidence', 0)*100)}%)"
        elif is_levine or fall_status.get("behavior") == "LEVINE_SIGN_DISTRESS" or fall_status.get("state") == "CORONARY_DISTRESS":
            badge_text = "💔 CONCERNING: LEVINE'S SIGN (CHEST PAIN DISTRESS)"
        elif risk_level == "CONCERNING":
            badge_text = f"⚠️ CONCERNING: RISK DETECTED ({int(fall_status.get('fall_confidence', 0)*100)}%)"
        elif risk_level == "UNUSUAL":
            badge_text = "⚠️ UNUSUAL: UNSTABLE / SEVERE TILT"
        else:
            badge_text = "🟢 NORMAL: ROUTINE MOBILITY"

        cv2.putText(frame, badge_text, (16, 56), cv2.FONT_HERSHEY_DUPLEX, 0.78, risk_color, 2, cv2.LINE_AA)
        
        # 3. Telemetry values in HUD (Right side)
        if features is not None:
            angle = features.get("spine_angle", 0.0)
            vy = features.get("vertical_velocity", 0.0)
            ar = features.get("aspect_ratio", 0.0)
            
            telemetry_str1 = f"Angle: {angle:.1f}°   Aspect: {ar:.2f}"
            telemetry_str2 = f"Speed: {vy:+.2f}   Incidents: {fall_status.get('total_falls', 0)}"
            
            cv2.putText(frame, telemetry_str1, (w - 340, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 220, 240), 1, cv2.LINE_AA)
            cv2.putText(frame, telemetry_str2, (w - 340, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 220, 240), 1, cv2.LINE_AA)
            
            # 4. Draw Bounding Box & Label
            bbox = features.get("bbox_px") if show_bbox else None
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), risk_color, 2)
                
                label = f"{risk_level} | {angle:.0f}°"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + tw + 8, max(22, y1)), risk_color, -1)
                cv2.putText(frame, label, (x1 + 4, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA)

            # 5. Draw Spine vector (mid-hip to mid-shoulder)
            hip_px = features.get("mid_hip_px") if show_skeleton else None
            sh_px = features.get("mid_shoulder_px") if show_skeleton else None
            if hip_px and sh_px:
                cv2.line(frame, hip_px, sh_px, (0, 255, 255), 3, cv2.LINE_AA)
                cv2.circle(frame, hip_px, 5, (0, 255, 255), -1)
                cv2.circle(frame, sh_px, 5, (0, 255, 255), -1)

        # 6. Draw Skeleton Lines
        if show_skeleton and keypoints and "all_landmarks" in keypoints:
            lms = keypoints["all_landmarks"]
            for p1_idx, p2_idx in POSE_CONNECTIONS:
                lm1 = lms[p1_idx]
                lm2 = lms[p2_idx]
                if lm1.visibility > 0.4 and lm2.visibility > 0.4:
                    pt1 = (int(lm1.x * w), int(lm1.y * h))
                    pt2 = (int(lm2.x * w), int(lm2.y * h))
                    cv2.line(frame, pt1, pt2, risk_color, 2, cv2.LINE_AA)
            
            for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 0]:
                lm = lms[idx]
                if lm.visibility > 0.4:
                    pt = (int(lm.x * w), int(lm.y * h))
                    cv2.circle(frame, pt, 4, (255, 255, 255), -1)
                    cv2.circle(frame, pt, 2, risk_color, -1)

        # 6b. Draw Levine Sign Cardiac Distress Beacon & Wrist Vector
        if is_levine:
            sternum_px = None
            if features and "sternum_px" in features:
                sternum_px = features["sternum_px"]
            elif features and "mid_shoulder_px" in features:
                sternum_px = features["mid_shoulder_px"]
            elif keypoints and "left_shoulder" in keypoints and "right_shoulder" in keypoints:
                ls, rs = keypoints["left_shoulder"], keypoints["right_shoulder"]
                sternum_px = (int((ls["x_norm"] + rs["x_norm"]) / 2 * w), int((ls["y_norm"] + rs["y_norm"]) / 2 * h))

            if sternum_px is not None:
                clutch_pts = []
                if keypoints:
                    for side in ("left_wrist", "right_wrist"):
                        kp = keypoints.get(side)
                        if kp and kp.get("visibility", 1.0) >= 0.2:
                            wpt = (kp["x_px"], kp["y_px"])
                            dist = np.hypot(kp["x_norm"] - (sternum_px[0] / max(1, w)), kp["y_norm"] - (sternum_px[1] / max(1, h)))
                            if dist <= 0.22:
                                clutch_pts.append(wpt)
                    if not clutch_pts:
                        for side in ("left_wrist", "right_wrist"):
                            kp = keypoints.get(side)
                            if kp and kp.get("visibility", 1.0) >= 0.2:
                                clutch_pts.append((kp["x_px"], kp["y_px"]))
                self.draw_levine_beacon(frame, sternum_px, clutch_pts)

        # 7. Flashing Border on Critical Inactivity or Fall
        if risk_level == "EMERGENCY":
            cv2.rectangle(frame, (0, 0), (w, h), COLOR_EMERGENCY, 8)
        elif risk_level == "CONCERNING":
            cv2.rectangle(frame, (0, 0), (w, h), COLOR_CONCERNING, 4)

        return frame

    def draw_hud_multi(
        self,
        frame: np.ndarray,
        persons: list,
        show_skeleton: bool = True,
        show_bbox: bool = True,
    ) -> np.ndarray:
        """
        Multi-person HUD (2-3 persons).
        persons: list of {track_id, bbox, keypoints, features, fall_status, inactivity_status, has_pose}
        Header badge/border reflects the worst risk across persons.
        Each person gets its own risk-colored bbox, P# label, spine vector and skeleton.
        """
        h, w, _ = frame.shape
        overlay = frame.copy()

        order = {"NORMAL": 0, "UNUSUAL": 1, "CONCERNING": 2, "EMERGENCY": 3}

        def person_risk(p: Dict[str, Any]) -> str:
            r = p.get("fall_status", {}).get("risk_level", "NORMAL")
            if p.get("inactivity_status", {}).get("is_inactive_alert", False):
                return "EMERGENCY"
            if p.get("fall_status", {}).get("state") == "FALLEN" and r != "EMERGENCY":
                return "CONCERNING"
            if (p.get("fall_status", {}).get("behavior") == "LEVINE_SIGN_DISTRESS" or
                    p.get("fall_status", {}).get("state") == "CORONARY_DISTRESS") and r not in ("EMERGENCY", "CONCERNING"):
                return "CONCERNING"
            return r

        risks = [person_risk(p) for p in persons] if persons else ["NORMAL"]
        worst = max(risks, key=lambda r: order.get(r, 0)) if risks else "NORMAL"
        risk_color = self.get_risk_color(worst)

        # 1. Top HUD header (same style as single-person)
        hud_h = 74
        cv2.rectangle(overlay, (0, 0), (w, hud_h), COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
        cv2.line(frame, (0, hud_h), (w, hud_h), risk_color, 2)
        cv2.putText(frame, f"ARUGA | MULTI-PERSON RISK (n={len(persons)})", (16, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 175, 190), 1, cv2.LINE_AA)

        # 2. Worst-risk badge + total incidents
        total_falls = sum(p.get("fall_status", {}).get("total_falls", 0) for p in persons)
        has_multi_levine = any(
            (p.get("fall_status", {}).get("behavior") == "LEVINE_SIGN_DISTRESS" or
             p.get("fall_status", {}).get("state") == "CORONARY_DISTRESS" or
             (p.get("features") and p["features"].get("is_levine_gesture", False)))
            for p in persons
        )
        if worst == "EMERGENCY":
            worst_dur = 0.0
            for p in persons:
                if person_risk(p) == "EMERGENCY":
                    worst_dur = max(worst_dur, p.get("inactivity_status", {}).get("inactive_duration", 0))
            badge_text = f"🚨 EMERGENCY: INACTIVE {worst_dur:.1f}s"
        elif worst == "CONCERNING":
            if total_falls > 0:
                badge_text = f"⚠️ CONCERNING: FALL DETECTED (total {total_falls})"
            elif has_multi_levine:
                badge_text = "💔 CONCERNING: CARDIAC DISTRESS / LEVINE'S SIGN"
            else:
                badge_text = "⚠️ CONCERNING: UNSTABLE POSTURE DETECTED"
        elif worst == "UNUSUAL":
            badge_text = "⚠️ UNUSUAL: UNSTABLE / SEVERE TILT"
        else:
            badge_text = "🟢 NORMAL: ROUTINE MOBILITY"
        cv2.putText(frame, badge_text, (16, 56), cv2.FONT_HERSHEY_DUPLEX, 0.72, risk_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Incidents: {total_falls}", (w - 170, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 220, 240), 1, cv2.LINE_AA)

        # 3. Per-person overlays
        for p in persons:
            tid = p.get("track_id", 0)
            prisk = person_risk(p)
            pcolor = self.get_risk_color(prisk)
            x1, y1, x2, y2 = p.get("bbox", [0, 0, 0, 0])
            features = p.get("features")
            keypoints = p.get("keypoints")
            angle = features.get("spine_angle", 0.0) if features else 0.0

            if show_bbox:
                cv2.rectangle(frame, (x1, y1), (x2, y2), pcolor, 2)
                label = f"P{tid} {prisk} | {angle:.0f}°"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + tw + 8, max(22, y1)), pcolor, -1)
                cv2.putText(frame, label, (x1 + 4, max(16, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA)
            else:
                # Still tag the person ID so tracks stay identifiable with overlays off
                cv2.putText(frame, f"P{tid}", (x1 + 4, max(16, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, pcolor, 1, cv2.LINE_AA)

            if show_skeleton and features:
                hip_px = features.get("mid_hip_px")
                sh_px = features.get("mid_shoulder_px")
                if hip_px and sh_px:
                    cv2.line(frame, hip_px, sh_px, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(frame, hip_px, 4, (0, 255, 255), -1)
                    cv2.circle(frame, sh_px, 4, (0, 255, 255), -1)

            if show_skeleton and keypoints and "all_landmarks" in keypoints:
                lms = keypoints["all_landmarks"]
                for p1_idx, p2_idx in POSE_CONNECTIONS:
                    if p1_idx >= len(lms) or p2_idx >= len(lms):
                        continue
                    lm1, lm2 = lms[p1_idx], lms[p2_idx]
                    if lm1.visibility > 0.4 and lm2.visibility > 0.4:
                        pt1 = (int(lm1.x * w), int(lm1.y * h))
                        pt2 = (int(lm2.x * w), int(lm2.y * h))
                        cv2.line(frame, pt1, pt2, pcolor, 2, cv2.LINE_AA)
                for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 0]:
                    if idx >= len(lms):
                        continue
                    lm = lms[idx]
                    if lm.visibility > 0.4:
                        pt = (int(lm.x * w), int(lm.y * h))
                        cv2.circle(frame, pt, 3, (255, 255, 255), -1)
                        cv2.circle(frame, pt, 2, pcolor, -1)

            # Levine's sign cardiac distress beacon & vector for person
            p_is_levine = (
                (features is not None and features.get("is_levine_gesture", False)) or
                p.get("fall_status", {}).get("behavior") == "LEVINE_SIGN_DISTRESS" or
                p.get("fall_status", {}).get("is_levine_sign", False) or
                p.get("fall_status", {}).get("state") == "CORONARY_DISTRESS"
            )
            if p_is_levine:
                sternum_px = None
                if features and "sternum_px" in features:
                    sternum_px = features["sternum_px"]
                elif features and "mid_shoulder_px" in features:
                    sternum_px = features["mid_shoulder_px"]
                elif keypoints and "left_shoulder" in keypoints and "right_shoulder" in keypoints:
                    ls, rs = keypoints["left_shoulder"], keypoints["right_shoulder"]
                    sternum_px = (int((ls["x_norm"] + rs["x_norm"]) / 2 * w), int((ls["y_norm"] + rs["y_norm"]) / 2 * h))

                if sternum_px is not None:
                    clutch_pts = []
                    if keypoints:
                        for side in ("left_wrist", "right_wrist"):
                            kp = keypoints.get(side)
                            if kp and kp.get("visibility", 1.0) >= 0.2:
                                wpt = (kp["x_px"], kp["y_px"])
                                dist = np.hypot(kp["x_norm"] - (sternum_px[0] / max(1, w)), kp["y_norm"] - (sternum_px[1] / max(1, h)))
                                if dist <= 0.22:
                                    clutch_pts.append(wpt)
                        if not clutch_pts:
                            for side in ("left_wrist", "right_wrist"):
                                kp = keypoints.get(side)
                                if kp and kp.get("visibility", 1.0) >= 0.2:
                                    clutch_pts.append((kp["x_px"], kp["y_px"]))
                    self.draw_levine_beacon(frame, sternum_px, clutch_pts, label=f"P{tid} LEVINE SIGN (CHEST PAIN)")

        # 4. Border reflects worst risk
        if worst == "EMERGENCY":
            cv2.rectangle(frame, (0, 0), (w, h), COLOR_EMERGENCY, 8)
        elif worst == "CONCERNING":
            cv2.rectangle(frame, (0, 0), (w, h), COLOR_CONCERNING, 4)

        return frame
