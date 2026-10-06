import numpy as np
import time
from collections import deque
from typing import Dict, Any, Optional, Tuple, List

class FeatureExtractor:
    """
    Extracts geometric, kinematic, and temporal features from pose keypoints:
    - Spine angle (degrees from vertical)
    - Keypoint bounding box aspect ratio (W / H)
    - Torso centroid vertical velocity (Vy) & acceleration
    - Joint mobility/motion energy (for inactivity detection)
    """
    def __init__(self, history_len: int = 15):
        self.history_len = history_len
        # Queue storing tuples of (timestamp, centroid_y_norm, spine_angle, keypoint_coords)
        self.history: deque = deque(maxlen=history_len)
        
    def reset(self):
        self.history.clear()

    def extract_features(self, keypoints: Dict[str, Any], current_time: Optional[float] = None) -> Dict[str, Any]:
        if current_time is None:
            current_time = time.time()
            
        ls = keypoints["left_shoulder"]
        rs = keypoints["right_shoulder"]
        lh = keypoints["left_hip"]
        rh = keypoints["right_hip"]
        lk = keypoints["left_knee"]
        rk = keypoints["right_knee"]
        la = keypoints["left_ankle"]
        ra = keypoints["right_ankle"]
        nose = keypoints["nose"]
        lw = keypoints.get("left_wrist")
        rw = keypoints.get("right_wrist")
        le = keypoints.get("left_elbow")
        re = keypoints.get("right_elbow")
        
        # 1. Midpoint calculations (normalized coords)
        mid_shoulder_x = (ls["x_norm"] + rs["x_norm"]) / 2.0
        mid_shoulder_y = (ls["y_norm"] + rs["y_norm"]) / 2.0
        
        mid_hip_x = (lh["x_norm"] + rh["x_norm"]) / 2.0
        mid_hip_y = (lh["y_norm"] + rh["y_norm"]) / 2.0
        
        mid_knee_x = (lk["x_norm"] + rk["x_norm"]) / 2.0
        mid_knee_y = (lk["y_norm"] + rk["y_norm"]) / 2.0
        
        mid_ankle_x = (la["x_norm"] + ra["x_norm"]) / 2.0
        mid_ankle_y = (la["y_norm"] + ra["y_norm"]) / 2.0
        
        # Centroid is torso center (midpoint between shoulders and hips)
        centroid_x = (mid_shoulder_x + mid_hip_x) / 2.0
        centroid_y = (mid_shoulder_y + mid_hip_y) / 2.0
        
        # Pixel coordinates
        frame_w = keypoints["frame_width"]
        frame_h = keypoints["frame_height"]
        centroid_px = (int(centroid_x * frame_w), int(centroid_y * frame_h))
        
        # 2. Spine Inclination Angle relative to vertical axis (0 deg = upright standing, 90 deg = horizontal lying down)
        # Vector from hip to shoulder:
        dx = mid_shoulder_x - mid_hip_x
        dy = mid_shoulder_y - mid_hip_y
        
        # In image coords, y increases downwards. If standing, shoulder is above hip (dy < 0).
        # We calculate angle with the vertical axis:
        spine_angle = np.degrees(np.arctan2(abs(dx), max(abs(dy), 1e-6)))
        
        # 3. Bounding Box & Aspect Ratio of prominent visible joints
        all_lms = keypoints.get("all_landmarks", [])
        if all_lms:
            xs = [lm.x for lm in all_lms if lm.visibility > 0.3]
            ys = [lm.y for lm in all_lms if lm.visibility > 0.3]
        else:
            xs = [ls["x_norm"], rs["x_norm"], lh["x_norm"], rh["x_norm"], lk["x_norm"], rk["x_norm"], la["x_norm"], ra["x_norm"]]
            ys = [ls["y_norm"], rs["y_norm"], lh["y_norm"], rh["y_norm"], lk["y_norm"], rk["y_norm"], la["y_norm"], ra["y_norm"]]
            
        if xs and ys:
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            bbox_w = max(0.01, max_x - min_x)
            bbox_h = max(0.01, max_y - min_y)
            aspect_ratio = bbox_w / bbox_h
            bbox_px = (
                int(max(0, min_x * frame_w)),
                int(max(0, min_y * frame_h)),
                int(min(frame_w, max_x * frame_w)),
                int(min(frame_h, max_y * frame_h))
            )
        else:
            aspect_ratio = 0.5
            bbox_px = (0, 0, frame_w, frame_h)
            
        # 4. Kinematic Motion & Vertical Velocity (Vy)
        # Positive Vy = moving downwards rapidly in image frame
        vertical_velocity = 0.0
        vertical_acceleration = 0.0
        motion_energy = 0.0 # Measure of movement for stillness detection
        
        # Major joint tracking for motion energy
        current_joint_pos = np.array([
            [mid_shoulder_x, mid_shoulder_y],
            [mid_hip_x, mid_hip_y],
            [nose["x_norm"], nose["y_norm"]],
            [mid_knee_x, mid_knee_y],
            [mid_ankle_x, mid_ankle_y]
        ])
        
        if len(self.history) > 0:
            prev_time, prev_y, prev_angle, prev_joints, prev_vy = self.history[-1]
            dt = max(0.001, current_time - prev_time)
            
            # Instantaneous downward velocity
            inst_vy = (centroid_y - prev_y) / dt

            # Smoothed velocity over a short fixed lookback (~0.35s).
            # NOTE: previously this used the whole history buffer, which diluted
            # fast drops whenever it was full of standing frames and made Vy
            # frame-rate dependent. Fixed lookback keeps drop response identical
            # at 10 FPS (Ryzen target) and 30 FPS (webcam).
            if len(self.history) >= 2:
                ref_time, ref_y = self.history[0][0], self.history[0][1]
                for entry in self.history:
                    if current_time - entry[0] > 0.35:
                        ref_time, ref_y = entry[0], entry[1]
                    else:
                        break
                window_dt = max(0.001, current_time - ref_time)
                vertical_velocity = (centroid_y - ref_y) / window_dt
            else:
                vertical_velocity = inst_vy
                
            vertical_acceleration = (vertical_velocity - prev_vy) / dt
            
            # Motion energy: mean Euclidean displacement of joints normalized per second
            disp = np.linalg.norm(current_joint_pos - prev_joints, axis=1)
            motion_energy = float(np.mean(disp) / dt)
        else:
            inst_vy = 0.0

        # Store in sliding history
        self.history.append((current_time, centroid_y, spine_angle, current_joint_pos, vertical_velocity))
        
        # 5. Head vs Hip relative height (normalized)
        # When standing, head_y < hip_y. When fallen or head on floor, head_y >= hip_y or very close
        head_hip_diff = mid_hip_y - nose["y_norm"]

        # 6. Levine's Sign Biometric Cues (Cardiac Distress)
        # Sternum midpoint: ((l11.x + l12.x)/2, (l11.y + l12.y)/2)
        sternum_x = mid_shoulder_x
        sternum_y = mid_shoulder_y

        d_left, dy_left = 999.0, 999.0
        if lw is not None and lw.get("visibility", 1.0) >= 0.2:
            d_left = float(np.hypot(lw["x_norm"] - sternum_x, lw["y_norm"] - sternum_y))
            dy_left = abs(float(lw["y_norm"]) - sternum_y)

        d_right, dy_right = 999.0, 999.0
        if rw is not None and rw.get("visibility", 1.0) >= 0.2:
            d_right = float(np.hypot(rw["x_norm"] - sternum_x, rw["y_norm"] - sternum_y))
            dy_right = abs(float(rw["y_norm"]) - sternum_y)

        if d_left == 999.0 and d_right == 999.0:
            wrist_dist, wrist_dy = 1.0, 1.0
        elif d_left <= d_right:
            wrist_dist, wrist_dy = d_left, dy_left
        else:
            wrist_dist, wrist_dy = d_right, dy_right

        # Flag is_levine_gesture: clutching against sternum + antalgic trunk flexion.
        # wrist_dy: a clutching wrist sits AT sternum height; a hand resting in
        # the lap during a slow slouch hangs >=0.10 below it (false-positive guard).
        is_levine_gesture = bool((wrist_dist <= 0.16) and (wrist_dy <= 0.05)
                                 and (20.0 <= spine_angle <= 48.0))
        
        return {
            "timestamp": current_time,
            "centroid_norm": (centroid_x, centroid_y),
            "centroid_px": centroid_px,
            "mid_shoulder_px": (int(mid_shoulder_x * frame_w), int(mid_shoulder_y * frame_h)),
            "mid_hip_px": (int(mid_hip_x * frame_w), int(mid_hip_y * frame_h)),
            "spine_angle": float(spine_angle),
            "aspect_ratio": float(aspect_ratio),
            "vertical_velocity": float(vertical_velocity),
            "vertical_acceleration": float(vertical_acceleration),
            "motion_energy": float(motion_energy),
            "head_hip_diff": float(head_hip_diff),
            "bbox_px": bbox_px,
            "sternum_px": (int(sternum_x * frame_w), int(sternum_y * frame_h)),
            "sternum_norm": (float(sternum_x), float(sternum_y)),
            "wrist_sternum_dist": float(wrist_dist),
            "left_wrist_sternum_dist": float(d_left if d_left != 999.0 else 1.0),
            "right_wrist_sternum_dist": float(d_right if d_right != 999.0 else 1.0),
            "is_levine_gesture": bool(is_levine_gesture)
        }

    extract = extract_features
