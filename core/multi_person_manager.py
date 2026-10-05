import time
from types import SimpleNamespace
from typing import Dict, Any, List, Optional

from core.pose_estimator import PoseEstimator
from core.feature_extractor import FeatureExtractor
from core.fall_detector import FallDetector
from core.inactivity_monitor import InactivityMonitor
from core.person_detector import PersonDetector
from core.person_tracker import CentroidTracker


def _remap_crop_keypoints(crop_keypoints: Dict[str, Any], x0: int, y0: int,
                          crop_w: int, crop_h: int,
                          full_w: int, full_h: int) -> Dict[str, Any]:
    """
    Convert crop-local normalized/px coords to full-frame coords.
    Rebuilds all_landmarks with adjusted x/y so the visualizer draws correctly.
    """
    def remap_point(pt: Dict[str, Any]) -> Dict[str, Any]:
        gx = (x0 + pt["x_norm"] * crop_w) / max(1, full_w)
        gy = (y0 + pt["y_norm"] * crop_h) / max(1, full_h)
        return {
            "x_norm": float(min(1.0, max(0.0, gx))),
            "y_norm": float(min(1.0, max(0.0, gy))),
            "z_norm": pt.get("z_norm", 0.0),
            "x_px": int(min(full_w - 1, max(0, x0 + pt["x_px"]))),
            "y_px": int(min(full_h - 1, max(0, y0 + pt["y_px"]))),
            "visibility": pt.get("visibility", 0.0),
        }

    out = {}
    for k, v in crop_keypoints.items():
        if isinstance(v, dict) and "x_norm" in v:
            out[k] = remap_point(v)
        elif k in ("frame_width", "frame_height", "all_landmarks"):
            continue
        else:
            out[k] = v
    out["frame_width"] = full_w
    out["frame_height"] = full_h

    # Remap full landmark list for skeleton drawing
    adj = []
    for lm in crop_keypoints.get("all_landmarks", []):
        gx = (x0 + lm.x * crop_w) / max(1, full_w)
        gy = (y0 + lm.y * crop_h) / max(1, full_h)
        adj.append(SimpleNamespace(x=float(gx), y=float(gy), z=lm.z, visibility=lm.visibility))
    out["all_landmarks"] = adj
    return out


class MultiPersonManager:
    """
    Multi-person pipeline (2-3 persons) for rapid prototyping.

    - PersonDetector (YOLOv8n ONNX, HOG fallback) runs every `detect_interval` frames.
    - CentroidTracker keeps stable track IDs between detections.
    - Each track owns its own PoseEstimator (temporal smoothing) +
      FeatureExtractor + FallDetector + InactivityMonitor.

    process(frame) -> list of per-person dicts:
      {track_id, bbox, has_pose, keypoints, features, fall_status, inactivity_status}
    """

    def __init__(
        self,
        max_persons: int = 3,
        angle_threshold: float = 58.0,
        velocity_threshold: float = 0.32,
        inactivity_timeout: float = 6.0,
        detect_interval: int = 5,
        model_complexity: int = 1,
        tracker_max_missing: int = 15,
        crop_padding: float = 0.15,
    ):
        self.max_persons = max(1, min(3, int(max_persons)))
        self.angle_threshold = angle_threshold
        self.velocity_threshold = velocity_threshold
        self.inactivity_timeout = inactivity_timeout
        self.detect_interval = max(1, int(detect_interval))
        self.model_complexity = model_complexity
        self.crop_padding = crop_padding

        self.detector = PersonDetector(max_persons=self.max_persons)
        self.tracker = CentroidTracker(max_missing=tracker_max_missing)
        self.persons: Dict[int, Dict[str, Any]] = {}
        self.frame_idx = 0
        self._last_boxes: List[List[int]] = []

    # ---------- config ----------

    def update_parameters(self, angle_threshold=None, velocity_threshold=None,
                          inactivity_timeout=None, max_persons=None, detect_interval=None):
        if angle_threshold is not None:
            self.angle_threshold = angle_threshold
        if velocity_threshold is not None:
            self.velocity_threshold = velocity_threshold
        if inactivity_timeout is not None:
            self.inactivity_timeout = inactivity_timeout
        if max_persons is not None:
            self.max_persons = max(1, min(3, int(max_persons)))
            self.detector.set_max_persons(self.max_persons)
        if detect_interval is not None:
            self.detect_interval = max(1, int(detect_interval))
        for p in self.persons.values():
            p["fall"].update_parameters(angle_threshold=self.angle_threshold,
                                        velocity_threshold=self.velocity_threshold)
            p["inact"].update_parameters(inactivity_timeout=self.inactivity_timeout)

    def reset(self):
        self.tracker.reset()
        self.persons.clear()
        self.frame_idx = 0
        self._last_boxes = []

    def close(self):
        for p in self.persons.values():
            try:
                p["pose"].close()
            except Exception:
                pass
        self.persons.clear()
        try:
            self.detector.close()
        except Exception:
            pass

    # ---------- per-track lifecycle ----------

    def _get_or_create_person(self, track_id: int) -> Dict[str, Any]:
        if track_id not in self.persons:
            self.persons[track_id] = {
                "pose": PoseEstimator(model_complexity=self.model_complexity),
                "extractor": FeatureExtractor(),
                "fall": FallDetector(angle_threshold=self.angle_threshold,
                                     velocity_threshold=self.velocity_threshold),
                "inact": InactivityMonitor(inactivity_timeout=self.inactivity_timeout),
            }
        return self.persons[track_id]

    def _evict_lost(self, active_ids):
        for tid in [t for t in self.persons if t not in active_ids]:
            try:
                self.persons[tid]["pose"].close()
            except Exception:
                pass
            del self.persons[tid]

    # ---------- main ----------

    def process(self, frame_bgr, current_time: Optional[float] = None) -> List[Dict[str, Any]]:
        import cv2
        if current_time is None:
            current_time = time.time()
        h, w = frame_bgr.shape[:2]
        self.frame_idx += 1

        # 1. Detect (every N frames) + track
        if self.frame_idx == 1 or (self.frame_idx % self.detect_interval == 0):
            boxes = self.detector.detect(frame_bgr)
            self._last_boxes = boxes
            tracked = self.tracker.update(boxes)
        else:
            # Reuse last assignments; age tracker only on detect frames.
            # Carry forward current tracks on last-known bboxes.
            tracked = sorted([(tid, list(t["bbox"])) for tid, t in self.tracker.tracks.items()])

        # Cap to max_persons (tracker may hold stale tracks briefly)
        tracked = tracked[: self.max_persons]
        active_ids = {tid for tid, _ in tracked}
        self._evict_lost(active_ids)

        results: List[Dict[str, Any]] = []
        for track_id, bbox in tracked:
            person = self._get_or_create_person(track_id)
            x1, y1, x2, y2 = [int(v) for v in bbox]
            # Expand crop with padding for shoulders/feet
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            px, py = int(bw * self.crop_padding), int(bh * self.crop_padding)
            cx0, cy0 = max(0, x1 - px), max(0, y1 - py)
            cx1, cy1 = min(w, x2 + px), min(h, y2 + py)
            crop = frame_bgr[cy0:cy1, cx0:cx1]
            crop_w, crop_h = max(1, cx1 - cx0), max(1, cy1 - cy0)

            fall_status = {"state": "NORMAL", "risk_level": "NORMAL",
                           "fall_confidence": 0.0, "total_falls": person["fall"].total_falls_detected}
            inactivity_status = {"is_inactive_alert": False, "inactive_duration": 0.0, "is_still": False}
            features = None
            keypoints = None
            has_pose = False

            if crop.size > 0 and crop_w >= 30 and crop_h >= 40:
                try:
                    has_pose, _, crop_kp = person["pose"].process_frame(crop)
                except Exception:
                    has_pose, crop_kp = False, None
                if has_pose and crop_kp:
                    keypoints = _remap_crop_keypoints(crop_kp, cx0, cy0, crop_w, crop_h, w, h)
                    features = person["extractor"].extract_features(keypoints, current_time=current_time)
                    temp_fall = {"state": person["fall"].current_state.value,
                                 "is_horizontal": features.get("spine_angle", 0) >= self.angle_threshold}
                    inactivity_status = person["inact"].process(features, temp_fall)
                    fall_status = person["fall"].process(
                        features, is_inactive=inactivity_status.get("is_inactive_alert", False))

            results.append({
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "has_pose": has_pose,
                "keypoints": keypoints,
                "features": features,
                "fall_status": fall_status,
                "inactivity_status": inactivity_status,
            })

        # Deterministic order
        results.sort(key=lambda r: r["track_id"])
        return results
