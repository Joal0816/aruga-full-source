import cv2
import numpy as np
try:
    import mediapipe as mp
except Exception:
    mp = None
from typing import Optional, Dict, Tuple, Any

class PoseEstimator:
    """
    Extracts 33 human body keypoints using MediaPipe Pose.
    Provides normalized and pixel coordinates for downstream fall/inactivity analysis.
    """
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        if mp is not None:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            self.pose = self.mp_pose.Pose(
                static_image_mode=static_image_mode,
                model_complexity=model_complexity,
                smooth_landmarks=smooth_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self._backend = None
        else:
            from core.yolo_pose_backend import YoloPoseBackend
            self.mp_pose = None
            self.mp_drawing = None
            self.mp_drawing_styles = None
            self.pose = None
            self._backend = YoloPoseBackend(max_persons=1)
        
    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[bool, Optional[Any], Optional[Dict[str, Any]]]:
        """
        Processes a BGR image frame.
        Returns:
            has_pose (bool): Whether a person/pose was detected.
            raw_landmarks: MediaPipe landmark list.
            keypoints (dict): Extracted key joint positions in normalized and pixel coords.
        """
        h, w, _ = frame_bgr.shape
        if self._backend is not None:
            persons = self._backend.infer(frame_bgr)
            if not persons:
                return False, None, None
            best = max(persons, key=lambda p: p["score"])
            kp = self._backend.to_keypoints(best, w, h)
            return True, kp.get("all_landmarks"), kp

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Performance optimization: mark image not writeable for mediapipe
        frame_rgb.flags.writeable = False
        results = self.pose.process(frame_rgb)
        frame_rgb.flags.writeable = True
        
        if not results.pose_landmarks:
            return False, None, None
            
        landmarks = results.pose_landmarks.landmark
        
        # Extract prominent anatomical landmarks
        # Key landmark indices in MediaPipe Pose:
        # 0: nose, 11: left_shoulder, 12: right_shoulder
        # 23: left_hip, 24: right_hip
        # 25: left_knee, 26: right_knee
        # 27: left_ankle, 28: right_ankle
        # 15: left_wrist, 16: right_wrist
        
        def to_point(lm):
            return {
                "x_norm": lm.x,
                "y_norm": lm.y,
                "z_norm": lm.z,
                "x_px": int(lm.x * w),
                "y_px": int(lm.y * h),
                "visibility": lm.visibility
            }
            
        keypoints = {
            "nose": to_point(landmarks[0]),
            "left_shoulder": to_point(landmarks[11]),
            "right_shoulder": to_point(landmarks[12]),
            "left_elbow": to_point(landmarks[13]),
            "right_elbow": to_point(landmarks[14]),
            "left_wrist": to_point(landmarks[15]),
            "right_wrist": to_point(landmarks[16]),
            "left_hip": to_point(landmarks[23]),
            "right_hip": to_point(landmarks[24]),
            "left_knee": to_point(landmarks[25]),
            "right_knee": to_point(landmarks[26]),
            "left_ankle": to_point(landmarks[27]),
            "right_ankle": to_point(landmarks[28]),
            "frame_width": w,
            "frame_height": h,
            "all_landmarks": landmarks
        }
        
        return True, results.pose_landmarks, keypoints

    def close(self):
        if self.pose:
            self.pose.close()
