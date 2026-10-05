import os
import cv2
import time
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any, Optional

class EventLogger:
    """
    Records fall and inactivity events, manages snapshot captures,
    and exports incident logs.
    """
    def __init__(self, output_dir: str = "alerts"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.events: List[Dict[str, Any]] = []
        self.last_logged_time = 0.0
        self._last_logged_by_key: Dict[tuple, float] = {}

    def log_event(
        self,
        event_type: str,
        features: Optional[Dict[str, Any]],
        confidence: float,
        frame: Optional[Any] = None,
        extra_note: str = "",
        person_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        now = time.time()
        # Cooldown: prevent logging identical events multiple times within 1.5s.
        # Keyed per (event_type, person_id) so person #2's fall isn't suppressed by person #1.
        key = (event_type, person_id)
        last = self._last_logged_by_key.get(key, 0.0)
        if (now - last) < 1.5 and len(self.events) > 0:
            prev = self.events[-1]
            if prev.get("event_type") == event_type and prev.get("person_id") == person_id:
                return prev
        self._last_logged_by_key[key] = now
        # Legacy global timestamp kept for single-person callers.
        if person_id is None:
            if (now - self.last_logged_time) < 1.5 and len(self.events) > 0 and self.events[-1]["event_type"] == event_type:
                return self.events[-1]
            self.last_logged_time = now
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = ""
        
        if frame is not None:
            tag = f"{event_type.lower()}"
            if person_id is not None:
                tag = f"p{person_id}_{tag}"
            snapshot_filename = f"{tag}_{filename_ts}.jpg"
            snapshot_path = os.path.join(self.output_dir, snapshot_filename)
            cv2.imwrite(snapshot_path, frame)
            
        angle = features.get("spine_angle", 0.0) if features else 0.0
        vy = features.get("vertical_velocity", 0.0) if features else 0.0
        ar = features.get("aspect_ratio", 0.0) if features else 0.0
        
        event_record = {
            "id": len(self.events) + 1,
            "timestamp": timestamp_str,
            "person_id": person_id if person_id is not None else "-",
            "event_type": event_type,
            "confidence": f"{int(confidence * 100)}%",
            "spine_angle": f"{angle:.1f}°",
            "vertical_velocity": f"{vy:+.2f}",
            "aspect_ratio": f"{ar:.2f}",
            "snapshot_path": snapshot_path,
            "note": extra_note
        }
        
        self.events.append(event_record)
        return event_record

    def get_dataframe(self) -> pd.DataFrame:
        if not self.events:
            return pd.DataFrame(columns=[
                "id", "timestamp", "person_id", "event_type", "confidence", "spine_angle", "vertical_velocity", "aspect_ratio", "note"
            ])
        df = pd.DataFrame(self.events)
        return df[[col for col in df.columns if col != "snapshot_path"]]

    def clear(self):
        self.events.clear()
        self.last_logged_time = 0.0
        self._last_logged_by_key.clear()
