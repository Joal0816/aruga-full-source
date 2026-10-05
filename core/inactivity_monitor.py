import time
from typing import Dict, Any, Optional

class InactivityMonitor:
    """
    Monitors unresponsiveness and prolonged stillness following a fall or collapse.
    Calculates elapsed inactive time and triggers emergency warnings.
    """
    def __init__(
        self,
        inactivity_timeout: float = 6.0, # seconds before triggering inactive alert
        motion_threshold: float = 0.08    # normalized joint motion energy threshold
    ):
        self.inactivity_timeout = inactivity_timeout
        self.motion_threshold = motion_threshold
        
        self.inactivity_start_time: Optional[float] = None
        self.is_inactive_alert: bool = False
        self.total_inactive_duration: float = 0.0
        self.alert_count: int = 0

    def update_parameters(self, inactivity_timeout: Optional[float] = None, motion_threshold: Optional[float] = None):
        if inactivity_timeout is not None:
            self.inactivity_timeout = inactivity_timeout
        if motion_threshold is not None:
            self.motion_threshold = motion_threshold

    def reset(self):
        self.inactivity_start_time = None
        self.is_inactive_alert = False
        self.total_inactive_duration = 0.0

    def process(self, features: Dict[str, Any], fall_status: Dict[str, Any]) -> Dict[str, Any]:
        now = features["timestamp"]
        motion_energy = features["motion_energy"]
        is_horizontal = fall_status["is_horizontal"]
        current_state = fall_status["state"]
        
        # Consider a candidate for inactivity if:
        # 1) The person is in FALLEN or INACTIVE_ALERT state OR
        # 2) The person is horizontal and motion is below threshold
        is_still = motion_energy < self.motion_threshold
        inactivity_condition = (current_state in ["FALLEN", "INACTIVE_ALERT"]) or (is_horizontal and is_still)
        
        if inactivity_condition:
            if self.inactivity_start_time is None:
                self.inactivity_start_time = now
            self.total_inactive_duration = now - self.inactivity_start_time
            
            if self.total_inactive_duration >= self.inactivity_timeout:
                if not self.is_inactive_alert:
                    self.alert_count += 1
                self.is_inactive_alert = True
        else:
            # Person stood up or moved significantly
            self.inactivity_start_time = None
            self.total_inactive_duration = 0.0
            self.is_inactive_alert = False

        severity = "NORMAL"
        if self.is_inactive_alert:
            severity = "CRITICAL" if self.total_inactive_duration > (self.inactivity_timeout * 1.8) else "WARNING"

        return {
            "is_inactive_alert": self.is_inactive_alert,
            "inactive_duration": float(self.total_inactive_duration),
            "inactivity_timeout": float(self.inactivity_timeout),
            "is_still": bool(is_still),
            "motion_energy": float(motion_energy),
            "severity": severity,
            "alert_count": self.alert_count
        }
