import time
from enum import Enum
from typing import Dict, Any, Optional

class SystemState(Enum):
    NORMAL = "NORMAL"
    PRE_FALL = "PRE_FALL"
    FALLEN = "FALLEN"
    INACTIVE_ALERT = "INACTIVE_ALERT"
    RECOVERED = "RECOVERED"
    CORONARY_DISTRESS = "CORONARY_DISTRESS"

class RiskLevel(Enum):
    NORMAL = "NORMAL"
    UNUSUAL = "UNUSUAL"
    CONCERNING = "CONCERNING"
    EMERGENCY = "EMERGENCY"

class FallDetector:
    """
    ARUGA Contextual Risk Assessment & Fall Detection Engine:
    - Monitors Spine Angle (theta)
    - Monitors Centroid Downward Velocity (Vy)
    - Monitors Bounding Box Aspect Ratio (AR)
    - Categorizes postures into 4 Risk Levels:
      1. NORMAL: Stable upright stance / routine movement
      2. UNUSUAL: Severe tilt, heavy leaning, stumble or descent
      3. CONCERNING: Horizontal position / person down on floor
      4. EMERGENCY: Prolonged stillness / unresponsiveness post-fall
    """
    def __init__(
        self,
        angle_threshold: float = 58.0,
        velocity_threshold: float = 0.32,
        aspect_ratio_threshold: float = 1.15,
        pre_fall_window: float = 1.0,
        recovery_confirm_duration: float = 1.2,
        levine_duration_threshold: float = 0.8
    ):
        self.angle_threshold = angle_threshold
        self.velocity_threshold = velocity_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.pre_fall_window = pre_fall_window
        self.recovery_confirm_duration = recovery_confirm_duration
        self.levine_duration_threshold = levine_duration_threshold
        
        self.current_state = SystemState.NORMAL
        self.risk_level = RiskLevel.NORMAL
        self.risk_description = "Normal upright posture & mobility"
        self.last_state_change = time.time()
        self.last_pre_fall_time = 0.0
        self.recovery_candidate_start = 0.0
        self.fall_timestamp: Optional[float] = None
        self.fall_confidence = 0.0
        self.alert_triggered = False
        self.total_falls_detected = 0

        # Levine's sign (coronary distress) tracking
        self.levine_start_time = 0.0
        self.last_levine_time = 0.0
        self.is_levine_sign = False
        self.behavior = "NORMAL"
        self.clinical_advisory = ""

    def update_parameters(
        self,
        angle_threshold: Optional[float] = None,
        velocity_threshold: Optional[float] = None,
        aspect_ratio_threshold: Optional[float] = None,
        levine_duration_threshold: Optional[float] = None
    ):
        if angle_threshold is not None:
            self.angle_threshold = angle_threshold
        if velocity_threshold is not None:
            self.velocity_threshold = velocity_threshold
        if aspect_ratio_threshold is not None:
            self.aspect_ratio_threshold = aspect_ratio_threshold
        if levine_duration_threshold is not None:
            self.levine_duration_threshold = levine_duration_threshold

    def reset(self):
        self.current_state = SystemState.NORMAL
        self.last_state_change = time.time()
        self.last_pre_fall_time = 0.0
        self.recovery_candidate_start = 0.0
        self.fall_timestamp = None
        self.fall_confidence = 0.0
        self.alert_triggered = False
        self.levine_start_time = 0.0
        self.last_levine_time = 0.0
        self.is_levine_sign = False
        self.behavior = "NORMAL"
        self.clinical_advisory = ""

    def process(self, features: Dict[str, Any], is_inactive: bool = False) -> Dict[str, Any]:
        """
        Evaluates current features and transitions state machine.
        """
        now = features["timestamp"]
        spine_angle = features["spine_angle"]
        vy = features["vertical_velocity"]
        ar = features["aspect_ratio"]
        
        is_horizontal = (spine_angle >= self.angle_threshold) or (ar >= self.aspect_ratio_threshold and spine_angle >= 45.0)
        is_fast_drop = vy >= self.velocity_threshold
        is_upright = (spine_angle < 38.0) and (ar < 0.9)
        
        # 1. Detect rapid downward descent (Pre-Fall / Falling motion)
        is_levine = features.get("is_levine_gesture", False)
        if is_levine:
            self.last_levine_time = now
            if self.levine_start_time == 0.0:
                self.levine_start_time = now
            if (now - self.levine_start_time) >= self.levine_duration_threshold:
                self.is_levine_sign = True
        else:
            if self.levine_start_time > 0.0 and (now - self.last_levine_time > 0.5):
                self.levine_start_time = 0.0
                self.is_levine_sign = False

        if is_fast_drop and (now - self.last_pre_fall_time > 0.5):
            self.last_pre_fall_time = now
            if self.current_state in [SystemState.NORMAL, SystemState.CORONARY_DISTRESS]:
                self.current_state = SystemState.PRE_FALL
                self.last_state_change = now

        # Check if pre-fall expired without landing horizontal
        if self.current_state == SystemState.PRE_FALL:
            if (now - self.last_state_change) > self.pre_fall_window and not is_horizontal:
                self.current_state = SystemState.CORONARY_DISTRESS if self.is_levine_sign else SystemState.NORMAL
                self.last_state_change = now

        # 2. Transition into FALLEN
        # Case A: Rapid drop followed immediately by horizontal orientation (Standard dynamic fall)
        recent_pre_fall = (now - self.last_pre_fall_time) <= self.pre_fall_window
        
        if is_horizontal:
            if self.current_state in [SystemState.NORMAL, SystemState.PRE_FALL, SystemState.RECOVERED, SystemState.CORONARY_DISTRESS]:
                if recent_pre_fall:
                    self.fall_confidence = min(0.98, 0.70 + (vy / max(self.velocity_threshold, 0.01)) * 0.15 + (spine_angle / 90.0) * 0.15)
                else:
                    # Slow fall / collapse / fainting (gradual slip to floor)
                    self.fall_confidence = min(0.85, 0.50 + (spine_angle / 90.0) * 0.35)
                
                self.current_state = SystemState.FALLEN
                self.fall_timestamp = now
                self.last_state_change = now
                self.alert_triggered = True
                self.total_falls_detected += 1

        # 3. Detect Recovery (Standing back up after fall)
        elif self.current_state in [SystemState.FALLEN, SystemState.INACTIVE_ALERT]:
            if is_upright:
                if self.recovery_candidate_start == 0.0:
                    self.recovery_candidate_start = now
                elif (now - self.recovery_candidate_start) >= self.recovery_confirm_duration:
                    self.current_state = SystemState.RECOVERED
                    self.last_state_change = now
                    self.recovery_candidate_start = 0.0
                    self.alert_triggered = False
            else:
                self.recovery_candidate_start = 0.0

        # 4. Transition from RECOVERED back to NORMAL after stabilization
        if self.current_state == SystemState.RECOVERED:
            if (now - self.last_state_change) > 2.0 and is_upright:
                self.current_state = SystemState.NORMAL
                self.last_state_change = now
                self.fall_confidence = 0.0

        # 5. Transition to/from CORONARY_DISTRESS
        if self.is_levine_sign:
            if self.current_state in [SystemState.NORMAL, SystemState.RECOVERED]:
                self.current_state = SystemState.CORONARY_DISTRESS
                self.last_state_change = now
        elif self.current_state == SystemState.CORONARY_DISTRESS:
            if not is_levine and is_upright:
                self.current_state = SystemState.NORMAL
                self.last_state_change = now

        duration_in_state = now - self.last_state_change

        # 6. Behavioral pattern & clinical advisory reporting
        if self.is_levine_sign:
            self.behavior = "LEVINE_SIGN_DISTRESS"
            self.clinical_advisory = "Levine's sign detected: clutches chest with forward trunk flexion (~50% MI Risk)"
        elif self.current_state == SystemState.FALLEN:
            self.behavior = "FALL_DETECTED"
            self.clinical_advisory = ""
        elif is_inactive or self.current_state == SystemState.INACTIVE_ALERT:
            self.behavior = "PROLONGED_STILLNESS"
            self.clinical_advisory = ""
        else:
            self.behavior = "NORMAL"
            self.clinical_advisory = ""

        # 7. Determine 4-Tier Contextual Risk Level
        if is_inactive or self.current_state == SystemState.INACTIVE_ALERT:
            self.risk_level = RiskLevel.EMERGENCY
            self.risk_description = "EMERGENCY: Prolonged unresponsiveness / stillness detected"
        elif self.current_state == SystemState.FALLEN or is_horizontal:
            self.risk_level = RiskLevel.CONCERNING
            self.risk_description = "CONCERNING: Fallen or horizontal posture detected"
        elif self.is_levine_sign or self.current_state == SystemState.CORONARY_DISTRESS:
            self.risk_level = RiskLevel.CONCERNING
            self.risk_description = "CONCERNING: Levine's sign detected: clutches chest with forward trunk flexion (~50% MI Risk)"
        elif self.current_state == SystemState.PRE_FALL or is_fast_drop or (spine_angle >= 36.0 and not is_upright) or ar >= 0.95:
            self.risk_level = RiskLevel.UNUSUAL
            self.risk_description = "UNUSUAL: Severe tilt, heavy leaning, or rapid transition"
        else:
            self.risk_level = RiskLevel.NORMAL
            self.risk_description = "NORMAL: Upright posture & routine mobility"

        return {
            "state": self.current_state.value,
            "state_enum": self.current_state,
            "risk_level": self.risk_level.value,
            "risk_level_enum": self.risk_level,
            "risk_description": self.risk_description,
            "duration_in_state": duration_in_state,
            "fall_confidence": float(self.fall_confidence),
            "alert_triggered": self.alert_triggered,
            "total_falls": self.total_falls_detected,
            "is_horizontal": is_horizontal,
            "is_fast_drop": is_fast_drop,
            "is_levine_sign": bool(self.is_levine_sign),
            "behavior": self.behavior,
            "clinical_advisory": self.clinical_advisory
        }

