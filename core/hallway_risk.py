from typing import Dict, Any, Optional

# Extended risk levels for hallway / waiting-area monitoring.
# NORMAL / UNUSUAL / CONCERNING / EMERGENCY keep their original meaning.
# RESTING: reclined + still in a bench zone with no preceding drop (napping visitor).
# SLUMP:   rapid drop ending reclined in a bench zone + still (possible faint — needs a check).
RISK_DESCRIPTIONS = {
    "NORMAL": "NORMAL: Upright posture & routine mobility",
    "RESTING": "RESTING: Reclined and still in seating area (no alarm)",
    "UNUSUAL": "UNUSUAL: Severe tilt, heavy leaning, or rapid transition",
    "SLUMP": "SLUMP: Sudden collapse into seating — check on person",
    "CONCERNING": "CONCERNING: Fallen or horizontal posture detected",
    "EMERGENCY": "EMERGENCY: Prolonged unresponsiveness / stillness detected",
}

RISK_ORDER = {"NORMAL": 0, "RESTING": 1, "UNUSUAL": 2, "SLUMP": 3, "CONCERNING": 4, "EMERGENCY": 5}


def seated_score(keypoints: Optional[Dict[str, Any]]) -> float:
    """0..1 estimate that the person is sitting (thighs horizontal).

    Standing: hips well above knees. Sitting: hips ~= knee height.
    Returns 0 when knees are occluded (bench!) rather than guessing.
    y grows downward, so standing gives knee_y - hip_y >> 0.
    """
    if not keypoints:
        return 0.0
    try:
        ls, rs = keypoints["left_shoulder"], keypoints["right_shoulder"]
        lh, rh = keypoints["left_hip"], keypoints["right_hip"]
        lk, rk = keypoints["left_knee"], keypoints["right_knee"]
    except (KeyError, TypeError):
        return 0.0
    if lk["visibility"] < 0.3 or rk["visibility"] < 0.3:
        return 0.0
    if lh["visibility"] < 0.2 or rh["visibility"] < 0.2:
        return 0.0
    torso = abs((ls["y_norm"] + rs["y_norm"]) / 2.0 - (lh["y_norm"] + rh["y_norm"]) / 2.0)
    torso = max(0.05, torso)
    rise = ((lk["y_norm"] + rk["y_norm"]) / 2.0 - (lh["y_norm"] + rh["y_norm"]) / 2.0) / torso
    # Standing rise ~= 0.8-1.2 (knee a full torso below hip); sitting rise ~= 0-0.3.
    if rise >= 0.6:
        return 0.0
    if rise <= 0.15:
        return 1.0
    return float((0.6 - rise) / 0.45)


class HallwayRisk:
    """Per-person fusion: kinematics (FSM) + zone + seated cues -> hallway risk.

    Conservative by design:
    - Zones can only *downgrade* CONCERNING (bench context), never upgrade
      a calm NORMAL into an alarm, and never suppress EMERGENCY.
    - A rapid drop is remembered briefly: reclined-in-bench shortly after a
      drop = SLUMP (needs check); reclined-in-bench with no drop = RESTING.
    """

    def __init__(self, drop_memory: float = 2.5):
        self.drop_memory = drop_memory  # seconds a fast drop stays suspicious
        self.last_drop_time: float = 0.0
        self._latched_slump: bool = False

    def reset(self):
        self.last_drop_time = 0.0
        self._latched_slump = False

    def assess(self, features: Optional[Dict[str, Any]],
               fall_status: Dict[str, Any],
               inactivity_status: Dict[str, Any],
               zone: Dict[str, Any],
               keypoints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        now = (features or {}).get("timestamp", 0.0)
        if fall_status.get("is_fast_drop"):
            self.last_drop_time = now
        recent_drop = (now - self.last_drop_time) <= self.drop_memory

        is_inactive = bool(inactivity_status.get("is_inactive_alert", False))
        state = fall_status.get("state", "NORMAL")
        base_risk = fall_status.get("risk_level", "NORMAL")
        if is_inactive:
            base_risk = "EMERGENCY"
        ztype = zone.get("type", "floor")
        seated = seated_score(keypoints)
        angle = (features or {}).get("spine_angle", 0.0)
        reclined = angle >= 45.0

        risk, downgraded = base_risk, False
        if base_risk == "CONCERNING" and not is_inactive:
            if ztype == "bench":
                # Reclined on seating: slump if it started with a drop, else resting.
                risk = "SLUMP" if (recent_drop or seated > 0.5) and reclined else "RESTING"
                # ...unless clearly flat-out horizontal (angle extreme = likely slid off).
                if angle >= 75.0:
                    risk = "CONCERNING"
                downgraded = risk != "CONCERNING"
            elif ztype == "ignore":
                risk = "UNUSUAL"
                downgraded = True

        # Tilted-but-not-horizontal in seating + still + no recent drop = napping, not a fall.
        is_still = bool(inactivity_status.get("is_still", False))
        if risk == "UNUSUAL" and ztype == "bench" and reclined \
                and not recent_drop and is_still and not is_inactive:
            risk, downgraded = "RESTING", True

        # Fresh fast drop in a bench zone deserves SLUMP even pre-horizontal.
        if risk in ("NORMAL", "UNUSUAL", "RESTING") and ztype == "bench" \
                and fall_status.get("is_fast_drop") and not is_inactive:
            risk, downgraded = "SLUMP", False

        # Latch SLUMP while the person stays reclined in seating: a slump must
        # not decay into RESTING just because the drop memory expired.
        if risk == "SLUMP":
            self._latched_slump = True
        elif not reclined or ztype != "bench":
            self._latched_slump = False
        if self._latched_slump and risk == "RESTING":
            risk, downgraded = "SLUMP", False

        return {
            "risk_level": risk,
            "risk_description": RISK_DESCRIPTIONS.get(risk, risk),
            "base_risk": base_risk,
            "downgraded": downgraded,
            "zone": zone.get("name", "?"),
            "zone_type": ztype,
            "seated_score": float(seated),
            "reclined": bool(reclined),
            "recent_drop": bool(recent_drop),
        }
