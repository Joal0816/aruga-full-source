"""
Unit and regression tests for Levine's Sign Biometric Detection & Coronary Distress State Machine.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.feature_extractor import FeatureExtractor
from core.fall_detector import FallDetector, SystemState, RiskLevel

FW, FH = 640, 480


def make_keypoints(sh_x=0.5, sh_y=0.3,
                   hip_x=0.5, hip_y=0.6,
                   lw_x=0.5, lw_y=0.32,
                   rw_x=0.5, rw_y=0.32,
                   spine_flex_dx=0.0):
    """
    Generate synthetic keypoint dict.
    spine_flex_dx controls horizontal offset between hips and shoulders, altering spine_angle.
    """
    def pt(x, y, vis=0.9):
        return {
            "x_norm": float(x),
            "y_norm": float(y),
            "z_norm": 0.0,
            "x_px": int(x * FW),
            "y_px": int(y * FH),
            "visibility": vis
        }

    return {
        "nose": pt(sh_x + spine_flex_dx * 0.5, sh_y - 0.1),
        "left_shoulder": pt(sh_x - 0.08 + spine_flex_dx, sh_y),
        "right_shoulder": pt(sh_x + 0.08 + spine_flex_dx, sh_y),
        "left_elbow": pt(sh_x - 0.06, (sh_y + hip_y) / 2),
        "right_elbow": pt(sh_x + 0.06, (sh_y + hip_y) / 2),
        "left_wrist": pt(lw_x, lw_y),
        "right_wrist": pt(rw_x, rw_y),
        "left_hip": pt(hip_x - 0.05, hip_y),
        "right_hip": pt(hip_x + 0.05, hip_y),
        "left_knee": pt(hip_x - 0.05, hip_y + 0.2),
        "right_knee": pt(hip_x + 0.05, hip_y + 0.2),
        "left_ankle": pt(hip_x - 0.05, hip_y + 0.35),
        "right_ankle": pt(hip_x + 0.05, hip_y + 0.35),
        "frame_width": FW,
        "frame_height": FH,
        "all_landmarks": []
    }


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}")
    return cond


def main():
    ok = True
    ext = FeatureExtractor()

    # ---- 1. Feature Extraction: Levine's Sign Biometrics ----
    print("Test: Levine's sign feature extraction geometry")

    # A: Clutching sternum with trunk flexion ~30 deg
    # dy = 0.3, dx = 0.3 * tan(30 deg) ~= 0.173
    kp_clutch = make_keypoints(sh_x=0.5, sh_y=0.3, hip_x=0.5, hip_y=0.6,
                               lw_x=0.68, lw_y=0.31, rw_x=0.65, rw_y=0.55,
                               spine_flex_dx=0.173)
    feats = ext.extract_features(kp_clutch, current_time=1.0)
    angle = feats["spine_angle"]
    w_dist = feats["wrist_sternum_dist"]
    is_gesture = feats["is_levine_gesture"]

    ok &= check("wrist to sternum is close", w_dist <= 0.16, f"(dist={w_dist:.3f})")
    ok &= check("spine angle in antalgic range 20-48 deg", 20.0 <= angle <= 48.0, f"(angle={angle:.1f} deg)")
    ok &= check("is_levine_gesture is True", is_gesture is True)

    # B: Hands at sides (normal standing)
    ext.reset()
    kp_normal = make_keypoints(sh_x=0.5, sh_y=0.3, hip_x=0.5, hip_y=0.6,
                               lw_x=0.35, lw_y=0.55, rw_x=0.65, rw_y=0.55,
                               spine_flex_dx=0.0)
    feats_norm = ext.extract_features(kp_normal, current_time=2.0)
    ok &= check("hands at sides gives is_levine_gesture False", feats_norm["is_levine_gesture"] is False)

    # C: Hands at chest but completely upright (spine_angle < 20 deg)
    ext.reset()
    kp_upright_clutch = make_keypoints(sh_x=0.5, sh_y=0.3, hip_x=0.5, hip_y=0.6,
                                       lw_x=0.5, lw_y=0.30, rw_x=0.5, rw_y=0.30,
                                       spine_flex_dx=0.0)
    feats_upright = ext.extract_features(kp_upright_clutch, current_time=3.0)
    ok &= check("chest clutch without trunk flexion is not Levine gesture",
                feats_upright["is_levine_gesture"] is False,
                f"(angle={feats_upright['spine_angle']:.1f})")

    # ---- 2. Fall Detector: State Transitions for Levine's Sign ----
    print("Test: FallDetector sustained coronary distress elevation")
    fd = FallDetector(levine_duration_threshold=0.8)

    # Initial frame with Levine gesture
    t = 100.0
    res = fd.process(feats, is_inactive=False)
    ok &= check("transient gesture does not immediately elevate before sustained threshold",
                res["state"] == "NORMAL" and res["is_levine_sign"] is False)

    # Sustain gesture past 0.8 seconds (e.g. 10 frames over 1.0 second)
    for i in range(1, 11):
        feats_t = dict(feats)
        feats_t["timestamp"] = t + i * 0.1
        res = fd.process(feats_t, is_inactive=False)

    ok &= check("sustained gesture triggers is_levine_sign", res["is_levine_sign"] is True)
    ok &= check("sustained gesture transitions state to CORONARY_DISTRESS", res["state"] == "CORONARY_DISTRESS")
    ok &= check("risk level elevated to CONCERNING", res["risk_level"] == "CONCERNING")
    ok &= check("behavior reports LEVINE_SIGN_DISTRESS", res["behavior"] == "LEVINE_SIGN_DISTRESS")
    ok &= check("clinical advisory provided", "Levine's sign detected" in res["clinical_advisory"])

    # ---- 3. FallDetector: Subsequent Fall from Coronary Distress ----
    print("Test: collapse from Coronary Distress transitions to FALLEN")
    # Sudden drop + horizontal posture
    fall_feats = {
        "timestamp": t + 2.0,
        "spine_angle": 75.0,
        "vertical_velocity": 0.45,
        "aspect_ratio": 1.5,
        "is_levine_gesture": True,
        "wrist_sternum_dist": 0.05
    }
    res_fall = fd.process(fall_feats, is_inactive=False)
    ok &= check("collapsing to floor transitions to FALLEN", res_fall["state"] == "FALLEN")
    ok &= check("alert is triggered", res_fall["alert_triggered"] is True)

    # ---- 4. Inactivity while fallen escalates to EMERGENCY ----
    print("Test: unresponsiveness escalates risk to EMERGENCY")
    res_inact = fd.process(fall_feats, is_inactive=True)
    ok &= check("inactivity escalates to EMERGENCY", res_inact["risk_level"] == "EMERGENCY")

    # ---- 5. Recovery to Normal when upright ----
    print("Test: recovery back to normal when upright and no distress")
    fd.reset()
    # Put in coronary distress
    for i in range(10):
        feats_t = dict(feats)
        feats_t["timestamp"] = t + i * 0.1
        fd.process(feats_t)
    assert fd.current_state == SystemState.CORONARY_DISTRESS

    # Release gesture and stand upright
    upright_feats = {
        "timestamp": t + 2.0,
        "spine_angle": 5.0,
        "vertical_velocity": 0.0,
        "aspect_ratio": 0.4,
        "is_levine_gesture": False,
        "wrist_sternum_dist": 0.35
    }
    # Wait past 0.5s grace period
    upright_feats["timestamp"] = t + 3.0
    res_rec = fd.process(upright_feats)
    ok &= check("releasing clutch and standing upright returns to NORMAL", res_rec["state"] == "NORMAL")

    print("\n" + ("ALL LEVINE SIGN TESTS PASS" if ok else "SOME TESTS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
