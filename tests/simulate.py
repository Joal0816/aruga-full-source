"""
ARUGA hallway simulation harness — validates the full detection chain with ZERO footage.

Feeds scripted skeleton trajectories (stand / walk / fall / lie / sit / slouch)
through the REAL pipeline objects in app order:
    FeatureExtractor -> InactivityMonitor -> FallDetector -> HallwayRisk (+zones)

Scenarios (virtual 10 FPS clock, so the whole suite runs in milliseconds):
  walk_normal   upright walking, floor zone            -> always NORMAL, no events
  sit_normal    stand -> sit upright -> still, bench   -> NORMAL, no events
  fall_floor    stand -> fast drop -> lie still 8s     -> CONCERNING then EMERGENCY + logs
  nap_bench     sit -> slow slouch (56 deg) -> still   -> settles to RESTING, no logs
  slump_bench   stand -> fast drop into slouch -> still-> latches SLUMP, no EMERGENCY, 0 falls

Run:  .\\.venv\\Scripts\\python.exe tests\\simulate.py
Exit code 0 = all scenarios pass.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.feature_extractor import FeatureExtractor
from core.fall_detector import FallDetector
from core.inactivity_monitor import InactivityMonitor
from core.hallway_risk import HallwayRisk
from core.zones import ZoneManager
from utils.logger import EventLogger

FW, FH = 640, 480
DT = 0.1  # virtual 10 FPS


def joints(nose, sh, hips, knees, ankles, vis=0.9):
    """Build a keypoint dict. sh/hips/... are (x, y); L/R get tiny offsets."""
    def p(xy, dx=0.0):
        return {"x_norm": xy[0] + dx, "y_norm": xy[1], "z_norm": 0.0,
                "x_px": int((xy[0] + dx) * FW), "y_px": int(xy[1] * FH), "visibility": vis}

    return {
        "nose": p(nose),
        "left_shoulder": p(sh, -0.03), "right_shoulder": p(sh, 0.03),
        "left_elbow": p(((sh[0] + hips[0]) / 2 - 0.05, (sh[1] + hips[1]) / 2)),
        "right_elbow": p(((sh[0] + hips[0]) / 2 + 0.05, (sh[1] + hips[1]) / 2)),
        "left_wrist": p((hips[0] - 0.06, hips[1])), "right_wrist": p((hips[0] + 0.06, hips[1])),
        "left_hip": p(hips, -0.02), "right_hip": p(hips, 0.02),
        "left_knee": p(knees, -0.02), "right_knee": p(knees, 0.02),
        "left_ankle": p(ankles, -0.02), "right_ankle": p(ankles, 0.02),
        "frame_width": FW, "frame_height": FH, "all_landmarks": [],
    }


def bbox_of(j):
    xs = [j[k]["x_norm"] for k in ("nose", "left_shoulder", "right_shoulder", "left_hip",
                                   "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")]
    ys = [j[k]["y_norm"] for k in ("nose", "left_shoulder", "right_shoulder", "left_hip",
                                   "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")]
    m = 0.04
    return [int((min(xs) - m) * FW), int((min(ys) - m) * FH),
            int((max(xs) + m) * FW), int((max(ys) + m) * FH)]


def lerp_pose(a, b, p):
    """Interpolate two joint-spec dicts {joint: (x, y)}."""
    return {k: (a[k][0] + (b[k][0] - a[k][0]) * p, a[k][1] + (b[k][1] - a[k][1]) * p) for k in a}


# ---------- pose specs ----------
STAND_F = {"nose": (0.30, 0.20), "sh": (0.30, 0.30), "hips": (0.30, 0.55),
           "knees": (0.30, 0.75), "ankles": (0.30, 0.95)}
LIE_F = {"nose": (0.68, 0.60), "sh": (0.60, 0.60), "hips": (0.35, 0.60),
         "knees": (0.25, 0.60), "ankles": (0.15, 0.62)}
STAND_B = {"nose": (0.65, 0.20), "sh": (0.65, 0.30), "hips": (0.65, 0.55),
           "knees": (0.65, 0.75), "ankles": (0.65, 0.95)}
SIT_B = {"nose": (0.65, 0.34), "sh": (0.65, 0.42), "hips": (0.65, 0.62),
         "knees": (0.79, 0.62), "ankles": (0.79, 0.90)}
SLOUCH_B = {"nose": (0.81, 0.46), "sh": (0.77, 0.52), "hips": (0.62, 0.62),
            "knees": (0.75, 0.63), "ankles": (0.75, 0.90)}


def hold(spec, seconds, jitter=0.0, t0=0.0, i0=0):
    frames = []
    n = int(seconds / DT)
    for i in range(n):
        j = {k: (v[0], v[1] + (jitter if (i0 + i) % 2 == 0 else -jitter)) for k, v in spec.items()}
        frames.append((t0 + (i0 + i) * DT, joints(**j)))
    return frames


def transition(a, b, seconds, t0=0.0, i0=0):
    frames = []
    n = max(2, int(seconds / DT))
    for i in range(n):
        p = (i + 1) / n
        spec = lerp_pose(a, b, p)
        frames.append((t0 + (i0 + i) * DT, joints(**spec)))
    return frames


def make_zones():
    z = ZoneManager()
    z.zones = []  # drop default; define explicit floor + bench
    z.add_zone("hall_floor", "floor", [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    z.add_zone("bench_row", "bench", [[0.45, 0.35], [1.0, 0.35], [1.0, 1.0], [0.45, 1.0]])
    return z


def run_scenario(frames, angle=58.0, vel=0.32, timeout=3.0):
    ext = FeatureExtractor()
    fall = FallDetector(angle_threshold=angle, velocity_threshold=vel)
    inact = InactivityMonitor(inactivity_timeout=timeout)
    risk = HallwayRisk()
    zones = make_zones()
    logger = EventLogger(output_dir=tempfile.mkdtemp(prefix="sim_alerts_"))
    timeline = []
    # Rising-edge logging (same pattern as the apps): one FALL log per fall
    # incident, one INACTIVITY log per stillness episode. The old
    # if-FALL-elif-INACTIVITY chain starved INACTIVITY while still FALLEN.
    logged_falls, logged_inactive = 0, False
    for t, kp in frames:
        feats = ext.extract_features(kp, current_time=t)
        temp = {"state": fall.current_state.value,
                "is_horizontal": feats.get("spine_angle", 0) >= angle}
        ins = inact.process(feats, temp)
        fs = fall.process(feats, is_inactive=ins.get("is_inactive_alert", False))
        if fs.get("state") == "FALLEN" and fall.total_falls_detected > logged_falls:
            logged_falls = fall.total_falls_detected
            logger.log_event("FALL_DETECTED", feats, fs.get("fall_confidence", 0.8), None)
        if ins.get("is_inactive_alert") and not logged_inactive:
            logged_inactive = True
            logger.log_event("INACTIVITY_EMERGENCY", feats, 1.0, None,
                             extra_note=f"Stationary for {ins.get('inactive_duration'):.1f}s")
        elif not ins.get("is_inactive_alert"):
            logged_inactive = False
        zone = zones.zone_for_person(bbox_of(kp), FW, FH, kp)
        hw = risk.assess(feats, fs, ins, zone, kp)
        timeline.append((t, hw["risk_level"], zone["type"], round(feats["spine_angle"], 1)))
    return {"timeline": timeline, "fall": fall, "inact": inact,
            "events": list(logger.events)}


def risks(timeline):
    return [r for _, r, _, _ in timeline]


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    return cond


def main():
    ok = True
    t = 1000.0  # virtual clock base (large so FSM debounce timers behave)

    # ---- 1. walk_normal ----
    print("Scenario: walk_normal (3s upright walking, floor)")
    frames = hold(STAND_F, 3.0, jitter=0.008, t0=t)
    r = run_scenario(frames)
    ok &= check("always NORMAL", set(risks(r["timeline"])) == {"NORMAL"})
    ok &= check("no events", len(r["events"]) == 0)

    # ---- 2. sit_normal ----
    print("Scenario: sit_normal (stand 1s -> sit 1s -> still 5s, bench)")
    f2 = hold(STAND_B, 1.0, t0=t)
    f2 += transition(STAND_B, SIT_B, 1.0, t0=t, i0=len(f2))
    f2 += hold(SIT_B, 5.0, t0=t, i0=len(f2))
    r = run_scenario(f2)
    ok &= check("final NORMAL", risks(r["timeline"])[-1] == "NORMAL")
    ok &= check("no CONCERNING/SLUMP/EMERGENCY", not (set(risks(r["timeline"])) & {"CONCERNING", "SLUMP", "EMERGENCY"}))
    ok &= check("no events", len(r["events"]) == 0)

    # ---- 3. fall_floor ----
    print("Scenario: fall_floor (stand 1s -> drop 0.3s -> lie still 8s, floor)")
    f3 = hold(STAND_F, 1.0, t0=t)
    f3 += transition(STAND_F, LIE_F, 0.3, t0=t, i0=len(f3))
    f3 += hold(LIE_F, 8.0, t0=t, i0=len(f3))
    r = run_scenario(f3)
    tl = risks(r["timeline"])
    ok &= check("FALLEN detected", r["fall"].total_falls_detected >= 1,
                f"(falls={r['fall'].total_falls_detected})")
    ok &= check("CONCERNING reached", "CONCERNING" in tl)
    ok &= check("escalates to EMERGENCY", tl[-1] == "EMERGENCY")
    types = [e["event_type"] for e in r["events"]]
    ok &= check("FALL_DETECTED logged", "FALL_DETECTED" in types, f"events={types}")
    ok &= check("INACTIVITY_EMERGENCY logged", "INACTIVITY_EMERGENCY" in types)

    # ---- 4. nap_bench ----
    print("Scenario: nap_bench (sit 1s -> slow 2s slouch to ~56deg -> still 8s, bench)")
    f4 = hold(SIT_B, 1.0, t0=t)
    f4 += transition(SIT_B, SLOUCH_B, 2.0, t0=t, i0=len(f4))
    f4 += hold(SLOUCH_B, 8.0, t0=t, i0=len(f4))
    r = run_scenario(f4)
    tl = risks(r["timeline"])
    ok &= check("settles to RESTING", tl[-1] == "RESTING", f"(final={tl[-1]})")
    ok &= check("never CONCERNING/SLUMP/EMERGENCY",
                not (set(tl) & {"CONCERNING", "SLUMP", "EMERGENCY"}))
    ok &= check("zero falls counted", r["fall"].total_falls_detected == 0)
    ok &= check("no events logged", len(r["events"]) == 0)

    # ---- 5. slump_bench ----
    print("Scenario: slump_bench (stand 1s -> fast 0.3s drop into slouch -> still 8s, bench)")
    f5 = hold(STAND_B, 1.0, t0=t)
    f5 += transition(STAND_B, SLOUCH_B, 0.3, t0=t, i0=len(f5))
    f5 += hold(SLOUCH_B, 8.0, t0=t, i0=len(f5))
    r = run_scenario(f5)
    tl = risks(r["timeline"])
    ok &= check("SLUMP raised", "SLUMP" in tl)
    ok &= check("SLUMP latches while still reclined", tl[-1] == "SLUMP", f"(final={tl[-1]})")
    ok &= check("zero floor-falls counted", r["fall"].total_falls_detected == 0)
    ok &= check("no EMERGENCY escalation (conservative)", "EMERGENCY" not in tl)

    print("\n" + ("ALL SCENARIOS PASS" if ok else "SOME SCENARIOS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
