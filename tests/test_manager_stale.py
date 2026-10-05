"""
Regression tests for hallway manager robustness (no footage needed).

  1. confirm-gating: sub-threshold boxes never spawn tracks (no ghosts),
     but DO maintain an existing track (fallen bodies score low).
  2. stale + latch: an alarming track that vanishes is held advancing,
     then latched unverified after the cap (frozen path needs no keypoints).
  3. hold + escalate while blind; cap expiry; overlap release.
  4. recovery handoff: stand-up overlapping a held alarm resolves it with
     RECOVERED instead of stacking phantom boxes.
  5. tracker IoU fallback rescues abrupt lying->standing geometry flips.
  6. pose-union bbox covers confident keypoints outside the raw box.

Run:  .\\.venv\\Scripts\\python.exe tests\\test_manager_stale.py
Exit code 0 = pass.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.hallway_manager import HallwayManager
from core.feature_extractor import FeatureExtractor
from core.fall_detector import FallDetector
from core.inactivity_monitor import InactivityMonitor
from core.hallway_risk import HallwayRisk

FW, FH = 640, 480


def standing_kpts(conf=0.9):
    k = np.zeros((17, 3), dtype=np.float32)
    pts = {0: (320, 96), 5: (300, 144), 6: (340, 144), 7: (290, 200), 8: (350, 200),
           9: (285, 260), 10: (355, 260), 11: (305, 264), 12: (335, 264),
           13: (305, 360), 14: (335, 360), 15: (305, 456), 16: (335, 456)}
    for idx, (x, y) in pts.items():
        k[idx] = (x, y, conf)
    return k


def lying_kpts(conf=0.9):
    """Clearly horizontal body (spine ~90 deg)."""
    k = np.zeros((17, 3), dtype=np.float32)
    pts = {0: (440, 290), 5: (370, 290), 6: (410, 290), 7: (340, 292), 8: (430, 292),
           9: (320, 294), 10: (450, 294), 11: (220, 290), 12: (250, 290),
           13: (160, 292), 14: (180, 292), 15: (100, 300), 16: (120, 300)}
    for idx, (x, y) in pts.items():
        k[idx] = (x, y, conf)
    return k


def make_mgr(det_score=0.9, kpt_conf=0.9):
    mgr = HallwayManager(max_persons=4, providers=["CPUExecutionProvider"])
    mgr.infer_calls = 0

    def fake_infer(frame):
        mgr.infer_calls += 1
        if fake_infer.on:
            return [{"bbox": [250, 50, 390, 470], "score": det_score,
                     "kpts": standing_kpts(kpt_conf)}]
        return []

    fake_infer.on = True
    mgr.backend.infer = fake_infer
    return mgr, fake_infer


def blank():
    return np.zeros((FH, FW, 3), dtype=np.uint8)


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    return cond


def main():
    ok = True
    t = 5000.0

    # ---- 1. confirm gating ----
    print("Test: weak boxes never spawn tracks")
    mgr, fake = make_mgr(det_score=0.30)
    out = None
    for i in range(5):
        out = mgr.process(blank(), current_time=t + i * 0.1)
    live = [p for p in out if not p.get("stale")]
    ok &= check("no track from 0.30 detections", len(live) == 0 and len(mgr.persons) == 0)

    print("Test: strong detection spawns, weak maintains")
    mgr, fake = make_mgr(det_score=0.90)
    out = mgr.process(blank(), current_time=t)
    live = [p for p in out if not p.get("stale")]
    ok &= check("track spawned at 0.90", len(live) == 1)
    tid = live[0]["track_id"]
    # Now drop to weak: same track must persist (maintain path)
    weak_hit = False
    orig_infer = mgr.backend.infer

    def weak_infer(frame):
        return [{"bbox": [250, 50, 390, 470], "score": 0.25, "kpts": standing_kpts(0.9)}]

    mgr.backend.infer = weak_infer
    for i in range(1, 6):
        out = mgr.process(blank(), current_time=t + i * 0.1)
    live = [p for p in out if not p.get("stale")]
    ok &= check("same track maintained at 0.25", len(live) == 1 and live[0]["track_id"] == tid)
    mgr.backend.infer = orig_infer

    # ---- 2. stale + latch (held entry; keypoints required to re-feed) ----
    print("Test: lost alarming track -> stale -> latched alarm")
    mgr, fake = make_mgr()
    mgr.persons[7] = {"extractor": FeatureExtractor(), "fall": FallDetector(),
                      "inact": InactivityMonitor(), "risk": HallwayRisk(),
                      "live_updates": 5,
                      "last_hallway": {"risk_level": "CONCERNING", "risk_description": "x",
                                       "base_risk": "CONCERNING", "downgraded": False,
                                       "zone": "hall_floor", "zone_type": "floor",
                                       "seated_score": 0.0, "reclined": True, "recent_drop": True},
                      "last_bbox": [250, 50, 390, 470],
                      "last_keypoints": mgr.backend.to_keypoints(
                          {"bbox": [250, 50, 390, 470], "score": 0.9,
                           "kpts": lying_kpts(0.9)}, FW, FH)}
    fake.on = False  # nobody visible from here on
    out = mgr.process(blank(), current_time=t)
    stale = [p for p in out if p.get("stale")]
    ok &= check("stale entry shown after loss", len(stale) == 1 and stale[0]["track_id"] == 7)
    ok &= check("no latch yet (within hold window)", len(mgr.lost_alarms) == 0)
    mgr.hold_max = 5.0
    out = mgr.process(blank(), current_time=t + 11.0)
    stale = [p for p in out if p.get("stale")]
    ok &= check("stale expired after timeout", len(stale) == 0)
    ok &= check("alarm latched as unverified",
                len(mgr.lost_alarms) == 1 and mgr.lost_alarms[0]["track_id"] == 7
                and mgr.lost_alarms[0]["risk_level"] == "CONCERNING")
    mgr.reset()
    ok &= check("reset clears latch", len(mgr.lost_alarms) == 0 and len(mgr.stale) == 0)

    # ---- 3. hold + escalate while blind (no pixel logic: advance always) ----
    print("Test: held track escalates CONCERNING -> EMERGENCY while blind")
    mgr, fake = make_mgr()
    det = {"bbox": [50, 200, 500, 380], "score": 0.9, "kpts": lying_kpts(0.9)}
    kp = mgr.backend.to_keypoints(det, FW, FH)
    bbox = [50, 200, 500, 380]
    mgr.persons[9] = {"extractor": FeatureExtractor(),
                      "fall": FallDetector(), "inact": InactivityMonitor(inactivity_timeout=6.0),
                      "risk": HallwayRisk(), "live_updates": 30,
                      "last_hallway": {"risk_level": "CONCERNING", "risk_description": "x",
                                       "base_risk": "CONCERNING", "downgraded": False,
                                       "zone": "hall_floor", "zone_type": "floor",
                                       "seated_score": 0.0, "reclined": True, "recent_drop": True},
                      "last_bbox": bbox, "last_keypoints": kp}
    fake.on = False  # detector blind from here on
    risks = []
    for i in range(80):  # 8 virtual seconds with zero detections
        out = mgr.process(blank(), current_time=t + i * 0.1)
        risks += [p["hallway"]["risk_level"] + ("^" if p.get("coasted") else "") for p in out]
    bare = [r.rstrip("^") for r in risks]
    ok &= check("held (coasted) while blind", any(r.endswith("^") for r in risks))
    ok &= check("escalated to EMERGENCY while blind", "EMERGENCY" in bare,
                f"(saw: {sorted(set(risks))})")
    ok &= check("no latch while holding (still present)", len(mgr.lost_alarms) == 0)

    print("Test: hold cap expiry -> latch")
    mgr.hold_max = 5.0
    for i in range(80, 140):  # +6s virtual: exceeds the 5s test cap
        out = mgr.process(blank(), current_time=t + i * 0.1)
    ok &= check("latched after hold cap",
                any(a["track_id"] == 9 for a in mgr.lost_alarms))

    print("Test: visibly-back person releases the hold silently")
    mgr2, fake2 = make_mgr()
    mgr2.persons[11] = {"extractor": FeatureExtractor(),
                        "fall": FallDetector(), "inact": InactivityMonitor(),
                        "risk": HallwayRisk(), "live_updates": 30,
                        "last_hallway": {"risk_level": "CONCERNING", "risk_description": "x",
                                         "base_risk": "CONCERNING", "downgraded": False,
                                         "zone": "hall_floor", "zone_type": "floor",
                                         "seated_score": 0.0, "reclined": True, "recent_drop": True},
                        "last_bbox": [250, 50, 390, 470],
                        "last_keypoints": mgr2.backend.to_keypoints(
                            {"bbox": [250, 50, 390, 470], "score": 0.9,
                             "kpts": standing_kpts(0.9)}, FW, FH)}
    fake2.on = False
    out = mgr2.process(blank(), current_time=t)
    ok &= check("held entry shown", any(p.get("stale") for p in out))

    def back_in_frame(frame):
        return [{"bbox": [250, 50, 390, 470], "score": 0.9, "kpts": standing_kpts(0.9)}]

    mgr2.backend.infer = back_in_frame
    out = mgr2.process(blank(), current_time=t + 0.1)
    ok &= check("hold released on reappearance",
                not any(p.get("stale") for p in out) and len(mgr2.lost_alarms) == 0)

    # ---- 4. recovery handoff: fall, vanish, stand up overlapping ----
    print("Test: stand-up overlapping a held alarm -> RECOVERED, no phantoms")
    mgr3, fake3 = make_mgr()
    lie_det = {"bbox": [50, 200, 500, 380], "score": 0.9, "kpts": lying_kpts(0.9)}
    mgr3.backend.infer = lambda frame: [dict(lie_det)]
    for i in range(10):
        out = mgr3.process(blank(), current_time=t + i * 0.1)
    live = [p for p in out if not p.get("stale")]
    ok &= check("fallen track alarming", len(live) == 1
                and live[0]["hallway"]["risk_level"] == "CONCERNING")
    old_tid = live[0]["track_id"]
    mgr3.backend.infer = lambda frame: []
    for i in range(10, 35):  # 25 blind frames: tracker coasts 20, then evicts -> stale hold
        out = mgr3.process(blank(), current_time=t + i * 0.1)
    ok &= check("held while blind", any(p.get("stale") for p in out))
    stand_det = {"bbox": [300, 60, 420, 470], "score": 0.9, "kpts": standing_kpts(0.9)}
    mgr3.backend.infer = lambda frame: [dict(stand_det)]
    for i in range(35, 41):
        out = mgr3.process(blank(), current_time=t + i * 0.1)
    recs = mgr3.pop_recoveries()
    ok &= check("recovery recorded P%d->P? " % old_tid,
                len(recs) == 1 and recs[0]["old_tid"] == old_tid,
                f"({recs})")
    ok &= check("no stale ghosts left", not any(p.get("stale") for p in out))
    ok &= check("no latch fired", len(mgr3.lost_alarms) == 0)
    live_now = [p for p in out if not p.get("stale")]
    ok &= check("new track upright NORMAL",
                len(live_now) == 1 and live_now[0]["hallway"]["risk_level"] == "NORMAL")

    # ---- 5. tracker IoU fallback ----
    print("Test: tracker holds ID across lying->standing flip")
    from core.person_tracker import CentroidTracker
    lying, stand = [0, 300, 400, 420], [300, 100, 400, 480]  # dist 165, IoU ~0.16
    tr0 = CentroidTracker(max_distance=140)
    tr0.update([lying])
    r0 = tr0.update([stand])
    ok &= check("distance-only tracker switches ID",
                sorted(t for t, _ in r0) == [1, 2], f"(tids={[t for t, _ in r0]})")
    tr1 = CentroidTracker(max_distance=140, iou_fallback=0.15)
    tr1.update([lying])
    r1 = tr1.update([stand])
    ok &= check("IoU fallback keeps ID", r1[0][0] == 1, f"({r0[0][0]} vs {r1[0][0]})")

    # ---- 6. pose-union bbox ----
    print("Test: bbox stretches to cover confident keypoints")
    from core.yolo_pose_backend import YoloPoseBackend
    b = YoloPoseBackend.__new__(YoloPoseBackend)
    b.conf_thresh, b.maintain_thresh, b.kpt_thresh, b.max_persons = 0.45, 0.20, 0.35, 8
    raw = np.zeros((56, 10), dtype=np.float32)
    raw[0:4, 0] = [320, 320, 100, 100]  # cx, cy, w, h (640 space == px here)
    raw[4, 0] = 0.9
    raw[5:8, 0] = [320, 100, 0.9]    # nose ABOVE the box
    raw[38:41, 0] = [300, 340, 0.9]  # hip11 gives the 4th confident joint
    raw[50:53, 0] = [320, 500, 0.9]  # ankle15 BELOW the box
    raw[53:56, 0] = [330, 505, 0.9]  # ankle16 BELOW the box
    dets = b._decode(raw, 1.0, 0, 0, 640, 640)
    ok &= check("one detection decoded", len(dets) == 1)
    x1, y1, x2, y2 = dets[0]["bbox"]
    ok &= check("union covers head+feet", y1 <= 100 and y2 >= 505,
                f"(box={dets[0]['bbox']})")

    print("\n" + ("ALL MANAGER TESTS PASS" if ok else "MANAGER TESTS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
