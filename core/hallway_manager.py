import time
from typing import Dict, Any, List, Optional

from core.yolo_pose_backend import YoloPoseBackend
from core.person_tracker import CentroidTracker
from core.feature_extractor import FeatureExtractor
from core.fall_detector import FallDetector
from core.inactivity_monitor import InactivityMonitor
from core.hallway_risk import HallwayRisk
from core.zones import ZoneManager


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / max(1.0, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


class HallwayManager:
    """
    Fully multi-person pipeline for clinic hallways / waiting areas.

    - ONE YOLO-pose inference per frame for all bodies (flat cost vs headcount).
    - Centroid tracker for stable P1..Pn IDs.
    - Per-ID FeatureExtractor + FallDetector + InactivityMonitor (reused core)
      + HallwayRisk fusion (kinematics x zone x seated cues).
    - ZoneManager (bench/floor/ignore polygons; default = whole-frame floor).

    process(frame) -> list of per-person dicts:
      {track_id, bbox, det_score, pose_score, weak_pose, keypoints, features,
       fall_status, inactivity_status, zone, hallway}
    """

    def __init__(
        self,
        max_persons: int = 6,
        angle_threshold: float = 58.0,
        velocity_threshold: float = 0.32,
        inactivity_timeout: float = 6.0,
        det_conf: float = 0.45,
        kpt_conf: float = 0.35,
        pose_min_score: float = 0.20,
        model_path: Optional[str] = None,
        providers: Optional[List[str]] = None,
        zones: Optional[ZoneManager] = None,
    ):
        self.max_persons = max(1, min(12, int(max_persons)))
        self.angle_threshold = angle_threshold
        self.velocity_threshold = velocity_threshold
        self.inactivity_timeout = inactivity_timeout
        self.pose_min_score = pose_min_score

        kwargs: Dict[str, Any] = {"conf_thresh": det_conf, "kpt_thresh": kpt_conf,
                                  "max_persons": self.max_persons, "providers": providers}
        if model_path:
            kwargs["model_path"] = model_path
        self.backend = YoloPoseBackend(**kwargs)
        self.tracker = CentroidTracker(max_missing=20, max_distance=140, iou_fallback=0.15)
        self.persons: Dict[int, Dict[str, Any]] = {}
        self.stale: Dict[int, Dict[str, Any]] = {}  # lost alarming tracks, held via re-feed
        self.hold_max = 120.0  # seconds a lost alarm is held before latching unverified
        self.min_live_updates = 3  # live frames before a track is hold/latch-eligible
        self.lost_alarms: List[Dict[str, Any]] = []  # alarming tracks gone longer; cleared on reset
        self.recoveries: List[Dict[str, Any]] = []  # drained by the app for RECOVERED logs
        self.recovery_iou = 0.15  # overlap gate for stand-up handoff
        self.recovery_upright_angle = 38.0  # new track must read upright to claim a recovery
        self.latch_recovery_window = 600.0  # seconds a latch stays clearable by handoff
        self.zones = zones or ZoneManager()

    def pop_recoveries(self) -> List[Dict[str, Any]]:
        """Drain pending stand-up handoffs {old_tid, new_tid, zone}."""
        out = self.recoveries
        self.recoveries = []
        return out

    # ---------- config ----------

    def update_parameters(self, angle_threshold=None, velocity_threshold=None,
                          inactivity_timeout=None, max_persons=None, det_conf=None):
        if angle_threshold is not None:
            self.angle_threshold = angle_threshold
        if velocity_threshold is not None:
            self.velocity_threshold = velocity_threshold
        if inactivity_timeout is not None:
            self.inactivity_timeout = inactivity_timeout
        if max_persons is not None:
            self.max_persons = max(1, min(12, int(max_persons)))
            self.backend.set_max_persons(self.max_persons)
        if det_conf is not None:
            self.backend.conf_thresh = det_conf
        for p in self.persons.values():
            p["fall"].update_parameters(angle_threshold=self.angle_threshold,
                                        velocity_threshold=self.velocity_threshold)
            p["inact"].update_parameters(inactivity_timeout=self.inactivity_timeout)

    def reset(self):
        self.tracker.reset()
        self.persons.clear()
        self.stale.clear()
        self.lost_alarms.clear()

    # ---------- per-track state ----------

    def _get(self, track_id: int) -> Dict[str, Any]:
        if track_id not in self.persons:
            self.persons[track_id] = {
                "extractor": FeatureExtractor(),
                "fall": FallDetector(angle_threshold=self.angle_threshold,
                                     velocity_threshold=self.velocity_threshold),
                "inact": InactivityMonitor(inactivity_timeout=self.inactivity_timeout),
                "risk": HallwayRisk(),
                "live_updates": 0,
            }
        return self.persons[track_id]

    def _fuse(self, person: Dict[str, Any], keypoints: Dict[str, Any],
              bbox, w: int, h: int, now: float):
        """One full FSM+fusion step. Shared by live, weak-coast and stale-hold paths."""
        feats = person["extractor"].extract_features(keypoints, current_time=now)
        temp = {"state": person["fall"].current_state.value,
                "is_horizontal": feats.get("spine_angle", 0) >= self.angle_threshold}
        ins = person["inact"].process(feats, temp)
        fs = person["fall"].process(feats, is_inactive=ins.get("is_inactive_alert", False))
        zone = self.zones.zone_for_person(bbox, w, h, keypoints)
        hw = person["risk"].assess(feats, fs, ins, zone, keypoints)
        return feats, fs, ins, zone, hw

    # ---------- main ----------

    def process(self, frame_bgr, current_time: Optional[float] = None) -> List[Dict[str, Any]]:
        if current_time is None:
            current_time = time.time()
        h, w = frame_bgr.shape[:2]

        detections = self.backend.infer(frame_bgr)
        det_by_box = {tuple(d["bbox"]): d for d in detections}
        tracked = self.tracker.update([d["bbox"] for d in detections])[: self.max_persons]

        # Confirm-gating: a never-seen ID only becomes a track on a strong
        # detection (>= confirm). Weak boxes may MAINTAIN existing tracks
        # (fallen bodies score low) but never SPAWN ghosts.
        confirmed: List = []
        for tid, bbox in tracked:
            det = det_by_box.get(tuple(bbox))
            if tid not in self.persons and (det is None or det["score"] < self.backend.conf_thresh):
                self.tracker.remove(tid)
                continue
            confirmed.append((tid, bbox))
        tracked = confirmed
        active = {tid for tid, _ in tracked}

        # Eviction into hold: a lost track that was alarming keeps its state
        # machine advancing off the last good skeleton (re-feed), instead of
        # vanishing mid-emergency. Entry bar: alarming risk + a few solid live
        # frames (a 1-frame glitch can never latch an alarm).
        for tid in [t for t in self.persons if t not in active]:
            p = self.persons[tid]
            last_hw = p.get("last_hallway", {})
            if last_hw.get("risk_level") in ("CONCERNING", "SLUMP", "EMERGENCY") \
                    and p.get("live_updates", 0) >= self.min_live_updates \
                    and p.get("last_keypoints") is not None:
                self.stale[tid] = {"bbox": p.get("last_bbox", [0, 0, 0, 0]),
                                   "hallway": last_hw, "lost_since": current_time,
                                   "person": p, "keypoints": p["last_keypoints"]}
            del self.persons[tid]
        for tid in [t for t, s in self.stale.items()
                    if current_time - s["lost_since"] > self.hold_max or t in active]:
            s = self.stale[tid]
            if tid not in active:
                # Re-acquired under a new ID? A live body overlapping the held
                # box means the person is back — drop silently, no false latch.
                reacquired = any(_iou(s["bbox"], b) > 0.3 for _, b in tracked)
                if not reacquired:
                    # Gone long AND was alarming: latch an unverified-alarm record.
                    # Cleared only by reset (Stop) — a fallen person who left view
                    # must be verified by a human, never silently forgotten.
                    self.lost_alarms = [a for a in self.lost_alarms if a["track_id"] != tid]
                    self.lost_alarms.append({"track_id": tid,
                                             "risk_level": s["hallway"]["risk_level"],
                                             "zone": s["hallway"].get("zone", "?"),
                                             "bbox": list(s["bbox"]),
                                             "lost_at": current_time})
            del self.stale[tid]

        out: List[Dict[str, Any]] = []
        for track_id, bbox in sorted(tracked):
            person = self._get(track_id)
            det = det_by_box.get(tuple(bbox))
            keypoints = self.backend.to_keypoints(det, w, h) if det else None
            pose_score = keypoints.get("pose_score", 0.0) if keypoints else 0.0
            weak = det is None or pose_score < self.pose_min_score

            use_kp, coasted = keypoints, False
            if weak:
                # Detector degraded but history alarming: hold the last good
                # skeleton instead of freezing (desktop parity).
                prev_hw = person.get("last_hallway", {})
                if prev_hw.get("risk_level") in ("CONCERNING", "SLUMP", "EMERGENCY") \
                        and person.get("live_updates", 0) >= self.min_live_updates \
                        and person.get("last_keypoints") is not None:
                    use_kp, coasted = person["last_keypoints"], True

            if use_kp is not None and (not weak or coasted):
                feats, fs, ins, zone, hw = self._fuse(person, use_kp, bbox, w, h, now=current_time)
            else:
                feats = None
                fs = {"state": person["fall"].current_state.value,
                      "risk_level": person["fall"].risk_level.value,
                      "fall_confidence": float(person["fall"].fall_confidence),
                      "total_falls": person["fall"].total_falls_detected,
                      "is_fast_drop": False}
                ins = {"is_inactive_alert": person["inact"].is_inactive_alert,
                       "inactive_duration": float(person["inact"].total_inactive_duration),
                       "is_still": False}
                zone = self.zones.zone_for_person(bbox, w, h, keypoints)
                hw = person["risk"].assess(feats, fs, ins, zone, keypoints)
            person["last_hallway"] = hw
            person["last_bbox"] = list(bbox)
            if det is not None and not weak:
                # Solid live sighting: earns hold-eligibility, refreshes re-feed state.
                person["live_updates"] = person.get("live_updates", 0) + 1
                person["last_keypoints"] = keypoints

            out.append({"track_id": track_id, "bbox": list(bbox),
                        "det_score": float(det["score"]) if det else 0.0,
                        "pose_score": float(pose_score), "weak_pose": bool(weak and not coasted),
                        "keypoints": use_kp if coasted else keypoints,
                        "features": feats,
                        "fall_status": fs, "inactivity_status": ins,
                        "zone": zone, "hallway": hw, "stale": False, "coasted": coasted})
        # Held entries: the detector is blind but the alarm state machine keeps
        # advancing off the last good skeleton — a motionless fallen person
        # still escalates to EMERGENCY. Re-acquisition or hold_max ends the hold.
        for tid, s in sorted(self.stale.items()):
            p, kp = s["person"], s["keypoints"]
            feats, fs, ins, zone, hw = self._fuse(p, kp, s["bbox"], w, h, now=current_time)
            s["hallway"] = hw
            out.append({"track_id": tid, "bbox": list(s["bbox"]),
                        "det_score": 0.0, "pose_score": 0.0, "weak_pose": False,
                        "keypoints": kp, "features": feats,
                        "fall_status": fs, "inactivity_status": ins,
                        "zone": zone, "hallway": hw,
                        "stale": True, "coasted": True,
                        "hold_for": round(current_time - s["lost_since"], 1)})

        # Recovery handoff: a live body overlapping a held/latched alarm box.
        # Upright overlap = the same person stood back up -> resolve with a
        # RECOVERED log instead of stacking phantom "last seen" boxes.
        # Alarming overlap = still down, re-detected -> live track takes over silently.
        live = [p for p in out if not p.get("stale")]

        def _resolve(bbox):
            best, best_iou = None, 0.0
            for p in live:
                iou = _iou(bbox, p["bbox"])
                if iou > self.recovery_iou and iou > best_iou:
                    best, best_iou = p, iou
            return best

        resolved = set()
        for tid in [t for t in self.stale]:
            hit = _resolve(self.stale[tid]["bbox"])
            if hit is None:
                continue
            hrisk = hit.get("hallway", {}).get("risk_level", "NORMAL")
            hfeats = hit.get("features") or {}
            if hrisk not in ("CONCERNING", "SLUMP", "EMERGENCY") \
                    and hfeats.get("spine_angle", 90.0) < self.recovery_upright_angle:
                self.recoveries.append({"old_tid": tid, "new_tid": hit["track_id"],
                                        "zone": hit.get("hallway", {}).get("zone", "?")})
            resolved.add(tid)
            del self.stale[tid]
        if resolved:
            # Resolved holds vanish immediately (no one-frame ghost).
            out = [p for p in out if not (p.get("stale") and p["track_id"] in resolved)]
        for a in [x for x in self.lost_alarms
                  if x.get("bbox")
                  and current_time - x.get("lost_at", 0) <= self.latch_recovery_window]:
            hit = _resolve(a["bbox"])
            if hit is None:
                continue
            hrisk = hit.get("hallway", {}).get("risk_level", "NORMAL")
            hfeats = hit.get("features") or {}
            if hrisk not in ("CONCERNING", "SLUMP", "EMERGENCY") \
                    and hfeats.get("spine_angle", 90.0) < self.recovery_upright_angle:
                self.recoveries.append({"old_tid": a["track_id"], "new_tid": hit["track_id"],
                                        "zone": hit.get("hallway", {}).get("zone", "?")})
                self.lost_alarms = [x for x in self.lost_alarms if x["track_id"] != a["track_id"]]
            elif hrisk in ("CONCERNING", "SLUMP", "EMERGENCY"):
                # Still down under a new ID: retire the stale latch, live alarm continues.
                self.lost_alarms = [x for x in self.lost_alarms if x["track_id"] != a["track_id"]]
        # NOTE: out[] still lists resolved stale entries this frame; they vanish
        # next frame now their state is deleted. One frame of overlap is harmless.
        return out
