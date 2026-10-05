"""
Batch-evaluate hallway_app's engine over video clips, headless (no GUI).

Runs HallwayManager over every .mp4/.avi in assets/eval_clips/ (see
tools/fetch_datasets.py) plus any extra paths given, and writes a CSV with
per-clip results: frames, persons seen, max risk reached, falls counted,
events logged. Human reviews the CSV against known clip labels.

Usage:
    .\\.venv\\Scripts\\python.exe tools\\eval_clips.py [extra_clip ...] [--out eval_report.csv]
                         [--angle 58] [--vel 0.32] [--timeout 6] [--max-persons 6]
"""

import argparse
import csv
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

from core.hallway_manager import HallwayManager

RISK_RANK = {"NORMAL": 0, "RESTING": 1, "UNUSUAL": 2, "SLUMP": 3, "CONCERNING": 4, "EMERGENCY": 5}


def eval_clip(path, mgr, max_frames=0, tail_seconds=8.0):
    """Score one clip. After EOF, the last frame is re-fed (virtual clock
    advancing) for tail_seconds to simulate the person staying put — this is
    what exercises the inactivity path on real pixels. Without it, clips that
    end right after the fall can never reach EMERGENCY."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"clip": path, "error": "cannot open"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    mgr.reset()
    frames, max_risk, max_persons, falls, events = 0, "NORMAL", 0, 0, 0
    logged_falls: dict = {}
    logged_inactive: set = set()

    def step(frame, n):
        nonlocal frames, max_risk, max_persons, falls, events
        frames += 1
        persons = mgr.process(frame, current_time=n / fps)
        max_persons = max(max_persons, len(persons))
        for p in persons:
            r = p["hallway"]["risk_level"]
            if RISK_RANK.get(r, 0) > RISK_RANK.get(max_risk, 0):
                max_risk = r
            tid = p["track_id"]
            if p["fall_status"].get("total_falls", 0) > logged_falls.get(tid, 0):
                logged_falls[tid] = p["fall_status"]["total_falls"]
                falls += 1
                events += 1
            if p["inactivity_status"].get("is_inactive_alert") and tid not in logged_inactive:
                logged_inactive.add(tid)
                events += 1

    t0 = time.time()
    n, last = 0, None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        if max_frames and n > max_frames:
            break
        last = frame
        step(frame, n)
    cap.release()
    if last is not None and tail_seconds > 0:
        for _ in range(int(tail_seconds * fps)):
            n += 1
            step(last, n)
    return {"clip": os.path.basename(path), "frames": frames,
            "seconds": round(time.time() - t0, 1),
            "fps": round(frames / max(0.01, time.time() - t0), 1),
            "max_persons": max_persons, "max_risk": max_risk,
            "falls": falls, "events": events, "error": ""}


def main():
    ap = argparse.ArgumentParser(description="Batch eval hallway engine on clips")
    ap.add_argument("extra", nargs="*", help="extra clip paths")
    ap.add_argument("--out", default="eval_report.csv")
    ap.add_argument("--angle", type=float, default=58.0)
    ap.add_argument("--vel", type=float, default=0.32)
    ap.add_argument("--timeout", type=float, default=6.0)
    ap.add_argument("--max-persons", type=int, default=6)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--tail-seconds", type=float, default=8.0,
                    help="re-feed last frame this long (virtual clock) to test inactivity")
    args = ap.parse_args()

    clips = sorted(glob.glob(os.path.join("assets", "eval_clips", "*.mp4")) +
                   glob.glob(os.path.join("assets", "eval_clips", "*.avi"))) + args.extra
    if not clips:
        print("No clips found. Run tools/fetch_datasets.py first (or pass paths).")
        return 1
    mgr = HallwayManager(max_persons=args.max_persons, angle_threshold=args.angle,
                         velocity_threshold=args.vel, inactivity_timeout=args.timeout)
    print(f"Backend: {mgr.backend.backend_name} | clips: {len(clips)}")
    rows = [eval_clip(c, mgr, args.max_frames, args.tail_seconds) for c in clips]
    for r in rows:
        print(f"  {r['clip']}: frames={r.get('frames')} fps={r.get('fps')} "
              f"persons={r.get('max_persons')} max_risk={r.get('max_risk')} "
              f"falls={r.get('falls')} events={r.get('events')} {r.get('error', '')}")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "frames", "seconds", "fps", "max_persons",
                                          "max_risk", "falls", "events", "error"])
        w.writeheader()
        w.writerows(rows)
    print(f"Report: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
