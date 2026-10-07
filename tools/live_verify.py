"""Automated real-world verification harness — runs ARUGA's REAL pipeline
against a live camera (or a file for self-test) and prints a PASS/FAIL
checklist mirroring VERIFICATION.md, plus the exact items a human must do.

Usage:
    python tools/live_verify.py --url rtsp://USER:PASS@IP:554/stream1 [--seconds 30]
    python tools/live_verify.py --file assets/synthetic_fall_demo.mp4   # no camera needed
    python tools/live_verify.py --discover                              # probe LAN for RTSP/Tapo

Exit code 0 = all automatable checks passed.
"""
import argparse
import os
import socket
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

from core.frame_pump import FramePump, SourceLost
from core.hallway_manager import HallwayManager

RISK_RANK = {"NORMAL": 0, "RESTING": 1, "UNUSUAL": 2, "SLUMP": 3,
             "CONCERNING": 4, "EMERGENCY": 5}
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((bool(cond), name, detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def human(item):
    RESULTS.append((None, item, ""))
    print(f"  [HUMAN] {item}")


def discover(subnet_prefix="192.168.1.", ports=(554, 8000)):
    print(f"[discover] probing {subnet_prefix}0/24 ports {ports} ...")
    hits = []
    for i in range(1, 255):
        h = f"{subnet_prefix}{i}"
        for p in ports:
            s = socket.socket(); s.settimeout(0.08)
            try:
                if s.connect_ex((h, p)) == 0:
                    hits.append(f"{h}:{p}")
            finally:
                s.close()
    print(f"[discover] open: {hits or 'NONE — camera off or different subnet?'}")
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="rtsp://user:pass@ip:554/stream1 (Tapo: Camera Account creds)")
    ap.add_argument("--file", help="video file instead of live camera (self-test)")
    ap.add_argument("--seconds", type=float, default=30.0, help="live analysis window")
    ap.add_argument("--discover", action="store_true")
    args = ap.parse_args()

    if args.discover:
        discover()
        if not args.url and not args.file:
            return 0

    source, kind = (args.file, "file") if args.file else (args.url, "rtsp")
    if not source:
        print("Need --url (live) or --file (self-test), or --discover.")
        return 2

    print(f"\n== A. Source ({kind}) ==")
    try:
        pump = FramePump(source, kind=kind, loop_files=False)
    except SourceLost as e:
        check("source opens", False, f"({e})")
        return 1
    check("source opens", True)

    mgr = HallwayManager(max_persons=6)
    deadline = time.time() + (0 if kind == "file" else args.seconds)
    frames = persons_max = pose_hits = 0
    max_risk, falls, events = "NORMAL", 0, 0
    logged_falls, logged_inactive, res_w, res_h = {}, set(), 0, 0
    stream_lost = None
    snapshot = None
    t_start = time.time()

    print(f"== B. Pipeline analysis ({'EOF' if kind=='file' else f'{args.seconds:.0f}s live'}) ==")
    try:
        for frame in pump.frames(alive=lambda: kind == "file" or time.time() < deadline):
            frames += 1
            res_h, res_w = frame.shape[:2]
            if snapshot is None:
                snapshot = frame.copy()
            persons = mgr.process(frame, current_time=time.time())
            persons_max = max(persons_max, len(persons))
            for p in persons:
                if p.get("features"):
                    pose_hits += 1
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
                elif not p["inactivity_status"].get("is_inactive_alert"):
                    logged_inactive.discard(tid)
            if kind != "file" and time.time() >= deadline:
                break
    except SourceLost as e:
        if kind == "file":
            print(f"  (file ended: {e})")
        else:
            stream_lost = str(e)
            check("stream stays up for window", False, f"({e})")

    dt = max(0.01, time.time() - t_start)
    check("frames received", frames >= (20 if kind == "file" else 10),
          f"({frames} frames, {res_w}x{res_h}, {frames/dt:.1f} fps process rate)")
    check("persons tracked", persons_max >= 1,
          f"(max {persons_max} concurrent, pose on {pose_hits}/{frames} frames)")
    if kind != "file":
        check("no false EMERGENCY on baseline scene", max_risk != "EMERGENCY" or falls > 0,
              f"(max risk reached: {max_risk})")
        check("stream survived the window", stream_lost is None)

    snap_path = "/tmp/aruga_live_verify_snapshot.jpg"
    if snapshot is not None:
        cv2.imwrite(snap_path, snapshot)
        print(f"  snapshot: {snap_path}")

    if kind == "file":
        print("== C. Alarm path (same engine, synthetic fall clip) ==")
        check("fall detected end-to-end", falls >= 1, f"(falls={falls}, events={events}, max_risk={max_risk})")
        check("risk escalated", RISK_RANK.get(max_risk, 0) >= RISK_RANK["CONCERNING"],
              f"(max_risk={max_risk})")

    print("\n== Items that need a human ==")
    human("Physical drill: walk in, drop/sit/lie in view of the live camera -> alarm within ~1s + EMERGENCY at 6s still")
    human("Audible beep check (ear) + 10-min Acknowledge button behavior in the running UI")
    human("Zone calibration approval: capture empty-hallway BG, draw bench/ignore zones (I can draft them from the snapshot)")
    if kind != "file":
        human("RTSP Camera Account credentials from the Tapo app (used for this run: pre-supplied)")
    else:
        human("Live run still pending: Tapo on LAN + Camera Account creds (Tapp app -> Advanced Settings -> Camera Account)")
    human("Upload VERIFICATION.md to aruga.joalvergs.tech (Cloudflare Pages deploy access)")

    ok = all(c for c, _, _ in RESULTS if c is not None)
    failed = [n for c, n, _ in RESULTS if c is False]
    print(f"\n{'ALL AUTOMATABLE CHECKS PASSED' if ok else 'FAILURES: ' + ', '.join(failed)}")
    print(f"snapshot for visual review: {snap_path if snapshot is not None else 'n/a'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
