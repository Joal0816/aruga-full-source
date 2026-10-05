"""
Tests for the shared frame pump (no hardware needed).

  - file source loops past EOF; loop_files=False raises SourceLost at EOF
  - dead webcam index exhausts tolerance and raises SourceLost
  - dead RTSP endpoint raises SourceLost (fast, closed localhost port)
  - frames() respects the alive callback

Run:  .\\.venv\\Scripts\\python.exe tests\\test_frame_pump.py
Exit code 0 = pass.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.frame_pump import FramePump, SourceLost, classify


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    return cond


def make_clip(path, n=8, size=(160, 120)):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    for i in range(n):
        out.write(np.full((size[1], size[0], 3), i * 30, dtype=np.uint8))
    out.release()


def main():
    import tempfile
    ok = True
    tmp = tempfile.mkdtemp(prefix="pump_")

    print("Test: classify()")
    ok &= check("rtsp/file/webcam",
                classify("rtsp://a/b") == "rtsp" and classify("http://a/b.mjpg") == "rtsp"
                and classify("c:\\x.mp4") == "file" and classify(0) == "webcam")

    print("Test: file loops past EOF")
    mp4 = os.path.join(tmp, "loop.mp4")
    make_clip(mp4)
    pump = FramePump(mp4, kind="file")
    valid = 0
    for _ in range(30):  # ~3.5 loop boundaries return None once each (seek frame)
        if pump.read() is not None:
            valid += 1
    pump.release()
    ok &= check("wraps without raising", valid >= 20, f"({valid} valid / 30 reads)")

    print("Test: loop_files=False ends the clip")
    pump = FramePump(mp4, kind="file", loop_files=False)
    n, ended = 0, False
    try:
        for _ in range(30):
            if pump.read() is not None:
                n += 1
    except SourceLost:
        ended = True
    pump.release()
    ok &= check("8 frames then SourceLost", n == 8 and ended, f"({n} frames)")

    print("Test: frames() honors alive()")
    pump = FramePump(mp4, kind="file")
    calls = {"n": 0}

    def alive():
        calls["n"] += 1
        return calls["n"] <= 5

    seen = sum(1 for _ in pump.frames(alive=alive))
    pump.release()
    ok &= check("stops after alive() False", seen == 5, f"({seen})")

    print("Test: dead webcam raises SourceLost quickly")
    try:
        pump = FramePump(99, kind="webcam", webcam_tolerance=2)
    except SourceLost as e:
        pump = None
        ok &= check("raised at open", "Cannot open" in str(e), f"({e})")
    if pump is not None:
        try:
            for _ in range(10):
                pump.read()
            ok &= check("raised on reads", False)
        except SourceLost as e:
            ok &= check("raised on reads", "Webcam" in str(e), f"({e})")
        pump.release()

    print("Test: dead RTSP raises SourceLost")
    try:
        pump = FramePump("rtsp://127.0.0.1:9/dead", kind="rtsp", rtsp_retries=1)
    except SourceLost as e:
        pump = None
        ok &= check("raised at open", "Cannot open" in str(e), f"({e})")
    if pump is not None:
        try:
            t0 = __import__("time").time()
            pump.read()
            for _ in range(6):
                pump.read()
            ok &= check("raised on reads", False)
        except SourceLost as e:
            ok &= check("raised on reads", True, f"({e})")
        pump.release()

    print("\n" + ("ALL PUMP TESTS PASS" if ok else "PUMP TESTS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
