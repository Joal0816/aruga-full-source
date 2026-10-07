"""Benchmark pose backends + the full hallway pipeline on real frames.

Run on an IDLE CPU (numbers are meaningless under load — kill other jobs
first). Reports ms/frame and fps for:
  1. PoseEstimator  — MediaPipe solutions when available (py<=3.12 env),
                      else the YOLO-ONNX fallback. Single-person path used
                      by app.py / desktop_app.py.
  2. YoloPoseBackend.infer — raw detect+decode (what hallway_app pays).
  3. HallwayManager.process — full chain: pose + tracking + features +
                      FSM + zones + risk (end-to-end analysis cost).

Usage:
    python tools/bench_pipeline.py [--clip assets/eval_clips/ur_fall-01.mp4]
                                   [--frames 40]
"""
import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_frames(path, n):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    frames = []
    while len(frames) < n:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        raise SystemExit(f"no frames in {path}")
    return frames


def bench(label, fn, frames):
    fn(frames[0])  # warmup
    t0 = time.time()
    hits = 0
    for fr in frames:
        hits += bool(fn(fr))
    dt = time.time() - t0
    ms = dt / len(frames) * 1000.0
    print(f"  {label:<46} {ms:7.1f} ms/frame  {len(frames)/dt:6.2f} fps"
          + (f"  (hits {hits}/{len(frames)})" if hits else ""))
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="assets/eval_clips/ur_fall-01.mp4")
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()
    frames = load_frames(args.clip, args.frames)
    print(f"clip={args.clip} frames={len(frames)} "
          f"cpu_count={os.cpu_count()}")

    from core.pose_estimator import PoseEstimator
    pe = PoseEstimator()
    backend = "mediapipe-solutions" if pe._backend is None else "yolo-onnx-fallback"
    bench(f"1. PoseEstimator ({backend})", pe.process_frame, frames)
    if hasattr(pe, "close"):
        pe.close()

    from core.yolo_pose_backend import YoloPoseBackend
    be = YoloPoseBackend(max_persons=6)
    bench("2. YoloPoseBackend.infer (raw)", be.infer, frames)

    from core.hallway_manager import HallwayManager
    mgr = HallwayManager(max_persons=6)
    mgr.process(frames[0], current_time=0.0)
    clock = [0.0]

    def step(fr):
        clock[0] += 0.1
        mgr.process(fr, current_time=clock[0])
        return True

    bench("3. HallwayManager.process (full)", step, frames)
    print("Done.")


if __name__ == "__main__":
    main()
