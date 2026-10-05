import cv2
import argparse
import time
import os
from core.pose_estimator import PoseEstimator
from core.feature_extractor import FeatureExtractor
from core.fall_detector import FallDetector
from core.inactivity_monitor import InactivityMonitor
from core.multi_person_manager import MultiPersonManager
from utils.visualizer import Visualizer
from utils.logger import EventLogger
from utils.synthetic_generator import generate_synthetic_fall_video

def run_pipeline(
    source: str = "0",
    angle_thresh: float = 58.0,
    vel_thresh: float = 0.32,
    inactivity_timeout: float = 6.0,
    mode: str = "single",
    max_persons: int = 3,
    detect_interval: int = 5,
    show_skeleton: bool = True,
    show_bbox: bool = True,
):
    # Handle synthetic source
    if source.lower() == "synthetic":
        synthetic_path = os.path.join("assets", "synthetic_fall_demo.mp4")
        if not os.path.exists(synthetic_path):
            print("Generating synthetic demo video...")
            generate_synthetic_fall_video(synthetic_path)
        source = synthetic_path
    elif source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{source}'")
        return

    # Optimize webcam buffer
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    multi_mode = str(mode).lower().startswith("multi")
    max_persons = max(1, min(3, int(max_persons)))

    pose_estimator = PoseEstimator() if not multi_mode else None
    feature_extractor = FeatureExtractor()
    fall_detector = FallDetector(angle_threshold=angle_thresh, velocity_threshold=vel_thresh)
    inactivity_monitor = InactivityMonitor(inactivity_timeout=inactivity_timeout)
    multi_manager = None
    if multi_mode:
        multi_manager = MultiPersonManager(
            max_persons=max_persons,
            angle_threshold=angle_thresh,
            velocity_threshold=vel_thresh,
            inactivity_timeout=inactivity_timeout,
            detect_interval=detect_interval,
        )
    visualizer = Visualizer()
    logger = EventLogger()

    print("\n==========================================================================")
    print("  ARUGA - AI-Assisted Recognition & Guided Contextual Risk Assessment")
    print("==========================================================================")
    print(f"Mode: {'MULTI-PERSON (up to %d)' % max_persons if multi_mode else 'SINGLE-PERSON'}")
    print("Postures Evaluated: NORMAL | UNUSUAL | CONCERNING | EMERGENCY")
    print("Controls: Press 'q' to quit, 'r' to reset system state.\n")

    fps_start_time = time.time()
    frame_counter = 0
    fps = 0.0
    logged_fall_count = 0
    logged_inactive = False
    logged_falls: dict = {}
    logged_inactive_ids: set = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str) and os.path.exists(source):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame_counter += 1
        now = time.time()
        if (now - fps_start_time) >= 1.0:
            fps = frame_counter / (now - fps_start_time)
            frame_counter = 0
            fps_start_time = now

        if multi_mode:
            persons = multi_manager.process(frame, current_time=now)
            for p in persons:
                tid = p["track_id"]
                feats, fs, ins = p["features"], p["fall_status"], p["inactivity_status"]
                if feats is None:
                    continue
                if fs.get("state") == "FALLEN" and fs.get("total_falls", 0) > logged_falls.get(tid, 0):
                    logged_falls[tid] = fs["total_falls"]
                    logger.log_event("FALL_DETECTED", feats, fs.get("fall_confidence", 0.8), frame, person_id=tid)
                    print(f"[P{tid}] FALL_DETECTED conf={fs.get('fall_confidence', 0):.0%}")
                if ins.get("is_inactive_alert") and tid not in logged_inactive_ids:
                    logged_inactive_ids.add(tid)
                    logger.log_event("INACTIVITY_EMERGENCY", feats, 1.0, frame,
                                     extra_note=f"P{tid} stationary for {ins.get('inactive_duration'):.1f}s",
                                     person_id=tid)
                    print(f"[P{tid}] INACTIVITY_EMERGENCY {ins.get('inactive_duration'):.1f}s")
                elif not ins.get("is_inactive_alert") and tid in logged_inactive_ids:
                    logged_inactive_ids.discard(tid)
            tids = {p["track_id"] for p in persons}
            logged_falls = {k: v for k, v in logged_falls.items() if k in tids}
            logged_inactive_ids &= tids
            display_frame = visualizer.draw_hud_multi(
                frame, persons, show_skeleton=show_skeleton, show_bbox=show_bbox)
        else:
            # 1. Pose estimation
            has_pose, raw_landmarks, keypoints = pose_estimator.process_frame(frame)

            features = None
            fall_status = {
                "state": "NORMAL",
                "risk_level": "NORMAL",
                "fall_confidence": 0.0,
                "total_falls": fall_detector.total_falls_detected
            }
            inactivity_status = {"is_inactive_alert": False, "inactive_duration": 0.0}

            if has_pose and keypoints:
                # 2. Extract features
                features = feature_extractor.extract_features(keypoints, current_time=now)

                # 3. Inactivity monitoring pre-check for risk escalation
                temp_fall = {"state": fall_detector.current_state.value, "is_horizontal": features.get("spine_angle", 0) >= angle_thresh}
                inactivity_status = inactivity_monitor.process(features, temp_fall)

                # 4. Contextual Risk & Fall detection
                fall_status = fall_detector.process(features, is_inactive=inactivity_status.get("is_inactive_alert", False))

                # Log events: one log per incident (rising edge)
                if fall_status.get("state") == "FALLEN" \
                        and fall_status.get("total_falls", 0) > logged_fall_count:
                    logged_fall_count = fall_status["total_falls"]
                    logger.log_event("FALL_DETECTED", features, fall_status.get("fall_confidence", 0.8), frame)
                if inactivity_status.get("is_inactive_alert") and not logged_inactive:
                    logged_inactive = True
                    logger.log_event("INACTIVITY_EMERGENCY", features, 1.0, frame, extra_note=f"Stationary for {inactivity_status.get('inactive_duration'):.1f}s")
                elif not inactivity_status.get("is_inactive_alert"):
                    logged_inactive = False

            # 5. Visual telemetry HUD
            display_frame = visualizer.draw_hud(
                frame, keypoints, features, fall_status, inactivity_status,
                show_skeleton=show_skeleton, show_bbox=show_bbox)

        # Display FPS
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (16, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("ARUGA | Contextual Risk Assessment", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            fall_detector.reset()
            inactivity_monitor.reset()
            feature_extractor.reset()
            logged_fall_count = 0
            logged_inactive = False
            logged_falls.clear()
            logged_inactive_ids.clear()
            if multi_manager is not None:
                multi_manager.reset()
            print("System state reset.")

    cap.release()
    if pose_estimator is not None:
        pose_estimator.close()
    if multi_manager is not None:
        multi_manager.close()
    cv2.destroyAllWindows()
    print("Pipeline closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARUGA - AI-Assisted Recognition and Understanding for Guided Contextual Risk Assessment")
    parser.add_argument("--source", type=str, default="synthetic", help="Webcam index (0), video path, or 'synthetic'")
    parser.add_argument("--angle-thresh", type=float, default=58.0, help="Spine angle threshold for fallen posture (degrees)")
    parser.add_argument("--vel-thresh", type=float, default=0.32, help="Centroid vertical velocity threshold")
    parser.add_argument("--inactivity-timeout", type=float, default=6.0, help="Inactivity alert timeout in seconds")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "multi"],
                        help="single = original 1-person pipeline; multi = 2-3 person detect+track")
    parser.add_argument("--max-persons", type=int, default=3, choices=[1, 2, 3],
                        help="Cap on tracked persons in multi mode")
    parser.add_argument("--detect-interval", type=int, default=5,
                        help="Run person detector every N frames in multi mode (higher = faster)")
    parser.add_argument("--hide-skeleton", action="store_true",
                        help="Hide skeleton / spine overlay")
    parser.add_argument("--hide-bbox", action="store_true",
                        help="Hide telemetry bounding box overlay")
    args = parser.parse_args()

    run_pipeline(
        source=args.source,
        angle_thresh=args.angle_thresh,
        vel_thresh=args.vel_thresh,
        inactivity_timeout=args.inactivity_timeout,
        mode=args.mode,
        max_persons=args.max_persons,
        detect_interval=args.detect_interval,
        show_skeleton=not args.hide_skeleton,
        show_bbox=not args.hide_bbox,
    )
