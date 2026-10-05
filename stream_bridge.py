"""
ARUGA Universal Stream & File Analysis Bridge
Translates RTSP (e.g. TP-Link Tapo C200/C230/C310/TC70), USB webcams, and uploaded media files
into real-time HTTP MJPEG video and JSON telemetry feeds for the ARUGA PWA.

Milestone 1 pipeline (canonical, multi-person):
    camera source -> core.frame_pump.FramePump -> core.hallway_manager.HallwayManager
    (YOLOv8n-pose ONNX + tracking + fall/inactivity/risk/zones) -> annotated MJPEG + telemetry.

No synthetic fallback: any camera/pipeline failure is published verbatim as a
telemetry status + error frame and retried for real.
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import urllib.parse

try:
    from voice_relay import VoiceRelay, handle_voice_websocket
    VOICE_AVAILABLE = True
except Exception as _voice_import_error:
    VOICE_AVAILABLE = False
    _voice_relay_error = _voice_import_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("aruga_bridge")

# Global pipeline state
latest_frame = None
latest_telemetry = {
    "status": "INITIALIZING",
    "spine_angle": 0.0,
    "velocity_y": 0.0,
    "stillness_energy": 0.0,
    "inactivity_timer": 0.0,
    "behavior_pattern": "NORMAL",
    "fps": 0.0,
    "source": "None",
    "timestamp": time.time()
}
frame_lock = threading.Lock()
running = True
voice_relay = None  # VoiceRelay instance (None when unavailable)

# One HallwayManager for uploaded-media analysis (ONNX load is expensive);
# its state is reset per upload and all use is serialized by the lock.
_upload_mgr = None
_upload_mgr_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser(description="ARUGA Tapo RTSP & Multi-Platform Bridge")
    parser.add_argument("--source", type=str, default="0",
                        help="Camera source: integer index (0), RTSP URL (rtsp://admin:pass@ip:554/stream1), or file path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host IP (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port (default 8080)")
    parser.add_argument("--tapo-ip", type=str, default="", help="Tapo camera IP (e.g. 192.168.1.50)")
    parser.add_argument("--tapo-user", type=str, default="", help="Tapo camera username")
    parser.add_argument("--tapo-pass", type=str, default="", help="Tapo camera password")
    parser.add_argument("--stream", type=str, choices=["stream1", "stream2"], default="stream2",
                        help="stream1=High-Def (1080p/2K), stream2=Low-Def/Fast (360p, recommended for high FPS inference)")
    parser.add_argument("--patient-audio", type=str, choices=["synthetic", "mic", "rtsp", "silence"],
                        default=None,
                        help="Patient audio source broadcast on ws://host:port/voice: synthetic demo speech, local mic (pulse), camera RTSP stream, or silence (default: synthetic when --source None, else rtsp for rtsp:// sources)")
    parser.add_argument("--voice-sink", type=str, choices=["auto", "pulse", "file"], default="auto",
                        help="Nurse microphone playback sink: auto (pulse via pacat, fall back to file), pulse (pacat), or file (append raw PCM to /tmp/aruga_nurse_speak.pcm)")
    return parser.parse_args()


def build_rtsp_url(ip, user, password, stream="stream2"):
    """Delegate to the canonical Tapo URL builder in core.camera_sources."""
    from core.camera_sources import build_tapo_rtsp_url
    return build_tapo_rtsp_url(ip, user, password, stream)


# ---------------------------------------------------------------------------
# Publishing helpers (thread-safe)
# ---------------------------------------------------------------------------

def _publish(frame_bgr, telemetry_update):
    """Encode a BGR frame as JPEG and publish it with telemetry."""
    global latest_frame
    try:
        import cv2
    except ImportError:
        return
    ret, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if ret:
        with frame_lock:
            latest_frame = jpeg.tobytes()
            latest_telemetry.update(telemetry_update)


def _error_frame(message: str):
    """Plain black frame carrying the REAL error text. No synthetic scene."""
    import numpy as np
    import cv2
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "ARUGA BRIDGE - NO SIGNAL", (16, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    for i, line in enumerate(message.splitlines()[:6]):
        cv2.putText(frame, line[:70], (16, 90 + 32 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return frame


def _report_source_error(status: str, detail: str, source_str):
    """Publish an actual failure state (no fake feed, no fake detections)."""
    logger.error(f"{status}: {detail}")
    frame = _error_frame(f"{status}: {detail}")
    _publish(frame, {
        "status": status,
        "error": str(detail),
        "source": str(source_str),
        "timestamp": time.time(),
        "track_count": 0,
        "persons": [],
        "risk": "UNKNOWN",
    })


def _interruptible_sleep(seconds: float):
    """Sleep in small slices so shutdown (running=False) is responsive."""
    end = time.time() + seconds
    while running and time.time() < end:
        time.sleep(0.2)


def _risk_rank(risk: str) -> int:
    order = {"NORMAL": 0, "RESTING": 1, "UNUSUAL": 2, "SLUMP": 3, "CONCERNING": 4, "EMERGENCY": 5}
    return order.get(risk, 0)


def _person_summary(p):
    return {
        "track_id": int(p.get("track_id", 0)),
        "risk": str(p.get("hallway", {}).get("risk_level", "NORMAL")),
        "zone": str(p.get("hallway", {}).get("zone", "?")),
        "fall_state": str(p.get("fall_status", {}).get("state", "NORMAL")),
        "inactive_duration": round(float(p.get("inactivity_status", {}).get("inactive_duration", 0.0)), 1),
        "bbox": [int(v) for v in p.get("bbox", [])],
    }


def _telemetry_from_persons(persons, fps, source_str, now, lost_alarms):
    """Map multi-person pipeline output onto the bridge telemetry schema.

    Keeps all legacy single-person keys (populated from the worst-risk person)
    and adds the multi-person keys.
    """
    from utils.hallway_overlay import worst_risk

    upd = {
        "fps": round(float(fps), 1),
        "source": str(source_str),
        "timestamp": now,
        "track_count": len(persons),
    }
    if not persons:
        upd.update({
            "status": "SEARCHING_PERSON",
            "spine_angle": 0.0,
            "velocity_y": 0.0,
            "stillness_energy": 0.0,
            "inactivity_timer": 0.0,
            "behavior_pattern": "NORMAL",
            "risk": "NONE",
            "persons": [],
        })
    else:
        worst = worst_risk(persons)
        focus = next((p for p in persons
                      if p.get("hallway", {}).get("risk_level") == worst), persons[0])
        feats = focus.get("features") or {}
        fs = focus.get("fall_status") or {}
        ins = focus.get("inactivity_status") or {}
        upd.update({
            "status": str(fs.get("state", "UNKNOWN")),
            "spine_angle": round(float(feats.get("spine_angle", 0.0)), 1),
            "velocity_y": round(float(feats.get("vertical_velocity", 0.0)), 2),
            "stillness_energy": round(float(feats.get("motion_energy", 0.0)), 2),
            "inactivity_timer": round(float(ins.get("inactive_duration", 0.0)), 1),
            "behavior_pattern": str(fs.get("behavior", "NORMAL")),
            "risk": worst,
            "persons": [_person_summary(p) for p in persons],
        })
    upd["lost_alarms"] = [
        {
            "track_id": int(a.get("track_id", 0)),
            "risk_level": str(a.get("risk_level", "?")),
            "zone": str(a.get("zone", "?")),
            "bbox": [int(v) for v in a.get("bbox", [])],
            "lost_at": float(a.get("lost_at", 0.0)),
        }
        for a in (lost_alarms or [])
    ]
    return upd


def _get_upload_manager():
    """Lazily create the (single) HallwayManager used for upload analysis."""
    global _upload_mgr
    if _upload_mgr is None:
        from core.hallway_manager import HallwayManager
        _upload_mgr = HallwayManager()
    return _upload_mgr


def analyze_uploaded_file(file_bytes):
    """Analyze an uploaded image or video with the canonical ARUGA pipeline.

    Returns real pipeline output only — no mock values.
    """
    try:
        import cv2
        import numpy as np
    except Exception as e:
        return {"success": False, "error": f"OpenCV/NumPy unavailable: {e}"}

    try:
        from core.frame_pump import FramePump, SourceLost
        from utils.hallway_overlay import worst_risk
    except Exception as e:
        return {"success": False, "error": f"ARUGA core modules unavailable: {e}"}

    # Try to decode as an image first
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is not None:
        h, w = img.shape[:2]
        try:
            with _upload_mgr_lock:
                mgr = _get_upload_manager()
                mgr.reset()
                persons = mgr.process(img, current_time=time.time())
                lost = list(mgr.lost_alarms)
        except Exception as e:
            return {"success": False, "error": f"inference failed: {e}"}
        return {
            "file_type": "image",
            "resolution": f"{w}x{h}",
            "persons_detected": len(persons),
            "analysis": {
                "persons": [_person_summary(p) for p in persons],
                "worst_risk": worst_risk(persons) if persons else "NONE",
                "lost_alarms": lost,
            },
            "timestamp": time.time(),
            "success": True,
        }

    # Otherwise treat as video: write to a temp file and pump frames through the pipeline
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".mp4", prefix="aruga_upload_")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            f.write(file_bytes)
        frames_evaluated = 0
        fall_frames = 0
        inactivity_frames = 0
        worst = "NONE"
        max_tracks = 0
        try:
            with _upload_mgr_lock:
                mgr = _get_upload_manager()
                mgr.reset()
                pump = FramePump(path, kind="file", loop_files=False)
                try:
                    while frames_evaluated < 120:
                        try:
                            frame = pump.read()
                        except SourceLost:
                            break  # end of clip
                        if frame is None:
                            continue
                        frames_evaluated += 1
                        persons = mgr.process(frame, current_time=time.time())
                        max_tracks = max(max_tracks, len(persons))
                        if persons:
                            r = worst_risk(persons)
                            if r != "NONE" and (worst == "NONE" or _risk_rank(r) > _risk_rank(worst)):
                                worst = r
                            if any(p.get("fall_status", {}).get("state") in ("FALLEN", "INACTIVE_ALERT")
                                   for p in persons):
                                fall_frames += 1
                            if any(p.get("inactivity_status", {}).get("is_inactive_alert")
                                   for p in persons):
                                inactivity_frames += 1
                finally:
                    pump.release()
        except SourceLost as e:
            return {"success": False, "error": f"cannot open uploaded video: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

        if frames_evaluated == 0:
            return {"success": False, "error": "no decodable frames in uploaded video"}
        return {
            "file_type": "video/stream",
            "size_bytes": len(file_bytes),
            "analysis": {
                "frames_evaluated": frames_evaluated,
                "fall_frames": fall_frames,
                "inactivity_frames": inactivity_frames,
                "max_tracked_persons": max_tracks,
                "overall_risk_index": worst,
            },
            "timestamp": time.time(),
            "success": True,
        }
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def camera_worker(source_str):
    """Capture -> FramePump -> HallwayManager (YOLOv8-pose) -> annotated frames + telemetry.

    Multi-person, zone-aware, reconnecting. No synthetic fallback: every
    failure is published verbatim to telemetry/logs and retried for real.
    """
    global latest_frame, latest_telemetry, running

    try:
        import cv2  # noqa: F401
        import numpy as np  # noqa: F401
    except ImportError:
        logger.error("OpenCV (cv2) or NumPy is not installed. Run: pip install opencv-python numpy")
        return

    # Model paths (assets/models/*.onnx) resolve relative to the repo — anchor CWD
    repo_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(repo_root)
    os.chdir(repo_root)

    # Canonical ARUGA pipeline (multi-person, zone-aware, YOLOv8n-pose ONNX)
    try:
        from core.frame_pump import FramePump, SourceLost
        from core.hallway_manager import HallwayManager
        from utils.hallway_overlay import draw_hallway_hud
    except Exception as e:
        _report_source_error("PIPELINE_ERROR", f"ARUGA core modules failed to load: {e}", source_str)
        return

    # Resolve the video source (webcam index / RTSP URL / file path)
    if not source_str or source_str == "None":
        _report_source_error(
            "SOURCE_UNAVAILABLE",
            "No video source configured. Pass --source 0 | rtsp://user:pass@ip:554/stream2 "
            "or --tapo-ip/--tapo-user/--tapo-pass.",
            source_str)
        return
    source = int(source_str) if source_str.isdigit() else source_str

    try:
        mgr = HallwayManager()
        logger.info("ARUGA pipeline ready: HallwayManager (YOLOv8n-pose ONNX, multi-person tracking).")
    except Exception as e:
        _report_source_error("PIPELINE_ERROR", f"YOLO pose pipeline unavailable: {e}", source_str)
        return

    prev_time = time.time()
    fps = 0.0

    while running:
        # (Re)open the capture through FramePump: TCP transport, timeouts, backoff
        try:
            pump = FramePump(source,
                             on_status=lambda m: logger.info(f"camera: {m}"),
                             abort=lambda: not running)
        except SourceLost as e:
            _report_source_error("SOURCE_UNAVAILABLE", str(e), source_str)
            _interruptible_sleep(5.0)
            continue

        logger.info(f"Connected to video source: {source_str}")
        try:
            while running:
                frame = pump.read()  # None = transient glitch; raises SourceLost when dead
                if frame is None:
                    continue

                curr_time = time.time()
                dt = curr_time - prev_time
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)
                prev_time = curr_time

                persons = mgr.process(frame, current_time=curr_time)
                for rec in mgr.pop_recoveries():
                    logger.info(f"RECOVERED track handoff: {rec}")

                telemetry_update = _telemetry_from_persons(
                    persons, fps, source_str, curr_time, mgr.lost_alarms)

                disp = draw_hallway_hud(frame, persons,
                                        extra_status=f"{telemetry_update['fps']} fps",
                                        lost_alarms=mgr.lost_alarms)
                _publish(disp, telemetry_update)

                # Cap processing loop at ~30 FPS
                time.sleep(max(0.001, 0.033 - (time.time() - curr_time)))
        except SourceLost as e:
            _report_source_error("SOURCE_LOST", f"{e} (reconnecting)", source_str)
            _interruptible_sleep(5.0)
        except Exception as e:
            _report_source_error("INFERENCE_ERROR", str(e), source_str)
            _interruptible_sleep(1.0)
        finally:
            pump.release()

    logger.info("Camera capture loop closed.")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class BridgeRequestHandler(BaseHTTPRequestHandler):
    def end_headers_with_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers_with_cors()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/voice":
            # Two-way voice relay (RFC 6455 WebSocket on the same port).
            # Runs the WS read loop in this handler thread — thread-per-client,
            # so long-lived connections never block the other endpoints.
            if voice_relay is not None and "websocket" in (self.headers.get("Upgrade", "") or "").lower():
                self.close_connection = True
                try:
                    handle_voice_websocket(self, voice_relay)
                except Exception as exc:
                    logger.warning(f"⚠️  Voice relay connection failed ({exc})")
            else:
                self.send_response(400)
                self.send_header("Content-Type", "text/plain")
                self.end_headers_with_cors()
                self.wfile.write(b"WebSocket upgrade required. Connect to ws://<host>:<port>/voice")

        elif parsed.path == "/video_feed":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers_with_cors()

            try:
                while running:
                    with frame_lock:
                        frame_bytes = latest_frame

                    if frame_bytes:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.04)
            except (ConnectionResetError, BrokenPipeError):
                pass

        elif parsed.path == "/video_frame":
            # Single JPEG snapshot endpoint (cross-platform: iOS Safari, Android, Web)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers_with_cors()
            with frame_lock:
                frame_bytes = latest_frame
            if frame_bytes:
                self.wfile.write(frame_bytes)
            else:
                self.wfile.write(b"")

        elif parsed.path == "/telemetry":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers_with_cors()
            with frame_lock:
                data = json.dumps(latest_telemetry).encode("utf-8")
            self.wfile.write(data)

        elif parsed.path == "/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers_with_cors()
            resp = json.dumps({"online": True, "app": "ARUGA Stream Bridge", "version": "2.0.0"}).encode("utf-8")
            self.wfile.write(resp)

        else:
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers_with_cors()
            self.wfile.write(b"Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/analyze_file":
            # Handle image/video file upload analysis
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_response(400)
                self.end_headers_with_cors()
                self.wfile.write(b"No data provided")
                return

            file_data = self.rfile.read(content_length)

            # Analyze file bytes with the canonical ARUGA pipeline (real results)
            result = self.process_uploaded_file(file_data)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers_with_cors()
            self.wfile.write(json.dumps(result).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers_with_cors()

    def process_uploaded_file(self, file_bytes):
        """Processes an uploaded image or video byte array and returns real analysis."""
        return analyze_uploaded_file(file_bytes)

    def log_message(self, format, *args):
        # Silence routine request logging in stdout
        return


def main():
    global voice_relay
    args = parse_args()

    # If Tapo credentials supplied, format RTSP URL automatically
    if args.tapo_ip and args.tapo_user and args.tapo_pass:
        source = build_rtsp_url(args.tapo_ip, args.tapo_user, args.tapo_pass, args.stream)
        logger.info(f"Constructed Tapo RTSP URL: rtsp://{args.tapo_user}:***@{args.tapo_ip}:554/{args.stream}")
    else:
        source = args.source

    # Resolve patient-audio default: synthetic when no camera, rtsp for rtsp://
    if args.patient_audio is None:
        src = str(source) if source else ""
        if src.startswith("rtsp://"):
            args.patient_audio = "rtsp"
        else:
            args.patient_audio = "synthetic"

    # Start camera worker thread
    cam_thread = threading.Thread(target=camera_worker, args=(source,), daemon=True)
    cam_thread.start()

    # Start two-way voice relay (patient audio broadcast + nurse mic sink)
    if VOICE_AVAILABLE:
        try:
            voice_relay = VoiceRelay(patient_mode=args.patient_audio,
                                     sink_mode=args.voice_sink,
                                     rtsp_url=str(source) if str(source).startswith("rtsp://") else "")
            voice_relay.start()
            logger.info(f"🎙 Voice relay enabled: ws://{args.host}:{args.port}/voice | patient= {args.patient_audio} | sink= {voice_relay.nurse_sink.effective_mode}")
        except Exception as exc:
            logger.warning(f"⚠️  Voice relay disabled ({exc}) — video endpoints unaffected")
            voice_relay = None
    else:
        logger.warning(f"⚠️  Voice relay unavailable ({_voice_relay_error}) — video endpoints unaffected")

    # Start HTTP & Streaming server
    server_address = (args.host, args.port)
    httpd = ThreadedHTTPServer(server_address, BridgeRequestHandler)
    logger.info(f"===========================================================")
    logger.info(f"🚀 ARUGA Stream Bridge live at http://{args.host}:{args.port}")
    logger.info(f"📹 Video Feed Endpoint: http://{args.host}:{args.port}/video_feed")
    logger.info(f"📊 Telemetry Endpoint:  http://{args.host}:{args.port}/telemetry")
    logger.info(f"📁 File Analysis API:   http://{args.host}:{args.port}/analyze_file")
    logger.info(f"===========================================================")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down bridge server...")
    finally:
        global running
        running = False
        cam_thread.join(timeout=3.0)
        httpd.server_close()

if __name__ == "__main__":
    main()
