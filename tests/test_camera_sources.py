"""
Tests for camera source management (no hardware needed).

  - RTSP URL validation accept/reject
  - Profile CRUD + persistence roundtrip (temp dir)
  - zones_path_for mapping (default keeps legacy path)
  - open_capture on a real (generated) video file
  - probe_usb_cameras never raises, returns well-formed rows

Run:  .\\.venv\\Scripts\\python.exe tests\\test_camera_sources.py
Exit code 0 = pass.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.camera_sources import (
    CameraProfiles, zones_path_for, validate_network_url,
    probe_usb_cameras, open_capture, slugify,
)


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    return cond


def main():
    import tempfile
    ok = True
    tmp = tempfile.mkdtemp(prefix="cams_")

    print("Test: RTSP URL validation")
    good = ["rtsp://admin:pw@192.168.1.50:554/stream1",
            "rtsp://192.168.0.10/live", "rtmp://x/y", "http://cam.local/mjpg"]
    bad = ["", "192.168.1.50/stream", "ftp://x/y", "rtsp:/broken"]
    ok &= check("accepts camera URLs", all(validate_network_url(u)[0] for u in good))
    ok &= check("rejects bad URLs", all(not validate_network_url(u)[0] for u in bad))

    print("Test: profiles CRUD + persistence")
    profs = CameraProfiles(path=os.path.join(tmp, "cameras.json"))
    profs.upsert({"name": "Hall USB", "kind": "usb", "index": 1})
    profs.upsert({"name": "Gate", "kind": "rtsp", "url": "rtsp://a@b/c"})
    profs.upsert({"name": "Clip", "kind": "file", "path": "c:\\x.mp4"})
    try:
        profs.upsert({"name": "Bad", "kind": "laser"})
        ok &= check("rejects bad kind", False)
    except ValueError:
        ok &= check("rejects bad kind", True)
    profs.active = "Gate"
    profs.save()
    profs2 = CameraProfiles(path=os.path.join(tmp, "cameras.json"))
    ok &= check("roundtrip names", profs2.names() == ["Hall USB", "Gate", "Clip"],
                f"({profs2.names()})")
    ok &= check("roundtrip active", profs2.active == "Gate")
    ok &= check("describe usb", profs2.describe(profs2.get("Hall USB")) == "USB camera index 1")
    profs2.delete("Gate")
    ok &= check("delete clears active", profs2.active is None and profs2.names() == ["Hall USB", "Clip"])
    profs3 = CameraProfiles(path=os.path.join(tmp, "nope.json"))
    ok &= check("missing file -> empty", profs3.names() == [] and profs3.active is None)

    print("Test: zone paths + slugs")
    ok &= check("default keeps legacy", zones_path_for("default") == os.path.join("assets", "zones.json"))
    ok &= check("profile gets own file", zones_path_for("Gate RTSP") == os.path.join("assets", "zones_gate_rtsp.json"))
    ok &= check("slugify", slugify("Hallway #1 (East)") == "hallway_1_east")

    print("Test: open_capture on generated clip")
    mp4 = os.path.join(tmp, "t.mp4")
    out = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for _ in range(5):
        out.write(np.zeros((120, 160, 3), dtype=np.uint8))
    out.release()
    cap = open_capture(mp4, "file")
    ok &= check("opens file", cap.isOpened())
    n = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        n += 1
    cap.release()
    ok &= check("reads 5 frames", n == 5, f"({n})")

    print("Test: USB probe is safe")
    rows = probe_usb_cameras(max_index=1)
    ok &= check("well-formed rows", isinstance(rows, list)
                and all(set(r) == {"index", "ok", "width", "height", "fps"} for r in rows))
    ok &= check("empty range -> empty", probe_usb_cameras(max_index=0) == [])

    print("Test: network stream end-to-end (local MJPEG-over-HTTP server)")
    import threading
    import time as _time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    payloads = []
    for i in range(10):
        img = np.full((120, 160, 3), i * 20, dtype=np.uint8)
        payloads.append(cv2.imencode(".jpg", img)[1].tobytes())

    class _H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            i = 0
            try:
                while True:
                    jpg = payloads[i % len(payloads)]
                    i += 1
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                    _time.sleep(0.05)
            except Exception:
                pass

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 8899), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        ncap = open_capture("http://127.0.0.1:8899/cam.mjpg", "rtsp")
        ok &= check("network stream opens", ncap.isOpened())
        n, shape = 0, None
        for _ in range(6):
            ret, fr = ncap.read()
            if ret:
                n += 1
                shape = fr.shape
        ncap.release()
        ok &= check("network frames flow", n >= 4 and shape == (120, 160, 3), f"({n} frames)")
    finally:
        srv.shutdown()

    print("\n" + ("ALL CAMERA TESTS PASS" if ok else "CAMERA TESTS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
