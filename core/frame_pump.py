"""Kind-aware frame pump: one reader for files, webcams and network streams.

UI-agnostic (no Tk/Streamlit imports) so the desktop, hallway and web apps
share identical source behavior:
  - file   : loops forever by default (loop_files=False to end at EOF instead)
  - rtsp   : tolerates blips, reconnects with backoff, else raises SourceLost
  - webcam : tolerates transient empty reads, else raises SourceLost
Abort (e.g. user pressed Stop mid-reconnect) surfaces as SourceLost too —
callers already stopping should swallow it.

Usage:
    pump = FramePump(source, kind)          # kind: webcam | rtsp | file | auto
    try:
        for frame in pump.frames(alive=lambda: running):
            ... process ...
    except SourceLost as e:
        ... show str(e), stop ...
    finally:
        pump.release()
"""

import time
import cv2
from typing import Callable, Iterator, Optional

from core.camera_sources import open_capture, NETWORK_SCHEMES


class SourceLost(Exception):
    """Raised when a source is unrecoverable (bad path, dead camera)."""


def classify(source) -> str:
    if isinstance(source, str) and source.lower().startswith(NETWORK_SCHEMES):
        return "rtsp"
    if isinstance(source, str):
        return "file"
    return "webcam"


class FramePump:
    def __init__(self, source, kind: str = "auto",
                 webcam_size=(640, 480),
                 webcam_tolerance: int = 30,
                 rtsp_retries: int = 5,
                 loop_files: bool = True,
                 on_status: Optional[Callable[[str], None]] = None,
                 abort: Callable[[], bool] = lambda: False):
        self.source = source
        self.kind = kind if kind != "auto" else classify(source)
        self.webcam_tolerance = webcam_tolerance
        self.rtsp_retries = rtsp_retries
        self.loop_files = loop_files
        self.on_status = on_status
        self.abort = abort
        self._fails = 0
        self.cap = open_capture(source, self.kind)
        if not self.cap.isOpened():
            raise SourceLost(f"Cannot open source: {source}")
        if self.kind == "webcam" and isinstance(source, int):
            self._tune_webcam()

    def _tune_webcam(self):
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception:
            pass

    def _reopen(self):
        try:
            self.cap.release()
        except Exception:
            pass
        self.cap = open_capture(self.source, self.kind)
        if self.kind == "webcam" and isinstance(self.source, int):
            self._tune_webcam()

    def _note(self, msg: str):
        if self.on_status is not None:
            try:
                self.on_status(msg)
            except Exception:
                pass

    def read(self):
        """Next frame, or None on a transient glitch (caller continues).

        Raises SourceLost when the source is dead.
        """
        ret, frame = self.cap.read()
        if ret:
            self._fails = 0
            return frame
        self._fails += 1
        if self.kind == "file":
            if not self.loop_files:
                raise SourceLost("Video ended.")
            if self._fails > 3:
                self._reopen()  # odd codec ignoring seeks
                self._fails = 0
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return None
        if self.kind == "rtsp":
            if self._fails <= 3:
                time.sleep(0.5)
                return None
            for attempt in range(1, self.rtsp_retries + 1):
                if self.abort():
                    raise SourceLost("Stopped by user.")
                self._note(f"Reconnecting stream (try {attempt}/{self.rtsp_retries})…")
                time.sleep(2)
                self._reopen()
                ok, fr = self.cap.read()
                if ok:
                    self._fails = 0
                    return fr
            raise SourceLost("Stream lost (reconnect failed). Check camera power/network.")
        if self._fails <= self.webcam_tolerance:
            time.sleep(0.05)
            return None
        raise SourceLost("Webcam stopped responding (30 empty reads).")

    def frames(self, alive: Callable[[], bool] = lambda: True) -> Iterator:
        while alive():
            frame = self.read()  # may raise SourceLost
            if frame is None:
                continue
            yield frame

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass
