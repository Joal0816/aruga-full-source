import os
import cv2
import numpy as np
from typing import List, Tuple

from core.pose_estimator import PoseEstimator


def _iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1)) * max(1, (ay2 - ay1))
    area_b = max(1, (bx2 - bx1)) * max(1, (by2 - by1))
    return inter / float(area_a + area_b - inter)


def _nms_merge(boxes: List[List[int]], iou_thresh: float = 0.4) -> List[List[int]]:
    """Merge overlapping box proposals (largest area wins)."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept: List[List[int]] = []
    for b in boxes:
        if all(_iou(b, k) < iou_thresh for k in kept):
            kept.append(b)
    return kept


class PersonDetector:
    """
    Multi-person box detector for rapid prototyping — zero new pip dependencies.

    Backends (auto-selected, best available first):
      1. YOLOv8n ONNX via cv2.dnn — ONLY if the user drops `yolov8n.onnx`
         into assets/models/ manually. Best accuracy, incl. lying persons.
      2. Tiled MediaPipe pose proposals (always available): runs a lightweight
         pose pass (complexity 0, detection mode) on the full frame + left/right
         overlapping vertical tiles and derives boxes from visible landmarks.
         Catches lying/fallen persons that HOG misses.
      3. HOG person detector (built into OpenCV): assists with upright/far
         persons where the pose pass fails.

    Returns boxes as [x1, y1, x2, y2] pixel coords, capped to max_persons.
    Known prototype limitation: heavily overlapping persons may merge into one box.
    """

    def __init__(
        self,
        max_persons: int = 3,
        conf_thresh: float = 0.45,
        nms_thresh: float = 0.45,
        input_size: int = 640,
        model_path: str = os.path.join("assets", "models", "yolov8n.onnx"),
        use_tiles: bool = True,
        use_hog: bool = True,
    ):
        self.max_persons = max(1, min(3, int(max_persons)))
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size
        self.model_path = model_path
        self.use_tiles = use_tiles
        self.use_hog = use_hog
        self.backend = "none"
        self.net = None
        self.hog = None
        self._tile_poses: List[PoseEstimator] = []
        self._init_backend()

    # ---------- backend init ----------

    def _init_backend(self):
        # 1. Optional YOLO drop-in (no auto-download; offline-safe)
        if os.path.exists(self.model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.backend = "onnx_yolo"
                print("[PersonDetector] Using YOLO ONNX backend.")
            except Exception as e:
                print(f"[PersonDetector] ONNX load failed ({e}); using built-in backends.")
                self.net = None
        else:
            print("[PersonDetector] No yolov8n.onnx in assets/models/ — "
                  "using built-in tiled-pose + HOG backends (no download needed).")

        # 2+3. Built-in backends (always available, no downloads)
        if self.use_tiles:
            try:
                # Stateless detection-mode estimators: full frame + 2 tiles.
                # Shared instances are only used inside detect() (sequential calls).
                self._tile_poses = [
                    PoseEstimator(static_image_mode=True, model_complexity=0)
                    for _ in range(3)
                ]
            except Exception as e:
                print(f"[PersonDetector] Tile pose init failed: {e}")
                self._tile_poses = []
        if self.use_hog:
            try:
                self.hog = cv2.HOGDescriptor()
                self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            except Exception as e:
                print(f"[PersonDetector] HOG init failed: {e}")
                self.hog = None
        if self.backend == "none" and (self._tile_poses or self.hog is not None):
            self.backend = "builtin"

    def set_max_persons(self, n: int):
        self.max_persons = max(1, min(3, int(n)))

    def close(self):
        for p in self._tile_poses:
            try:
                p.close()
            except Exception:
                pass

    # ---------- public API ----------

    def detect(self, frame_bgr: np.ndarray) -> List[List[int]]:
        """Returns list of [x1, y1, x2, y2] boxes (up to max_persons)."""
        if self.backend == "onnx_yolo":
            try:
                return self._detect_onnx(frame_bgr)
            except Exception as e:
                print(f"[PersonDetector] ONNX detect error: {e}")
        proposals: List[List[int]] = []
        if self._tile_poses:
            proposals += self._detect_pose_tiles(frame_bgr)
        if self.hog is not None:
            proposals += self._detect_hog(frame_bgr)
        merged = _nms_merge(proposals)
        return merged[: self.max_persons]

    # ---------- backend: tiled MediaPipe pose proposals ----------

    def _pose_box(self, estimator: PoseEstimator, img: np.ndarray,
                  ox: int, full_w: int, full_h: int) -> List[int]:
        """Run one pose pass on img (offset ox in full frame), return global box or []."""
        try:
            has_pose, _, kp = estimator.process_frame(img)
        except Exception:
            return []
        if not has_pose or kp is None:
            return []
        lms = kp.get("all_landmarks", [])
        xs = [lm.x for lm in lms if lm.visibility > 0.3]
        ys = [lm.y for lm in lms if lm.visibility > 0.3]
        if len(xs) < 4:
            return []
        ih, iw = img.shape[:2]
        # Tile-local norm -> full-frame pixels
        x1 = int(max(0, ox + min(xs) * iw))
        y1 = 0  # tiles span full height, so y maps directly
        x2 = int(min(full_w, ox + max(xs) * iw))
        y1 = int(max(0, min(ys) * full_h))
        y2 = int(min(full_h, max(ys) * full_h))
        # Expand slightly for head/feet margins
        pad = int(0.08 * max(1, y2 - y1))
        y1 = max(0, y1 - pad)
        y2 = min(full_h, y2 + pad)
        padx = int(0.05 * max(1, x2 - x1))
        x1 = max(0, x1 - padx)
        x2 = min(full_w, x2 + padx)
        if x2 - x1 < 20 or y2 - y1 < 40:
            return []
        return [x1, y1, x2, y2]

    def _detect_pose_tiles(self, frame_bgr: np.ndarray) -> List[List[int]]:
        h, w = frame_bgr.shape[:2]
        boxes: List[List[int]] = []
        # (x_offset, estimator_idx, image)
        views: List[Tuple[int, int, np.ndarray]] = [(0, 0, frame_bgr)]
        if w >= 320 and len(self._tile_poses) >= 3:
            half = w // 2
            overlap = w // 6  # generous overlap so center persons appear in both tiles
            left = frame_bgr[:, : half + overlap]
            right = frame_bgr[:, half - overlap:]
            views += [(0, 1, left), (half - overlap, 2, right)]
        for ox, ei, img in views:
            if ei >= len(self._tile_poses):
                break
            b = self._pose_box(self._tile_poses[ei], img, ox, w, h)
            if b:
                boxes.append(b)
        return boxes

    # ---------- backend: HOG (upright assist) ----------

    def _detect_hog(self, frame_bgr: np.ndarray) -> List[List[int]]:
        h, w = frame_bgr.shape[:2]
        try:
            rects, _ = self.hog.detectMultiScale(frame_bgr, winStride=(8, 8), padding=(8, 8), scale=1.05)
        except Exception:
            return []
        boxes = []
        for (x, y, bw, bh) in rects:
            pad_w, pad_h = int(bw * 0.1), int(bh * 0.07)
            x1 = max(0, x + pad_w)
            y1 = max(0, y + pad_h)
            x2 = min(w, x + bw - pad_w)
            y2 = min(h, y + bh - pad_h)
            if x2 - x1 >= 20 and y2 - y1 >= 40:
                boxes.append([x1, y1, x2, y2])
        return boxes

    # ---------- backend: optional YOLO ONNX drop-in ----------

    def _detect_onnx(self, frame_bgr: np.ndarray) -> List[List[int]]:
        h, w = frame_bgr.shape[:2]
        size = self.input_size
        scale = min(size / w, size / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame_bgr, (nw, nh))
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (size, size), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = np.array(self.net.forward()).squeeze()
        if out.shape[0] == 84:
            out = out.T  # -> (8400, 84): [cx, cy, bw, bh, 80 classes]

        boxes, scores = [], []
        for row in out:
            person_score = float(row[4])  # COCO class 0 = person
            if person_score >= self.conf_thresh:
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                cx = (cx - pad_x) / scale
                cy = (cy - pad_y) / scale
                bw = bw / scale
                bh = bh / scale
                x1 = int(max(0, cx - bw / 2))
                y1 = int(max(0, cy - bh / 2))
                x2 = int(min(w, cx + bw / 2))
                y2 = int(min(h, cy + bh / 2))
                if x2 - x1 < 20 or y2 - y1 < 40:
                    continue
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(person_score)

        if not boxes:
            return []
        idxs = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thresh, self.nms_thresh)
        if len(idxs) == 0:
            return []
        idxs = np.array(idxs).flatten()
        ranked = sorted(idxs, key=lambda i: scores[i], reverse=True)[: self.max_persons]
        return [[boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]] for i in ranked]
