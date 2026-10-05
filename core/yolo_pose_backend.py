import os
import time
import cv2
import numpy as np
from types import SimpleNamespace
from typing import List, Dict, Any, Optional

try:
    import onnxruntime as ort
    _ORT_OK = True
except Exception:
    ort = None
    _ORT_OK = False

# COCO-17 indices in YOLO-pose output
COCO = {
    "nose": 0,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

# Skeleton pairs (COCO indices) for overlay drawing
COCO_PAIRS = [
    (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]


class YoloPoseBackend:
    """
    Single-pass multi-person detect+pose for low-end hardware.

    - Model: YOLOv8n-pose ONNX (~13MB, vendored, AGPL-3.0 Ultralytics weights).
    - Runtime: ONNX Runtime. Execution providers are *benchmarked* at startup
      (CUDA if onnxruntime-gpu is installed, else DirectML for AMD/Intel iGPUs
      on Windows, else CPU) and the fastest wins. Scales from Ryzen 3 iGPU
      up to gaming-laptop CUDA by just swapping the ORT package / model file.
    - Output per person: bbox (full-frame px), score, COCO-17 keypoints
      (full-frame px + conf), plus a FeatureExtractor-compatible keypoint dict.

    Output tensor layout (Ultralytics pose export): (1, 56, 8400) with
    rows 0-3 = cx,cy,w,h | row 4 = person score | rows 5-55 = 17 x (x,y,conf).
    All coordinates are in 640-letterbox space and mapped back here.
    """

    def __init__(
        self,
        model_path: str = os.path.join("assets", "models", "yolov8n-pose.onnx"),
        conf_thresh: float = 0.45,
        maintain_thresh: float = 0.20,
        kpt_thresh: float = 0.35,
        input_size: int = 640,
        max_persons: int = 8,
        providers: Optional[List[str]] = None,  # None = auto benchmark
        benchmark_iters: int = 6,
    ):
        if not _ORT_OK:
            raise RuntimeError("onnxruntime is not installed (pip install onnxruntime-directml)")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Pose model not found: {model_path}. "
                "It ships with the repo (assets/models/) — re-clone or re-run setup.")
        self.model_path = model_path
        self.conf_thresh = conf_thresh      # to CONFIRM a new track
        self.maintain_thresh = maintain_thresh  # to MAINTAIN an existing track
        self.low_light_boost = False  # CLAHE contrast lift for dim hallways (toggle live)
        try:
            self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        except Exception:
            self._clahe = None
        self.kpt_thresh = kpt_thresh
        self.input_size = input_size
        self.max_persons = max_persons

        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, os.cpu_count() or 4)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available = ort.get_available_providers()
        if providers is None:
            # Prefer order for benchmarking: CUDA (gaming laptops) > DirectML (AMD/Intel iGPU) > CPU
            candidates = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
                          if p in available]
        else:
            candidates = [p for p in providers if p in available] or ["CPUExecutionProvider"]

        self.backend_name = "CPU"
        if len(candidates) == 1:
            self.session = ort.InferenceSession(model_path, sess_options=so, providers=candidates)
            self.backend_name = self._short(candidates[0])
        else:
            self.session, self.backend_name = self._benchmark(so, candidates, iters=benchmark_iters)
        print(f"[YoloPose] backend: {self.backend_name} "
              f"(tried {[self._short(c) for c in candidates]})")

    @staticmethod
    def _short(provider: str) -> str:
        return {"CUDAExecutionProvider": "CUDA", "DmlExecutionProvider": "DirectML",
                "CPUExecutionProvider": "CPU"}.get(provider, provider)

    def _benchmark(self, sess_options, candidates: List[str], iters: int = 6):
        dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
        best, best_ms, best_name = None, float("inf"), "CPU"
        for prov in candidates:
            try:
                sess = ort.InferenceSession(self.model_path, sess_options=sess_options, providers=[prov])
                sess.run(None, {"images": dummy})  # warmup (esp. DirectML shader compile)
                t0 = time.time()
                for _ in range(iters):
                    sess.run(None, {"images": dummy})
                ms = (time.time() - t0) / iters * 1000.0
                print(f"[YoloPose]   {self._short(prov)}: {ms:.1f} ms/infer")
                if ms < best_ms:
                    best, best_ms, best_name = sess, ms, self._short(prov)
            except Exception as e:
                print(f"[YoloPose]   {self._short(prov)} failed ({e}), skipped")
        if best is None:
            best = ort.InferenceSession(self.model_path, sess_options=sess_options,
                                        providers=["CPUExecutionProvider"])
        return best, best_name

    def set_max_persons(self, n: int):
        self.max_persons = max(1, min(12, int(n)))

    # ---------- inference ----------

    def infer(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Full pipeline: preprocess -> ORT -> decode+NMS -> person dicts."""
        h, w = frame_bgr.shape[:2]
        size = self.input_size
        scale = min(size / w, size / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame_bgr, (nw, nh))
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

        if self.low_light_boost and self._clahe is not None:
            try:
                lab = cv2.cvtColor(canvas, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
                canvas = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            except Exception:
                pass

        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = np.array(self.session.run(None, {"images": blob})[0]).squeeze()  # (56, 8400)
        if out.shape[0] != 56:
            out = out.T
        return self._decode(out, scale, pad_x, pad_y, w, h)

    def _decode(self, out: np.ndarray, scale: float, pad_x: int, pad_y: int,
                full_w: int, full_h: int) -> List[Dict[str, Any]]:
        def unmap_x(x):
            return float(min(full_w - 1, max(0, (x - pad_x) / scale)))

        def unmap_y(y):
            return float(min(full_h - 1, max(0, (y - pad_y) / scale)))

        boxes, scores, kept = [], [], []
        for i in range(out.shape[1]):
            score = float(out[4, i])
            if score < self.maintain_thresh:
                continue
            cx, cy, bw, bh = (unmap_x(out[0, i]), unmap_y(out[1, i]),
                              float(out[2, i] / scale), float(out[3, i] / scale))
            x1, y1 = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2))
            x2, y2 = int(min(full_w, cx + bw / 2)), int(min(full_h, cy + bh / 2))
            if x2 - x1 < 20 or y2 - y1 < 40:
                continue
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)
            kept.append(i)

        if not boxes:
            return []
        idxs = cv2.dnn.NMSBoxes(boxes, scores, self.maintain_thresh, 0.45)
        if len(idxs) == 0:
            return []
        ranked = sorted(np.array(idxs).flatten(), key=lambda j: scores[j], reverse=True)
        ranked = ranked[: self.max_persons]

        persons = []
        for j in ranked:
            i = kept[j]
            x, y, bw, bh = boxes[j]
            kpts = np.zeros((17, 3), dtype=np.float32)
            for k in range(17):
                kpts[k, 0] = unmap_x(out[5 + k * 3, i])
                kpts[k, 1] = unmap_y(out[5 + k * 3 + 1, i])
                kpts[k, 2] = float(out[5 + k * 3 + 2, i])
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            # Pose-union refinement: nano boxes often clip lying bodies
            # (head/feet cut). Stretch the box to cover confident keypoints.
            vis = kpts[kpts[:, 2] >= self.kpt_thresh]
            if len(vis) >= 4:
                m = int(0.08 * max(1, max(x2 - x1, y2 - y1)))
                x1 = int(max(0, min(x1, vis[:, 0].min() - m)))
                y1 = int(max(0, min(y1, vis[:, 1].min() - m)))
                x2 = int(min(full_w, max(x2, vis[:, 0].max() + m)))
                y2 = int(min(full_h, max(y2, vis[:, 1].max() + m)))
            persons.append({
                "bbox": [x1, y1, x2, y2],
                "score": scores[j],
                "kpts": kpts,  # 17x3: x_px, y_px, conf (full-frame coords)
            })
        return persons

    # ---------- mapping to existing pipeline format ----------

    def to_keypoints(self, person: Dict[str, Any], full_w: int, full_h: int) -> Dict[str, Any]:
        """Build a FeatureExtractor-compatible keypoint dict from COCO-17."""
        kpts = person["kpts"]

        def pt(idx: int) -> Dict[str, Any]:
            x, y, c = float(kpts[idx, 0]), float(kpts[idx, 1]), float(kpts[idx, 2])
            return {"x_norm": x / max(1, full_w), "y_norm": y / max(1, full_h),
                    "z_norm": 0.0, "x_px": int(x), "y_px": int(y), "visibility": c}

        kp = {name: pt(idx) for name, idx in COCO.items()}
        kp["frame_width"] = full_w
        kp["frame_height"] = full_h
        kp["all_landmarks"] = [
            SimpleNamespace(x=float(kpts[k, 0] / max(1, full_w)),
                            y=float(kpts[k, 1] / max(1, full_h)),
                            z=0.0, visibility=float(kpts[k, 2]))
            for k in range(17)
        ]
        # Overall pose quality from the joints the FSM actually uses
        core = [kpts[COCO[n], 2] for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
        kp["pose_score"] = float(np.mean(core))
        return kp
