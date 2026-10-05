import numpy as np
from typing import List, Dict, Tuple


class CentroidTracker:
    """
    Minimal centroid tracker (no scipy/sklearn needed).
    Greedy nearest-neighbour matching on box centroids, with an optional
    second-pass IoU fallback that catches abrupt geometry flips the centroid
    gate misses (e.g. lying-wide box -> standing-tall box on recovery).

    update(detections) -> list of (track_id, bbox)
    bbox format: [x1, y1, x2, y2]
    """

    def __init__(self, max_missing: int = 15, max_distance: int = 120,
                 iou_fallback: float = 0.0):
        self.max_missing = max_missing
        self.max_distance = max_distance
        self.iou_fallback = iou_fallback
        self.next_id = 1
        self.tracks: Dict[int, Dict] = {}  # id -> {centroid, bbox, missing}

    def reset(self):
        self.next_id = 1
        self.tracks.clear()

    def remove(self, track_id: int):
        """Drop one track (e.g. sub-confirm-threshold detection for a new ID)."""
        self.tracks.pop(track_id, None)

    @staticmethod
    def _centroid(bbox) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _iou(a, b) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area = max(1, (a[2] - a[0])) * max(1, (a[3] - a[1]))
        area += max(1, (b[2] - b[0])) * max(1, (b[3] - b[1]))
        return inter / max(1.0, area - inter)

    def update(self, detections: List[List[int]]) -> List[Tuple[int, List[int]]]:
        det_centroids = [self._centroid(b) for b in detections]

        # No existing tracks: register all
        if not self.tracks:
            out = []
            for bbox in detections:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"centroid": self._centroid(bbox), "bbox": list(bbox), "missing": 0}
                out.append((tid, list(bbox)))
            return out

        # No detections: age all tracks, keep survivors on last-known bbox
        if not detections:
            dead = []
            for tid, t in self.tracks.items():
                t["missing"] += 1
                if t["missing"] > self.max_missing:
                    dead.append(tid)
            for tid in dead:
                del self.tracks[tid]
            return sorted([(tid, list(t["bbox"])) for tid, t in self.tracks.items()])

        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid]["centroid"] for tid in track_ids]

        # Distance matrix (tracks x detections)
        dist = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
        for i, tc in enumerate(track_centroids):
            for j, dc in enumerate(det_centroids):
                dist[i, j] = float(np.hypot(tc[0] - dc[0], tc[1] - dc[1]))

        matched_tracks, matched_dets = set(), set()
        pairs = []
        # Greedy: repeatedly take smallest distance pair under threshold
        flat = [(dist[i, j], i, j) for i in range(len(track_ids)) for j in range(len(detections))]
        flat.sort(key=lambda x: x[0])
        for d, i, j in flat:
            if d > self.max_distance:
                break
            if i in matched_tracks or j in matched_dets:
                continue
            matched_tracks.add(i)
            matched_dets.add(j)
            pairs.append((track_ids[i], j))

        # Second pass: IoU fallback for abrupt geometry flips (recovery).
        # Only unmatched-vs-unmatched, best-IoU-first, so crowds still resolve
        # by distance first and this merely rescues the leftovers.
        if self.iou_fallback > 0:
            cands = []
            for i, tid in enumerate(track_ids):
                if i in matched_tracks:
                    continue
                for j, bbox in enumerate(detections):
                    if j in matched_dets:
                        continue
                    cands.append((self._iou(self.tracks[tid]["bbox"], bbox), i, j))
            cands.sort(key=lambda x: x[0], reverse=True)
            for iou, i, j in cands:
                if iou < self.iou_fallback:
                    break
                if i in matched_tracks or j in matched_dets:
                    continue
                matched_tracks.add(i)
                matched_dets.add(j)
                pairs.append((track_ids[i], j))

        out: List[Tuple[int, List[int]]] = []
        # Update matched
        for tid, j in pairs:
            bbox = list(detections[j])
            self.tracks[tid] = {"centroid": det_centroids[j], "bbox": bbox, "missing": 0}
            out.append((tid, bbox))
        # Age unmatched tracks
        dead = []
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self.tracks[tid]["missing"] += 1
                if self.tracks[tid]["missing"] > self.max_missing:
                    dead.append(tid)
                else:
                    out.append((tid, list(self.tracks[tid]["bbox"])))
        for tid in dead:
            del self.tracks[tid]
        # Register unmatched detections as new tracks
        for j, bbox in enumerate(detections):
            if j not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"centroid": det_centroids[j], "bbox": list(bbox), "missing": 0}
                out.append((tid, list(bbox)))

        out.sort(key=lambda x: x[0])
        return out
