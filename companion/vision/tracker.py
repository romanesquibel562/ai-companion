"""Lightweight IOU tracker: assigns stable track_ids to detections across frames."""
from __future__ import annotations

from companion.vision.types import BoundingBox, Detection


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter / float(area_a + area_b - inter)


class IOUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 10):
        self._iou_threshold = iou_threshold
        self._max_misses = max_misses
        self._next_id = 1
        self._tracks: dict[int, tuple[BoundingBox, int]] = {}  # id -> (last_box, misses)

    def update(self, detections: list[Detection]) -> list[Detection]:
        unmatched_track_ids = set(self._tracks.keys())

        for det in detections:
            best_id, best_iou = None, 0.0
            for track_id in unmatched_track_ids:
                box, _ = self._tracks[track_id]
                score = _iou(box, det.box)
                if score > best_iou:
                    best_id, best_iou = track_id, score

            if best_id is not None and best_iou >= self._iou_threshold:
                det.track_id = best_id
                self._tracks[best_id] = (det.box, 0)
                unmatched_track_ids.discard(best_id)
            else:
                det.track_id = self._next_id
                self._tracks[self._next_id] = (det.box, 0)
                self._next_id += 1

        for track_id in unmatched_track_ids:
            box, misses = self._tracks[track_id]
            misses += 1
            if misses > self._max_misses:
                del self._tracks[track_id]
            else:
                self._tracks[track_id] = (box, misses)

        return detections
