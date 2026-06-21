"""Draws detection boxes/labels onto a frame for the 'see through its eyes' view."""
from __future__ import annotations

import cv2
import numpy as np

from companion.vision.types import Detection

OWNER_COLOR = (0, 200, 0)
UNKNOWN_PERSON_COLOR = (0, 165, 255)
OBJECT_COLOR = (255, 180, 0)


def _color_for(det: Detection) -> tuple[int, int, int]:
    if det.label != "person":
        return OBJECT_COLOR
    if det.is_owner:
        return OWNER_COLOR
    return UNKNOWN_PERSON_COLOR


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        color = _color_for(det)
        box = det.box
        cv2.rectangle(out, (box.x1, box.y1), (box.x2, box.y2), color, 2)

        label = det.label
        if det.label == "person":
            label = "owner" if det.is_owner else ("unknown" if det.is_owner is False else "person")
        if det.track_id is not None:
            label = f"{label} #{det.track_id}"
        if det.distance_m is not None:
            label += f" {det.distance_m:.1f}m"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (box.x1, box.y1 - th - 6), (box.x1 + tw + 4, box.y1), color, -1)
        cv2.putText(
            out, label, (box.x1 + 2, box.y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )
    return out
