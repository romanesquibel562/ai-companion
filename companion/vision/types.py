"""Shared data types passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


@dataclass
class Detection:
    box: BoundingBox
    label: str
    confidence: float
    track_id: int | None = None
    is_owner: bool | None = None  # set by the face recognizer for "person" detections
    distance_m: float | None = None


@dataclass
class Frame:
    image: "object"  # numpy.ndarray, kept untyped to avoid a hard numpy import here
    timestamp: float
    detections: list[Detection] = field(default_factory=list)
