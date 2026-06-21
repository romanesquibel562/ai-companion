"""Object detector backends. Swap via config.detector.backend."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

from companion.config import DetectorConfig
from companion.vision.types import BoundingBox, Detection


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on one frame and return detections."""


class MockDetector(Detector):
    """Returns a single fake 'person' box that drifts across the frame, for dev without a Hailo HAT."""

    def __init__(self, config: DetectorConfig):
        self._t = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self._t += 1
        h, w = frame.shape[:2]
        size = min(h, w) // 3
        cx = int((w - size) * (0.5 + 0.5 * math.sin(self._t / 30)))
        cy = (h - size) // 2
        box = BoundingBox(cx, cy, cx + size, cy + size)
        return [Detection(box=box, label="person", confidence=0.9)]


class HailoYoloV8Detector(Detector):
    """Real backend for the Hailo AI HAT+. Requires `hailo_platform` and a compiled .hef model.

    TODO (on-device): load the HEF via hailo_platform.VDevice, run inference, parse
    YOLOv8 output tensors into Detection boxes using config.model_path and
    config.confidence_threshold.
    """

    def __init__(self, config: DetectorConfig):
        if not config.model_path:
            raise ValueError("detector.model_path must point to a compiled .hef file")
        self._config = config
        # from hailo_platform import VDevice, HEF  # local import: not installed off-device
        raise NotImplementedError(
            "Wire up the Hailo VDevice + HEF inference here once running on the Pi."
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        raise NotImplementedError


def create_detector(config: DetectorConfig) -> Detector:
    if config.backend == "mock":
        return MockDetector(config)
    if config.backend == "hailo_yolov8":
        return HailoYoloV8Detector(config)
    raise ValueError(f"Unknown detector backend: {config.backend}")
