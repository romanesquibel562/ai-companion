"""Camera backends. Swap via config.camera.backend."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np

from companion.config import CameraConfig


class Camera(ABC):
    @abstractmethod
    def read(self) -> np.ndarray:
        """Return one BGR frame as a numpy array."""

    def close(self) -> None:
        pass


class MockCamera(Camera):
    """Synthetic frames for development without a Pi Camera attached."""

    def __init__(self, config: CameraConfig):
        self._w, self._h = config.resolution
        self._t0 = time.monotonic()

    def read(self) -> np.ndarray:
        frame = np.zeros((self._h, self._w, 3), dtype=np.uint8)
        # a slowly moving gray square so the dashboard has visible motion to confirm the stream is live
        t = time.monotonic() - self._t0
        size = 80
        cx = int((self._w - size) * (0.5 + 0.5 * np.sin(t)))
        frame[40 : 40 + size, cx : cx + size] = (90, 90, 90)
        return frame


class PiCamera2Camera(Camera):
    """Real Camera Module 3 backend. Requires `picamera2` installed on the Pi."""

    def __init__(self, config: CameraConfig):
        from picamera2 import Picamera2  # local import: not installed off-device

        self._picam2 = Picamera2()
        cfg = self._picam2.create_video_configuration(
            main={"size": config.resolution, "format": "RGB888"}
        )
        self._picam2.configure(cfg)
        self._picam2.start()

    def read(self) -> np.ndarray:
        import cv2

        rgb = self._picam2.capture_array()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        self._picam2.stop()


def create_camera(config: CameraConfig) -> Camera:
    if config.backend == "mock":
        return MockCamera(config)
    if config.backend == "picamera2":
        return PiCamera2Camera(config)
    raise ValueError(f"Unknown camera backend: {config.backend}")
