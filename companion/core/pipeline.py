"""Orchestrates one frame through: camera -> detector -> tracker -> face recognizer -> addons -> overlay."""
from __future__ import annotations

import time

from companion.addons.registry import AddonRegistry
from companion.config import CompanionConfig
from companion.vision.camera import Camera, create_camera
from companion.vision.detector import Detector, create_detector
from companion.vision.face_recognizer import FaceRecognizer, create_face_recognizer
from companion.vision.overlay import draw_detections
from companion.vision.tracker import IOUTracker
from companion.vision.types import Frame


class Pipeline:
    def __init__(self, config: CompanionConfig, addon_registry: AddonRegistry | None = None):
        self.config = config
        self.camera: Camera = create_camera(config.camera)
        self.detector: Detector = create_detector(config.detector)
        self.face_recognizer: FaceRecognizer = create_face_recognizer(config.face_recognizer)
        self.tracker = IOUTracker()
        self.addons = addon_registry or AddonRegistry()

    def step(self) -> Frame:
        """Run one full cycle and return the frame with detections (image not yet annotated)."""
        image = self.camera.read()
        detections = self.detector.detect(image)
        detections = self.tracker.update(detections)
        self.face_recognizer.identify(image, detections)

        frame = Frame(image=image, timestamp=time.time(), detections=detections)
        self.addons.on_frame(frame)
        return frame

    def step_annotated(self):
        """Convenience for consumers (e.g. the web server) that just want a drawable frame."""
        frame = self.step()
        return draw_detections(frame.image, frame.detections), frame

    def close(self) -> None:
        self.camera.close()
