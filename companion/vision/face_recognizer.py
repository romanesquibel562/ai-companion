"""Owner face recognition. Marks 'person' detections as is_owner True/False/None (unknown).

Local-only, lightweight by design: no cloud calls. Default backend is a no-op
('mock') until you enroll an owner face and pick a real backend.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from companion.config import FACES_DIR, FaceRecognizerConfig
from companion.vision.types import Detection


class FaceRecognizer(ABC):
    @abstractmethod
    def identify(self, frame: np.ndarray, detections: list[Detection]) -> None:
        """Mutate `detections` in place, setting is_owner on person boxes."""

    def enroll(self, name: str, face_image: np.ndarray) -> None:
        raise NotImplementedError


class MockFaceRecognizer(FaceRecognizer):
    """Leaves is_owner unset (None = 'unknown'). Safe default with no owner enrolled."""

    def __init__(self, config: FaceRecognizerConfig):
        self._config = config

    def identify(self, frame: np.ndarray, detections: list[Detection]) -> None:
        return


class OpenCVLBPHFaceRecognizer(FaceRecognizer):
    """Real backend using opencv-contrib's LBPH face classifier.

    Lightweight enough for the Pi 5 CPU to run alongside Hailo-accelerated
    detection. Requires `opencv-contrib-python` (provides cv2.face) and at
    least one enrolled owner photo in data/faces/<owner_name>/.

    TODO (on-device): call enroll() with a few owner photos before first run,
    or drop face crops into data/faces/<owner_name>/ and call train().
    """

    def __init__(self, config: FaceRecognizerConfig):
        import cv2

        self._config = config
        self._cv2 = cv2
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._recognizer = cv2.face.LBPHFaceRecognizer_create()
        self._trained = False
        self._owner_label = 1
        self.train()

    def train(self) -> None:
        owner_dir = FACES_DIR / self._config.owner_name
        if not owner_dir.exists():
            return
        images, labels = [], []
        for path in owner_dir.glob("*.*"):
            img = self._cv2.imread(str(path), self._cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(self._owner_label)
        if images:
            self._recognizer.train(images, np.array(labels))
            self._trained = True

    def identify(self, frame: np.ndarray, detections: list[Detection]) -> None:
        if not self._trained:
            return
        gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        for det in detections:
            if det.label != "person":
                continue
            crop = gray[det.box.y1 : det.box.y2, det.box.x1 : det.box.x2]
            faces = self._cascade.detectMultiScale(crop, 1.1, 4)
            if len(faces) == 0:
                det.is_owner = None
                continue
            fx, fy, fw, fh = faces[0]
            face_img = crop[fy : fy + fh, fx : fx + fw]
            label, confidence = self._recognizer.predict(face_img)
            # LBPH: lower distance = more confident match
            det.is_owner = label == self._owner_label and confidence < 70

    def enroll(self, name: str, face_image: np.ndarray) -> None:
        owner_dir = FACES_DIR / name
        owner_dir.mkdir(parents=True, exist_ok=True)
        existing = list(owner_dir.glob("*.png"))
        out_path = owner_dir / f"{len(existing) + 1:03d}.png"
        self._cv2.imwrite(str(out_path), face_image)
        self.train()


def create_face_recognizer(config: FaceRecognizerConfig) -> FaceRecognizer:
    if config.backend == "mock":
        return MockFaceRecognizer(config)
    if config.backend == "opencv_lbph":
        return OpenCVLBPHFaceRecognizer(config)
    raise ValueError(f"Unknown face recognizer backend: {config.backend}")
