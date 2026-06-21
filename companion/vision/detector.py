"""Object detector backends. Swap via config.detector.backend."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

import cv2
import numpy as np

from companion.config import DetectorConfig
from companion.vision.coco_labels import COCO_CLASSES
from companion.vision.types import BoundingBox, Detection


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on one frame and return detections."""

    def close(self) -> None:
        pass


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
    """Real backend for the Hailo AI HAT+.

    Runs a YOLOv8 model with on-chip NMS post-processing (the Hailo model zoo
    .hef files at /usr/share/hailo-models/ already bake this in, so the
    output vstream gives finished per-class boxes — no manual NMS/decoding).

    Output format confirmed empirically against this exact model: a list
    (batch) of length 1, containing a list of 80 per-class entries, each a
    list of [ymin, xmin, ymax, xmax, score] in normalized 0-1 coords
    relative to the letterboxed 640x640 input.
    """

    def __init__(self, config: DetectorConfig):
        if not config.model_path:
            raise ValueError("detector.model_path must point to a compiled .hef file")
        from hailo_platform import (
            HEF,
            ConfigureParams,
            FormatType,
            HailoStreamInterface,
            InferVStreams,
            InputVStreamParams,
            OutputVStreamParams,
            VDevice,
        )

        self._config = config
        self._hef = HEF(config.model_path)
        self._target = VDevice()
        configure_params = ConfigureParams.create_from_hef(
            self._hef, interface=HailoStreamInterface.PCIe
        )
        self._network_group = self._target.configure(self._hef, configure_params)[0]
        network_group_params = self._network_group.create_params()

        self._input_info = self._hef.get_input_vstream_infos()[0]
        self._output_info = self._hef.get_output_vstream_infos()[0]
        self._input_h, self._input_w = self._input_info.shape[:2]

        input_params = InputVStreamParams.make(
            self._network_group, quantized=True, format_type=FormatType.UINT8
        )
        output_params = OutputVStreamParams.make(
            self._network_group, quantized=False, format_type=FormatType.FLOAT32
        )

        self._infer_pipeline = InferVStreams(self._network_group, input_params, output_params)
        self._infer_pipeline.__enter__()
        self._activated_network = self._network_group.activate(network_group_params)
        self._activated_network.__enter__()

    def _letterbox(self, rgb: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = rgb.shape[:2]
        scale = min(self._input_w / w, self._input_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(rgb, (new_w, new_h))
        canvas = np.full((self._input_h, self._input_w, 3), 114, dtype=np.uint8)
        pad_left, pad_top = (self._input_w - new_w) // 2, (self._input_h - new_h) // 2
        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
        return canvas, scale, pad_left, pad_top

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        canvas, scale, pad_left, pad_top = self._letterbox(rgb)
        batch = np.expand_dims(canvas, axis=0)

        results = self._infer_pipeline.infer({self._input_info.name: batch})
        per_class = results[self._output_info.name][0]

        detections: list[Detection] = []
        for class_id, class_detections in enumerate(per_class):
            if len(class_detections) == 0:
                continue
            label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id)
            for ymin, xmin, ymax, xmax, score in class_detections:
                if score < self._config.confidence_threshold:
                    continue
                x1 = (xmin * self._input_w - pad_left) / scale
                y1 = (ymin * self._input_h - pad_top) / scale
                x2 = (xmax * self._input_w - pad_left) / scale
                y2 = (ymax * self._input_h - pad_top) / scale
                box = BoundingBox(
                    x1=max(0, int(x1)),
                    y1=max(0, int(y1)),
                    x2=min(w, int(x2)),
                    y2=min(h, int(y2)),
                )
                detections.append(Detection(box=box, label=label, confidence=float(score)))
        return detections

    def close(self) -> None:
        self._activated_network.__exit__(None, None, None)
        self._infer_pipeline.__exit__(None, None, None)
        self._target.release()


def create_detector(config: DetectorConfig) -> Detector:
    if config.backend == "mock":
        return MockDetector(config)
    if config.backend == "hailo_yolov8":
        return HailoYoloV8Detector(config)
    raise ValueError(f"Unknown detector backend: {config.backend}")
