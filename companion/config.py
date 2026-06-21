"""Central config for the companion. Loads config.yaml, falls back to defaults."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
FACES_DIR = ROOT_DIR / "data" / "faces"


@dataclass
class CameraConfig:
    backend: str = "mock"  # "mock" | "picamera2"
    resolution: tuple[int, int] = (640, 480)
    fps: int = 30


@dataclass
class DetectorConfig:
    backend: str = "mock"  # "mock" | "hailo_yolov8"
    model_path: str = ""
    confidence_threshold: float = 0.5


@dataclass
class FaceRecognizerConfig:
    backend: str = "mock"  # "mock" | "opencv_lbph" | future embedding backend
    owner_name: str = "owner"


@dataclass
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class CompanionConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    face_recognizer: FaceRecognizerConfig = field(default_factory=FaceRecognizerConfig)
    web: WebConfig = field(default_factory=WebConfig)
    addons: list[str] = field(default_factory=list)


def _merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Path = CONFIG_PATH) -> CompanionConfig:
    cfg = CompanionConfig()
    if not path.exists():
        return cfg

    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    if "camera" in raw:
        cfg.camera = CameraConfig(**_merge(cfg.camera.__dict__.copy(), raw["camera"]))
    if "detector" in raw:
        cfg.detector = DetectorConfig(**_merge(cfg.detector.__dict__.copy(), raw["detector"]))
    if "face_recognizer" in raw:
        cfg.face_recognizer = FaceRecognizerConfig(
            **_merge(cfg.face_recognizer.__dict__.copy(), raw["face_recognizer"])
        )
    if "web" in raw:
        cfg.web = WebConfig(**_merge(cfg.web.__dict__.copy(), raw["web"]))
    if "addons" in raw:
        cfg.addons = list(raw["addons"])
    return cfg
