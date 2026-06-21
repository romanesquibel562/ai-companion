"""Addon interface for future hardware/software extensions (screen, speaker, sensors, LLM hooks).

Subclass Addon and implement the hooks you need; the rest default to no-ops.
Register new addons by name in companion/addons/registry.py and list them in
config.yaml under `addons:`.
"""
from __future__ import annotations

from companion.vision.types import Frame


class Addon:
    name: str = "addon"

    def on_start(self) -> None:
        """Called once when the pipeline starts."""

    def on_frame(self, frame: Frame) -> None:
        """Called after every processed frame, with detections already attached."""

    def on_stop(self) -> None:
        """Called once when the pipeline shuts down."""
