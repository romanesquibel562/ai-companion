"""Loads and dispatches to enabled addons, by name from config.yaml's `addons:` list.

Add new addons here as they're built, e.g.:
    "screen_display": ScreenDisplayAddon,
    "speaker": SpeakerAddon,
"""
from __future__ import annotations

from companion.addons.base import Addon
from companion.vision.types import Frame

ADDON_CLASSES: dict[str, type[Addon]] = {
    # "screen_display": ScreenDisplayAddon,
}


class AddonRegistry:
    def __init__(self, enabled_names: list[str] | None = None):
        self._addons: list[Addon] = []
        for name in enabled_names or []:
            cls = ADDON_CLASSES.get(name)
            if cls is None:
                raise ValueError(f"Unknown addon '{name}'. Available: {list(ADDON_CLASSES)}")
            self._addons.append(cls())
        for addon in self._addons:
            addon.on_start()

    def on_frame(self, frame: Frame) -> None:
        for addon in self._addons:
            addon.on_frame(frame)

    def stop(self) -> None:
        for addon in self._addons:
            addon.on_stop()
