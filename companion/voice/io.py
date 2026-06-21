"""Local voice output via Piper TTS, pitched/slowed for a flat, deliberate
HAL-9000-style delivery. Fully offline.

Voice input (STT) is not implemented yet — no microphone is connected to the
Pi. `listen()` raises until that hardware arrives; `speak()` works today
(audio just has nowhere to play until a speaker is connected either).
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from companion.config import ROOT_DIR, VoiceConfig


class VoiceIO:
    def listen(self) -> str:
        raise NotImplementedError

    def speak(self, text: str) -> None:
        raise NotImplementedError


class MockVoiceIO(VoiceIO):
    """No audio. Lets the rest of the app call speak()/listen() safely with no hardware."""

    def listen(self) -> str:
        raise NotImplementedError("No microphone connected — voice input is not wired up.")

    def speak(self, text: str) -> None:
        return


class PiperVoiceIO(VoiceIO):
    def __init__(self, config: VoiceConfig):
        self._config = config
        model_path = Path(config.model_path)
        self._model_path = model_path if model_path.is_absolute() else ROOT_DIR / model_path

    def listen(self) -> str:
        raise NotImplementedError("No microphone connected — voice input is not wired up.")

    def speak(self, text: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_wav = Path(tmp) / "raw.wav"
            tuned_wav = Path(tmp) / "tuned.wav"

            subprocess.run(
                [
                    "python3", "-m", "piper",
                    "-m", str(self._model_path),
                    "--length-scale", str(self._config.length_scale),
                    "-f", str(raw_wav),
                ],
                input=text.encode(),
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(raw_wav),
                    "-af", f"asetrate=22050*{self._config.pitch_scale},aresample=22050",
                    str(tuned_wav),
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(["aplay", "-q", str(tuned_wav)], check=True, capture_output=True)


def create_voice_io(config: VoiceConfig) -> VoiceIO:
    if config.backend == "mock":
        return MockVoiceIO()
    if config.backend == "piper":
        return PiperVoiceIO(config)
    raise ValueError(f"Unknown voice backend: {config.backend}")
