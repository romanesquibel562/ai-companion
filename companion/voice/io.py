"""Local voice I/O.

speak() — Piper TTS, pitched/slowed for a flat HAL-9000 delivery. Fully offline.
listen() — Vosk STT, records from USB mic via ALSA, fully offline.
"""
from __future__ import annotations

import json
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
    def listen(self) -> str:
        raise NotImplementedError("No microphone — voice input not available.")

    def speak(self, text: str) -> None:
        return


class PiperVoiceIO(VoiceIO):
    def __init__(self, config: VoiceConfig):
        self._config = config
        model_path = Path(config.model_path)
        self._model_path = model_path if model_path.is_absolute() else ROOT_DIR / model_path
        stt_path = Path(config.stt_model_path)
        self._stt_model_path = stt_path if stt_path.is_absolute() else ROOT_DIR / stt_path
        self._vosk_model = None

    def _get_vosk_model(self):
        if self._vosk_model is None:
            from vosk import Model
            self._vosk_model = Model(str(self._stt_model_path))
        return self._vosk_model

    def listen(self) -> str:
        from vosk import KaldiRecognizer
        model = self._get_vosk_model()
        rec = KaldiRecognizer(model, 16000)

        # Record from USB mic via ALSA, resample to 16000 Hz
        proc = subprocess.run(
            [
                "arecord",
                "-D", self._config.mic_device,
                "-f", "S16_LE",
                "-r", "16000",
                "-c", "1",
                "-d", str(self._config.listen_seconds),
                "-t", "raw",
                "-q",
            ],
            capture_output=True,
            check=True,
        )

        rec.AcceptWaveform(proc.stdout)
        result = json.loads(rec.FinalResult())
        return result.get("text", "").strip()

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
