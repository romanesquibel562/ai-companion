"""Phase 3 placeholder: voice input (STT, e.g. whisper.cpp) and output (TTS, e.g. piper).

Not wired into the pipeline yet. Planned shape: a wake-word or push-to-talk
loop that transcribes speech -> CompanionLLM.chat() -> speaks the reply.
"""


class VoiceIO:
    def __init__(self):
        raise NotImplementedError("Phase 3: wire up STT/TTS here.")

    def listen(self) -> str:
        raise NotImplementedError

    def speak(self, text: str) -> None:
        raise NotImplementedError
