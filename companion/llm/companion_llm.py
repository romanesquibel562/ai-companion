"""Local LLM conversation via Ollama. Backs the terminal panel on the web
dashboard — this is the interactive interface until mic/speaker/screen
hardware is bought, at which point voice I/O calls the same chat() method.
"""
from __future__ import annotations

from companion.config import LLMConfig

SYSTEM_PROMPT = (
    "You are the AI companion running on a Raspberry Pi 5 with computer vision. "
    "Your owner is named Roman. The 'Current scene' message tells you who is "
    "currently in view, based on real-time face recognition. If it says a person "
    "is identified as Roman, that means Roman himself is the one talking to you "
    "right now — address him by name when it's natural to. If the scene says "
    "the person is unrecognized or not clearly visible, don't assume it's Roman. "
    "Be concise and helpful, especially with code questions."
)


class CompanionLLM:
    def __init__(self, config: LLMConfig):
        self._config = config

    def chat(self, message: str, scene_context: str | None = None) -> str:
        raise NotImplementedError


class MockCompanionLLM(CompanionLLM):
    """No Ollama call. Echoes back so the terminal panel UI is testable without a model."""

    def chat(self, message: str, scene_context: str | None = None) -> str:
        return f"(mock llm) you said: {message}"


class OllamaCompanionLLM(CompanionLLM):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        from ollama import Client

        self._client = Client(host=config.host)

    def chat(self, message: str, scene_context: str | None = None) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if scene_context:
            messages.append({"role": "system", "content": f"Current scene: {scene_context}"})
        messages.append({"role": "user", "content": message})
        response = self._client.chat(model=self._config.model, messages=messages)
        return response["message"]["content"]


def create_companion_llm(config: LLMConfig) -> CompanionLLM:
    if config.backend == "mock":
        return MockCompanionLLM(config)
    if config.backend == "ollama":
        return OllamaCompanionLLM(config)
    raise ValueError(f"Unknown llm backend: {config.backend}")
