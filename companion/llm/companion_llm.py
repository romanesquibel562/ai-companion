"""Phase 2 placeholder: local LLM via Ollama for conversation/coding help.

Not wired into the pipeline yet. Planned shape:

    from ollama import Client
    client = Client()
    response = client.chat(model="llama3.1", messages=[...])

This will eventually take scene context from companion.vision (e.g. "owner
detected", "unknown person at door") plus user text/voice input, so the
companion can comment on what it sees and hold a conversation.
"""


class CompanionLLM:
    def __init__(self, model: str = "llama3.1"):
        self.model = model
        raise NotImplementedError("Phase 2: wire up Ollama client here.")

    def chat(self, message: str, scene_context: str | None = None) -> str:
        raise NotImplementedError
