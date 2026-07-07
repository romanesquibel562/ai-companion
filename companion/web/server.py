"""FastAPI dashboard: MJPEG stream of the annotated camera feed ('see through its eyes')."""
from __future__ import annotations

import threading
import time

import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request

from companion.addons.registry import AddonRegistry
from companion.config import CompanionConfig, load_config
from companion.core.pipeline import Pipeline
from companion.llm.companion_llm import create_companion_llm
from companion.voice.io import create_voice_io


class ChatRequest(BaseModel):
    message: str


class SpeakRequest(BaseModel):
    text: str

TEMPLATES = Jinja2Templates(directory=str(__file__.rsplit("/", 1)[0] + "/templates"))


class PipelineWorker:
    """Runs the pipeline in a background thread and exposes the latest annotated frame."""

    def __init__(self, config: CompanionConfig):
        self._pipeline = Pipeline(config, AddonRegistry(config.addons))
        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_detections = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            annotated, frame = self._pipeline.step_annotated()
            ok, jpeg = cv2.imencode(".jpg", annotated)
            if ok:
                with self._lock:
                    self._latest_jpeg = jpeg.tobytes()
                    self._latest_detections = frame.detections
            time.sleep(1 / max(self._pipeline.config.camera.fps, 1))

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def latest_detections(self):
        with self._lock:
            return list(self._latest_detections)

    def stop(self) -> None:
        self._stop.set()
        self._pipeline.close()


def create_app(config: CompanionConfig | None = None) -> FastAPI:
    config = config or load_config()
    app = FastAPI(title="AI Companion")
    worker = PipelineWorker(config)
    worker.start()
    app.state.worker = worker
    llm = create_companion_llm(config.llm)
    voice = create_voice_io(config.voice)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return TEMPLATES.TemplateResponse(request, "index.html", {})

    @app.get("/stream")
    def stream():
        def generate():
            while True:
                jpeg = worker.latest_jpeg()
                if jpeg is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    )
                time.sleep(1 / max(config.camera.fps, 1))

        return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/detections")
    def detections():
        return [
            {
                "label": d.label,
                "is_owner": d.is_owner,
                "track_id": d.track_id,
                "confidence": d.confidence,
                "distance_m": d.distance_m,
            }
            for d in worker.latest_detections()
        ]

    @app.post("/chat")
    def chat(req: ChatRequest):
        detections = worker.latest_detections()
        parts = []
        for d in detections:
            if d.label != "person":
                parts.append(d.label)
            elif d.is_owner is True:
                parts.append("person identified as Roman, your owner")
            elif d.is_owner is False:
                parts.append("person present, but face does not match Roman (unrecognized)")
            else:
                parts.append("person present, face not clearly visible to confirm identity")
        scene_context = ", ".join(parts) or "nothing in view"
        reply = llm.chat(req.message, scene_context=scene_context)
        return {"reply": reply}

    @app.post("/speak")
    def speak(req: SpeakRequest):
        try:
            voice.speak(req.text)
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @app.post("/listen")
    def listen():
        try:
            text = voice.listen()
            return {"ok": True, "text": text}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @app.on_event("shutdown")
    def shutdown():
        worker.stop()

    return app
