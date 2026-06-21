"""FastAPI dashboard: MJPEG stream of the annotated camera feed ('see through its eyes')."""
from __future__ import annotations

import threading
import time

import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from companion.addons.registry import AddonRegistry
from companion.config import CompanionConfig, load_config
from companion.core.pipeline import Pipeline

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

    @app.on_event("shutdown")
    def shutdown():
        worker.stop()

    return app
