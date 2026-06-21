# AI Companion

Raspberry Pi 5 + Hailo AI HAT+ (26 TOPS) + Camera Module 3 companion: live
detection dashboard now, local LLM (Ollama) and voice I/O next.

## Structure

- `companion/vision/` — camera, detector, face recognizer, tracker, overlay. Each has a
  `mock` backend (no hardware needed) and a real backend stub for the Pi/Hailo.
- `companion/core/pipeline.py` — wires camera → detector → tracker → face recognizer → addons → overlay.
- `companion/addons/` — plugin interface for future hardware (screen, speaker, sensors).
- `companion/web/` — FastAPI dashboard, streams the annotated feed at `/stream`.
- `companion/llm/`, `companion/voice/` — placeholders for phase 2 (Ollama chat) and phase 3 (voice I/O).
- `config.yaml` — switch any backend between `mock` and its real hardware implementation.

## Run (dev, no hardware)

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Open http://localhost:8000 — you'll see a synthetic moving box standing in for camera + detection.

## Move to real hardware (on the Pi)

1. Install `picamera2` and `hailo-platform` (Pi OS + Hailo SDK provide these).
2. In `config.yaml`, set `camera.backend: picamera2`, `detector.backend: hailo_yolov8`
   (and point `detector.model_path` at a compiled `.hef`), `face_recognizer.backend: opencv_lbph`.
3. Enroll your face: drop a few photos into `data/faces/owner/`, or call
   `FaceRecognizer.enroll()` from a short script.
4. Fill in the `TODO` in `companion/vision/detector.py`'s `HailoYoloV8Detector` to run
   real Hailo inference and parse YOLOv8 output into `Detection` boxes.

## Roadmap

- [x] Scaffold: pipeline, addons, web dashboard, mock backends
- [ ] Real Hailo YOLOv8 inference + Picamera2 capture on-device
- [ ] Owner face enrollment flow
- [ ] Distance estimation addon (carried over from the previous project)
- [ ] Ollama-backed conversation (`companion/llm`)
- [ ] Voice I/O (`companion/voice`)
- [ ] First physical addon (screen or extra sensor) via `companion/addons`
