# Commands

Copy/paste reference for running and debugging this project. All commands
assume you're already `cd ~/ai_companion`.

## Run the dashboard

```bash
~/ai_companion/scripts/run.sh
```

This always kills any leftover instance from a previous run first, so you
won't hit "Device or resource busy" from a stale process still holding the
camera.

Then open `http://192.168.5.121:8000` in a browser (use whichever IP/hostname
you're currently connected through).

Stop it with `Ctrl+C` in the terminal it's running in.

## Run main.py directly (without the cleanup wrapper)

```bash
cd ~/ai_companion
source .venv/bin/activate
python3 main.py
```

This is the actual entrypoint — `scripts/run.sh` is just a convenience
wrapper around this same command. Use this form if you've already killed
any stale process yourself and just want to start the app.

## Run it in the background (survives terminal close)

```bash
cd ~/ai_companion
source .venv/bin/activate
nohup python3 main.py > /tmp/companion_server.log 2>&1 &
disown
```

Check its log:

```bash
tail -f /tmp/companion_server.log
```

Stop it:

```bash
pkill -f "python3 main.py"
```

## Camera-only test (no web server, no detector)

```bash
cd ~/ai_companion
source .venv/bin/activate
python3 scripts/test_camera.py
```

Saves frames to `scripts/camera_test_output/frame_0.jpg` ... `frame_4.jpg`.

## Enroll your face (owner recognition)

```bash
cd ~/ai_companion
source .venv/bin/activate
python3 scripts/enroll_face.py --count 20
```

Sit in front of the camera and look at it — it saves 20 cropped face photos
to `data/faces/owner/`. Move your head slightly between captures for variety
(angle, lighting). Run with `pkill -f "python3 main.py"` first if the
dashboard is currently using the camera.

Then switch `face_recognizer.backend` to `opencv_lbph` in `config.yaml` and
restart with `scripts/run.sh`. Detections should now show "owner" in a green
box for you, and "unknown" in orange for anyone else.

To add more photos later (e.g. better lighting), just re-run the script —
it appends, it doesn't overwrite.

## Chat brain (terminal panel on the dashboard)

**Active.** The web dashboard has a terminal-style chat panel next to the
camera feed — type a question, get a real answer (including code) from a
local Ollama model (`qwen2.5-coder:3b`, set in `config.yaml` under `llm:`).
This is the interactive interface until mic/speaker/screen hardware exists.

Swap models any time:

```bash
ollama pull <model-name>
```

then update `llm.model` in `config.yaml` and restart with `scripts/run.sh`.

Check Ollama is running and see installed models:

```bash
systemctl is-active ollama
ollama list
```

## Voice output (HAL-9000-style TTS)

**Active in software, silent until a speaker is connected.** Every chat
reply is also sent to `/speak`, which runs it through Piper TTS
(`data/voices/en_US-norman-medium.onnx`) and pitches/slows it down for a
flat, deliberate delivery (`voice.length_scale` / `voice.pitch_scale` in
`config.yaml`). Right now `aplay` has no real output device (Pi only has
HDMI audio, no speaker yet), so `/speak` calls succeed at generating audio
but fail at playback — that's expected, not a bug. Once a USB
speaker/headset is plugged in, it should just start working.

Test it directly:

```bash
curl -s -X POST http://localhost:8000/speak -H "Content-Type: application/json" \
  -d '{"text":"Good evening. All systems are functioning normally."}'
```

Try a different voice or tuning:

```bash
cd ~/ai_companion
source .venv/bin/activate
python3 -m piper.download_voices <voice-name> --download-dir data/voices
```

Browse available voices at the Piper voices list, then update
`voice.model_path` in `config.yaml` to point at the new `.onnx` file.

## Fix "Device or resource busy" (camera won't start)

Find what's holding the camera:

```bash
sudo fuser -v /dev/media0
```

If it's a leftover Python process, kill it (replace `<PID>` with the number
from the output above):

```bash
kill <PID>
```

If it's `pipewire`/`wireplumber` (desktop audio/video manager grabbing the
camera):

```bash
systemctl --user stop wireplumber pipewire pipewire.socket
```

## Check live detections (what the model currently sees)

```bash
curl -s http://localhost:8000/detections
```

Returns the current frame's detections as JSON. If you see odd labels
("motorcycle", "airplane", etc.) alongside "person", that's model noise from
background clutter at the default confidence threshold, not a bug. Raise
`detector.confidence_threshold` in `config.yaml` (e.g. `0.4` → `0.6`) to cut
down false positives; restart with `scripts/run.sh` after changing it.

## Hailo hardware checks

```bash
hailortcli scan                      # is the device detected?
hailortcli fw-control identify       # chip variant / firmware version
hailortcli parse-hef /usr/share/hailo-models/yolov8s_h8l.hef   # model input/output shapes
```

## Git

```bash
git status
git add .
git commit -m "your message here"
```

## Re-create the venv from scratch (if it gets corrupted)

```bash
cd ~/ai_companion
rm -rf .venv
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`--system-site-packages` is required — `picamera2` and `hailo_platform` are
installed system-wide via `apt`, not pip.
