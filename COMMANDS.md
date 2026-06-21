# Commands

Copy/paste reference for running and debugging this project. All commands
assume you're already `cd ~/ai_companion`.

## Run the dashboard

```bash
cd ~/ai_companion
source .venv/bin/activate
python3 main.py
```

Then open `http://192.168.5.121:8000` in a browser (use whichever IP/hostname
you're currently connected through).

Stop it with `Ctrl+C` in the terminal it's running in.

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
