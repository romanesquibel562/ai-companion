#!/usr/bin/env bash
# Always kills any previous instance before starting, so the camera lock
# from a leftover process never blocks the new one.
set -e
cd "$(dirname "$0")/.."

pkill -f "python3 main.py" 2>/dev/null || true
sleep 1

source .venv/bin/activate
exec python3 main.py
