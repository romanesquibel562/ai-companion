"""Standalone real-camera test, independent of the mock pipeline/config.

Captures a few frames directly from the Camera Module 3 via picamera2,
saves them to scripts/camera_test_output/, and prints basic diagnostics.
Run this first when debugging "black screen" issues, to isolate whether
the problem is the camera itself or something further down the pipeline.

Usage:
    source .venv/bin/activate
    python3 scripts/test_camera.py
"""
import time
from pathlib import Path

import cv2
from picamera2 import Picamera2

OUT_DIR = Path(__file__).parent / "camera_test_output"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(1)  # let auto-exposure/white-balance settle

    print("Capturing 5 frames...")
    for i in range(5):
        rgb = picam2.capture_array()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        mean_brightness = bgr.mean()
        out_path = OUT_DIR / f"frame_{i}.jpg"
        cv2.imwrite(str(out_path), bgr)
        print(f"  frame_{i}.jpg  shape={bgr.shape}  mean_brightness={mean_brightness:.1f}")
        time.sleep(0.5)

    picam2.stop()
    print(f"\nSaved frames to {OUT_DIR}")
    print("If mean_brightness is near 0 for every frame, the lens cap may still be on")
    print("or the sensor isn't getting light. If frames look fine here but the")
    print("dashboard is still black, the issue is in companion/vision/camera.py or")
    print("config.yaml (camera.backend must be 'picamera2', not 'mock').")


if __name__ == "__main__":
    main()
