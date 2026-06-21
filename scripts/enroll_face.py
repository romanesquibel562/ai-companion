"""Enroll the owner's face for the opencv_lbph face recognizer.

Captures frames from the real camera, detects a face with the same Haar
cascade the recognizer uses at runtime, and saves cropped grayscale faces to
data/faces/<owner_name>/. Run this BEFORE switching
face_recognizer.backend to opencv_lbph in config.yaml — otherwise there's
nothing to train on.

Usage:
    source .venv/bin/activate
    python3 scripts/enroll_face.py
    python3 scripts/enroll_face.py --count 30 --name owner
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
from picamera2 import Picamera2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from companion.config import FACES_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="owner", help="owner_name from config.yaml")
    parser.add_argument("--count", type=int, default=20, help="number of face photos to capture")
    args = parser.parse_args()

    owner_dir = FACES_DIR / args.name
    owner_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(owner_dir.glob("*.png")))

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    print(f"Capturing {args.count} face photos for '{args.name}'. Look at the camera,")
    print("move your head slightly between captures for variety. Ctrl+C to stop early.")

    saved = 0
    try:
        while saved < args.count:
            rgb = picam2.capture_array()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) == 0:
                print("  no face detected, hold still...")
                time.sleep(0.3)
                continue

            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            face_crop = gray[fy : fy + fh, fx : fx + fw]
            out_path = owner_dir / f"{existing + saved + 1:03d}.png"
            cv2.imwrite(str(out_path), face_crop)
            saved += 1
            print(f"  saved {out_path.name} ({saved}/{args.count})")
            time.sleep(0.4)
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        picam2.stop()

    print(f"\nDone. {saved} face photos saved to {owner_dir}")
    print("Next: set face_recognizer.backend to 'opencv_lbph' in config.yaml, then")
    print("restart the dashboard (scripts/run.sh) to start recognizing the owner.")


if __name__ == "__main__":
    main()
