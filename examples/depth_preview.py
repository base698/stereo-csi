#!/usr/bin/env python3
"""Live stereo depth preview: prints center-of-frame depth once per second.

Run on the Jetson after calibrating:

  python3 examples/depth_preview.py \
      --calibration calibration/output/stereo_calibration.npz \
      --baseline-override 52.5

--baseline-override is the PHYSICAL lens-center spacing you measured with
calipers; the solved calibration baseline is often off by several mm and the
physical measurement gives truer absolute depth.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stereo_csi import CameraSource, CSICameraCapture, StereoDepthService


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--baseline-override", type=float, default=None)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--camera-id", type=int, default=0)
    args = ap.parse_args()

    camera = CameraSource(
        enabled=True,
        camera_id=args.camera_id,
        use_csi=True,
        stereo_mode=True,
        invert_camera=False,
        output_width=args.width,
        output_height=args.height,
        video_fps=30,
        csi_capture_factory=CSICameraCapture,
    )
    camera.initialize()

    depth = StereoDepthService(
        calibration_file=args.calibration,
        baseline_override=args.baseline_override,
        image_size=(args.width, args.height),
    )
    depth.load_calibration()

    cx, cy = args.width // 2, args.height // 2
    print(f"Sampling depth at frame center ({cx}, {cy}) - ctrl-c to stop")
    try:
        while True:
            left, right = camera.read_frames()
            if left is None or right is None:
                time.sleep(0.1)
                continue
            measurements = depth.calculate_depths(left, right, [(cx, cy)])
            m = measurements[0] if measurements else None
            if m is not None and m.depth_mm:
                print(f"depth: {m.depth_mm:6.0f} mm   "
                      f"disparity: {m.disparity_px:5.1f} px   "
                      f"confidence: {m.confidence:.2f}")
            else:
                print("depth: no valid measurement (low texture?)")
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()


if __name__ == "__main__":
    main()
