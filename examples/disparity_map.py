#!/usr/bin/env python3
"""Dense disparity heat map - the classic depth-camera demo image.

Computes disparity for every pixel of a stereo pair, colorizes it
(near = warm, far = cool), and writes a side-by-side of the left frame and
the heat map.

  python3 examples/disparity_map.py \
      --calibration calibration/output/stereo_calibration.npz \
      --left calibration/images/left/img_042.jpg \
      --right calibration/images/right/img_042.jpg \
      --out disparity.jpg

Omit --left/--right to capture a live pair from the cameras instead.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from stereo_csi import StereoDepthService
from stereo_csi.dense import colorize_disparity, compute_dense_disparity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--left")
    ap.add_argument("--right")
    ap.add_argument("--out", default="disparity.jpg")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    if args.left and args.right:
        left = cv2.imread(args.left)
        right = cv2.imread(args.right)
    else:
        from stereo_csi import CameraSource, CSICameraCapture
        cam = CameraSource(enabled=True, camera_id=0, use_csi=True, stereo_mode=True,
                           invert_camera=False, video_fps=30,
                           output_width=args.width, output_height=args.height,
                           csi_capture_factory=CSICameraCapture)
        cam.initialize()
        left = right = None
        deadline = time.time() + 15
        while time.time() < deadline:
            left, right = cam.read_frames()
            if left is not None and right is not None:
                break
            time.sleep(0.3)
        cam.close()
        if left is None or right is None:
            raise SystemExit("no frames from cameras (are they in use by another process?)")

    depth = StereoDepthService(calibration_file=args.calibration,
                               image_size=(left.shape[1], left.shape[0]))
    depth.load_calibration()

    rect_left, disp = compute_dense_disparity(depth, left, right)
    heat = colorize_disparity(disp)
    side = np.hstack([rect_left, heat])
    cv2.imwrite(args.out, side)
    print(f"wrote {args.out}  ({side.shape[1]}x{side.shape[0]})")


if __name__ == "__main__":
    main()
