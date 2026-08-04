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


def dense_disparity(depth, left, right):
    """Rectify a pair and run SGBM over the full frame."""
    rect_left, rect_right = depth.rectify_stereo_frames(left, right)
    gray_l = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

    matcher = cv2.StereoSGBM_create(
        minDisparity=-192,
        numDisparities=256,
        blockSize=5,
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        disp12MaxDiff=1,
    )
    disp = matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0
    return rect_left, disp


def colorize(disp):
    """Normalize valid disparities and apply a heat colormap."""
    valid = disp > disp.min() + 1
    vis = np.zeros_like(disp)
    if valid.any():
        lo, hi = np.percentile(disp[valid], [5, 95])
        vis = np.clip((np.abs(disp) - abs(lo)) / max(1e-6, abs(hi) - abs(lo)), 0, 1)
    vis8 = (vis * 255).astype(np.uint8)
    heat = cv2.applyColorMap(vis8, cv2.COLORMAP_TURBO)
    heat[~valid] = (30, 30, 30)  # matcher found nothing: dark gray, not a lie
    return heat


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

    rect_left, disp = dense_disparity(depth, left, right)
    heat = colorize(disp)
    side = np.hstack([rect_left, heat])
    cv2.imwrite(args.out, side)
    print(f"wrote {args.out}  ({side.shape[1]}x{side.shape[0]})")


if __name__ == "__main__":
    main()
