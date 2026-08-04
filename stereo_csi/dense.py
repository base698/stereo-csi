"""Dense disparity computation and heat-map colorization.

Shared by examples/disparity_map.py (one-shot) and examples/depth_web.py
(live panel). Computes disparity for every pixel, optionally at reduced
resolution so it can run at interactive rates on the Jetson CPU.
"""
from __future__ import annotations

import cv2
import numpy as np


def compute_dense_disparity(depth_service, left, right, downscale=1):
    """Rectify a stereo pair and run SGBM over the full frame.

    Returns (rect_left, disparity) at 1/downscale resolution. The disparity
    values are in downscaled-pixel units — fine for visualization; use the
    point-wise service for metric measurements.
    """
    rect_left, rect_right = depth_service.rectify_stereo_frames(left, right)
    if downscale > 1:
        size = (rect_left.shape[1] // downscale, rect_left.shape[0] // downscale)
        rect_left = cv2.resize(rect_left, size, interpolation=cv2.INTER_AREA)
        rect_right = cv2.resize(rect_right, size, interpolation=cv2.INTER_AREA)

    gray_l = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

    # search range mirrors the calibration's shifted rectification, scaled
    num_disp = max(16, (256 // downscale) // 16 * 16)
    matcher = cv2.StereoSGBM_create(
        minDisparity=-(192 // downscale),
        numDisparities=num_disp,
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


def colorize_disparity(disp):
    """Normalize valid disparities and apply a heat colormap (near=warm)."""
    valid = disp > disp.min() + 1
    vis = np.zeros_like(disp)
    if valid.any():
        lo, hi = np.percentile(np.abs(disp[valid]), [5, 95])
        vis = np.clip((np.abs(disp) - lo) / max(1e-6, hi - lo), 0, 1)
    heat = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat[~valid] = (30, 30, 30)  # no match: dark gray, not a lie
    return heat
