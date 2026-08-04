"""Camera calibration, rectification, and stereo depth processing."""

from __future__ import annotations

import os
import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class StereoPointMeasurement:
    """Depth and quality for one point from a shared stereo disparity pass."""

    depth_mm: float
    disparity_px: float
    valid_ratio: float
    disparity_iqr_px: float
    texture_std: float
    confidence: float
    point_camera_mm: np.ndarray
    covariance_camera: np.ndarray


class StereoDepthService:
    """Own camera calibration state and calculate depth from stereo frames."""

    def __init__(
        self,
        calibration_file=None,
        baseline_override=None,
        min_texture_std=4.0,
        min_valid_mm=0.0,
        max_valid_mm=6000.0,
        image_size=(640, 480),
    ):
        self.calibration_file = calibration_file
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_enabled = False
        self.stereo_calibration_enabled = False
        self.K1 = None
        self.D1 = None
        self.K2 = None
        self.D2 = None
        self.R = None
        self.T = None
        self.baseline = None
        self.baseline_override = baseline_override
        self.stereo_matcher = None
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.Q = None
        self.stereo_map_left = None
        self.stereo_map_right = None
        self.stereo_focal_length = None
        self.stereo_min_disparity = 0
        self.stereo_num_disparities = 192
        self.stereo_disparity_sign = 1
        self.depth_min_texture_std = float(min_texture_std)
        self.depth_min_valid_mm = float(min_valid_mm)
        self.depth_max_valid_mm = float(max_valid_mm)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.last_depth_debug = "not computed"

        if self.calibration_file:
            self.load_calibration()

    def load_calibration(self):
        """Load camera calibration from file (single or stereo)"""
        try:
            # Check if file exists
            if not os.path.exists(self.calibration_file):
                print(f"⚠ Calibration file not found: {self.calibration_file}")
                self.calibration_enabled = False
                return

            calib_data = np.load(self.calibration_file)

            # Check if it's stereo calibration
            if 'K1' in calib_data and 'K2' in calib_data:
                # Stereo calibration
                self.K1 = calib_data['K1']
                self.D1 = calib_data['D1']
                self.K2 = calib_data['K2']
                self.D2 = calib_data['D2']
                self.R = calib_data['R']
                self.T = calib_data['T']
                self.baseline = calib_data['baseline']
                self.stereo_calibration_enabled = True
                self.calibration_enabled = True

                # Use left camera parameters for single-camera fallback
                self.camera_matrix = self.K1
                self.dist_coeffs = self.D1
                self.init_stereo_rectification()

                # Create stereo matcher for disparity calculation.
                # StereoSGBM is more accurate than StereoBM.
                self.stereo_matcher = cv2.StereoSGBM_create(
                    minDisparity=self.stereo_min_disparity,
                    numDisparities=self.stereo_num_disparities,  # Must be divisible by 16
                    blockSize=5,
                    P1=8 * 3 * 5**2,  # Smoothness penalty
                    P2=32 * 3 * 5**2,
                    disp12MaxDiff=1,
                    uniquenessRatio=10,
                    speckleWindowSize=100,
                    speckleRange=32,
                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                )

                # Get RMS error from calibration
                rms_error = calib_data.get('rms_error', 0)

                # Apply baseline override if provided
                baseline_from_calib = self.baseline
                if self.baseline_override is not None:
                    self.baseline = self.baseline_override
                    print(f"✓ Stereo calibration loaded: {self.calibration_file}")
                    print(f"  RMS error: {rms_error:.3f} pixels")
                    print(f"  Baseline (calibrated): {baseline_from_calib:.2f} mm")
                    print(f"  Baseline (OVERRIDDEN): {self.baseline:.2f} mm ⚠")
                    print(f"  Left focal length: fx={self.K1[0,0]:.2f}, fy={self.K1[1,1]:.2f}")
                    print(f"  Right focal length: fx={self.K2[0,0]:.2f}, fy={self.K2[1,1]:.2f}")
                else:
                    print(f"✓ Stereo calibration loaded: {self.calibration_file}")
                    print(f"  RMS error: {rms_error:.3f} pixels")
                    print(f"  Baseline: {self.baseline:.2f} mm")
                    print(f"  Left focal length: fx={self.K1[0,0]:.2f}, fy={self.K1[1,1]:.2f}")
                    print(f"  Right focal length: fx={self.K2[0,0]:.2f}, fy={self.K2[1,1]:.2f}")
                if rms_error > 1.0:
                    print(f"  ⚠ Stereo RMS is high ({rms_error:.3f}px); distance may be inaccurate until recalibrated")
            else:
                # Single camera calibration
                self.camera_matrix = calib_data['camera_matrix']
                self.dist_coeffs = calib_data['dist_coeffs']
                self.calibration_enabled = True
                rms_error = calib_data.get('rms_error', 0)
                print(f"✓ Single camera calibration loaded: {self.calibration_file}")
                print(f"  RMS error: {rms_error:.3f} pixels")
                print(f"  Focal length: fx={self.camera_matrix[0,0]:.2f}, fy={self.camera_matrix[1,1]:.2f}")
        except Exception as e:
            print(f"⚠ Failed to load calibration: {e}")
            self.calibration_enabled = False
            self.stereo_calibration_enabled = False

    def init_stereo_rectification(self, image_size=None):
        """Build rectification maps so stereo disparity is computed on aligned frames."""
        try:
            image_size = self.image_size if image_size is None else image_size
            self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
                self.K1, self.D1,
                self.K2, self.D2,
                image_size,
                self.R, self.T,
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=-1
            )

            center_point = np.array([[[image_size[0] / 2.0, image_size[1] / 2.0]]], dtype=np.float32)
            rectified_center = cv2.undistortPoints(
                center_point, self.K1, self.D1, R=self.R1, P=self.P1
            )[0, 0]
            rectified_shift_x = (image_size[0] / 2.0) - float(rectified_center[0])
            rectified_shift_y = (image_size[1] / 2.0) - float(rectified_center[1])
            if abs(rectified_shift_x) > 1.0 or abs(rectified_shift_y) > 1.0:
                self.P1[0, 2] += rectified_shift_x
                self.P2[0, 2] += rectified_shift_x
                self.P1[1, 2] += rectified_shift_y
                self.P2[1, 2] += rectified_shift_y

            self.stereo_map_left = cv2.initUndistortRectifyMap(
                self.K1, self.D1, self.R1, self.P1, image_size, cv2.CV_16SC2
            )
            self.stereo_map_right = cv2.initUndistortRectifyMap(
                self.K2, self.D2, self.R2, self.P2, image_size, cv2.CV_16SC2
            )

            self.stereo_focal_length = float(self.P1[0, 0])
            rectified_baseline = abs(float(self.P2[0, 3]) / float(self.P2[0, 0]))
            if self.baseline_override is None:
                self.baseline = rectified_baseline

            # P2[0,3] sign tells us which disparity direction to expect after rectification.
            # Some camera orderings produce negative valid disparities.
            if float(self.P2[0, 3]) > 0:
                self.stereo_disparity_sign = -1
                self.stereo_min_disparity = -192
                self.stereo_num_disparities = 256
            else:
                self.stereo_disparity_sign = 1
                self.stereo_min_disparity = 0
                self.stereo_num_disparities = 192

            print("✓ Stereo rectification initialized")
            print(f"  Rectified focal length: {self.stereo_focal_length:.2f}px")
            print(f"  Rectified baseline: {rectified_baseline:.2f} mm")
            print(f"  Rectified image shift: x={rectified_shift_x:.1f}px, y={rectified_shift_y:.1f}px")
            print(f"  Disparity search: min={self.stereo_min_disparity}, num={self.stereo_num_disparities}, sign={self.stereo_disparity_sign:+d}")
        except Exception as e:
            print(f"⚠ Failed to initialize stereo rectification: {e}")
            self.stereo_calibration_enabled = False
            self.stereo_map_left = None
            self.stereo_map_right = None

    def rectify_stereo_frames(self, frame_left, frame_right):
        """Rectify left/right frames before stereo matching."""
        if self.stereo_map_left is None or self.stereo_map_right is None:
            return frame_left, frame_right

        try:
            left = cv2.remap(
                frame_left,
                self.stereo_map_left[0],
                self.stereo_map_left[1],
                cv2.INTER_LINEAR
            )
            right = cv2.remap(
                frame_right,
                self.stereo_map_right[0],
                self.stereo_map_right[1],
                cv2.INTER_LINEAR
            )
            return left, right
        except Exception as e:
            print(f"Error rectifying stereo frames: {e}")
            return frame_left, frame_right

    def rectify_stereo_point(self, x, y):
        """Map a point from the raw left image into rectified stereo coordinates."""
        if self.R1 is None or self.P1 is None:
            return int(x), int(y)

        try:
            point = np.array([[[float(x), float(y)]]], dtype=np.float32)
            rectified = cv2.undistortPoints(point, self.K1, self.D1, R=self.R1, P=self.P1)
            rx, ry = rectified[0, 0]
            return int(round(rx)), int(round(ry))
        except Exception as e:
            print(f"Error rectifying stereo point: {e}")
            return int(x), int(y)

    def raw_pixel_depth_to_camera(self, x, y, depth_mm):
        """Back-project a raw left-camera pixel using the raw camera model."""
        intrinsics = self.K1 if self.K1 is not None else self.camera_matrix
        if intrinsics is None:
            intrinsics = self.P1[:, :3] if self.P1 is not None else None
        if intrinsics is None:
            raise ValueError("camera intrinsics are not loaded")

        if self.D1 is not None and self.K1 is not None:
            point = np.array([[[float(x), float(y)]]], dtype=np.float32)
            normalized = cv2.undistortPoints(point, self.K1, self.D1)[0, 0]
            return np.array(
                [
                    float(normalized[0]) * depth_mm,
                    float(normalized[1]) * depth_mm,
                    depth_mm,
                ],
                dtype=float,
            )

        fx = float(intrinsics[0, 0])
        fy = float(intrinsics[1, 1])
        cx = float(intrinsics[0, 2])
        cy = float(intrinsics[1, 2])
        return np.array(
            [
                (float(x) - cx) * depth_mm / fx,
                (float(y) - cy) * depth_mm / fy,
                depth_mm,
            ],
            dtype=float,
        )

    def undistort_frame(self, frame, use_left=True):
        """Apply camera calibration to undistort frame"""
        if not self.calibration_enabled:
            return frame

        try:
            if self.stereo_calibration_enabled:
                # Use appropriate camera matrix for stereo
                K = self.K1 if use_left else self.K2
                D = self.D1 if use_left else self.D2
                return cv2.undistort(frame, K, D)
            else:
                return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        except Exception as e:
            print(f"Error undistorting frame: {e}")
            return frame

    def _sample_depth_measurement(
        self,
        gray_left,
        disparity,
        x,
        y,
        raw_x=None,
        raw_y=None,
    ):
        """Sample one rectified point from an already-computed disparity map."""
        if x < 0 or x >= disparity.shape[1] or y < 0 or y >= disparity.shape[0]:
            self.last_depth_debug = f"rectified point out of frame ({x}, {y})"
            return None

        texture_radius = 16
        tx1 = max(0, x - texture_radius)
        tx2 = min(gray_left.shape[1], x + texture_radius + 1)
        ty1 = max(0, y - texture_radius)
        ty2 = min(gray_left.shape[0], y + texture_radius + 1)
        texture_std = float(np.std(gray_left[ty1:ty2, tx1:tx2]))
        if texture_std < self.depth_min_texture_std:
            self.last_depth_debug = f"low texture {texture_std:.1f}"
            return None

        valid = None
        window_size = 0
        for radius in (4, 8, 16, 32):
            x1 = max(0, x - radius)
            x2 = min(disparity.shape[1], x + radius + 1)
            y1 = max(0, y - radius)
            y2 = min(disparity.shape[0], y + radius + 1)
            window = disparity[y1:y2, x1:x2]
            window_size = int(window.size)
            if self.stereo_disparity_sign < 0:
                candidate = window[
                    (window < -1) & (window >= self.stereo_min_disparity)
                ]
            else:
                candidate = window[window > 0]
            if candidate.size >= 8:
                valid = np.abs(candidate.astype(float))
                break

        if valid is None:
            self.last_depth_debug = f"no valid disparity near ({x}, {y})"
            return None

        disparity_px = float(np.median(valid))
        if disparity_px <= 0:
            self.last_depth_debug = f"invalid disparity {disparity_px:.2f}px"
            return None

        focal_length = float(self.stereo_focal_length or self.K1[0, 0])
        depth_mm = (focal_length * float(self.baseline)) / disparity_px
        if self.depth_min_valid_mm > 0 and depth_mm < self.depth_min_valid_mm:
            self.last_depth_debug = f"depth too near {depth_mm / 1000.0:.2f}m"
            return None
        if self.depth_max_valid_mm > 0 and depth_mm > self.depth_max_valid_mm:
            self.last_depth_debug = f"depth too far {depth_mm / 1000.0:.2f}m"
            return None

        q25, q75 = np.percentile(valid, [25, 75])
        disparity_iqr = float(q75 - q25)
        valid_ratio = float(valid.size / max(1, window_size))
        texture_quality = 1.0
        if self.depth_min_texture_std > 0:
            texture_quality = min(
                1.0,
                texture_std / (2.0 * self.depth_min_texture_std),
            )
        spread_quality = math.exp(-disparity_iqr / max(disparity_px, 1.0))
        confidence = max(
            0.0,
            min(1.0, valid_ratio * texture_quality * spread_quality),
        )

        point_camera = self.raw_pixel_depth_to_camera(
            x if raw_x is None else raw_x,
            y if raw_y is None else raw_y,
            depth_mm,
        )
        intrinsics = self.K1 if self.K1 is not None else self.camera_matrix
        if intrinsics is None:
            intrinsics = self.P1[:, :3] if self.P1 is not None else np.eye(3)
        fx = float(intrinsics[0, 0])
        fy = float(intrinsics[1, 1])
        disparity_sigma = max(0.5, disparity_iqr / 1.349)
        depth_sigma = max(
            5.0,
            focal_length * float(self.baseline) * disparity_sigma
            / (disparity_px ** 2),
        )
        lateral_sigma_x = max(3.0, depth_mm * 1.5 / fx)
        lateral_sigma_y = max(3.0, depth_mm * 1.5 / fy)
        covariance_camera = np.diag(
            [lateral_sigma_x ** 2, lateral_sigma_y ** 2, depth_sigma ** 2]
        )
        self.last_depth_debug = f"disp {disparity_px:.2f}px"
        return StereoPointMeasurement(
            depth_mm=depth_mm,
            disparity_px=disparity_px,
            valid_ratio=valid_ratio,
            disparity_iqr_px=disparity_iqr,
            texture_std=texture_std,
            confidence=confidence,
            point_camera_mm=point_camera,
            covariance_camera=covariance_camera,
        )

    def calculate_depths(self, frame_left, frame_right, points):
        """Calculate many point depths with one rectification/disparity pass."""
        points = list(points)
        if not self.stereo_calibration_enabled or self.stereo_matcher is None:
            self.last_depth_debug = "stereo disabled"
            return [None for _ in points]
        if not points:
            return []

        try:
            rectified_left, rectified_right = self.rectify_stereo_frames(
                frame_left, frame_right
            )
            gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)
            disparity = (
                self.stereo_matcher.compute(gray_left, gray_right).astype(np.float32)
                / 16.0
            )
            measurements = []
            for raw_x, raw_y in points:
                x, y = self.rectify_stereo_point(raw_x, raw_y)
                measurements.append(
                    self._sample_depth_measurement(
                        gray_left,
                        disparity,
                        x,
                        y,
                        raw_x=raw_x,
                        raw_y=raw_y,
                    )
                )
            return measurements
        except Exception as exc:
            self.last_depth_debug = f"error: {exc}"
            print(f"Error calculating depth: {exc}")
            return [None for _ in points]

    def calculate_depth(self, frame_left, frame_right, x, y):
        """Compatibility wrapper returning only depth for one image point."""
        measurements = self.calculate_depths(frame_left, frame_right, [(x, y)])
        measurement = measurements[0] if measurements else None
        return measurement.depth_mm if measurement is not None else None
