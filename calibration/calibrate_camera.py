#!/usr/bin/env python3
"""
Camera Calibration Script with Visualization
Run on host machine to calibrate camera(s) from captured images
Shows checkerboard detection lines and reprojection errors
"""

import cv2
import numpy as np
import glob
import argparse
import os
from pathlib import Path

def calibrate_single_camera(images_path, pattern_size=(9, 6), square_size=25.0,
                            visualize=True, output_dir='calibration_output'):
    """
    Calibrate a single camera with visualization

    Args:
        images_path: Path pattern for images (e.g., 'calib/*.jpg')
        pattern_size: Checkerboard internal corners (width, height)
        square_size: Size of checkerboard squares in mm
        visualize: Show detection results
        output_dir: Where to save calibration results

    Returns:
        Camera matrix, distortion coefficients, reprojection error
    """
    print("\n" + "="*70)
    print("SINGLE CAMERA CALIBRATION")
    print("="*70)

    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points (3D points in real world space)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Scale by square size

    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world
    imgpoints = []  # 2D points in image plane

    # Get list of images
    images = sorted(glob.glob(images_path))

    if not images:
        print(f"ERROR: No images found at {images_path}")
        return None, None, None

    print(f"Found {len(images)} images")
    print(f"Pattern: {pattern_size[0]}x{pattern_size[1]} corners")
    print(f"Square size: {square_size}mm\n")

    os.makedirs(output_dir, exist_ok=True)
    vis_dir = f"{output_dir}/visualizations"
    os.makedirs(vis_dir, exist_ok=True)

    successful = 0
    img_size = None

    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"  [{i+1}/{len(images)}] ✗ Could not read {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]

        # Find checkerboard corners
        print(f"  [{i+1}/{len(images)}] Processing {Path(fname).name}...", end=' ')
        ret, corners = cv2.findChessboardCorners(gray, pattern_size,
                                                  cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                  cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                  cv2.CALIB_CB_FAST_CHECK)

        if ret:
            # Refine corner positions to sub-pixel accuracy
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(corners_refined)

            print(f"✓ FOUND")

            # Visualize
            if visualize:
                vis_img = img.copy()
                cv2.drawChessboardCorners(vis_img, pattern_size, corners_refined, ret)

                # Add info text
                cv2.putText(vis_img, f"Image {i+1}/{len(images)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(vis_img, "Pattern DETECTED", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Save visualization
                vis_path = f"{vis_dir}/detected_{i:03d}.jpg"
                cv2.imwrite(vis_path, vis_img)

            successful += 1
        else:
            print(f"✗ PATTERN NOT FOUND")

    print(f"\nSuccessfully processed: {successful}/{len(images)} images")

    if successful < 10:
        print("\nWARNING: Need at least 10 images for good calibration!")
        print("Capture more images with varied angles and distances.")
        return None, None, None

    # Calibrate camera
    print("\nRunning calibration...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i],
                                          camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    mean_error /= len(objpoints)

    print("\n" + "="*70)
    print("CALIBRATION RESULTS")
    print("="*70)
    print(f"RMS reprojection error: {mean_error:.3f} pixels")
    if mean_error < 0.5:
        print("  → Excellent calibration!")
    elif mean_error < 1.0:
        print("  → Good calibration")
    else:
        print("  → Warning: Consider recalibrating for better accuracy")

    print(f"\nCamera Matrix:\n{camera_matrix}")
    print(f"\nDistortion Coefficients:\n{dist_coeffs}")

    print(f"\nFocal length: fx={camera_matrix[0,0]:.2f}, fy={camera_matrix[1,1]:.2f}")
    print(f"Optical center: cx={camera_matrix[0,2]:.2f}, cy={camera_matrix[1,2]:.2f}")

    # Save calibration
    calib_file = f"{output_dir}/camera_calibration.npz"
    np.savez(calib_file,
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             rvecs=rvecs,
             tvecs=tvecs,
             rms_error=mean_error,
             pattern_size=pattern_size,
             square_size=square_size)

    print(f"\n✓ Calibration saved to: {calib_file}")
    print(f"✓ Visualizations saved to: {vis_dir}/")

    # Create undistortion example
    if visualize and len(images) > 0:
        create_undistortion_comparison(images[0], camera_matrix, dist_coeffs, output_dir)

    return camera_matrix, dist_coeffs, mean_error


def create_undistortion_comparison(image_path, camera_matrix, dist_coeffs, output_dir):
    """Create before/after undistortion comparison"""
    img = cv2.imread(image_path)
    if img is None:
        return

    h, w = img.shape[:2]

    # Get optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    # Undistort
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Create side-by-side comparison
    comparison = np.hstack([img, undistorted])

    # Add labels
    cv2.putText(comparison, "ORIGINAL (Distorted)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(comparison, "UNDISTORTED", (w + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw grid lines to show distortion correction
    for y in range(0, h, h//10):
        cv2.line(comparison, (0, y), (w, y), (255, 0, 0), 1)
    for x in range(0, w, w//10):
        cv2.line(comparison, (x, 0), (x, h), (255, 0, 0), 1)

    for y in range(0, h, h//10):
        cv2.line(comparison, (w, y), (2*w, y), (0, 255, 0), 1)
    for x in range(w, 2*w, w//10):
        cv2.line(comparison, (x, 0), (x, h), (0, 255, 0), 1)

    output_path = f"{output_dir}/undistortion_comparison.jpg"
    cv2.imwrite(output_path, comparison)
    print(f"✓ Undistortion example saved to: {output_path}")


def calibrate_stereo(left_images, right_images, pattern_size=(9, 6),
                    square_size=25.0, output_dir='calibration_output'):
    """
    Calibrate stereo camera pair

    Args:
        left_images: Path pattern for left camera images
        right_images: Path pattern for right camera images
        pattern_size: Checkerboard pattern size
        square_size: Square size in mm
        output_dir: Output directory

    Returns:
        Stereo calibration parameters
    """
    print("\n" + "="*70)
    print("STEREO CAMERA CALIBRATION")
    print("="*70)

    # First calibrate individual cameras
    print("\n1. Calibrating LEFT camera...")
    K1, D1, error1 = calibrate_single_camera(left_images, pattern_size, square_size,
                                             visualize=True,
                                             output_dir=f"{output_dir}/left")

    print("\n2. Calibrating RIGHT camera...")
    K2, D2, error2 = calibrate_single_camera(right_images, pattern_size, square_size,
                                             visualize=True,
                                             output_dir=f"{output_dir}/right")

    if K1 is None or K2 is None:
        print("\nERROR: Individual camera calibrations failed")
        return None

    print("\n3. Computing stereo calibration...")

    # Load matching image pairs
    left_imgs = sorted(glob.glob(left_images))
    right_imgs = sorted(glob.glob(right_images))

    # Prepare object points
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints_L = []
    imgpoints_R = []

    img_size = None

    for left_path, right_path in zip(left_imgs, right_imgs):
        img_L = cv2.imread(left_path)
        img_R = cv2.imread(right_path)

        if img_L is None or img_R is None:
            continue

        gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
        gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)
        img_size = gray_L.shape[::-1]

        # Find corners in both images
        ret_L, corners_L = cv2.findChessboardCorners(gray_L, pattern_size)
        ret_R, corners_R = cv2.findChessboardCorners(gray_R, pattern_size)

        if ret_L and ret_R:
            corners_L = cv2.cornerSubPix(gray_L, corners_L, (11, 11), (-1, -1), criteria)
            corners_R = cv2.cornerSubPix(gray_R, corners_R, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints_L.append(corners_L)
            imgpoints_R.append(corners_R)

    print(f"Found {len(objpoints)} matching image pairs")

    # Stereo calibration
    ret, M1, D1, M2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_L, imgpoints_R,
        K1, D1, K2, D2, img_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    baseline = np.linalg.norm(T)

    print("\n" + "="*70)
    print("STEREO CALIBRATION RESULTS")
    print("="*70)
    print(f"Stereo RMS error: {ret:.3f} pixels")
    print(f"Baseline (distance between cameras): {baseline:.2f}mm")
    print(f"Rotation between cameras:\n{R}")
    print(f"Translation between cameras:\n{T}")

    # Save stereo calibration
    stereo_file = f"{output_dir}/stereo_calibration.npz"
    np.savez(stereo_file,
             K1=M1, D1=D1, K2=M2, D2=D2,
             R=R, T=T, E=E, F=F,
             baseline=baseline,
             rms_error=ret,
             pattern_size=pattern_size,
             square_size=square_size)

    print(f"\n✓ Stereo calibration saved to: {stereo_file}")

    return M1, D1, M2, D2, R, T


def main():
    parser = argparse.ArgumentParser(
        description="Camera calibration with visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single camera calibration
  python tools/vision/calibration/calibrate_camera.py --input "calib_images/*.jpg" --pattern 9x6

  # Stereo camera calibration
  python tools/vision/calibration/calibrate_camera.py --stereo \\
      --left "calib_stereo/left/*.jpg" \\
      --right "calib_stereo/right/*.jpg" \\
      --pattern 9x6

  # With custom square size (in mm)
  python tools/vision/calibration/calibrate_camera.py --input "calib/*.jpg" --pattern 9x6 --square-size 40
        """
    )

    parser.add_argument('--input', '-i', type=str,
                       help='Input images path pattern (e.g., "images/*.jpg")')
    parser.add_argument('--stereo', action='store_true',
                       help='Perform stereo calibration')
    parser.add_argument('--left', type=str,
                       help='Left camera images for stereo (e.g., "left/*.jpg")')
    parser.add_argument('--right', type=str,
                       help='Right camera images for stereo (e.g., "right/*.jpg")')
    parser.add_argument('--pattern', '-p', type=str, default='9x6',
                       help='Checkerboard pattern (default: 9x6 internal corners)')
    parser.add_argument('--square-size', '-s', type=float, default=25.0,
                       help='Checkerboard square size in mm (default: 25.0)')
    parser.add_argument('--output', '-o', type=str, default='calibration_output',
                       help='Output directory (default: calibration_output)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')

    args = parser.parse_args()

    # Parse pattern size
    pattern_parts = args.pattern.lower().split('x')
    if len(pattern_parts) != 2:
        print("ERROR: Pattern must be in format WIDTHxHEIGHT (e.g., 9x6)")
        return

    pattern_size = (int(pattern_parts[0]), int(pattern_parts[1]))

    # Run calibration
    if args.stereo:
        if not args.left or not args.right:
            print("ERROR: --left and --right required for stereo calibration")
            return

        calibrate_stereo(args.left, args.right, pattern_size,
                        args.square_size, args.output)
    else:
        if not args.input:
            print("ERROR: --input required for single camera calibration")
            return

        calibrate_single_camera(args.input, pattern_size, args.square_size,
                               visualize=not args.no_viz, output_dir=args.output)

    print("\n✓ Calibration complete!")

if __name__ == "__main__":
    main()
