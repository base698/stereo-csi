#!/usr/bin/env python3
"""
Headless Camera Calibration Image Capture for Jetson
Captures calibration images via SSH with text feedback (no display needed)
Web server mode available for viewing camera feed in browser
"""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import os
import argparse
import time
import threading
import io
import sys
from datetime import datetime
from pathlib import Path

CSI_SENSOR_WIDTH = 1640
CSI_SENSOR_HEIGHT = 1232

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Try importing CSI camera helper
try:
    from stereo_csi.csi_camera import CSICameraCapture
    CSI_HELPER_AVAILABLE = True
    print("✓ CSICameraCapture module loaded")
except ImportError as e:
    CSI_HELPER_AVAILABLE = False
    print(f"⚠ CSICameraCapture not available: {e}")

# Optional FastAPI for web server mode
try:
    from fastapi import FastAPI, Response
    from fastapi.responses import HTMLResponse, StreamingResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("Warning: FastAPI not available. Web server mode disabled.")


def gstreamer_pipeline(sensor_id, width, height, fps=30, flip_method=0):
    """Build the Jetson CSI pipeline used by the calibration helper."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){CSI_SENSOR_WIDTH}, "
        f"height=(int){CSI_SENSOR_HEIGHT}, format=(string)NV12, "
        f"framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, "
        "format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )


def capture_calibration_images(camera_id, output_dir, pattern_size=(9, 6),
                               num_images=30, use_csi=False, stereo_mode=False,
                               width=640, height=480):
    """
    Capture calibration images with text feedback for headless operation

    Args:
        camera_id: Camera device ID (or ignored if use_csi)
        output_dir: Directory to save images
        pattern_size: Checkerboard internal corners (width, height)
        num_images: Target number of images to capture
        use_csi: Use CSI camera with GStreamer
        stereo_mode: Capture from two cameras for stereo calibration
        width: Output frame width
        height: Output frame height
    """
    os.makedirs(output_dir, exist_ok=True)

    # Open camera(s)
    if use_csi:
        print("Opening CSI camera...")
        # Use CSI camera helper (subprocess+GStreamer)
        if CSI_HELPER_AVAILABLE:
            cap = CSICameraCapture(
                sensor_id=0,
                width=width,
                height=height,
                fps=30,
                flip_method=0
            )
            cap.start()
            print(f"✓ CSI Camera initialized with subprocess+GStreamer ({width}x{height} @ 30 FPS)")
        else:
            # Fallback to cv2.VideoCapture with GStreamer
            gst_pipeline = gstreamer_pipeline(0, width, height)
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            print("✓ CSI Camera initialized with cv2.VideoCapture+GStreamer")

        cap2 = None
        if stereo_mode:
            print("Opening second CSI camera...")
            if CSI_HELPER_AVAILABLE:
                cap2 = CSICameraCapture(
                    sensor_id=1,
                    width=width,
                    height=height,
                    fps=30,
                    flip_method=0
                )
                cap2.start()
                print("✓ Second CSI Camera initialized with subprocess+GStreamer")
            else:
                gst_pipeline2 = gst_pipeline.replace("sensor-id=0", "sensor-id=1")
                cap2 = cv2.VideoCapture(gst_pipeline2, cv2.CAP_GSTREAMER)
                print("✓ Second CSI Camera initialized with cv2.VideoCapture+GStreamer")
    else:
        print(f"Opening USB camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        # Set format to MJPEG if available (better compatibility, prevents green image)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap2 = None
        if stereo_mode:
            print(f"Opening second USB camera {camera_id + 1}...")
            cap2 = cv2.VideoCapture(camera_id + 1)
            # Set format to MJPEG if available (better compatibility, prevents green image)
            cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap2.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return

    if stereo_mode and not cap2.isOpened():
        print("ERROR: Could not open second camera")
        return

    print("\n" + "="*60)
    print("CALIBRATION IMAGE CAPTURE (Headless Mode)")
    print("="*60)
    print(f"Pattern size: {pattern_size[0]}x{pattern_size[1]} internal corners")
    print(f"Frame size: {width}x{height}")
    print(f"Target images: {num_images}")
    print(f"Output directory: {output_dir}")
    print("\nInstructions:")
    print("  - Move checkerboard to different positions and angles")
    print("  - Pattern must be fully visible in frame")
    print("  - Press ENTER to capture when pattern is detected")
    print("  - Type 'q' and ENTER to quit")
    print("="*60 + "\n")

    captured_count = 0

    # Create subdirectories for stereo
    if stereo_mode:
        os.makedirs(f"{output_dir}/left", exist_ok=True)
        os.makedirs(f"{output_dir}/right", exist_ok=True)

    while captured_count < num_images:
        # Capture frame(s)
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to capture frame")
            break

        ret2, frame2 = (cap2.read() if stereo_mode else (None, None))
        if stereo_mode and not ret2:
            print("ERROR: Failed to capture from second camera")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if stereo_mode else None

        # Find checkerboard corners
        found, corners = cv2.findChessboardCorners(gray, pattern_size,
                                                    cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                    cv2.CALIB_CB_NORMALIZE_IMAGE)

        found2 = False
        if stereo_mode and found:
            found2, corners2 = cv2.findChessboardCorners(gray2, pattern_size,
                                                         cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                         cv2.CALIB_CB_NORMALIZE_IMAGE)

        # Status update
        status = "✓ PATTERN DETECTED" if found else "✗ No pattern"
        if stereo_mode:
            status2 = "✓ DETECTED" if found2 else "✗ Not detected"
            status = f"Left: {status} | Right: {status2}"

        print(f"\r[{captured_count}/{num_images}] {status}  ", end='', flush=True)

        # Wait for user input
        import select
        import sys

        # Non-blocking input check (Unix/Linux only)
        if select.select([sys.stdin], [], [], 0.1)[0]:
            user_input = sys.stdin.readline().strip()

            if user_input.lower() == 'q':
                print("\n\nCapture cancelled by user")
                break

            # ENTER pressed - try to capture
            if found and (not stereo_mode or found2):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                if stereo_mode:
                    # Save both images
                    left_path = f"{output_dir}/left/img_{captured_count:03d}_{timestamp}.jpg"
                    right_path = f"{output_dir}/right/img_{captured_count:03d}_{timestamp}.jpg"
                    cv2.imwrite(left_path, frame)
                    cv2.imwrite(right_path, frame2)
                    print(f"\n  ✓ Saved: {left_path}")
                    print(f"  ✓ Saved: {right_path}")
                else:
                    # Save single image
                    filename = f"{output_dir}/img_{captured_count:03d}_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"\n  ✓ Saved: {filename}")

                captured_count += 1
            else:
                print("\n  ✗ Cannot capture - pattern not detected in both cameras")

    cap.release()
    if cap2:
        cap2.release()

    print("\n\n" + "="*60)
    print(f"Capture complete: {captured_count} images saved")
    print(f"Copy '{output_dir}' to your host machine for calibration")
    print("="*60)

class WebCalibrationCapture:
    """Web-based calibration capture with browser interface"""

    def __init__(self, camera_id=0, output_dir='calibration_images', pattern_size=(9, 6),
                 use_csi=False, stereo_mode=False, width=640, height=480):
        self.camera_id = camera_id
        self.output_dir = output_dir
        self.pattern_size = pattern_size
        self.use_csi = use_csi
        self.stereo_mode = stereo_mode
        self.width = int(width)
        self.height = int(height)

        # State
        self.captured_count = 0
        self.latest_frame = None
        self.latest_frame_left = None
        self.latest_frame_right = None
        self.latest_clean_frame = None
        self.latest_clean_frame_left = None
        self.latest_clean_frame_right = None
        self.pattern_detected = False
        self.pattern_detected_left = False
        self.pattern_detected_right = False
        self.lock = threading.Lock()
        self.running = True

        # Setup output directories
        os.makedirs(output_dir, exist_ok=True)
        if stereo_mode:
            os.makedirs(f"{output_dir}/left", exist_ok=True)
            os.makedirs(f"{output_dir}/right", exist_ok=True)

        # Open cameras
        self.cap = None
        self.cap2 = None
        self._open_cameras()

        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _open_cameras(self):
        """Open camera(s)"""
        if self.use_csi:
            print(f"Opening CSI camera (CSI_HELPER_AVAILABLE={CSI_HELPER_AVAILABLE})...")
            use_csi_helper = CSI_HELPER_AVAILABLE

            # Use CSI camera helper (subprocess+GStreamer)
            if use_csi_helper:
                try:
                    self.cap = CSICameraCapture(
                        sensor_id=0,
                        width=self.width,
                        height=self.height,
                        fps=30,
                        flip_method=0
                    )
                    self.cap.start()
                    print(
                        "✓ CSI Camera initialized with subprocess+GStreamer "
                        f"({self.width}x{self.height} @ 30 FPS)"
                    )
                except Exception as e:
                    print(f"✗ Failed to initialize CSICameraCapture: {e}")
                    print("Falling back to cv2.VideoCapture...")
                    use_csi_helper = False  # Force fallback for second camera too

            if not use_csi_helper:
                # Fallback to cv2.VideoCapture with GStreamer
                print("Using cv2.VideoCapture with GStreamer pipeline...")
                gst_pipeline = gstreamer_pipeline(0, self.width, self.height)
                self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                if self.cap.isOpened():
                    print("✓ CSI Camera initialized with cv2.VideoCapture+GStreamer")
                else:
                    print("✗ Failed to open CSI camera with cv2.VideoCapture")

            if self.stereo_mode:
                if use_csi_helper:
                    try:
                        self.cap2 = CSICameraCapture(
                            sensor_id=1,
                            width=self.width,
                            height=self.height,
                            fps=30,
                            flip_method=0
                        )
                        self.cap2.start()
                        print("✓ Second CSI Camera initialized with subprocess+GStreamer")
                    except Exception as e:
                        print(f"✗ Failed to initialize second CSICameraCapture: {e}")
                else:
                    gst_pipeline2 = gstreamer_pipeline(1, self.width, self.height)
                    self.cap2 = cv2.VideoCapture(gst_pipeline2, cv2.CAP_GSTREAMER)
                    if self.cap2.isOpened():
                        print("✓ Second CSI Camera initialized with cv2.VideoCapture+GStreamer")
                    else:
                        print("✗ Failed to open second CSI camera")
        else:
            self.cap = cv2.VideoCapture(self.camera_id)
            # Set format to MJPEG if available (better compatibility, prevents green image)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            if self.stereo_mode:
                self.cap2 = cv2.VideoCapture(self.camera_id + 1)
                # Set format to MJPEG if available (better compatibility, prevents green image)
                self.cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap2.set(cv2.CAP_PROP_FPS, 30)

    def _capture_loop(self):
        """Capture frames and detect pattern (runs at ~5 FPS)"""
        last_capture_time = 0
        min_capture_interval = 1.0  # Minimum 1 second between auto-captures

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.2)
                continue

            ret2, frame2 = (self.cap2.read() if self.stereo_mode else (None, None))
            clean_frame = frame.copy()
            clean_frame2 = frame2.copy() if ret2 else None

            # Convert to grayscale and find pattern
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, self.pattern_size,
                                                       cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                       cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                       cv2.CALIB_CB_FAST_CHECK)

            # Draw pattern if found
            if found:
                cv2.drawChessboardCorners(frame, self.pattern_size, corners, found)

            # Add overlay
            self._add_overlay(frame, found, "LEFT" if self.stereo_mode else None)

            found2 = False
            if self.stereo_mode and ret2:
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                found2, corners2 = cv2.findChessboardCorners(gray2, self.pattern_size,
                                                             cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                             cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                             cv2.CALIB_CB_FAST_CHECK)
                if found2:
                    cv2.drawChessboardCorners(frame2, self.pattern_size, corners2, found2)
                self._add_overlay(frame2, found2, "RIGHT")

            # Auto-save when pattern detected (with rate limiting)
            current_time = time.time()
            if current_time - last_capture_time >= min_capture_interval:
                should_save = False
                if self.stereo_mode:
                    should_save = found and found2
                else:
                    should_save = found

                if should_save:
                    # Save images without drawing overlays
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                    if self.stereo_mode:
                        left_path = f"{self.output_dir}/left/img_{self.captured_count:03d}_{timestamp}.jpg"
                        right_path = f"{self.output_dir}/right/img_{self.captured_count:03d}_{timestamp}.jpg"
                        if clean_frame2 is not None:
                            cv2.imwrite(left_path, clean_frame)
                            cv2.imwrite(right_path, clean_frame2)
                            print(f"[AUTO] Saved {self.captured_count}: {left_path} & {right_path}")
                            self.captured_count += 1
                    else:
                        filename = f"{self.output_dir}/img_{self.captured_count:03d}_{timestamp}.jpg"
                        cv2.imwrite(filename, clean_frame)
                        print(f"[AUTO] Saved {self.captured_count}: {filename}")
                        self.captured_count += 1

                    if should_save:
                        last_capture_time = current_time

            # Update state
            with self.lock:
                if self.stereo_mode:
                    self.latest_frame_left = frame.copy()
                    self.latest_frame_right = frame2.copy() if ret2 else None
                    self.latest_clean_frame_left = clean_frame.copy()
                    self.latest_clean_frame_right = clean_frame2.copy() if clean_frame2 is not None else None
                    self.pattern_detected_left = found
                    self.pattern_detected_right = found2
                else:
                    self.latest_frame = frame.copy()
                    self.latest_clean_frame = clean_frame.copy()
                    self.pattern_detected = found

            # Limit to ~5 FPS
            time.sleep(0.2)

    def _add_overlay(self, frame, pattern_found, label=None):
        """Add status overlay to frame"""
        h, w = frame.shape[:2]

        # Status text
        status = "PATTERN DETECTED" if pattern_found else "No pattern"
        color = (0, 255, 0) if pattern_found else (0, 0, 255)

        cv2.putText(frame, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if label:
            cv2.putText(frame, label, (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Capture count
        cv2.putText(frame, f"Captured: {self.captured_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Crosshair
        cv2.line(frame, (w//2 - 20, h//2), (w//2 + 20, h//2), (0, 255, 0), 1)
        cv2.line(frame, (w//2, h//2 - 20), (w//2, h//2 + 20), (0, 255, 0), 1)

    def capture_image(self):
        """Capture current frame(s) if pattern detected"""
        with self.lock:
            if self.stereo_mode:
                if not self.pattern_detected_left or not self.pattern_detected_right:
                    return False, "Pattern not detected in both cameras"

                if self.latest_clean_frame_left is None or self.latest_clean_frame_right is None:
                    return False, "Frames not available"

                # Save both images
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                left_path = f"{self.output_dir}/left/img_{self.captured_count:03d}_{timestamp}.jpg"
                right_path = f"{self.output_dir}/right/img_{self.captured_count:03d}_{timestamp}.jpg"

                cv2.imwrite(left_path, self.latest_clean_frame_left)
                cv2.imwrite(right_path, self.latest_clean_frame_right)

                self.captured_count += 1
                return True, f"Saved {Path(left_path).name} and {Path(right_path).name}"

            else:
                if not self.pattern_detected:
                    return False, "Pattern not detected"

                if self.latest_clean_frame is None:
                    return False, "Frame not available"

                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{self.output_dir}/img_{self.captured_count:03d}_{timestamp}.jpg"
                cv2.imwrite(filename, self.latest_clean_frame)

                self.captured_count += 1
                return True, f"Saved {Path(filename).name}"

    def get_frame_jpeg(self, stereo_side=None):
        """Get current frame as JPEG bytes"""
        with self.lock:
            if self.stereo_mode:
                if stereo_side == "left":
                    frame = self.latest_frame_left
                elif stereo_side == "right":
                    frame = self.latest_frame_right
                else:
                    # Combined side-by-side
                    if self.latest_frame_left is not None and self.latest_frame_right is not None:
                        frame = cv2.hconcat([self.latest_frame_left, self.latest_frame_right])
                    else:
                        frame = None
            else:
                frame = self.latest_frame

        if frame is None:
            # Return blank frame
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(frame, "No frame", (self.width // 3, self.height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpeg.tobytes()

    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        if self.cap:
            self.cap.release()
        if self.cap2:
            self.cap2.release()


def start_web_server(camera_id=0, output_dir='calibration_images', pattern_size=(9, 6),
                     use_csi=False, stereo_mode=False, port=8000,
                     width=640, height=480):
    """Start web server for calibration capture"""

    if not FASTAPI_AVAILABLE:
        print("ERROR: FastAPI not installed. Install with: pip install fastapi uvicorn")
        return

    app = FastAPI()

    # Initialize capture system
    capture_system = WebCalibrationCapture(camera_id, output_dir, pattern_size,
                                           use_csi, stereo_mode, width, height)

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Serve main page"""
        stereo_html = ""
        if stereo_mode:
            stereo_html = f"""
            <h2>Stereo Cameras</h2>
            <div style="display: flex; gap: 20px;">
                <div>
                    <h3>Left Camera</h3>
                    <img src="/stream/left" width="{width}" height="{height}" style="border: 2px solid #333; max-width: 48vw; height: auto;">
                </div>
                <div>
                    <h3>Right Camera</h3>
                    <img src="/stream/right" width="{width}" height="{height}" style="border: 2px solid #333; max-width: 48vw; height: auto;">
                </div>
            </div>
            <h3>Combined View</h3>
            """

        html = f"""
        <html>
        <head>
            <title>Camera Calibration Capture</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #1e1e1e; color: #fff; }}
                h1 {{ color: #4CAF50; }}
                button {{
                    padding: 15px 30px;
                    font-size: 18px;
                    margin: 10px;
                    cursor: pointer;
                    border: none;
                    border-radius: 5px;
                }}
                #captureBtn {{ background: #4CAF50; color: white; }}
                #captureBtn:disabled {{ background: #666; cursor: not-allowed; }}
                #status {{
                    padding: 10px;
                    margin: 10px 0;
                    border-radius: 5px;
                    font-weight: bold;
                }}
                .success {{ background: #4CAF50; }}
                .error {{ background: #f44336; }}
                .info {{ background: #2196F3; }}
            </style>
        </head>
        <body>
            <h1>📷 Camera Calibration Capture</h1>
            <p>Pattern: {pattern_size[0]}x{pattern_size[1]} internal corners</p>
            <p>Frame: {width}x{height}</p>
            <p>Output: {output_dir}/</p>

            <div id="status" class="info">Status: Ready - Position checkerboard and click Capture</div>

            <button id="captureBtn" onclick="captureImage()">📸 CAPTURE IMAGE</button>
            <button onclick="location.reload()">🔄 Refresh</button>

            {stereo_html}

            <h2>Camera Feed</h2>
            <img src="/stream" width="{width * 2 if stereo_mode else width}" height="{height}"
                 style="border: 2px solid #333; max-width: 100%; height: auto;">

            <h2>Instructions</h2>
            <ul>
                <li>Move checkerboard to different positions and angles</li>
                <li>Wait for "PATTERN DETECTED" (green text) in video</li>
                <li>Click <strong>CAPTURE IMAGE</strong> button</li>
                <li>Capture 30-50 images with varied orientations</li>
                <li>Copy '{output_dir}' to host for calibration</li>
            </ul>

            <script>
                async function captureImage() {{
                    const btn = document.getElementById('captureBtn');
                    const status = document.getElementById('status');

                    btn.disabled = true;
                    status.className = 'info';
                    status.textContent = 'Capturing...';

                    try {{
                        const response = await fetch('/capture', {{ method: 'POST' }});
                        const data = await response.json();

                        if (data.success) {{
                            status.className = 'success';
                            status.textContent = `✓ ${{data.message}} - Total: ${{data.count}}`;
                        }} else {{
                            status.className = 'error';
                            status.textContent = `✗ ${{data.message}}`;
                        }}
                    }} catch (error) {{
                        status.className = 'error';
                        status.textContent = '✗ Error: ' + error.message;
                    }}

                    setTimeout(() => {{ btn.disabled = false; }}, 500);
                }}

                // Auto-refresh status every 2 seconds
                setInterval(async () => {{
                    try {{
                        const response = await fetch('/status');
                        const data = await response.json();
                        const currentStatus = document.getElementById('status');
                        if (!currentStatus.textContent.includes('Capturing')) {{
                            currentStatus.className = 'info';
                            currentStatus.textContent = `Images captured: ${{data.count}} | Pattern: ${{data.pattern_detected ? '✓ Detected' : '✗ Not detected'}}`;
                        }}
                    }} catch (error) {{}}
                }}, 2000);
            </script>
        </body>
        </html>
        """
        return html

    @app.get("/stream/{side}")
    @app.get("/stream")
    async def stream(side: str = None):
        """Stream camera feed as MJPEG"""
        def generate():
            while True:
                frame_bytes = capture_system.get_frame_jpeg(side)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.2)  # 5 FPS

        return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/capture")
    async def capture():
        """Capture image"""
        success, message = capture_system.capture_image()
        return {
            "success": success,
            "message": message,
            "count": capture_system.captured_count
        }

    @app.get("/status")
    async def status():
        """Get current status"""
        with capture_system.lock:
            if stereo_mode:
                pattern_detected = capture_system.pattern_detected_left and capture_system.pattern_detected_right
            else:
                pattern_detected = capture_system.pattern_detected

        return {
            "count": capture_system.captured_count,
            "pattern_detected": pattern_detected
        }

    print("\n" + "="*70)
    print("WEB CALIBRATION CAPTURE SERVER")
    print("="*70)
    print(f"Server running at: http://localhost:{port}")
    print(f"Access from other device: http://<jetson-ip>:{port}")
    print(f"Output directory: {output_dir}")
    print("="*70 + "\n")

    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    finally:
        capture_system.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Capture calibration images (headless mode for SSH)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Web server mode (recommended - view in browser)
  python tools/vision/calibration/capture_calibration.py --web --camera 0 --output calib_images
  python tools/vision/calibration/capture_calibration.py --web --use-csi --stereo --output calib_stereo
  python tools/vision/calibration/capture_calibration.py --web --use-csi --stereo --width 960 --height 720 --output calib_stereo_960

  # Headless text mode
  python tools/vision/calibration/capture_calibration.py --camera 0 --output calib_images
  python tools/vision/calibration/capture_calibration.py --use-csi --stereo --output calib_stereo
        """
    )

    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--output', '-o', type=str, default='calibration_images',
                       help='Output directory (default: calibration_images)')
    parser.add_argument('--pattern', '-p', type=str, default='9x6',
                       help='Checkerboard pattern (default: 9x6 internal corners)')
    parser.add_argument('--count', '-n', type=int, default=30,
                       help='Number of images to capture (default: 30)')
    parser.add_argument('--use-csi', action='store_true', default=True,
                       help='Use CSI camera with GStreamer (Jetson, default: True)')
    parser.add_argument('--use-usb', action='store_true',
                       help='Use USB camera instead of CSI')
    parser.add_argument('--stereo', action='store_true',
                       help='Capture from two cameras for stereo calibration')
    parser.add_argument('--web', action='store_true',
                       help='Start web server (view in browser)')
    parser.add_argument('--port', type=int, default=8000,
                       help='Web server port (default: 8000)')
    parser.add_argument('--width', type=int, default=640,
                       help='Captured frame width in pixels (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                       help='Captured frame height in pixels (default: 480)')

    args = parser.parse_args()

    # Parse pattern size
    pattern_parts = args.pattern.lower().split('x')
    if len(pattern_parts) != 2:
        print("ERROR: Pattern must be in format WIDTHxHEIGHT (e.g., 9x6)")
        return

    pattern_size = (int(pattern_parts[0]), int(pattern_parts[1]))

    # Handle use-csi vs use-usb flags
    use_csi = args.use_csi and not args.use_usb

    if args.web:
        # Web server mode
        start_web_server(
            camera_id=args.camera,
            output_dir=args.output,
            pattern_size=pattern_size,
            use_csi=use_csi,
            stereo_mode=args.stereo,
            port=args.port,
            width=args.width,
            height=args.height
        )
    else:
        # Headless text mode
        capture_calibration_images(
            camera_id=args.camera,
            output_dir=args.output,
            pattern_size=pattern_size,
            num_images=args.count,
            use_csi=use_csi,
            stereo_mode=args.stereo,
            width=args.width,
            height=args.height
        )

if __name__ == "__main__":
    main()
