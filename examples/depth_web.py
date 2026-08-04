#!/usr/bin/env python3
"""Static depth UI: live left-camera view, a crosshair, and the distance to
whatever is under it. No model, no inference loop - just stereo depth.

Run on the Jetson:

  python3 examples/depth_web.py \
      --calibration calibration/output/stereo_calibration.npz \
      --baseline-override 52.5

Then open http://<jetson>:8011 - the crosshair sits at frame center and the
readout shows the measured distance there (plus disparity/confidence).
Click anywhere on the image to move the measurement point.
"""
import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from stereo_csi import CameraSource, CSICameraCapture, StereoDepthService  # noqa: E402

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>stereo depth</title>
<style>
 body{background:#111;color:#eee;font-family:monospace;text-align:center;margin:0;padding:14px}
 #wrap{position:relative;display:inline-block;cursor:crosshair}
 img{max-width:96vw;max-height:78vh;display:block}
 #cross{position:absolute;width:34px;height:34px;margin:-17px 0 0 -17px;pointer-events:none}
 #cross:before,#cross:after{content:"";position:absolute;background:#22FF88}
 #cross:before{left:16px;top:0;width:2px;height:34px}
 #cross:after{top:16px;left:0;width:34px;height:2px}
 #depth{font-size:34px;margin:12px;color:#22FF88}
 #detail{color:#888;font-size:15px}
</style></head><body>
<div id=wrap><img id=view src=/frame><div id=cross></div></div>
<div id=depth>--</div>
<div id=detail></div>
<script>
 const v=document.getElementById("view"),c=document.getElementById("cross"),
       d=document.getElementById("depth"),t=document.getElementById("detail");
 let px=0.5, py=0.5;
 function placeCross(){c.style.left=(px*v.clientWidth)+"px";c.style.top=(py*v.clientHeight)+"px";}
 v.onload=placeCross; window.onresize=placeCross;
 document.getElementById("wrap").onclick=e=>{
   const r=v.getBoundingClientRect();
   px=(e.clientX-r.left)/r.width; py=(e.clientY-r.top)/r.height; placeCross();
 };
 setInterval(()=>{v.src="/frame?t="+Date.now();},350);
 setInterval(async()=>{
   const r=await fetch(`/depth?x=${px.toFixed(4)}&y=${py.toFixed(4)}`);
   const m=await r.json();
   if(m.depth_mm){d.textContent=(m.depth_mm/1000).toFixed(2)+" m";
     t.textContent=`disparity ${m.disparity_px.toFixed(1)}px · confidence ${m.confidence.toFixed(2)}`;}
   else{d.textContent="--";t.textContent=m.reason||"no valid measurement (low texture?)";}
 },500);
</script></body></html>"""


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.left = None
        self.right = None
        self.jpeg = b""


def capture_loop(camera, state):
    while True:
        left, right = camera.read_frames()
        if left is None:
            time.sleep(0.05)
            continue
        ok, buf = cv2.imencode(".jpg", left, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with state.lock:
            state.left = left
            state.right = right
            if ok:
                state.jpeg = buf.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--baseline-override", type=float, default=None)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--camera-id", type=int, default=0)
    ap.add_argument("--port", type=int, default=8011)
    args = ap.parse_args()

    camera = CameraSource(
        enabled=True, camera_id=args.camera_id, use_csi=True, stereo_mode=True,
        invert_camera=False, video_fps=30,
        output_width=args.width, output_height=args.height,
        csi_capture_factory=CSICameraCapture,
    )
    camera.initialize()

    depth = StereoDepthService(
        calibration_file=args.calibration,
        baseline_override=args.baseline_override,
        image_size=(args.width, args.height),
    )
    depth.load_calibration()

    state = State()
    threading.Thread(target=capture_loop, args=(camera, state), daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="text/plain"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body if isinstance(body, bytes) else body.encode())

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/frame":
                with state.lock:
                    jpeg = state.jpeg
                self._send(200 if jpeg else 503, jpeg or "no frame yet", "image/jpeg")
            elif url.path == "/depth":
                q = parse_qs(url.query)
                fx = float(q.get("x", [0.5])[0])
                fy = float(q.get("y", [0.5])[0])
                with state.lock:
                    left, right = state.left, state.right
                if left is None or right is None:
                    self._send(200, json.dumps({"depth_mm": None, "reason": "no frames yet"}),
                               "application/json")
                    return
                x = int(fx * args.width)
                y = int(fy * args.height)
                ms = depth.calculate_depths(left, right, [(x, y)])
                m = ms[0] if ms else None
                if m is None or not m.depth_mm:
                    payload = {"depth_mm": None}
                else:
                    payload = {"depth_mm": m.depth_mm, "disparity_px": m.disparity_px,
                               "confidence": m.confidence}
                self._send(200, json.dumps(payload), "application/json")
            else:
                self._send(200, PAGE, "text/html")

    print(f"depth UI on http://0.0.0.0:{args.port}  (crosshair at click point)")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
