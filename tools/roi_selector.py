"""
tools/roi_selector.py
=====================
Standalone helper to pick the ROI polygon by clicking on an actual frame
from your video, instead of hand-guessing pixel coordinates.

It does NOT import anything from config/settings.py or the rest of the
pipeline (those pull in torch/transformers, which would be a heavy,
unnecessary dependency for what is just "click some points and print
them out"). It only needs opencv-python + numpy.

Why it resizes the frame first
-------------------------------
main.py always does:

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

before checking is_inside_roi() / drawing the ROI. If you click points
on the video's native resolution (e.g. a 1920x1080 camera) but the
pipeline actually runs detection on a resized 640x360 frame, your ROI
will land in the wrong place. So this tool resizes to the SAME
FRAME_WIDTH x FRAME_HEIGHT your pipeline uses before you click,
regardless of what resolution the source video actually is — that's
what makes it work identically for any input video size.

By default it reads FRAME_WIDTH / FRAME_HEIGHT straight out of
config/settings.py with a lightweight regex (no heavy imports). You can
also override them with --width/--height if you want to try a
different working resolution.

Usage
-----
    python tools/roi_selector.py --video /path/to/video.mp4
    python tools/roi_selector.py --video /path/to/video.mp4 --width 640 --height 360
    python tools/roi_selector.py --video 0                      # webcam

Controls (shown in the window title bar / printed on start)
-------------------------------------------------------------
    Left click   add a polygon point
    Right click  undo the last point
    r            reset all points
    n            jump ahead to a later frame (in case frame 0 is blank/dark)
    p            jump back to an earlier frame
    c            re-capture / freeze the currently shown frame to draw on
    ENTER        finish and print the ROI_POINTS block to the terminal
    q / ESC      quit without printing

Output
------
Prints a block in exactly the format config/settings.py expects, e.g.:

    ROI_POINTS = np.array([
        (3, 226),
        (37, 231),
        (202, 185),
        ...
    ])

so you can copy/paste it straight over the existing ROI_POINTS in
config/settings.py. It's also written to tools/roi_points_output.txt.
"""

import os
import re
import sys
import json
import argparse

import cv2
import numpy as np


DEFAULT_WIDTH  = 640
DEFAULT_HEIGHT = 360

SETTINGS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "settings.py",
)

# config/settings.py auto-loads the ROI polygon from exactly this file
# on every startup (video, webcam, or future RTSP — identically) — see
# config.settings._load_roi_points(). No copy/paste into settings.py is
# required any more; running this tool and confirming a selection is
# the whole workflow.
ROI_JSON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "roi.json",
)


def save_roi_json(points, path=ROI_JSON_PATH):
    """Persist the selected polygon as {"points": [[x, y], ...]} —
    the exact shape config.settings._load_roi_points() expects."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"points": [[int(x), int(y)] for x, y in points]}
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)
    return path


def read_frame_size_from_settings():
    """
    Lightweight regex read of FRAME_WIDTH / FRAME_HEIGHT out of
    config/settings.py, without importing that module (which would
    drag in torch/transformers just to read two integers).
    Falls back to DEFAULT_WIDTH/DEFAULT_HEIGHT if the file or the
    values can't be found.
    """
    width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
    try:
        with open(SETTINGS_PATH, "r") as f:
            content = f.read()
        w_match = re.search(r"FRAME_WIDTH\s*=\s*(\d+)", content)
        h_match = re.search(r"FRAME_HEIGHT\s*=\s*(\d+)", content)
        if w_match:
            width = int(w_match.group(1))
        if h_match:
            height = int(h_match.group(1))
    except OSError:
        pass
    return width, height


class ROISelector:
    def __init__(self, video_source, width, height):
        self.width  = width
        self.height = height

        # allow "0", "1" etc. for webcam indices, otherwise treat as a path
        source = int(video_source) if str(video_source).isdigit() else video_source

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {video_source}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.current_frame_idx = 0
        self.base_frame  = None   # resized frame currently being drawn on
        self.points      = []

        self.window_name = "ROI Selector  (L-click=add  R-click=undo  r=reset  n/p=frame  c=capture  ENTER=finish  q=quit)"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        self._grab_frame(0)

    # ------------------------------------------------------------------
    def _grab_frame(self, frame_idx):
        """Seek to frame_idx, read it, resize to (width, height), and
        make it the frame we're currently drawing the polygon on."""
        if self.total_frames > 0:
            frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            print(f"[WARN] Could not read frame {frame_idx}; keeping previous frame.")
            return
        self.current_frame_idx = frame_idx
        # This is the resize that matters: it must match main.py's
        # cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT)) so the points
        # you click line up with what the pipeline sees at runtime,
        # no matter what resolution the source video actually is.
        self.base_frame = cv2.resize(frame, (self.width, self.height))

    # ------------------------------------------------------------------
    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()

    # ------------------------------------------------------------------
    def _render(self):
        display = self.base_frame.copy()

        for i, pt in enumerate(self.points):
            cv2.circle(display, pt, 4, (0, 0, 255), -1)
            cv2.putText(display, str(i), (pt[0] + 6, pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            if i > 0:
                cv2.line(display, self.points[i - 1], pt, (255, 0, 0), 2)

        if len(self.points) > 2:
            cv2.line(display, self.points[-1], self.points[0], (255, 0, 0), 1)

        info = (f"frame {self.current_frame_idx}/{max(self.total_frames - 1, 0)}  "
                f"size {self.width}x{self.height}  points {len(self.points)}")
        cv2.putText(display, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(self.window_name, display)

    # ------------------------------------------------------------------
    def run(self):
        print("\nControls: Left-click=add point | Right-click=undo | "
              "r=reset | n=next frame | p=prev frame | c=recapture | "
              "ENTER=finish | q/ESC=quit\n")

        step = max(1, (self.total_frames or 300) // 100)  # ~1% jumps

        while True:
            self._render()
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 10):          # ENTER
                if len(self.points) < 3:
                    print("[INFO] Need at least 3 points to form a polygon — keep clicking.")
                    continue
                break
            elif key in (27, ord('q')):  # ESC or q
                self.points = []
                break
            elif key == ord('r'):
                self.points = []
            elif key == ord('n'):
                self._grab_frame(self.current_frame_idx + step)
            elif key == ord('p'):
                self._grab_frame(self.current_frame_idx - step)
            elif key == ord('c'):
                self._grab_frame(self.current_frame_idx)

        cv2.destroyAllWindows()
        self.cap.release()
        return self.points


def format_roi_block(points):
    lines = ["ROI_POINTS = np.array(["]
    for (x, y) in points:
        lines.append(f"    ({x}, {y}),")
    lines.append("])")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Interactively select an ROI polygon on a video frame.")
    parser.add_argument("--video", required=True, help="Path to video file, or a webcam index like 0.")
    parser.add_argument("--width", type=int, default=None,
                         help="Working frame width (defaults to FRAME_WIDTH read from config/settings.py).")
    parser.add_argument("--height", type=int, default=None,
                         help="Working frame height (defaults to FRAME_HEIGHT read from config/settings.py).")
    args = parser.parse_args()

    cfg_width, cfg_height = read_frame_size_from_settings()
    width  = args.width  if args.width  is not None else cfg_width
    height = args.height if args.height is not None else cfg_height

    print(f"[INFO] Using working resolution {width}x{height} "
          f"(this must match FRAME_WIDTH/FRAME_HEIGHT in config/settings.py).")

    selector = ROISelector(args.video, width, height)
    points = selector.run()

    if not points:
        print("\n[INFO] No polygon selected — nothing printed.")
        sys.exit(0)

    block = format_roi_block(points)
    print("\n" + "=" * 60)
    print("Paste this over ROI_POINTS in config/settings.py:")
    print("=" * 60)
    print(block)
    print("=" * 60)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roi_points_output.txt")
    with open(out_path, "w") as f:
        f.write(block + "\n")
    print(f"\n[INFO] Also saved to {out_path}")

    json_path = save_roi_json(points)
    print(f"[INFO] ROI saved to {json_path} — the app now loads this "
          f"automatically on startup (video, webcam, or RTSP). No manual "
          f"paste into config/settings.py needed.")


if __name__ == "__main__":
    main()