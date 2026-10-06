"""Find subtitles that are already burned into a video, using macOS's built-in text detection."""
from pathlib import Path

import cv2
import numpy as np

BINS = 100


def _text_boxes(frame: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Text boxes as (x, y, w, h) in 0-1 coordinates, origin top-left. Works for any script."""
    import Foundation
    import Vision

    ok, png = cv2.imencode(".png", frame)
    if not ok:
        return []
    data = Foundation.NSData.dataWithBytes_length_(png.tobytes(), len(png))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    req = Vision.VNDetectTextRectanglesRequest.alloc().init()
    ok, _ = handler.performRequests_error_([req], None)
    if not ok:
        return []
    boxes = []
    for obs in req.results() or []:
        (x, y), (w, h) = obs.boundingBox()
        boxes.append((x, 1 - y - h, w, h))  # Vision's origin is bottom-left
    return boxes


def detect_band(video: Path, samples: int = 30) -> tuple[float, float] | None:
    """Return (top, bottom) of the subtitle band as fractions of frame height, or None.

    Subtitles are lines of text that are horizontally centred, sit in the lower part of the
    frame and show up in a lot of frames. Corner logos, titles and HUD text don't match all three.
    """
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    scale = 960 / w if w > 960 else 1.0
    hits = np.zeros(BINS)
    used = 0
    for idx in np.linspace(0, n - 1, samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        used += 1
        if scale != 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        rows = np.zeros(BINS, bool)
        for x, y, bw, bh in _text_boxes(frame):
            cx, cy = x + bw / 2, y + bh / 2
            if cy < 0.45 or not (0.3 <= cx <= 0.7) or bh > 0.2:
                continue
            rows[int(y * BINS):min(BINS, int((y + bh) * BINS) + 1)] = True
        hits += rows
    cap.release()
    if not used:
        return None

    persistent = hits / used >= 0.25
    if not persistent.any():
        return None
    # largest run of persistent rows (tolerating 2-row gaps between subtitle lines)
    runs, start, gap = [], None, 0
    for i, on in enumerate(np.append(persistent, [False] * 3)):
        if on:
            start = i if start is None else start
            gap, end = 0, i
        elif start is not None:
            gap += 1
            if gap > 2:
                runs.append((start, end))
                start = None
    top, bottom = max(runs, key=lambda r: r[1] - r[0])
    return max(0.0, top / BINS - 0.02), min(1.0, (bottom + 1) / BINS + 0.02)


def removal_filter(band: tuple[float, float], w: int, h: int) -> tuple[str, str]:
    """ffmpeg filter that removes the band. Near the bottom it's cropped off cleanly;
    elsewhere it's blurred out. Returns (filter, description)."""
    top, bottom = band
    if bottom >= 0.8 and top >= 0.6:
        keep = int(h * top) // 2 * 2
        return f"crop={w}:{keep}:0:0", "cropped off"
    y, bh = int(h * top), max(2, int(h * (bottom - top)) // 2 * 2)
    return (f"split[m][s];[s]crop={w}:{bh}:0:{y},boxblur=30:3[b];[m][b]overlay=0:{y}",
            "blurred")
