"""Cut a clip, reframe it to vertical around the speaker's face, and burn in animated captions."""
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .subtitles import detect_band, removal_filter

FONTS_DIR = Path(__file__).resolve().parent.parent / "fonts"
_SYSTEM_FONTS = [  # linked into FONTS_DIR so libass finds them all in one place
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Kohinoor.ttc",   # Devanagari (Hindi) with a real Bold weight
    "/System/Library/Fonts/GeezaPro.ttc",   # Arabic script (Urdu, Arabic) with Bold
]
# Areas the TikTok / Instagram Reels / YouTube Shorts apps cover with their own UI (1080x1920).
# Our text stays inside the area all three leave free.
SAFE_TOP = 280      # tabs / header
SAFE_BOTTOM = 450   # username, description, music, nav bar
SAFE_SIDE = 150     # like/comment/share buttons on the right (mirrored left so text stays centred)

FONT = "Arial Black"
FONT_DEVANAGARI = "Kohinoor Devanagari"
FONT_ARABIC = "Geeza Pro"
FONT_UNICODE = "Arial Unicode MS"  # fallback for Cyrillic, Greek, etc.
EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]")


def _fonts_dir() -> str:
    FONTS_DIR.mkdir(exist_ok=True)
    for f in _SYSTEM_FONTS:
        link = FONTS_DIR / Path(f).name
        if Path(f).exists() and not link.is_symlink():
            link.symlink_to(f)
    return str(FONTS_DIR)


def _pick_font(text: str) -> tuple[str, float]:
    """Font family for the dominant script, plus a size multiplier."""
    letters = [ch for ch in text if ch.isalpha()]
    if all(ord(ch) < 0x250 for ch in letters):
        return FONT, 1.0
    if sum("\u0900" <= ch <= "\u097f" for ch in letters) > len(letters) / 2:
        return FONT_DEVANAGARI, 1.35  # Devanagari glyphs sit small in the em box
    if sum("\u0600" <= ch <= "\u06ff" or "\u0750" <= ch <= "\u077f" for ch in letters) > len(letters) / 2:
        return FONT_ARABIC, 1.2
    return FONT_UNICODE, 1.0

FACE_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
FACE_MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
                  "face_detection_yunet/face_detection_yunet_2023mar.onnx")


def _face_detector(w: int, h: int):
    if not FACE_MODEL.exists():
        import urllib.request
        FACE_MODEL.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL)
    det = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (w, h), 0.7)
    det.setInputSize((w, h))
    return det


# ---------- cutting ----------

def cut(src: Path, start: float, end: float, out: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src),
         "-t", f"{end - start:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
         "-c:a", "aac", "-b:a", "192k", str(out)],
        check=True,
    )
    return out


# ---------- face tracking ----------

def track_faces(video: Path, sample_every: int = 5) -> tuple[np.ndarray, dict]:
    """Return a smoothed horizontal center (pixels) for every frame, plus face stats for layout choice."""
    cap = cv2.VideoCapture(str(video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    scale = 640 / w if w > 640 else 1.0
    sw, sh = int(w * scale), int(h * scale)
    det = _face_detector(sw, sh)
    samples, widths, sampled, idx = {}, [], 0, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % sample_every == 0:
            _, frame = cap.retrieve()
            sampled += 1
            small = cv2.resize(frame, (sw, sh)) if scale != 1 else frame
            _, faces = det.detect(small)
            if faces is not None and len(faces):
                # biggest face = the person on camera most prominently
                x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])[:4]
                samples[idx] = (x + fw / 2) / scale
                widths.append(fw / sw)
        idx += 1
    cap.release()
    n = max(idx, 1)

    stats = {
        "coverage": len(samples) / max(sampled, 1),          # how often a face is visible
        "size": float(np.median(widths)) if widths else 0.0,  # face width / frame width
        "position": float(np.median(list(samples.values()))) / w if samples else 0.5,
    }
    if not samples:
        return np.full(n, w / 2), stats

    # interpolate between detections, then smooth so the "camera" glides instead of jitters
    keys = np.array(sorted(samples))
    vals = np.array([samples[k] for k in keys])
    centers = np.interp(np.arange(n), keys, vals)
    win = max(1, int(fps * 0.8))
    kernel = np.ones(win) / win
    padded = np.pad(centers, (win, win), mode="edge")
    centers = np.convolve(padded, kernel, mode="same")[win:-win]

    # deadzone: only move when the subject drifts noticeably
    crop_w = h * 9 / 16
    dead = crop_w * 0.08
    cam = centers[0]
    out = np.empty(n)
    for i, c in enumerate(centers):
        if abs(c - cam) > dead:
            cam += (c - cam) * 0.12
        out[i] = cam
    return out, stats


def choose_layout(stats: dict) -> str:
    """Crop to the face only for talking-head shots: a clearly visible, reasonably large,
    roughly centred face. Gameplay with a corner facecam, slides, overlays, etc. get 'fit'."""
    if stats["coverage"] >= 0.6 and stats["size"] >= 0.07 and 0.2 <= stats["position"] <= 0.8:
        return "crop"
    return "fit"


# ---------- captions (ASS subtitles) ----------

def _ts(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _esc(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")")


def caption_events(words: list[dict], style: str = "Cap") -> list[str]:
    """ASS events for word-by-word captions: short chunks, the spoken word highlighted and popped."""
    lines = []
    # group words into short chunks (max 3 words / 1.4s / gap break)
    chunks, cur = [], []
    for w in words:
        if cur and (len(cur) >= 3 or w["e"] - cur[0]["s"] > 1.4 or w["s"] - cur[-1]["e"] > 0.6):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)

    for ci, chunk in enumerate(chunks):
        chunk_end = chunk[-1]["e"]
        next_start = chunks[ci + 1][0]["s"] if ci + 1 < len(chunks) else chunk_end + 0.5
        chunk_end = min(max(chunk_end, chunk[-1]["s"] + 0.25), next_start)
        for wi, w in enumerate(chunk):
            start = w["s"]
            end = chunk[wi + 1]["s"] if wi + 1 < len(chunk) else chunk_end
            if end <= start:
                continue
            parts = []
            for j, other in enumerate(chunk):
                txt = _esc(other["w"].upper())
                if j == wi:
                    parts.append(f"{{\\c&H00E5FF&\\fscx112\\fscy112}}{txt}{{\\c&HFFFFFF&\\fscx100\\fscy100}}")
                else:
                    parts.append(txt)
            pop = "{\\t(0,80,\\fscx105\\fscy105)\\t(80,160,\\fscx100\\fscy100)}" if wi == 0 else ""
            lines.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},{style},,0,0,0,,{pop}{' '.join(parts)}")
    return lines


def build_ass(words: list[dict], title: str, width: int, height: int, layout: str, path: Path):
    """Word-by-word highlighted captions plus a hook title for the first seconds."""
    vertical = layout != "landscape"
    title = EMOJI.sub("", title).strip()  # libass can't draw colour emoji
    font, fk = _pick_font(title + "".join(w["w"] for w in words))
    k = height / 1920 if vertical else height / 1080
    font_size = int((86 if vertical else 64) * k * fk)
    # crop: lower third of the face shot; fit: under the video, just above the bottom safe line
    margin_v = int({"crop": 560, "fit": SAFE_BOTTOM, "vertical": 560}.get(layout, 90) * k)
    side = int((SAFE_SIDE if vertical else 60) * k)
    hook_size = int((78 if vertical else 54) * k * fk)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,{max(3, int(7 * k))},{max(1, int(3 * k))},2,{side},{side},{margin_v},1
Style: Hook,{font},{hook_size},&H00000000,&H00000000,&H0000E5FF,&H0000E5FF,1,0,0,0,100,100,0,0,3,{int(22 * k)},0,8,{side},{side},{int((SAFE_TOP if vertical else 60) * k)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    if title:
        lines.append(f"Dialogue: 1,{_ts(0.2)},{_ts(3.2)},Hook,,0,0,0,,{{\\fad(150,250)}}{_esc(title)}")

    lines += caption_events(words)
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


def words_in_range(segments: list[dict], start: float, end: float) -> list[dict]:
    out = []
    for s in segments:
        for w in s["words"]:
            if w["s"] >= start - 0.05 and w["e"] <= end + 0.3:
                out.append({"w": w["w"], "s": round(w["s"] - start, 3), "e": round(w["e"] - start, 3)})
    return out


# ---------- final render ----------

def _encode(work: Path, raw: Path, vf: str, out: Path):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", raw.name, "-vf", vf,
         "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
         "-c:a", "copy", "-movflags", "+faststart", str(out.resolve())],
        check=True, cwd=work,
    )


def render_clip(src: Path, clip: dict, segments: list[dict], out: Path, work: Path,
                vertical: bool = True, captions: bool = True, layout: str = "auto",
                remove_subs: bool = False, log=print) -> str:
    """Render one clip. layout: 'auto' | 'crop' (follow the face) | 'fit' (whole frame on blurred bg).
    Returns the layout actually used."""
    work.mkdir(parents=True, exist_ok=True)
    raw = cut(src, clip["start"], clip["end"], work / f"{out.stem}_raw.mp4")

    if remove_subs:
        log("Looking for subtitles in the original video…", replace=True)
        band = detect_band(raw)
        if band:
            cap = cv2.VideoCapture(str(raw))
            fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            vf, how = removal_filter(band, fw, fh)
            clean = work / f"{out.stem}_clean.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-filter_complex", vf,
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-c:a", "copy", str(clean)],
                check=True,
            )
            raw.unlink()
            raw = clean.rename(raw)
            clip["original_subs"] = how
            log(f"Original subtitles {how}", replace=True)

    cap = cv2.VideoCapture(str(raw))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    if vertical:
        out_w, out_h = 1080, 1920
    else:
        out_w, out_h = (w // 2) * 2, (h // 2) * 2

    def subtitles(layout: str) -> str:
        if not captions:
            return "null"
        ass = work / f"{out.stem}.ass"
        words = words_in_range(segments, clip["start"], clip["end"])
        build_ass(words, clip.get("title", ""), out_w, out_h, layout, ass)
        return f"subtitles={ass.name}:fontsdir={_fonts_dir()}"

    if not vertical:
        _encode(work, raw, subtitles("landscape"), out)
        return "landscape"

    if w / h <= 9 / 16 + 0.01:  # already vertical
        _encode(work, raw, f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
                           f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:black,{subtitles('vertical')}", out)
        return "vertical"

    centers = None
    if layout in ("auto", "crop"):
        log("Tracking the speaker…", replace=True)
        centers, stats = track_faces(raw)
        if layout == "auto":
            layout = choose_layout(stats)

    if layout == "fit":
        # whole frame, centred, over a blurred zoomed copy of itself
        _encode(work, raw,
                f"split[a][b];"
                f"[a]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h},"
                f"boxblur=24:2,eq=brightness=-0.12[bg];"
                f"[b]scale={out_w}:-2[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,{subtitles('fit')}", out)
        return "fit"

    crop_w = int(round(h * 9 / 16)) // 2 * 2
    subs = subtitles("crop")

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", f"{fps}", "-i", "-",
         "-i", raw.name, "-map", "0:v", "-map", "1:a?", "-vf", subs,
         "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
         "-c:a", "copy", "-shortest", "-movflags", "+faststart", str(out.resolve())],
        stdin=subprocess.PIPE, cwd=work,
    )
    cap = cv2.VideoCapture(str(raw))
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cx = centers[min(i, len(centers) - 1)]
            x0 = int(np.clip(cx - crop_w / 2, 0, w - crop_w))
            crop = frame[:, x0:x0 + crop_w]
            ff.stdin.write(cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4).tobytes())
            i += 1
    finally:
        cap.release()
        ff.stdin.close()
        ff.wait()
    if ff.returncode != 0:
        raise RuntimeError("ffmpeg failed while rendering the clip")
    return "crop"


def thumbnail(video: Path, out: Path, at: float = 1.0):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{at}", "-i", str(video),
         "-frames:v", "1", "-vf", "scale=360:-2", str(out)],
        check=True,
    )
