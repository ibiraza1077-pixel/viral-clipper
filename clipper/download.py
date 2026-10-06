import re
import shutil
import subprocess
import time
from pathlib import Path

import yt_dlp


def download(url: str, dest_dir: Path, log=print) -> tuple[Path, str]:
    """Download a video with yt-dlp. Returns (path, title)."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    def hook(d):
        if d["status"] == "downloading":
            pct = re.sub(r"\x1b\[[0-9;]*m", "", d.get("_percent_str", "")).strip()
            if pct:
                log(f"Downloading… {pct}", replace=True)

    opts = {
        "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(dest_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "progress_hooks": [hook],
        "ffmpeg_location": shutil.which("ffmpeg"),
    }
    # YouTube hands out the odd 403 / throttle when many videos are fetched in a row; it clears on retry
    for attempt in range(1, 4):
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
            break
        except yt_dlp.utils.DownloadError as e:
            if attempt == 3 or not re.search(r"403|429|Forbidden|timed out|Connection|throttl", str(e)):
                raise
            for p in dest_dir.glob("source.*"):
                p.unlink(missing_ok=True)
            log(f"Download blocked ({str(e)[:60]}…), retrying {attempt + 1}/3 in {20 * attempt}s")
            time.sleep(20 * attempt)
    path = next(dest_dir.glob("source.*"))
    return path, info.get("title") or "video"


_LISTY = re.compile(r"[?&]list=|/playlist|youtube\.com/(@|channel/|c/|user/)|tiktok\.com/@[^/]+/?$|twitch\.tv/[^/]+/videos")


def expand(url: str, limit: int = 5) -> list[str]:
    """Turn a channel or playlist link into links to its latest `limit` videos.
    Any other link comes back unchanged."""
    if not _LISTY.search(url):
        return [url]
    if re.search(r"youtube\.com/(@[^/?]+|channel/[^/?]+|c/[^/?]+|user/[^/?]+)/?$", url):
        url = url.rstrip("/") + "/videos"  # channel home lists tabs, not videos
    opts = {"extract_flat": "in_playlist", "playlistend": limit, "quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if info.get("_type") != "playlist":
        return [info.get("webpage_url") or url]
    out = []
    for e in info.get("entries") or []:
        link = e.get("url") or e.get("webpage_url")
        if link and not link.startswith("http") and e.get("id"):
            link = f"https://www.youtube.com/watch?v={e['id']}"
        if link:
            out.append(link)
    return out[:limit]


def extract_audio(video: Path, out: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
         "-vn", "-ac", "1", "-ar", "16000", str(out)],
        check=True,
    )
    return out


def probe(video: Path) -> dict:
    """Width, height, fps, duration via OpenCV (avoids needing ffprobe)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return {"width": w, "height": h, "fps": fps, "duration": frames / fps if fps else 0}
