import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # must be set before cv2 is imported

import static_ffmpeg  # noqa: E402

# Make ffmpeg and deno (YouTube's JS runtime for yt-dlp) visible to every subprocess.
static_ffmpeg.add_paths(weak=True)
_venv_bin = str(Path(sys.executable).parent)
if _venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _venv_bin + os.pathsep + os.environ.get("PATH", "")

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
