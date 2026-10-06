"""Terminal usage:  .venv/bin/python -m clipper <link-or-file> [--clips 5] [--landscape] [--no-captions]"""
import argparse
from pathlib import Path

from dotenv import load_dotenv

from . import OUTPUT_DIR, pipeline

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ap = argparse.ArgumentParser(description="Turn a long video into viral short clips.")
ap.add_argument("source", help="Video link (YouTube, TikTok, X, …) or a local file path")
ap.add_argument("--clips", type=int, default=5)
ap.add_argument("--layout", choices=["auto", "crop", "fit"], default="auto",
                help="Vertical layout: crop = follow the face, fit = whole frame on blurred background")
ap.add_argument("--landscape", action="store_true", help="Keep 16:9 instead of 9:16")
ap.add_argument("--no-captions", action="store_true")
ap.add_argument("--remove-subs", action="store_true", help="Remove subtitles already burned into the video")
ap.add_argument("--min", type=int, default=20, help="Min clip length (s)")
ap.add_argument("--max", type=int, default=75, help="Max clip length (s)")
a = ap.parse_args()

local = Path(a.source).expanduser()
opts = {"num_clips": a.clips, "aspect": "landscape" if a.landscape else a.layout,
        "captions": not a.no_captions, "remove_subs": a.remove_subs, "min_len": a.min, "max_len": a.max}
job = pipeline.new_job(a.source, opts)
print(f"Job {job['id']}")

_orig_save = pipeline.save
_seen = [0]


def _print_save(j):
    _orig_save(j)
    for entry in j["log"][_seen[0]:]:
        print("  " + entry["msg"])
    _seen[0] = len(j["log"])


pipeline.save = _print_save
job = pipeline.run(job, local if local.exists() else None)
print(f"\nClips saved in: {OUTPUT_DIR / job['id']}")
for c in job["clips"]:
    print(f"  [{c['score']}] {c['file']} — {c['reason']}")
