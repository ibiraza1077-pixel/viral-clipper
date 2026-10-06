"""End-to-end job: link or file in, ready-to-post clips out."""
import json
import re
import shutil
import time
import traceback
import uuid
from pathlib import Path

from . import OUTPUT_DIR
from .download import download, extract_audio, probe
from .picker import backend_name, pick_clips
from .render import render_clip, thumbnail
from .transcribe import transcribe

STAGES = ["queued", "downloading", "transcribing", "picking", "rendering", "done"]
STORY_STAGES = ["queued", "writing", "voicing", "drawing", "rendering", "done"]


def new_job(source: str, options: dict, batch: str | None = None) -> dict:
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    job = {
        "id": job_id, "source": source, "options": options, "status": "queued", "batch": batch,
        "title": None, "log": [], "clips": [], "error": None,
        "created": time.time(), "progress": 0,
    }
    (OUTPUT_DIR / job_id).mkdir(parents=True)
    save(job)
    return job


def save(job: dict):
    (OUTPUT_DIR / job["id"] / "job.json").write_text(json.dumps(job, indent=2))


def load(job_id: str) -> dict | None:
    p = OUTPUT_DIR / job_id / "job.json"
    return json.loads(p.read_text()) if p.exists() else None


def list_jobs() -> list[dict]:
    jobs = []
    for p in OUTPUT_DIR.glob("*/job.json"):
        try:
            jobs.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            pass
    return sorted(jobs, key=lambda j: -j["created"])


def mark_interrupted():
    """Jobs left unfinished by a previous server run will never resume; say so instead of hanging."""
    for j in list_jobs():
        if j["status"] not in ("done", "error"):
            j["status"], j["error"] = "error", "Interrupted: the app was closed or the Mac went to sleep before this finished. Story videos can be retried; submit clip jobs again."
            save(j)


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()[:50] or "clip"


def _reporters(job: dict):
    def log(msg, replace=False):
        if replace and job["log"] and job["log"][-1].get("replace"):
            job["log"][-1] = {"t": time.time(), "msg": msg, "replace": True}
        else:
            job["log"].append({"t": time.time(), "msg": msg, "replace": replace})
        save(job)

    def stage(name, progress):
        job["status"], job["progress"] = name, progress
        save(job)

    return log, stage


def _fail(job: dict, d: Path, log, e: Exception):
    job["status"], job["error"] = "error", str(e)
    log(f"Error: {e}")
    (d / "error.log").write_text(traceback.format_exc())
    save(job)


def run_story(job: dict):
    """Story video: topic in, narrated cartoon video out (see clipper/story.py)."""
    from .story import make_story

    d = OUTPUT_DIR / job["id"]
    log, stage = _reporters(job)
    try:
        clip = make_story(job, d, log, stage)
        thumbnail(d / clip["file"], d / (Path(clip["file"]).stem + ".jpg"), at=2.0)
        clip["thumb"] = Path(clip["file"]).stem + ".jpg"
        job["clips"] = [clip]
        stage("done", 100)
        log("All done ✨")
    except Exception as e:  # surface every failure to the UI
        _fail(job, d, log, e)
    return job


def run(job: dict, uploaded: Path | None = None):
    d = OUTPUT_DIR / job["id"]
    work = d / "work"
    opts = job["options"]
    log, stage = _reporters(job)

    try:
        # 1. get the video
        stage("downloading", 5)
        if uploaded:
            src, title = uploaded, uploaded.stem
            log(f"Using uploaded file {uploaded.name}")
        else:
            log(f"Downloading {job['source']}")
            src, title = download(job["source"], work, log=log)
        job["title"] = title
        info = probe(src)
        log(f"Got “{title}” — {info['duration'] / 60:.1f} min, {info['width']}×{info['height']}")

        # 2. transcribe (cached per job)
        stage("transcribing", 20)
        tpath = d / "transcript.json"
        if tpath.exists():
            transcript = json.loads(tpath.read_text())
        else:
            log("Transcribing with Whisper (first run downloads the model, ~1.6 GB)…")
            audio = extract_audio(src, work / "audio.wav")
            transcript = transcribe(audio)
            tpath.write_text(json.dumps(transcript))
        segs = transcript["segments"]
        if not segs:
            raise RuntimeError("No speech found in this video — clips are picked from what's said.")
        log(f"Transcribed {sum(len(s['words']) for s in segs)} words ({transcript.get('language')})")

        # 3. pick viral moments
        stage("picking", 45)
        log(f"Finding the most viral moments with {backend_name()}…")
        clips = pick_clips(
            segs, title, info["duration"], n=int(opts.get("num_clips", 5)),
            min_len=int(opts.get("min_len", 20)), max_len=int(opts.get("max_len", 75)),
            instructions=opts.get("instructions", ""), log=log,
        )
        log(f"Picked {len(clips)} clips")

        # 4. render
        stage("rendering", 55)
        aspect = opts.get("aspect", "auto")
        vertical = aspect != "landscape"
        layout = aspect if aspect in ("auto", "crop", "fit") else "auto"
        for i, clip in enumerate(clips, 1):
            name = f"{i:02d}-{_slug(clip['title'])}"
            log(f"Rendering clip {i}/{len(clips)}: {clip['title']}")
            out = d / f"{name}.mp4"
            clip["layout"] = render_clip(src, clip, segs, out, work, vertical=vertical, layout=layout,
                                         captions=opts.get("captions", True),
                                         remove_subs=opts.get("remove_subs", False), log=log)
            thumbnail(out, d / f"{name}.jpg")
            clip.update({"file": out.name, "thumb": f"{name}.jpg", "duration": round(clip["end"] - clip["start"], 1)})
            job["clips"].append(clip)
            job["progress"] = 55 + int(45 * i / len(clips))
            save(job)

        stage("done", 100)
        log("All done ✨")
    except Exception as e:  # surface every failure to the UI
        _fail(job, d, log, e)
    finally:
        # keep the downloaded source for re-runs only if asked; raw cuts are always disposable
        for p in work.glob("*_raw.mp4"):
            p.unlink(missing_ok=True)
        if not opts.get("keep_source"):
            shutil.rmtree(work, ignore_errors=True)
    return job
