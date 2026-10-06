import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from clipper import OUTPUT_DIR, ROOT  # noqa: E402
from clipper import pipeline  # noqa: E402
from clipper.download import expand  # noqa: E402
from clipper.picker import backend_name  # noqa: E402
from clipper.story import STORY_PYTHON  # noqa: E402

app = FastAPI(title="Viral Clipper")
worker = ThreadPoolExecutor(max_workers=1)  # one video at a time keeps the Mac responsive
pipeline.mark_interrupted()


class JobRequest(BaseModel):
    url: str
    num_clips: int = 5
    aspect: str = "auto"
    captions: bool = True
    remove_subs: bool = False
    min_len: int = 20
    max_len: int = 75
    instructions: str = ""


class BatchRequest(JobRequest):
    url: str = ""
    urls: list[str]
    per_channel: int = 5  # how many of the latest videos to take from a channel/playlist link


def _options(r) -> dict:
    return {
        "num_clips": max(1, min(15, r.num_clips)), "aspect": r.aspect, "captions": r.captions, "remove_subs": r.remove_subs,
        "min_len": r.min_len, "max_len": max(r.min_len + 5, r.max_len), "instructions": r.instructions[:4000],
    }


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/status")
def status():
    return {"ai": backend_name()}


@app.post("/api/jobs")
def create_job(req: JobRequest):
    if not req.url.startswith(("http://", "https://")):
        raise HTTPException(400, "Paste a full link starting with https://")
    job = pipeline.new_job(req.url, _options(req))
    worker.submit(pipeline.run, job)
    return job


@app.post("/api/batch")
def create_batch(req: BatchRequest):
    """Queue many links at once. Channel and playlist links expand to their latest videos."""
    links = [u.strip() for u in req.urls if u.strip()]
    bad = [u for u in links if not u.startswith(("http://", "https://"))]
    if bad:
        raise HTTPException(400, f"Not a link: {bad[0][:80]}")
    if not links:
        raise HTTPException(400, "Paste at least one link")
    videos, seen = [], set()
    for link in links:
        try:
            found = expand(link, max(1, min(25, req.per_channel)))
        except Exception as e:
            raise HTTPException(400, f"Couldn't read {link[:80]}: {str(e)[:200]}")
        videos += [v for v in found if not (v in seen or seen.add(v))]
    if len(videos) > 50:
        raise HTTPException(400, f"That's {len(videos)} videos. Queue 50 or fewer at a time.")
    batch = uuid.uuid4().hex[:6]
    jobs = []
    for v in videos:
        job = pipeline.new_job(v, _options(req), batch=batch)
        worker.submit(pipeline.run, job)
        jobs.append(job)
    return {"batch": batch, "jobs": jobs}


class StoryRequest(BaseModel):
    topic: str
    series: str = "Every Level of a Footballer's Career"
    part: int = 1
    voice: str = "bm_george"
    notes: str = ""


@app.post("/api/story")
def create_story(req: StoryRequest):
    if not req.topic.strip():
        raise HTTPException(400, "Say what this part is about, e.g. 'Sunday league to semi-pro'")
    if not STORY_PYTHON.exists():
        raise HTTPException(400, "Story videos need the .venv-story environment (see README)")
    opts = {"kind": "story", "series": req.series.strip()[:80], "part": max(1, min(50, req.part)),
            "voice": req.voice, "notes": req.notes[:2000], "aspect": "vertical"}
    job = pipeline.new_job(req.topic.strip()[:200], opts)
    worker.submit(pipeline.run_story, job)
    return job


@app.post("/api/jobs/{job_id}/retry")
def retry(job_id: str):
    """Re-run a failed or interrupted story job. Its script, voice and finished pictures are reused."""
    j = pipeline.load(job_id)
    if not j:
        raise HTTPException(404, "Job not found")
    if j["options"].get("kind") != "story" or j["status"] != "error":
        raise HTTPException(400, "Only failed story videos can be retried")
    j.update(status="queued", error=None, progress=0)
    j["log"].append({"t": time.time(), "msg": "Retrying, reusing everything already made…", "replace": False})
    pipeline.save(j)
    worker.submit(pipeline.run_story, j)
    return j


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), num_clips: int = Form(5), aspect: str = Form("auto"),
                 captions: bool = Form(True), remove_subs: bool = Form(False), min_len: int = Form(20), max_len: int = Form(75),
                 instructions: str = Form("")):
    req = JobRequest(url=file.filename or "upload", num_clips=num_clips, aspect=aspect,
                     captions=captions, remove_subs=remove_subs, min_len=min_len, max_len=max_len,
                     instructions=instructions)
    job = pipeline.new_job(file.filename or "upload", _options(req))
    work = OUTPUT_DIR / job["id"] / "work"
    work.mkdir(parents=True, exist_ok=True)
    dest = work / Path(file.filename or "upload.mp4").name
    with dest.open("wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    worker.submit(pipeline.run, job, dest)
    return job


@app.get("/api/jobs")
def jobs():
    return pipeline.list_jobs()


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    j = pipeline.load(job_id)
    if not j:
        raise HTTPException(404, "Job not found")
    return j


app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")
