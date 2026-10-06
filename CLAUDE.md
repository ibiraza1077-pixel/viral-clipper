# Viral Clipper

Local Mac app (Apple Silicon) that turns long videos into vertical short clips with animated captions, and makes AI-narrated cartoon "story" videos. FastAPI backend + single-page UI. Everything runs on this Mac; the only paid/remote part is Claude picking clips and writing scripts.

## Run

- App: `.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8765` (or the `viral-clipper` entry in `.claude/launch.json`; users double-click `Start Viral Clipper.command`).
- CLI: `.venv/bin/python -m clipper <url-or-file> --clips 5` (see `clipper/__main__.py` for flags).
- Tests: `.venv/bin/python -m pytest -q` (fast, no network or models; they use a temp output folder). Add tests for new logic. CI (`.github/workflows/ci.yml`) runs them plus a Semgrep scan on every push.
- Tests don't cover downloading, Whisper or rendering. For those, run a short job end to end and check `output/<job-id>/job.json` and the `.mp4`.
- Dev tools: `uv pip install --python .venv/bin/python -r requirements-dev.txt`.

## Two Python environments — never mix them

- `.venv` — the clipper and web app (fastapi, yt-dlp, mlx-whisper, opencv, anthropic). Pinned in `requirements.txt`.
- `.venv-story` — heavy local models for story videos (mflux / Z-Image-Turbo, mlx-audio / Kokoro). Pinned in `requirements-story.txt`. Only `clipper/story_worker.py` runs here, as a subprocess that prints `PROGRESS ...` lines; the main app never imports these packages.
- Install with `uv pip install --python <venv>/bin/python ...`.

## Layout

- `app.py` — HTTP API. One-worker `ThreadPoolExecutor`: jobs run strictly one at a time on purpose (16 GB Mac).
- `clipper/pipeline.py` — job lifecycle. A job is a folder `output/<id>/` holding `job.json`; there is no database. `save()` is called on every log line so the UI can poll it. Unfinished jobs are marked interrupted on startup.
- Clip flow: `download.py` (yt-dlp, channel/playlist expansion) → `transcribe.py` (Whisper, word timestamps) → `picker.py` (Claude picks clips as structured output) → `render.py` (ffmpeg cut, YuNet face tracking, ASS captions) + `subtitles.py` (detect/remove burned-in subs).
- Story flow: `story.py` (Claude writes script → worker voices + draws → Whisper times words → ffmpeg assembles). `series/*.json` stores each series' locked character, colour and past episodes so parts stay consistent — this is user data, keep it.
- `static/index.html` — the whole UI, no build step.

## Conventions

- AI backend (`picker.ask`): uses `ANTHROPIC_API_KEY` from `.env` if set, otherwise shells out to the local `claude` CLI. Keep both paths working. Model id lives in `picker.MODEL`.
- Caption/title text must stay inside the safe area constants in `render.py` (platform UI overlays).
- Memory is the main constraint: load one model at a time, release it, call `mx.clear_cache()` between images.
- Errors inside a job must be caught and written to the job (`_fail`) so the UI shows them, never crash the worker.
- User-facing messages are plain, friendly English.

## Don't commit

`.venv*`, `output/`, `.env`, `fonts/` and `models/` (both recreated at runtime). See `.gitignore`.
