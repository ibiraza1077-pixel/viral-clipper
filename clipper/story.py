"""Story videos: a topic in, a 60-90s narrated cartoon video out.

Claude writes the script, Kokoro (local) reads it, Z-Image-Turbo (local) draws each scene,
Whisper times the captions and ffmpeg puts it together. Everything runs on this Mac for free.
The voice and image models live in .venv-story and run through clipper/story_worker.py.
"""
import json
import subprocess
import wave
from pathlib import Path

from pydantic import BaseModel, Field

from . import ROOT
from .picker import ask
from .render import EMOJI, _esc, _fonts_dir, _pick_font, _ts, caption_events
from .transcribe import transcribe

STORY_PYTHON = ROOT / ".venv-story" / "bin" / "python"
WORKER = Path(__file__).resolve().parent / "story_worker.py"

W, H = 1080, 1920
IMG_W, IMG_H = 1080, 1200          # on-screen size of each scene picture
GEN_W, GEN_H = 576, 640            # generated size (multiples of 16, same shape), upscaled for display
IMG_Y = 530                        # picture sits under the title block
GAP = 0.25                         # silence between scenes, seconds
FPS = 30
BG = "#0e3b1f"                     # default background: dark pitch green
SERIES_DIR = ROOT / "series"       # one JSON per series: its locked-in character, colour and past episodes

# The "explainer" look of the big "every level" channels: bald round heads, dot eyes, thick outlines.
STYLE = ("Simple 2D explainer cartoon in the style of animated YouTube infographic videos: characters with bald "
         "round heads, small dot eyes, simple eyebrows, plain flat-coloured clothes, thick black outlines, flat "
         "colours, light hand-drawn texture, plain detailed backgrounds.")
IMAGE_MODEL = "mflux-community/z-image-turbo-mflux-q4"  # 4-bit fits in 16 GB without swapping

VOICES = {"bm_george": "George (British, warm)", "bm_lewis": "Lewis (British, deep)",
          "bm_daniel": "Daniel (British, quick)", "bf_emma": "Emma (British)"}


class Beat(BaseModel):
    text: str = Field(description="4-12 consecutive spoken words, exactly as read aloud. A scene's beats joined "
                                  "with spaces form its full narration.")
    image_prompt: str = Field(description="The picture shown while these exact words are spoken. Illustrate them "
                                          "literally. Concrete and visual. Write 'the hero' for the main character.")
    reveal: str = Field(description="Big on-screen text when a key number lands, max 4 words, e.g. '£50 A MATCH'. "
                                    "Use on about one beat per scene; empty string otherwise.")


class Scene(BaseModel):
    beats: list[Beat]

    @property
    def narration(self) -> str:
        return " ".join(b.text.strip() for b in self.beats)


class StoryScript(BaseModel):
    series_title: str = Field(description="Series name for the title bar, max 40 characters")
    episode_title: str = Field(description="What this part covers, max 32 characters, e.g. 'Sunday League to Semi-Pro'")
    character: str = Field(description="A short fixed visual description of the main character, reused in every "
                                       "picture: bald round head (every character is bald), build, and a distinctive "
                                       "outfit for this job with clear colours. Keep it under 25 words.")
    background: str = Field(default=BG, description="Hex colour for the video background behind the pictures: a dark, rich colour "
                                        "that suits the job, e.g. '#0e3b1f' pitch green for football, '#0b2545' navy "
                                        "for aviation, '#0f3b3a' hospital teal for medicine.")
    scenes: list[Scene]
    caption: str = Field(description="Ready-to-post caption, 1-2 lines, ends with a question to drive comments")
    hashtags: list[str] = Field(description="5-7 hashtags without the # symbol")


PROMPT = """You write scripts for a faceless TikTok / YouTube Shorts / Instagram Reels channel.

Series: {series}
This video: Part {part}: {topic}
{continuity}{notes}
Format: "Every Level" videos. The viewer climbs a ladder of levels, and each level reveals what life is really like
and what it pays. Reference: "Your Life at Every Level in McDonald's", a series with millions of views.

Write a 60-90 second narration (170-230 words in total) split into 8-11 scenes (one per level or idea).
Split every scene into beats of 4-12 words. Each beat gets its own picture, so the picture changes every
2-3 seconds, like a picture book that keeps up with the narrator. Aim for 30-40 beats in total.

Rules:
- Scene 1 is the hook. In the first sentence, give a surprising number or a bold claim that makes people stay.
- Second person ("you") throughout, present tense, British English, a dry and witty tone. Short, punchy sentences.
- Every level needs a concrete money figure or a hard stat, with "around" or "about" where it varies. Use
  realistic, current UK figures. If research notes are given below, base every number on them. Never invent
  precise facts about named real people.
- Something new every scene: a number, a twist or a reality check. No filler.
- The last scene ends with a cliffhanger into the next part and asks viewers to comment.
- Narration is read aloud by a text-to-speech voice: write numbers the way they should be spoken
  ("fifty quid a match", "one point two million a year"), and use no symbols, emoji or abbreviations.
- Pictures illustrate the exact words of their beat. "Fifty quid a match" shows a hand holding a fifty pound
  note; "three hour coach trip" shows a clock and a coach on a motorway. Mix it up: about half the beats show
  the hero, the rest are close-ups of objects, places, documents, crowds or other people.
- A picture may contain at most 3 short words of text, only when it's naturally on an object (a price tag, a
  sign, a contract, a payslip). Put that exact text in quotes in the image_prompt.
- Describe every picture fully on its own (it's drawn without seeing the others).
"""


def series_path(series: str) -> Path:
    slug = "".join(c if c.isalnum() else "-" for c in series.lower()).strip("-")[:60] or "series"
    return SERIES_DIR / f"{slug}.json"


def load_series(series: str) -> dict:
    p = series_path(series)
    return json.loads(p.read_text()) if p.exists() else {}


def save_series(series: str, script: "StoryScript", part: int, topic: str):
    """Lock in the look after the first part, and remember every episode so later parts continue the story."""
    SERIES_DIR.mkdir(exist_ok=True)
    data = load_series(series)
    data.setdefault("character", script.character)
    data.setdefault("background", script.background)
    eps = {e["part"]: e for e in data.get("episodes", [])}
    eps[part] = {"part": part, "topic": topic, "title": script.episode_title}
    data["episodes"] = sorted(eps.values(), key=lambda e: e["part"])
    series_path(series).write_text(json.dumps(data, indent=2))


def write_script(series: str, topic: str, part: int, notes: str = "") -> StoryScript:
    known = load_series(series)
    continuity = ""
    if known.get("character"):
        continuity += (f"\nThe main character is already established; use exactly this description: "
                       f"{known['character']}\nBackground colour is already set: {known['background']}\n")
    earlier = [e for e in known.get("episodes", []) if e["part"] < part]
    if earlier:
        continuity += ("Earlier parts (pick up where the last one left off, and don't repeat their levels):\n"
                       + "".join(f"- Part {e['part']}: {e['title']} ({e['topic']})\n" for e in earlier))
    notes = f"\nResearch notes and creator notes:\n{notes.strip()}\n" if notes.strip() else ""
    script = ask(PROMPT.format(series=series, topic=topic, part=part, continuity=continuity, notes=notes),
                 StoryScript)
    if known.get("character"):  # never let a later part drift from the locked-in look
        script.character, script.background = known["character"], known["background"]
    return script


# ---------- local models ----------

def run_worker(job: dict, work: Path, log, stage=None) -> None:
    spec = work / "worker_job.json"
    spec.write_text(json.dumps(job))
    # caffeinate keeps the Mac from idle-sleeping mid-job (closing the lid on battery still sleeps it)
    proc = subprocess.Popen(["caffeinate", "-i", str(STORY_PYTHON), "-u", str(WORKER), str(spec)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = []
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("PROGRESS"):
            _, what, n = line.split()
            log({"voice": "Recording voiceover", "image": "Drawing picture"}[what] + f" {n}…", replace=True)
            if stage and what == "image":  # pictures are the slow part, so they drive the progress bar
                done, total = map(int, n.split("/"))
                stage("drawing", 35 + int(50 * done / total))
        elif line:
            tail = (tail + [line])[-15:]
    if proc.wait() != 0:
        raise RuntimeError("Voice/image generation failed:\n" + "\n".join(tail[-6:]))


def wav_seconds(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


# ---------- assembly ----------

def join_split_words(words: list[dict]) -> list[dict]:
    """Whisper splits "1,000" into "1" + ",000" and "90%" into "90" + "%"; put them back together."""
    out = []
    for w in words:
        if out and w["w"][:1] in ",.%'’" and w["s"] - out[-1]["e"] < 0.3:
            out[-1] = {**out[-1], "w": out[-1]["w"] + w["w"], "e": w["e"]}
        else:
            out.append(w)
    return out


def build_story_ass(script: StoryScript, part: int, words: list[dict], reveals: list[tuple], path: Path):
    title = EMOJI.sub("", script.series_title).strip()
    font, fk = _pick_font(title + " ".join(w["w"] for w in words))
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Title,{font},{int(76 * fk)},&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,1,0,0,0,100,100,0,0,3,18,0,8,110,110,322,1
Style: Part,{font},{int(42 * fk)},&H00000000,&H00000000,&H0000E5FF,&H0000E5FF,1,0,0,0,100,100,0,0,3,12,0,8,150,150,246,1
Style: Cap,{font},{int(86 * fk)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,7,3,2,150,150,560,1
Style: Reveal,{font},{int(118 * fk)},&H0000E5FF,&H0000E5FF,&H00000000,&H90000000,1,0,0,0,100,100,0,0,1,10,5,5,120,120,0,1
"""
    end = words[-1]["e"] + 1.5 if words else 60
    lines = [
        f"Dialogue: 2,{_ts(0)},{_ts(end)},Title,,0,0,0,,{_esc(title)}",
        f"Dialogue: 2,{_ts(0)},{_ts(end)},Part,,0,0,0,,{_esc(f'PART {part} · {script.episode_title}'.upper())}",
    ]
    for start, stop, text in reveals:  # big number pops in over the picture, top third
        lines.append(f"Dialogue: 1,{_ts(start + 0.3)},{_ts(stop)},Reveal,,0,0,0,,"
                     f"{{\\pos({W // 2},{IMG_Y + 140})\\fscx40\\fscy40\\t(0,180,\\fscx110\\fscy110)"
                     f"\\t(180,300,\\fscx100\\fscy100)\\fad(0,200)}}{_esc(text.upper())}")
    lines += caption_events(words)
    path.write_text(header + "\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
                    + "\n".join(lines) + "\n", encoding="utf-8")


MOVES = ["in", "right", "out", "left"]  # rotate the camera move so consecutive shots never feel the same


def shot_clip(image: Path, seconds: float, out: Path, move: str):
    """A gentle camera move on one still: push in, pull out, or a slow pan. Upscaled 3x first so
    zoompan's whole-pixel steps are a third of an output pixel, which keeps the motion smooth."""
    frames = max(2, round(seconds * FPS))
    zmax = 1.08
    if move == "in":
        z, x = f"1+{zmax - 1}*on/{frames}", "iw/2-(iw/zoom/2)"
    elif move == "out":
        z, x = f"{zmax}-{zmax - 1}*on/{frames}", "iw/2-(iw/zoom/2)"
    elif move == "right":
        z, x = f"{zmax}", f"(iw-iw/zoom)*on/{frames}"
    else:
        z, x = f"{zmax}", f"(iw-iw/zoom)*(1-on/{frames})"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(image), "-vf",
         f"scale={IMG_W * 3}:{IMG_H * 3}:flags=lanczos,zoompan=z='{z}':x='{x}':y='ih/2-(ih/zoom/2)'"
         f":d={frames}:s={IMG_W}x{IMG_H}:fps={FPS}",
         "-frames:v", str(frames), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
         "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )


def beat_times(script: "StoryScript", scene_starts: list[float], scene_lens: list[float],
               words: list[dict]) -> list[float]:
    """When each beat's picture should appear. Within a scene, time is shared out by how much text each beat
    has, then each cut is moved onto the nearest word Whisper heard starting, so pictures change on a word."""
    starts = [w["s"] for w in words]
    out = []
    for sc, t0, length in zip(script.scenes, scene_starts, scene_lens):
        sizes = [len(b.text) + 1 for b in sc.beats]
        acc = 0
        for i, size in enumerate(sizes):
            t = t0 + length * acc / sum(sizes)
            if i and starts:
                near = min(starts, key=lambda s: abs(s - t))
                t = near - 0.05 if abs(near - t) < 0.7 else t
            out.append(max(t, out[-1] + 0.4) if out else 0.0)
            acc += size
    return out


def _bg_colour(value: str) -> str:
    v = (value or "").strip().lstrip("#")
    return "0x" + v if len(v) == 6 and all(c in "0123456789abcdefABCDEF" for c in v) else BG.replace("#", "0x")


def assemble(script: StoryScript, part: int, work: Path, out: Path, log):
    n = len(script.scenes)
    durations = [wav_seconds(work / f"v{i:02d}.wav") + GAP for i in range(n)]

    log("Joining the voiceover…", replace=True)
    inputs, filt = [], []
    for i in range(n):
        inputs += ["-i", str(work / f"v{i:02d}.wav")]
        filt.append(f"[{i}:a]aresample=48000,apad=pad_dur={GAP}[a{i}]")
    filt.append("".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1,loudnorm=I=-14:TP=-1.5[out]")
    narration = work / "narration.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(filt),
                    "-map", "[out]", "-ar", "48000", str(narration)], check=True)

    log("Timing the captions…", replace=True)
    segs = transcribe(narration)["segments"]
    words = join_split_words([w for s in segs for w in s["words"]])

    scene_lens = [d - GAP for d in durations]
    scene_starts = [sum(durations[:i]) for i in range(n)]
    total = sum(durations) + 0.6
    beats = [b for sc in script.scenes for b in sc.beats]
    times = beat_times(script, scene_starts, scene_lens, words) + [total]

    reveals = [(times[i], times[i + 1] - 0.05, b.reveal.strip()) for i, b in enumerate(beats) if b.reveal.strip()]
    ass = work / "story.ass"
    build_story_ass(script, part, words, reveals, ass)

    # cut at whole frames so the shots add up exactly to the narration
    frames = [round(t * FPS) for t in times]
    for i in range(len(beats)):
        log(f"Animating shot {i + 1}/{len(beats)}…", replace=True)
        shot_clip(work / f"i{i:02d}.png", (frames[i + 1] - frames[i]) / FPS, work / f"s{i:02d}.mp4",
                  MOVES[i % len(MOVES)])
    (work / "scenes.txt").write_text("".join(f"file 's{i:02d}.mp4'\n" for i in range(len(beats))))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", "scenes.txt",
                    "-c", "copy", "pictures.mp4"], check=True, cwd=work)

    log("Rendering the final video…", replace=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"color=c={_bg_colour(script.background)}:s={W}x{H}:r={FPS}:d={total:.2f}",
         "-i", "pictures.mp4", "-i", narration.name,
         "-filter_complex",
         f"[0:v]vignette=PI/4[bg];[1:v]pad={IMG_W + 12}:{IMG_H + 12}:6:6:white[pic];"
         f"[bg][pic]overlay=(W-w)/2:{IMG_Y - 6}:eof_action=repeat,"
         f"subtitles={ass.name}:fontsdir={_fonts_dir()}[v]",
         "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
         str(out.resolve())],
        check=True, cwd=work,
    )
    return round(total, 1)


def make_story(job: dict, d: Path, log, stage) -> dict:
    """Runs one story job end to end. Returns the clip entry for job['clips']."""
    opts = job["options"]
    work = d / "work"
    work.mkdir(parents=True, exist_ok=True)
    part = int(opts.get("part", 1))

    stage("writing", 10)
    spath = d / "script.json"
    if spath.exists():  # re-runs reuse the script instead of writing a new one
        script = StoryScript.model_validate_json(spath.read_text())
    else:
        log("Writing the script with Claude…")
        script = write_script(opts["series"], job["source"], part, opts.get("notes", ""))
        spath.write_text(script.model_dump_json(indent=2))
    save_series(opts["series"], script, part, job["source"])
    job["title"] = f"{script.series_title} · Part {part}: {script.episode_title}"
    beats = [b for sc in script.scenes for b in sc.beats]
    log(f"Script ready: {len(script.scenes)} scenes, {len(beats)} shots, "
        f"{sum(len(s.narration.split()) for s in script.scenes)} words")

    stage("voicing", 25)
    voice = opts.get("voice", "bm_george")
    n = len(script.scenes)
    style = f"{STYLE} The main character, 'the hero', is {script.character}."
    worker_job = {
        "voice": voice, "speed": float(opts.get("speed", 1.1)),
        "tts": [{"text": s.narration, "out": str(work / f"v{i:02d}.wav")} for i, s in enumerate(script.scenes)],
        "width": GEN_W, "height": GEN_H, "steps": 5, "image_model": IMAGE_MODEL,
        # one seed for every picture keeps the drawing style and character steadier across scenes
        "images": [{"prompt": f"{style} {b.image_prompt}", "out": str(work / f"i{i:02d}.png"), "seed": 4242}
                   for i, b in enumerate(beats)],
    }
    log(f"Recording the voice and drawing {len(beats)} pictures on this Mac…")
    stage("drawing", 35)
    run_worker(worker_job, work, log, stage)

    stage("rendering", 85)
    slug = "".join(c if c.isalnum() else "-" for c in script.episode_title.lower()).strip("-")[:40]
    out = d / f"part-{part}-{slug}.mp4"
    duration = assemble(script, part, work, out, log)
    return {
        "file": out.name, "title": f"Part {part}: {script.episode_title}", "score": None,
        "reason": f"{len(beats)} shots · voice: {VOICES.get(voice, voice)}", "caption": script.caption,
        "hashtags": script.hashtags, "duration": duration, "start": 0, "end": duration,
    }
