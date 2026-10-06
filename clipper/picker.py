"""Ask Claude which moments of the transcript have the best shot at going viral."""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from pydantic import BaseModel, Field

MODEL = "claude-opus-5-5"


class Clip(BaseModel):
    start: float = Field(description="Clip start time in seconds, taken from the transcript timestamps")
    end: float = Field(description="Clip end time in seconds")
    title: str = Field(description="Short punchy on-screen hook title, max 8 words")
    score: int = Field(description="Viral potential 0-100")
    reason: str = Field(description="One or two sentences on why this clip will perform")
    caption: str = Field(description="Ready-to-post social caption, 1-2 lines")
    hashtags: list[str] = Field(description="4-6 relevant hashtags without the # symbol")


class ClipPlan(BaseModel):
    clips: list[Clip]


PROMPT = """You are an expert short-form video editor who has grown multiple TikTok, YouTube Shorts and Instagram Reels accounts to millions of followers.

Below is a timestamped transcript of a video titled "{title}" ({duration:.0f} seconds long). Each line is `[start-end] text`, in seconds.

Pick the {n} moments most likely to go viral as standalone short clips.

What makes a great clip:
- The first 1-3 seconds hook the viewer: a bold claim, a question, conflict, surprise, or a strong emotion. Never start mid-thought or on filler ("so", "um", "and yeah").
- It is self-contained: someone who never saw the full video understands it.
- It has a payoff: a punchline, insight, reveal, story resolution or quotable line. End right after the payoff, not trailing off.
- High-engagement themes: controversial or contrarian takes, relatable struggles, money, relationships, funny moments, surprising facts, actionable advice, emotional stories.
- Length {min_len}-{max_len} seconds. Shorter is better when the idea is complete.

Rules:
- start and end must line up with the transcript timestamps (start at the beginning of a line, end at the end of a line).
- Clips must not overlap.
- Order clips from highest to lowest score. Be honest with scores; not every clip is a 90.
- Write the title, caption and hashtags in the same language as the video.
{instructions}
Transcript:
{transcript}
"""


def format_transcript(segments: list[dict]) -> str:
    return "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments)


def build_prompt(segments, title, duration, n, min_len, max_len, instructions="") -> str:
    if instructions.strip():
        instructions = ("\nCampaign rules from the person posting these clips. Follow them exactly; "
                        "they override the guidance above:\n" + instructions.strip() + "\n")
    return PROMPT.format(
        title=title, duration=duration, n=n, min_len=min_len, max_len=max_len,
        instructions=instructions, transcript=format_transcript(segments),
    )


class SetupError(RuntimeError):
    """A problem retrying won't fix (not logged in, no AI configured)."""


def _pick_via_api(prompt: str, schema=ClipPlan):
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
        output_format=schema,
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise RuntimeError(f"Claude API returned no usable answer (stop_reason={response.stop_reason})")
    return response.parsed_output


def _pick_via_claude_code(prompt: str, schema=ClipPlan):
    """Uses the locally installed Claude Code CLI (your Claude subscription, no API key)."""
    claude = shutil.which("claude") or str(Path.home() / ".local/bin/claude")
    proc = subprocess.run(
        [claude, "-p", "--output-format", "json",
         "--json-schema", json.dumps(schema.model_json_schema())],
        input=prompt, capture_output=True, text=True, timeout=900,
    )
    # the result is one JSON object, but warnings can land on stdout around it
    lines = [ln for ln in proc.stdout.splitlines() if ln.lstrip().startswith("{")]
    try:
        data = json.loads(lines[-1] if lines else proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"Claude Code failed: {proc.stderr.strip() or proc.stdout.strip()[:500]}")
    if data.get("is_error"):
        msg = str(data.get("result", "unknown error"))
        if any(k in msg.lower() for k in ("login", "authenticat", "oauth")):
            msg += (" — open Terminal, run `claude`, type /login once, then try again. "
                    "(Or put an ANTHROPIC_API_KEY in the .env file.)")
            raise SetupError(f"Claude Code: {msg}")
        raise RuntimeError(f"Claude Code: {msg}")
    if data.get("structured_output"):
        return schema.model_validate(data["structured_output"])
    text = data.get("result", "")
    start, end = text.find("{"), text.rfind("}")
    if start == -1:
        raise RuntimeError(f"Claude Code returned no JSON: {text[:300]}")
    return schema.model_validate_json(text[start:end + 1])


def backend_name() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "Claude API"
    if shutil.which("claude") or (Path.home() / ".local/bin/claude").exists():
        return "Claude Code"
    return "none"


def ask(prompt: str, schema):
    """Send a prompt to whichever Claude backend is set up and get back a `schema` instance."""
    backend = backend_name()
    if backend == "Claude API":
        return _pick_via_api(prompt, schema)
    if backend == "Claude Code":
        return _pick_via_claude_code(prompt, schema)
    raise SetupError("No AI available: set ANTHROPIC_API_KEY in .env or install Claude Code.")


def pick_clips(segments, title, duration, n=5, min_len=20, max_len=75, instructions="",
               attempts=3, log=print) -> list[dict]:
    prompt = build_prompt(segments, title, duration, n, min_len, max_len, instructions)
    backend = backend_name()
    pick = {"Claude API": _pick_via_api, "Claude Code": _pick_via_claude_code}.get(backend)
    if not pick:
        raise SetupError("No AI available: set ANTHROPIC_API_KEY in .env or install Claude Code.")
    for attempt in range(1, attempts + 1):
        try:
            plan = pick(prompt)
            clips = snap_and_clean([c.model_dump() for c in plan.clips], segments, duration, max_len)
            if clips:
                return clips
            err = RuntimeError("The AI didn't return any usable clips.")
        except SetupError:
            raise
        except Exception as e:  # empty replies, bad JSON, timeouts, rate limits
            err = e
        if attempt < attempts:
            log(f"AI hiccup ({str(err)[:80]}), retrying {attempt + 1}/{attempts}…")
            time.sleep(10 * attempt)
    raise err


def snap_and_clean(clips, segments, duration, max_len) -> list[dict]:
    """Snap clip edges to transcript segment boundaries and drop overlaps."""
    starts = [s["start"] for s in segments]
    ends = [s["end"] for s in segments]
    out = []
    for c in sorted(clips, key=lambda c: -c["score"]):
        s = min(starts, key=lambda t: abs(t - c["start"]))
        e = min((t for t in ends if t > s), key=lambda t: abs(t - c["end"]), default=c["end"])
        e = min(e, s + max_len + 15, duration)
        # small padding so the first/last word isn't clipped
        s, e = max(0.0, s - 0.15), min(duration, e + 0.35)
        if e - s < 5 or any(s < o["end"] and e > o["start"] for o in out):
            continue
        out.append({**c, "start": round(s, 2), "end": round(e, 2)})
    return out
