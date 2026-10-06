from pathlib import Path

MODEL = "mlx-community/whisper-large-v3-turbo"


def transcribe(audio: Path) -> dict:
    """Word-level transcript using Whisper on Apple Silicon (MLX)."""
    import mlx_whisper

    result = mlx_whisper.transcribe(
        str(audio),
        path_or_hf_repo=MODEL,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    segments = []
    for s in result.get("segments", []):
        words = [
            {"w": w["word"].strip(), "s": round(w["start"], 2), "e": round(w["end"], 2)}
            for w in s.get("words", [])
            if w["word"].strip()
        ]
        if not words:
            continue
        segments.append({
            "start": words[0]["s"],
            "end": words[-1]["e"],
            "text": s["text"].strip(),
            "words": words,
        })
    return {"language": result.get("language"), "segments": segments}
