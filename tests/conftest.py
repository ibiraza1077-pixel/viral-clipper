"""Point every test at a throwaway output folder, so the real output/ (and any running jobs) are never touched."""
import pytest

import clipper
from clipper import pipeline


@pytest.fixture(autouse=True)
def output_dir(tmp_path, monkeypatch):
    out = tmp_path / "output"
    out.mkdir()
    monkeypatch.setattr(clipper, "OUTPUT_DIR", out)
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", out)
    return out


def seg(start, end, *words):
    """A transcript segment whose words are spread evenly across it."""
    step = (end - start) / len(words)
    ws = [{"w": w, "s": round(start + i * step, 2), "e": round(start + (i + 1) * step, 2)} for i, w in enumerate(words)]
    return {"start": start, "end": end, "text": " ".join(words), "words": ws}
