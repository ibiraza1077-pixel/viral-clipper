from clipper.picker import build_prompt, format_transcript, snap_and_clean

from .conftest import seg

SEGMENTS = [seg(0, 10, "a"), seg(10, 20, "b"), seg(20, 30, "c"), seg(30, 40, "d"), seg(40, 50, "e")]


def clip(start, end, score=50):
    return {"start": start, "end": end, "score": score, "title": "t"}


def test_snaps_to_segment_edges_with_padding():
    [c] = snap_and_clean([clip(9.4, 29.2)], SEGMENTS, duration=50, max_len=75)
    assert c["start"] == 9.85   # snapped to 10, minus 0.15 lead-in
    assert c["end"] == 30.35    # snapped to 30, plus 0.35 tail


def test_drops_overlaps_keeping_higher_score():
    out = snap_and_clean([clip(0, 20, score=40), clip(10, 30, score=90)], SEGMENTS, 50, 75)
    assert [c["score"] for c in out] == [90]


def test_keeps_non_overlapping_clips_best_first():
    out = snap_and_clean([clip(0, 10, score=40), clip(20, 40, score=90)], SEGMENTS, 50, 75)
    assert [c["score"] for c in out] == [90, 40]


def test_drops_clips_under_five_seconds():
    short = [seg(0, 2, "x"), seg(2, 4, "y")]
    assert snap_and_clean([clip(0, 2)], short, duration=4, max_len=75) == []


def test_never_runs_past_video_end():
    [c] = snap_and_clean([clip(30, 60)], SEGMENTS, duration=50, max_len=75)
    assert c["end"] <= 50


def test_caps_length_near_max_len():
    [c] = snap_and_clean([clip(0, 50)], SEGMENTS, duration=50, max_len=20)
    assert c["end"] - c["start"] <= 20 + 15 + 0.5


def test_format_transcript():
    assert format_transcript([seg(1, 2.5, "hello", "there")]) == "[1.0-2.5] hello there"


def test_campaign_rules_included_only_when_given():
    base = build_prompt(SEGMENTS, "T", 50, 3, 20, 75)
    assert "Campaign rules" not in base
    ruled = build_prompt(SEGMENTS, "T", 50, 3, 20, 75, instructions="max 30 seconds")
    assert "Campaign rules" in ruled and "max 30 seconds" in ruled
