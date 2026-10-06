import re

from clipper.render import _esc, _ts, build_ass, caption_events, choose_layout, words_in_range

from .conftest import seg


def w(text, s, e):
    return {"w": text, "s": s, "e": e}


def test_timestamps():
    assert _ts(0) == "0:00:00.00"
    assert _ts(-3) == "0:00:00.00"
    assert _ts(3725.5) == "1:02:05.50"


def test_escape_strips_ass_override_braces():
    assert _esc(r"{\b1}hi\N") == "(b1)hiN"


def test_layout_crops_only_clear_centred_faces():
    assert choose_layout({"coverage": 0.9, "size": 0.1, "position": 0.5}) == "crop"
    assert choose_layout({"coverage": 0.3, "size": 0.1, "position": 0.5}) == "fit"   # face rarely visible
    assert choose_layout({"coverage": 0.9, "size": 0.02, "position": 0.5}) == "fit"  # tiny facecam
    assert choose_layout({"coverage": 0.9, "size": 0.1, "position": 0.95}) == "fit"  # off in a corner


def test_words_in_range_rebases_to_clip_start():
    segs = [seg(0, 4, "one", "two"), seg(10, 12, "three")]
    out = words_in_range(segs, 2, 5)
    assert out == [{"w": "two", "s": 0.0, "e": 2.0}]


def test_captions_chunk_three_words_and_highlight_each():
    words = [w("a", 0, .2), w("b", .2, .4), w("c", .4, .6), w("d", .6, .8)]
    events = caption_events(words)
    assert len(events) == 4                      # one event per spoken word
    assert "A B C" in re.sub(r"\{[^}]*\}", "", events[0])
    assert re.sub(r"\{[^}]*\}", "", events[3]).endswith("D")  # 4th word starts a new chunk


def test_captions_break_on_pause():
    events = caption_events([w("hi", 0, .3), w("there", 2.0, 2.3)])
    assert re.sub(r"\{[^}]*\}", "", events[0]).endswith("HI")


def test_build_ass_strips_emoji_from_hook(tmp_path):
    path = tmp_path / "c.ass"
    build_ass([w("hey", 0, .5)], "Big news 🔥", 1080, 1920, "crop", path)
    text = path.read_text()
    assert "Big news" in text and "🔥" not in text
    assert "PlayResX: 1080" in text
