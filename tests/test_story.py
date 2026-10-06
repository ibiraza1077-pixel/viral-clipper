from clipper.download import _LISTY
from clipper.story import _bg_colour, join_split_words


def test_rejoins_numbers_whisper_splits():
    words = [{"w": "1", "s": 0, "e": .2}, {"w": ",000", "s": .2, "e": .5}, {"w": "pounds", "s": .6, "e": 1}]
    assert [w["w"] for w in join_split_words(words)] == ["1,000", "pounds"]


def test_does_not_join_across_a_pause():
    words = [{"w": "end", "s": 0, "e": .2}, {"w": ".", "s": 1.0, "e": 1.1}]
    assert len(join_split_words(words)) == 2


def test_background_colour_falls_back_on_bad_input():
    assert _bg_colour("#2a0f3d") == "0x2a0f3d"
    assert _bg_colour("purple") == "0x0e3b1f"
    assert _bg_colour("") == "0x0e3b1f"


def test_channel_and_playlist_links_are_expanded():
    for url in ["https://www.youtube.com/@creator", "https://youtube.com/playlist?list=PL1",
                "https://www.youtube.com/watch?v=x&list=PL1", "https://www.tiktok.com/@someone"]:
        assert _LISTY.search(url), url
    for url in ["https://www.youtube.com/watch?v=abc", "https://www.tiktok.com/@someone/video/123"]:
        assert not _LISTY.search(url), url
