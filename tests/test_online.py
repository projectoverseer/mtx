"""The offline half of `mtx enrich`: matching, genre voting, cross-checks.

Nothing here touches the network.  The provider modules are exercised through
a cache primed with recorded responses, which is also how the disk cache is
meant to behave in production: a second run over an enriched folder should
make no requests at all.
"""

from __future__ import annotations

import json

import pytest

from mtx.online import _tempo_check, enrich, local_facts
from mtx.online.genre import collect, collect_tags, normalise, umbrella
from mtx.online.http import Client
from mtx.online.match import (best, primary_artist, score_candidate,
                              search_artist, search_title, simplify_title,
                              title_score)


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------

def test_radio_edit_loses_to_the_album_cut():
    """The case that motivated candidate scoring.

    `bad guy`'s ISRC returns three MusicBrainz recordings and lists the 175 s
    radio edit first.  Taking `recordings[0]` would attach the wrong duration
    and the wrong credits to the track.
    """
    local = {"duration_s": 194.088, "title": "bad guy", "artist": "Billie Eilish"}
    candidates = [
        {"id": "radio", "title": "bad guy", "duration_s": 175.0},
        {"id": "album", "title": "bad guy", "duration_s": 194.0},
        {"id": "other", "title": "Bad Guy", "duration_s": 194.087},
    ]
    winner, scored = best(local, candidates, by_isrc=True)
    assert winner is not None
    assert winner["id"] in ("album", "other")
    assert scored[-1]["id"] == "radio"
    assert scored[-1]["match"]["score"] < scored[0]["match"]["score"]


def test_packaging_suffixes_do_not_break_a_title_match():
    assert simplify_title("bad guy (Official Audio)") == "bad guy"
    assert simplify_title("Circles - Remastered 2020") == "circles"
    assert title_score("As It Was", "As It Was (Official Video)") > 0.9
    assert title_score("Heat Waves", "Blinding Lights") < 0.6


def test_artist_order_does_not_matter():
    a = score_candidate({"duration_s": 180.0, "title": "luther",
                         "artist": "Kendrick Lamar, SZA"},
                        {"duration_s": 180.0, "title": "luther",
                         "artist": "SZA & Kendrick Lamar"})
    assert a["artist_score"] > 0.8
    assert a["score"] > 0.9


def test_a_wrong_duration_sinks_the_score_even_with_a_perfect_title():
    r = score_candidate({"duration_s": 194.0, "title": "bad guy"},
                        {"duration_s": 120.0, "title": "bad guy"})
    assert r["duration_score"] == 0.0
    assert r["score"] < 0.5
    assert any("duration differs" in n for n in r["notes"])


def test_search_terms_drop_packaging_but_stay_readable():
    """A search box is not a comparison.

    `Fly Me To The Moon (Dolby Atmos)` is a 2022 Atmos reissue whose ISRC no
    database carries, so it falls through to search -- and finds nothing under
    its full name. Stripping the packaging is what makes the fallback work,
    but the term still has to look like a title, not a folded key.
    """
    assert search_title("Fly Me To The Moon (Dolby Atmos)") == "Fly Me To The Moon"
    assert search_title("Circles - Remastered 2020") == "Circles"
    # Only known packaging words go. A parenthetical that is part of the title
    # has to survive, or the search becomes wrong in the other direction:
    # "Marea" and "Marea (we've lost dancing)" are not the same question.
    assert search_title("Marea (we’ve lost dancing)") == "Marea (we’ve lost dancing)"
    assert search_title("Sunflower (Spider-Man: Into the Spider-Verse)") ==         "Sunflower (Spider-Man: Into the Spider-Verse)"
    # Never returns empty: a title that is entirely packaging stays as it was.
    assert search_title("(Official Audio)") == "(Official Audio)"


def test_multi_artist_tags_are_split_for_databases_that_want_one_name():
    assert primary_artist("Frank Sinatra / Count Basie") == "Frank Sinatra"
    assert primary_artist("Kendrick Lamar, SZA") == "Kendrick Lamar"
    assert primary_artist("The Kid LAROI feat. Justin Bieber") == "The Kid LAROI"
    assert search_artist("Frank Sinatra / Count Basie") == "Frank Sinatra, Count Basie"


# ---------------------------------------------------------------------------
# genre
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,want", [
    ("Hip-Hop/Rap", "hip hop"),
    ("RnB", "r&b"),
    ("Alternative & Indie", "alternative"),
    ("Film Soundtracks", "soundtrack"),
    ("Miscellaneous", ""),
    ("Other", ""),
])
def test_normalise_repairs_shop_spellings(raw, want):
    assert normalise(raw) == want


@pytest.mark.parametrize("name,bucket", [
    ("alternative pop", "pop"),
    ("ambient pop", "pop"),      # head-final: pop, not electronic
    ("pop rock", "rock"),        # head-final: rock, not pop
    ("electropop", "pop"),       # no word boundary before "pop"
    ("tech house", "electronic"),
    ("alternative r&b", "r&b/soul"),
    ("trap", "hip hop"),
    ("conscious hip hop", "hip hop"),
])
def test_umbrella_is_head_final(name, bucket):
    assert umbrella(name) == bucket


def test_a_granular_source_outvotes_a_single_shop_label():
    """The bug this scoring exists to avoid.

    Deezer and Apple each return the one word `Alternative`.  MusicBrainz
    returns nine genres, the top one being `alternative pop`.  Sharing each
    source's weight across its own votes would hand the win to the coarse
    label purely because it was alone.
    """
    got = collect({
        "musicbrainz:recording": [{"name": "alternative pop", "count": 4},
                                  {"name": "pop", "count": 1},
                                  {"name": "electropop", "count": 1},
                                  {"name": "trap", "count": 1}],
        "musicbrainz:release-group": [{"name": "alternative pop", "count": 7},
                                      {"name": "pop", "count": 6},
                                      {"name": "electropop", "count": 5}],
        "deezer:album": [{"name": "Alternative"}],
        "itunes:track": [{"name": "Alternative"}],
    })
    assert got["primary"] == "alternative pop"
    assert got["umbrella"] == "pop"
    names = [r["name"] for r in got["ranked"]]
    assert names.index("alternative pop") < names.index("alternative")
    assert got["ranked"][0]["confidence"] == 1.0


def test_every_genre_carries_the_sources_that_voted_for_it():
    got = collect({"musicbrainz:recording": [{"name": "soul", "count": 2}],
                   "deezer:album": [{"name": "R&B"}]})
    by_name = {r["name"]: r for r in got["ranked"]}
    assert by_name["soul"]["sources"] == ["musicbrainz:recording"]
    assert by_name["r&b"]["sources"] == ["deezer:album"]
    assert got["source_count"] == 2


def test_no_sources_is_reported_not_guessed():
    got = collect({})
    assert got["available"] is False
    assert got["primary"] is None
    assert got["ranked"] == []


def test_descriptive_tags_drop_genres_and_shelf_labels():
    tags = collect_tags({"lastfm:track": [
        {"name": "electropop", "count": 100},   # a genre
        {"name": "dark", "count": 80},
        {"name": "nocturnal", "count": 40},
        {"name": "jahrescharts 2024", "count": 30},   # somebody's list
        {"name": "plattentests.de", "count": 20},     # a review site
        {"name": "2019", "count": 10},                # a year
    ]})
    names = [t["name"] for t in tags]
    assert names == ["dark", "nocturnal"]


# ---------------------------------------------------------------------------
# cross-checks
# ---------------------------------------------------------------------------

def _analysis(bpm, confidence="low"):
    return {"structure": {"tempo": {"bpm": bpm, "confidence": confidence}},
            "audio": {"duration_s": 194.088},
            "tags": {"named": {}, "all": {}}}


def test_agreement_promotes_a_low_confidence_tempo():
    got = _tempo_check(_analysis(134.78), 135.11, "deezer")
    assert got["verdict"] == "agree"
    assert got["resolved_confidence"] == "high"
    assert got["resolved_bpm"] == 135.11


def test_double_time_is_named_as_an_octave_error_not_a_disagreement():
    got = _tempo_check(_analysis(103.22), 204.7, "deezer")
    assert got["verdict"] == "octave"
    # The relationship is certain; which level to call "the tempo" is not, so
    # both readings survive and the confidence stays honest about that.
    assert got["resolved_confidence"] == "medium"
    assert got["alternate_bpm"] == 103.22
    assert got["resolved_bpm"] == 204.7


def test_a_real_disagreement_stays_low_confidence_and_keeps_the_local_value():
    got = _tempo_check(_analysis(108.74), 117.79, "deezer")
    assert got["verdict"] == "disagree"
    assert got["resolved_bpm"] == 108.74
    assert got["resolved_confidence"] == "low"


def test_no_published_tempo_leaves_the_estimate_untouched():
    got = _tempo_check(_analysis(120.0, "medium"), None, "deezer")
    assert got["verdict"] == "unavailable"
    assert got["resolved_bpm"] == 120.0
    assert got["resolved_confidence"] == "medium"


# ---------------------------------------------------------------------------
# the client and the whole section
# ---------------------------------------------------------------------------

def test_offline_client_answers_from_cache_and_never_calls_out(tmp_path):
    client = Client(cache_dir=str(tmp_path), user_agent="mtx/test", offline=True)
    url = "https://musicbrainz.org/ws/2/isrc/X?fmt=json"
    client._write_cache(url, {"url": url, "body": {"recordings": []}, "error": None})
    body, err = client.get_json(url)
    assert err is None and body == {"recordings": []}
    assert client.stats["hit"] == 1

    body, err = client.get_json("https://musicbrainz.org/ws/2/isrc/Y?fmt=json")
    assert body is None and "offline" in err
    assert client.stats["skipped"] == 1


def test_local_facts_reads_both_tag_shapes():
    got = local_facts({"audio": {"duration_s": 194.088},
                       "tags": {"named": {"title": "bad guy", "artist": "Billie Eilish",
                                          "genre": "Pop"},
                                "all": {"ISRC": "USUM71900764"}}})
    assert got["isrc"] == "USUM71900764"
    assert got["title"] == "bad guy"
    assert got["duration_s"] == 194.088
    assert got["genre_tag"] == "Pop"


def test_enrich_offline_still_returns_a_well_formed_section(tmp_path):
    """A total lookup failure must produce a usable section, not an exception."""
    got = enrich(_analysis(120.0), cache_dir=str(tmp_path), offline=True,
                 providers=("musicbrainz", "deezer", "itunes"), version="0.0.0")
    assert got["providers_available"] == []
    assert got["genres"]["available"] is False
    assert got["match_confidence"] == 0.0
    assert got["cross_checks"]["tempo"]["verdict"] == "unavailable"
    assert got["errors"]
    json.dumps(got)  # the section has to survive a round trip to disk


def test_a_provider_that_raises_does_not_lose_the_others(tmp_path, monkeypatch):
    from mtx.online import deezer
    monkeypatch.setattr(deezer, "lookup",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    got = enrich(_analysis(120.0), cache_dir=str(tmp_path), offline=True,
                 providers=("deezer", "itunes"), version="0.0.0")
    assert any("boom" in e for e in got["errors"])
    assert "itunes" in got
