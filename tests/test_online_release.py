"""Choosing the right release, and the right recording under a shared ISRC.

Every case here is a defect that shipped.  They share a shape: nothing raised,
nothing scored badly, and the answer was wrong -- which is the only kind of
bug that matters in a corpus meant to be evidence.
"""

from __future__ import annotations

import pytest

from mtx.online.lastfm import lookup as lastfm_lookup
from mtx.online.match import (best, earliest_date, pad_date, primary_artist,
                              search_title)
from mtx.online.musicbrainz import (_has_album, _issued_as_single,
                                    _pick_release, _release_groups,
                                    _song_first_release)


def release(title, *, date=None, status="Official", rg_title=None,
            primary="Album", secondary=(), rg_date=None, rid=None):
    return {
        "id": rid or f"rel-{title}-{date}",
        "title": title, "date": date, "status": status, "country": "XW",
        "release-group": {
            "id": f"rg-{rg_title or title}",
            "title": rg_title or title,
            "primary-type": primary,
            "secondary-types": list(secondary),
            "first-release-date": rg_date if rg_date is not None else date,
        },
    }


# ---------------------------------------------------------------------------
# date precision
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,want", [
    ("1999", "1999-01-01"),
    ("1999-06", "1999-06-01"),
    ("1999-06-08", "1999-06-08"),
    ("", "9999-99-99"),
    (None, "9999-99-99"),
])
def test_a_short_date_is_padded_to_its_earliest_meaning(raw, want):
    assert pad_date(raw) == want


def test_a_year_does_not_outrank_a_dated_release_in_the_same_year():
    """The `Scar Tissue` defect, in one line.

    Sorted as plain strings, `"1999"` comes before `"1999-06-08"`, so a
    year-only bootleg compilation beat the album it was compiled from and
    dated the song from the bootleg.
    """
    assert earliest_date(["1999", "1999-06-08"]) == "1999-06-08"


def test_the_earliest_year_still_wins_over_a_precise_later_one():
    assert earliest_date(["1999", "1998-12-31"]) == "1998-12-31"


def test_no_dates_is_none_not_a_guess():
    assert earliest_date([]) is None
    assert earliest_date([None, ""]) is None


# ---------------------------------------------------------------------------
# picking the release
# ---------------------------------------------------------------------------

def test_an_official_album_beats_a_bootleg_compilation():
    releases = [
        release("Best", date="1999", status="Bootleg",
                rg_title="200% Best Hits", secondary=("Compilation",)),
        release("Californication", date="1999-06-08"),
    ]
    assert _pick_release(releases, "")["title"] == "Californication"


def test_the_album_the_file_claims_wins_over_an_earlier_single():
    releases = [
        release("Scar Tissue", date="1999-05-31", primary="Single"),
        release("Californication", date="1999-06-08"),
    ]
    got = _pick_release(releases, "Californication")
    assert got["title"] == "Californication"


def test_an_exact_album_match_beats_a_containing_one():
    """`Heat Waves` is a substring of `Heat Waves (expansion pack)`.

    A remix EP whose name contains the album's name is not the album, so
    containment cannot be allowed to tie with equality.
    """
    releases = [
        release("Heat Waves (expansion pack)", date="2021-02-05",
                primary="Single", secondary=("Remix",)),
        release("Heat Waves", date="2020-06-29", primary="Single"),
    ]
    assert _pick_release(releases, "Heat Waves")["title"] == "Heat Waves"


def test_the_first_pressing_of_the_chosen_record_is_the_one_described():
    releases = [
        release("Californication", date="2000", rg_date="1999-06-08",
                rid="reissue"),
        release("Californication", date="1999-06-08", rg_date="1999-06-08",
                rid="original"),
    ]
    assert _pick_release(releases, "Californication")["id"] == "original"


def test_a_soundtrack_is_not_a_repackage():
    """For a song written for a film the soundtrack *is* the first release."""
    releases = [
        release("Wicked: For Good", date="2025-11-21",
                secondary=("Soundtrack",)),
        release("Now That's What I Call Music 2026", date="2026-01-01",
                secondary=("Compilation",)),
    ]
    assert _pick_release(releases, "")["title"] == "Wicked: For Good"


# ---------------------------------------------------------------------------
# what the packaging history says about the song
# ---------------------------------------------------------------------------

def test_a_song_is_dated_from_its_earliest_non_bootleg_release_group():
    releases = [
        release("Greatest Hits", date="2010", secondary=("Compilation",)),
        release("Californication", date="1999-06-08"),
        release("Fake", date="1990", status="Bootleg"),
    ]
    assert _song_first_release(releases) == "1999-06-08"


def test_issued_as_single_asks_about_a_single_named_after_the_song():
    """A B-side appears on a single without ever having been one."""
    on_a_single = [release("Scar Tissue", date="1999-05-31", primary="Single")]
    b_side_only = [release("Californication", date="1999-05-31",
                           primary="Single")]
    assert _issued_as_single(on_a_single, "Scar Tissue") is True
    assert _issued_as_single(b_side_only, "Scar Tissue") is False


def test_nothing_to_judge_from_is_none_not_false():
    """A missing measurement and a negative one must never look the same."""
    assert _issued_as_single([], "Scar Tissue") is None
    assert _issued_as_single([release("X", date="1999")], "") is None


def test_release_groups_are_deduplicated_and_ordered_by_date():
    releases = [
        release("Later", date="2010", rg_title="Comp", secondary=("Compilation",)),
        release("Album", date="1999-06-08"),
        release("Album, other pressing", date="2001", rg_title="Album",
                rg_date="1999-06-08"),
    ]
    groups = _release_groups(releases)
    assert [g["title"] for g in groups] == ["Album", "Comp"]


def test_has_album_is_forgiving_about_punctuation_and_case():
    releases = [release("The Art Of Loving", date="2025")]
    assert _has_album(releases, "the art of loving")
    assert not _has_album(releases, "Dreamland")
    assert not _has_album(releases, "")


# ---------------------------------------------------------------------------
# breaking a tie between two recordings under one ISRC
# ---------------------------------------------------------------------------

def test_two_identical_candidates_do_not_tie_on_api_order():
    """The `OLIVIA` defect.

    An ISRC returned two recordings of the same length and title.  Both scored
    exactly 1.00, `sort` is stable, and the winner was therefore whichever row
    MusicBrainz serialised first -- which credited the track to an unrelated
    artist and dated it from a compilation two years later.
    """
    local = {"duration_s": 209.0, "title": "Nice To Each Other",
             "artist": "Olivia Dean", "album": "The Art of Loving"}
    candidates = [
        {"id": "wrong", "title": "Nice to Each Other", "duration_s": 209.0,
         "artist": "OLIVIA", "first_release_date": "2026-03-27"},
        {"id": "right", "title": "Nice to Each Other", "duration_s": 209.0,
         "artist": "Olivia Dean", "first_release_date": "2025-05-30"},
    ]
    winner, _ = best(local, candidates, by_isrc=True)
    assert winner["id"] == "right"


def test_the_album_the_file_came_from_breaks_a_tie_before_the_date_does():
    local = {"duration_s": 200.0, "title": "Song", "artist": "A",
             "album": "The Album"}
    candidates = [
        {"id": "comp-only", "title": "Song", "duration_s": 200.0, "artist": "A",
         "first_release_date": "1990-01-01", "release_titles": ["Big Hits 90"]},
        {"id": "on-album", "title": "Song", "duration_s": 200.0, "artist": "A",
         "first_release_date": "1999-01-01", "release_titles": ["The Album"]},
    ]
    winner, _ = best(local, candidates, by_isrc=True)
    assert winner["id"] == "on-album"


def test_ranking_is_stable_when_nothing_distinguishes_the_rows():
    local = {"duration_s": 200.0, "title": "Song", "artist": "A"}
    candidates = [{"id": "b", "title": "Song", "duration_s": 200.0},
                  {"id": "a", "title": "Song", "duration_s": 200.0}]
    first, _ = best(local, candidates, by_isrc=True)
    second, _ = best(local, list(reversed(candidates)), by_isrc=True)
    assert first["id"] == second["id"] == "a"


# ---------------------------------------------------------------------------
# asking a name-indexed database a question it can answer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,want", [
    ("Ariana Grande (ft. Pharell Willians)", "Ariana Grande"),
    ("Post Malone [feat. Swae Lee]", "Post Malone"),
    ("Taylor Swift / Ed Sheeran / Future", "Taylor Swift"),
    ("Drake, 21 Savage", "Drake"),
    # A comma inside a name is not a separator, and splitting it asks about an
    # artist called "Tyler" -- who exists, with a play count 7,000x too small.
    ("Tyler, The Creator", "Tyler, The Creator"),
])
def test_a_lead_credit_survives_the_shapes_a_tag_comes_in(raw, want):
    assert primary_artist(raw) == want


def test_lastfm_refuses_a_row_credited_to_somebody_else(monkeypatch, tmp_path):
    """Autocorrect answers any string with something, and it looks like data."""
    monkeypatch.setenv("LASTFM_API_KEY", "x")
    calls = []

    class FakeClient:
        last_fetched_utc = "2026-09-03T00:00:00Z"

        def get_json(self, url, headers=None):
            calls.append(url)
            if "Someone+Else" in url or "Someone%20Else" in url:
                return {"track": {"name": "Song",
                                  "artist": {"name": "Someone Else"},
                                  "listeners": "2", "playcount": "2"}}, None
            if "method=track.getInfo" in url:
                return {"track": {"name": "Song", "artist": {"name": "Real"},
                                  "listeners": "9", "playcount": "900"}}, None
            return {}, None

    got = lastfm_lookup(FakeClient(), {"artist": "Real", "title": "Song"})
    assert got["track"]["playcount"] == 900
    assert got["track"]["artist"] == "Real"


def test_lastfm_falls_back_to_the_name_musicbrainz_resolved(monkeypatch):
    monkeypatch.setenv("LASTFM_API_KEY", "x")

    class FakeClient:
        last_fetched_utc = "2026-09-03T00:00:00Z"

        def get_json(self, url, headers=None):
            if "K-391" in url or "K%2D391" in url:
                return {"track": {"name": "Play", "artist": {"name": "K-391"},
                                  "listeners": "1", "playcount": "905765"}}, None
            return {}, None

    got = lastfm_lookup(FakeClient(), {
        "artist": "Alan Walker", "title": "Play",
        "resolved_artist": "K-391", "resolved_title": "Play"})
    assert got["matched_by"] == "musicbrainz"
    assert got["track"]["playcount"] == 905765


def test_search_title_keeps_a_readable_name():
    assert search_title("bad guy (Official Audio)") == "bad guy"


# ---------------------------------------------------------------------------
# the reader and the writer, tested together
# ---------------------------------------------------------------------------

def test_cohort_reads_the_keys_enrichment_actually_writes(tmp_path):
    """A contract test, because the alternative already cost us the corpus.

    `mtx cohort` looked for `online["genre"]` and `online["release"]`.  Enrich
    writes `online["genres"]` and nests the release under the provider.  Both
    sides had passing tests -- each against its own invented fixture -- and
    every cohort in a 1,321-track corpus silently fell back to the shop's own
    genre tag.  Nothing failed, and the answer was wrong.
    """
    import json

    from mtx.cohort import labels_for
    from mtx.online import enrich

    folder = tmp_path / "track"
    folder.mkdir()
    analysis = {
        "audio": {"duration_s": 200.0},
        "tags": {"named": {"title": "T", "artist": "A", "genre": "Pop",
                           "date": "1994-05-01"}, "all": {}},
        "structure": {"tempo": {"bpm": 120.0, "confidence": "medium"}},
    }
    section = enrich(analysis, cache_dir=str(tmp_path / "cache"), offline=True,
                     providers=("musicbrainz",), version="0.0.0")

    # Nothing was reachable, so these are empty -- but they must be the keys
    # the reader looks under, or the fallback chain is never exercised.
    assert "genres" in section
    assert "cross_checks" in section and "release_date" in section["cross_checks"]

    section["genres"] = {"available": True, "primary": "grunge",
                         "umbrella": "rock", "ranked": []}
    section["cross_checks"]["release_date"]["earliest"] = "1994-09-27"
    (folder / "online.json").write_text(
        json.dumps(section, ensure_ascii=False), encoding="utf-8")

    got = labels_for(analysis, str(folder), catalogue="Nirvana")
    assert (got["genre"], got["genre_source"]) == ("rock", "online")
    assert (got["year"], got["year_source"]) == (1994, "online")
    assert got["artist"] == "Nirvana"
