"""Who a catalogue folder is, and the two ways that went wrong.

**A rejected credit left its MBID behind.** `resolve_one` takes the most
common credit across a folder's tracks, then decides whether that credit is
this folder's artist. When it decides no, it kept the folder's name -- and
kept the credit's MBID anyway. The `Alan Walker` folder holds one track
credited to `K-391`, so it ended up with Alan Walker's name and K-391's id:
a row that is populated, correctly typed, internally consistent to every
reader, and about two different people.

**A folder no recording matched had no identity at all.** Every MBID here
otherwise comes from a recording match, so four Vietnamese artists -- each a
single track, each with `no candidate recording` against an ISRC MusicBrainz
does not carry -- were left as bare folder names. MusicBrainz knows all four
perfectly well; nobody had asked it directly.
"""

from __future__ import annotations

import os
import sys

import pytest

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "tools")
sys.path.insert(0, TOOLS)

identity = pytest.importorskip("identity")


def fact(name, mbid):
    return {"mb_name": name, "mb_mbid": mbid}


# --- a credit that is not this folder's artist -------------------------------

def test_a_rejected_credit_does_not_leave_its_mbid_behind():
    """The regression: one artist's name paired with another's id."""
    got = identity.resolve_one("Alan Walker", [fact("K-391", "k391-mbid")])

    assert got["name"] == "Alan Walker", "the folder name is kept"
    assert got["mbid"] is None, "and K-391's id must not be kept with it"
    assert got["rejected_credit"] == "K-391"
    assert got["rejected_mbid"] == "k391-mbid", "recorded, not silently dropped"


def test_an_accepted_credit_keeps_its_mbid():
    """The ordinary case, which must not regress into dropping ids."""
    got = identity.resolve_one("Drake", [fact("Drake", "drake-mbid")] * 3)

    assert got["name"] == "Drake"
    assert got["mbid"] == "drake-mbid"
    assert got["source"] == "musicbrainz"


def test_a_misspelt_folder_is_corrected_and_keeps_the_id():
    """One dropped letter is a typo, not a different person."""
    got = identity.resolve_one("Stepen Sanchez",
                               [fact("Stephen Sanchez", "sanchez-mbid")])

    assert got["name"] == "Stephen Sanchez"
    assert got["mbid"] == "sanchez-mbid"
    assert got["renamed"] is True


def test_a_folder_with_no_credits_stays_a_folder():
    got = identity.resolve_one("Some Artist", [{"mb_name": "", "mb_mbid": None}])

    assert got["mbid"] is None
    assert got["source"] == "folder"


# --- asking MusicBrainz about the artist directly ----------------------------

class _Client:
    def __init__(self, answer):
        self.answer = answer
        self.asked = []

    def get_json(self, url, headers=None):
        self.asked.append(url)
        return self.answer, None


def test_an_artist_search_resolves_a_folder_no_recording_matched():
    client = _Client({"artists": [
        {"score": 100, "name": "Lê Hiếu", "id": "le-hieu-mbid", "country": "VN"}]})

    got = identity.search_artist(client, "Lê Hiếu")

    assert got["mbid"] == "le-hieu-mbid"
    assert got["country"] == "VN"


def test_a_low_scoring_hit_is_refused():
    """A wrong MBID merges two catalogues in every within-artist comparison."""
    client = _Client({"artists": [
        {"score": 62, "name": "Le Hieu", "id": "someone-else"}]})

    assert identity.search_artist(client, "Lê Hiếu") is None


def test_a_high_score_on_a_different_name_is_refused():
    """The score is MusicBrainz's opinion; the name still has to match."""
    client = _Client({"artists": [
        {"score": 100, "name": "Completely Different", "id": "nope"}]})

    assert identity.search_artist(client, "Lê Hiếu") is None


def test_an_accented_name_matches_its_unaccented_spelling():
    """Squashing is what makes `Tiên Tiên` and `Tien Tien` one artist."""
    client = _Client({"artists": [
        {"score": 95, "name": "Tiên Tiên", "id": "tien-mbid", "country": "VN"}]})

    assert identity.search_artist(client, "Tien Tien")["mbid"] == "tien-mbid"


def test_no_results_is_not_an_error():
    assert identity.search_artist(_Client({"artists": []}), "Nobody") is None


def test_the_search_only_runs_where_there_is_no_id(monkeypatch, tmp_path):
    """It exists for folders with no vote; it must not second-guess a vote."""
    calls = []
    monkeypatch.setattr(identity, "search_artist",
                        lambda c, n: calls.append(n) or None)
    artists = {
        "Drake": {"name": "Drake", "mbid": "drake-mbid", "source": "musicbrainz"},
        "Unknown": {"name": "Unknown", "mbid": None, "source": "folder"},
    }
    identity.resolve_unheard(str(tmp_path), artists)

    assert calls == ["Unknown"]
