"""The Discogs query, which asked for a conjunction the index cannot answer.

Measured against the live API, for Adele's `He Won't Go` from `21`:

    artist + track + release_title  ->   0 results
    artist + track                  ->   2
    artist + release_title          ->  10

The old query sent the first of those whenever the file carried an album tag,
which is 425 of 1,321 tracks. Every one recorded `no results` -- which reads
as *Discogs does not have this record*, and meant the opposite. Label,
catalogue number, pressing credits and styles were all missing from a third
of the corpus because of a query shape.

The album query leads now, because what Discogs is for here is the pressing,
and a label and a catalogue number are properties of a release rather than of
a track. The track query is the fallback for a standalone single with no
album to name.

These tests use a stub client: the behaviour under test is which queries are
sent and in what order, which needs no network and no token.
"""

from __future__ import annotations

import urllib.parse

import pytest

from mtx.online import discogs


class _Client:
    """Answers a scripted set of queries and records every URL asked for."""

    def __init__(self, answers=None):
        self.urls = []
        self.answers = answers or {}
        self.last_fetched_utc = "2026-09-06T00:00:00Z"

    def get_json(self, url, headers=None):
        self.urls.append(url)
        params = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
        if "/releases/" in url:
            return ({"id": 1, "title": "21", "year": 2011, "country": "US",
                     "labels": [{"name": "XL Recordings", "catno": "XLCD 520"}],
                     "formats": [{"name": "CD"}], "tracklist": [],
                     "genres": ["Pop"], "styles": ["Soul"]}, None)
        for kind, count in self.answers.items():
            if _shape(params) == kind:
                return ({"results": [{"id": 1, "title": "Adele - 21",
                                      "year": 2011}] * count}, None)
        return ({"results": []}, None)


def _shape(params: dict) -> str:
    keys = [k for k in ("barcode", "artist", "track", "release_title", "q")
            if params.get(k)]
    return "+".join(keys)


def shapes(client) -> list[str]:
    return [_shape(dict(urllib.parse.parse_qsl(urllib.parse.urlparse(u).query)))
            for u in client.urls if "/database/search" in u]


@pytest.fixture(autouse=True)
def token(monkeypatch):
    monkeypatch.setenv("DISCOGS_TOKEN", "test-token")


LOCAL = {"artist": "Adele", "title": "He Won't Go", "album": "21"}


def test_the_album_query_comes_before_the_track_query():
    """Label and catalogue number are release properties, so ask by release."""
    client = _Client({"artist+release_title": 10})

    got = discogs.lookup(client, LOCAL)

    assert got["available"] is True
    assert got["matched_by"] == "artist+release"
    assert shapes(client)[0] == "artist+release_title"


def test_track_and_release_title_are_never_sent_together():
    """The regression: that conjunction returns nothing, every time."""
    client = _Client({})

    discogs.lookup(client, LOCAL)

    for shape in shapes(client):
        assert not ("track" in shape and "release_title" in shape), \
            f"{shape} is the query that returned 0 on the live API"


def test_a_single_with_no_album_still_gets_a_track_query():
    """`Alan Walker/Play` has no album tag and must not be skipped."""
    client = _Client({"artist+track": 2})

    got = discogs.lookup(client, {"artist": "Alan Walker", "title": "Play",
                                  "album": None})

    assert got["available"] is True
    assert got["matched_by"] == "artist+track"


def test_the_track_query_is_the_fallback_when_the_album_misses():
    client = _Client({"artist+track": 2})

    got = discogs.lookup(client, LOCAL)

    assert got["matched_by"] == "artist+track"
    assert shapes(client) == ["artist+release_title", "artist+track"]


def test_a_barcode_is_tried_first_when_there_is_one():
    client = _Client({"barcode": 1})

    got = discogs.lookup(client, dict(LOCAL, barcode="602547961730"))

    assert got["matched_by"] == "barcode"
    assert shapes(client) == ["barcode"], "an exact hit ends the search"


def test_a_featured_credit_is_reduced_to_the_lead_artist():
    """Discogs indexes the credited artist; the raw tag matches nothing."""
    client = _Client({"artist+release_title": 3})

    discogs.lookup(client, {"artist": "Adele (feat. Someone)",
                            "title": "He Won't Go", "album": "21"})

    asked = dict(urllib.parse.parse_qsl(
        urllib.parse.urlparse(client.urls[0]).query))
    assert asked["artist"] == "Adele"


def test_the_catalogue_number_is_read_off_the_label():
    """It lives on the label entry and was never extracted at all.

    So "no Discogs release" and "a release whose catalogue number we never
    read" were indistinguishable from the corpus, and the one field a pressing
    is actually identified by was absent from every row that did match.
    """
    client = _Client({"artist+release_title": 1})

    got = discogs.lookup(client, LOCAL)

    assert got["release"]["catalogue_numbers"] == ["XLCD 520"]
    assert got["release"]["labels"] == ["XL Recordings"]


def test_a_missing_token_short_circuits(monkeypatch):
    monkeypatch.delenv("DISCOGS_TOKEN", raising=False)
    client = _Client({"artist+release_title": 5})

    got = discogs.lookup(client, LOCAL)

    assert got["available"] is False
    assert client.urls == [], "no request should be made without a token"
