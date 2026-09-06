"""Discogs browse buckets, which are lists wearing a single label.

Discogs files every release under one of fifteen top-level genres, and several
of those are not genres at all: `Folk, World, & Country` is three stapled
together for a shop's browse menu. No record is in it.

Notion rejects a comma in a select option outright, so the push quietly swapped
it for a semicolon. That kept the table working and left `folk; world; &
country` sitting in the filter menu as a category matching 38 tracks and
describing none of them -- and, worse, meant those tracks cast no vote for
`folk`, or `country`, which are the genres they are actually in.

The hard part is that the separator does not tell you anything: `Funk / Soul`
is two genres and `Drum & Bass` is one. Only knowing the music separates them,
so the exceptions are a list rather than a rule, and the list says so.
"""

from __future__ import annotations

from mtx.online.genre import collect, split_bucket


def test_a_browse_bucket_becomes_its_parts():
    assert split_bucket("Folk, World, & Country") == ["Folk", "World", "Country"]


def test_a_slash_bucket_becomes_its_parts():
    assert split_bucket("Funk / Soul") == ["Funk", "Soul"]


def test_a_genre_whose_name_contains_an_ampersand_survives():
    """Splitting these would invent `drum`, `bass` and `roll`."""
    assert split_bucket("Drum & Bass") == ["Drum & Bass"]
    assert split_bucket("Rock & Roll") == ["Rock & Roll"]
    assert split_bucket("Rhythm & Blues") == ["Rhythm & Blues"]


def test_a_plain_genre_is_untouched():
    assert split_bucket("Electronic") == ["Electronic"]
    assert split_bucket("Hip Hop") == ["Hip Hop"]


def test_an_empty_label_yields_nothing():
    assert split_bucket("") == []
    assert split_bucket(None) == []


def test_a_split_never_returns_nothing():
    """Whatever the separators do, some label has to come back."""
    assert split_bucket("&") == ["&"]
    assert split_bucket("a, b") == ["a, b"], "fragments under three characters"


def test_the_parts_each_get_a_vote():
    """The point of the exercise: those tracks are in folk, and in country."""
    got = collect({"discogs": [{"name": "Folk, World, & Country", "count": 1}]})

    names = {g["name"] for g in got["ranked"]}
    assert {"folk", "world", "country"} <= names
    assert not any("," in n for n in names), \
        "no comma may reach a Notion select option"


def test_an_indivisible_genre_still_votes_as_itself():
    got = collect({"discogs": [{"name": "Drum & Bass", "count": 1}]})

    names = {g["name"] for g in got["ranked"]}
    assert "drum" not in names and "bass" not in names
