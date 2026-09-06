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


def test_an_ampersand_inside_a_word_is_not_a_separator():
    """The regression this rule introduced before the corpus caught it.

    A bucket is written `Folk, World, & Country`, with spaces.  A genre is
    written `R&B`, without.  Splitting on a bare ampersand turned
    `Contemporary R&B` into `contemporary r` on 283 tracks -- a category that
    reads like a real one and is a fragment of a word.
    """
    assert split_bucket("Contemporary R&B") == ["Contemporary R&B"]
    assert split_bucket("R&B") == ["R&B"]
    assert split_bucket("contemporary r&b") == ["contemporary r&b"]


def test_a_spaced_ampersand_is_still_a_separator():
    assert split_bucket("Stage & Screen") == ["Stage", "Screen"]


def test_the_lowercased_bucket_splits_too():
    """`normalise` runs first, so this is the spelling that actually arrives."""
    assert split_bucket("folk, world, & country") == ["folk", "world", "country"]


def test_a_slash_bucket_splits_without_needing_spaces():
    """`normalise` tightens ` / ` to `/` before a label reaches the splitter.

    Every slashed name in this corpus is a bucket rather than a genre --
    `funk/soul` on 415 tracks, `films/games` on 52, `rnb/swing` on 51 -- so
    those tracks were voting for a category no record is in, and casting no
    vote for `funk` or `soul`, which is what they are.
    """
    assert split_bucket("funk/soul") == ["funk", "soul"]
    assert split_bucket("rnb/swing") == ["rnb", "swing"]
    assert split_bucket("techno/house") == ["techno", "house"]
