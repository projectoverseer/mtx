"""What a neighbour list is for, and the two ways this one lied.

The list answers "what should I A/B against". Five names, ranked. Two things
went wrong with that, and neither raised anything.

The corpus holds sixteen rows that are eight recordings filed twice. The
percentiles knew -- `mark_duplicates` marks them and populations count one --
and the cosine did not, so `How Deep Is Your Love` spent two of its five slots
on the same Queen master. A list of five where two are one record is a list of
four, silently.

The second is worse because it looks fine. Every list has five rows whatever
the numbers beside them, so `Get Lucky`, which finds its own radio edit at
0.993, and `How Deep Is Your Love`, whose best match anywhere in 1,321 tracks
is 0.382, present identically. One of those is a reference and the other is
the least-unrelated record in the room, and the reader cannot tell which
without the corpus distribution in their head.
"""

from __future__ import annotations

import numpy as np

from mtx.cohort import _neighbours, mark_duplicates


def track(artist, title, key=None, vec=None):
    return {"artist": artist, "title": title, "recording_key": key,
            "sha256": title, "duration_s": 200.0, "vec": vec}


def build(rows):
    """Run the embedding path over rows carrying a `vec`."""
    mark_duplicates(rows)
    embeddings = [np.asarray(r["vec"], dtype=float) for r in rows]
    zmat = np.zeros((len(rows), 1))
    _neighbours(rows, zmat, embeddings, 5)
    return rows


def names(row):
    return [(n["artist"], n["title"]) for n in row["neighbours"]["list"]]


def test_one_recording_filed_twice_takes_one_slot_not_two():
    """The defect, in the shape it appeared: Queen twice in a top five."""
    rows = [
        track("Calvin Harris", "How Deep Is Your Love", "hdiyl", [1.0, 0.0, 0.0]),
        track("Queen", "You're My Best Friend", "queen", [0.9, 0.2, 0.0]),
        track("Queen", "You're My Best Friend (Remaster)", "queen", [0.9, 0.2, 0.0]),
        track("Harry Styles", "Golden", "golden", [0.5, 0.6, 0.0]),
        track("Jonas Brothers", "Sucker", "sucker", [0.2, 0.9, 0.1]),
        track("Daft Punk", "Around the World", "atw", [0.1, 0.4, 0.9]),
    ]
    build(rows)

    got = names(rows[0])
    assert len(got) == len(set(got)), "a recommended record appears once"
    assert sum(1 for a, _ in got if a == "Queen") == 1


def test_the_copy_that_is_recommended_is_the_primary():
    """Whichever copy the percentiles count is the one the list names."""
    rows = [
        track("A", "seed", "seed", [1.0, 0.0]),
        track("Queen", "short", "queen", [0.9, 0.1]),
        track("Queen", "long", "queen", [0.9, 0.1]),
    ]
    rows[1]["duration_s"] = 180.0
    rows[2]["duration_s"] = 300.0
    build(rows)

    assert names(rows[0]) == [("Queen", "long")]


def test_a_duplicate_still_gets_a_list_of_its_own():
    """It gets a percentile too. It is a track someone will open."""
    rows = [
        track("Queen", "short", "queen", [0.9, 0.1]),
        track("Queen", "long", "queen", [0.9, 0.1]),
        track("A", "other", "other", [0.1, 0.9]),
    ]
    rows[0]["duration_s"] = 180.0
    rows[1]["duration_s"] = 300.0
    build(rows)

    assert rows[0]["neighbours"]["list"], "the secondary is not left empty"


def test_a_track_never_recommends_its_own_recording():
    """Two masters of one performance are not an A/B against anything."""
    rows = [
        track("Queen", "short", "queen", [1.0, 0.0]),
        track("Queen", "long", "queen", [1.0, 0.0]),
        track("A", "other", "other", [0.1, 0.9]),
    ]
    rows[0]["duration_s"] = 180.0
    rows[1]["duration_s"] = 300.0
    build(rows)

    for r in rows[:2]:
        assert ("Queen", "short") not in names(r)
        assert ("Queen", "long") not in names(r)


def test_a_different_recording_by_the_same_artist_is_fair_game():
    """`mark_duplicates` keys on the recording, not the credit.

    `Get Lucky`'s radio edit and album cut are separate MusicBrainz recordings,
    and them finding each other at 0.993 is the evidence the embedding works
    at all. Excluding by artist would have thrown that away.
    """
    rows = [
        track("Daft Punk", "Get Lucky (Radio Edit)", "edit", [1.0, 0.0]),
        track("Daft Punk", "Get Lucky", "album", [0.99, 0.01]),
        track("A", "other", "other", [0.0, 1.0]),
    ]
    build(rows)

    assert names(rows[0])[0] == ("Daft Punk", "Get Lucky")


def test_the_strength_of_the_nearest_match_travels_with_the_list():
    rows = [
        track("A", "twin one", "one", [1.0, 0.0, 0.0]),
        track("A", "twin two", "two", [0.99, 0.01, 0.0]),
        track("B", "loner", "three", [0.0, 0.0, 1.0]),
        track("C", "mid", "four", [0.5, 0.5, 0.0]),
    ]
    build(rows)

    near = rows[0]["neighbours"]["nearest_similarity"]
    far = rows[2]["neighbours"]["nearest_similarity"]
    assert near is not None and far is not None
    assert near > far, "the pair that matches is nearer than the outlier"


def test_the_percentile_places_that_strength_in_the_corpus():
    """0.38 means nothing until you know the corpus median is 0.58."""
    rows = [
        track("A", "twin one", "one", [1.0, 0.0, 0.0]),
        track("A", "twin two", "two", [0.99, 0.01, 0.0]),
        track("B", "loner", "three", [0.0, 0.0, 1.0]),
        track("C", "mid", "four", [0.5, 0.5, 0.0]),
    ]
    build(rows)

    pct = [r["neighbours"]["nearest_percentile"] for r in rows]
    assert all(0.0 <= p <= 100.0 for p in pct)
    assert rows[2]["neighbours"]["nearest_percentile"] == min(pct), \
        "the track with nothing near it ranks lowest"


def test_a_lonely_track_is_still_given_its_five_names():
    """Reporting the weakness is not the same as withholding the answer."""
    rows = [
        track("A", "one", "one", [1.0, 0.0, 0.0]),
        track("A", "two", "two", [0.99, 0.01, 0.0]),
        track("B", "loner", "three", [0.0, 0.0, 1.0]),
    ]
    build(rows)

    assert names(rows[2]), "the outlier still gets neighbours"


def test_the_z_space_fallback_names_the_keys_it_cannot_fill():
    """A missing key and a null read the same in Notion; they are not.

    The fallback measures distance, not cosine, and has no corpus ranking to
    place it against. Leaving the keys out would make a reader think the
    strength had been measured and lost.
    """
    rows = [track("A", "one"), track("B", "two")]
    zmat = np.arange(12.0).reshape(2, 6)
    _neighbours(rows, zmat, [None, None], 5)

    for r in rows:
        assert r["neighbours"]["basis"].startswith("mean per-metric")
        assert r["neighbours"]["nearest_similarity"] is None
        assert r["neighbours"]["nearest_percentile"] is None


def test_the_fallback_also_declines_to_recommend_a_duplicate():
    rows = [
        track("A", "seed", "seed"),
        track("Queen", "short", "queen"),
        track("Queen", "long", "queen"),
    ]
    rows[1]["duration_s"] = 180.0
    rows[2]["duration_s"] = 300.0
    mark_duplicates(rows)
    zmat = np.array([[0.0] * 6, [0.1] * 6, [0.1] * 6])
    _neighbours(rows, zmat, [None, None, None], 5)

    got = names(rows[0])
    assert sum(1 for a, _ in got if a == "Queen") <= 1
