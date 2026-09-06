"""One recording, counted once, however many masters of it are on disk.

Eight recordings appear twice in this corpus -- a single and its album cut,
different sha256s because they are genuinely different masters of one
performance. Both are worth keeping: two masters of one recording is the only
A/B in the library where the song itself is held constant.

But they are one recording. Counted twice they vote twice in every percentile
and in the artist's own median, and with n=78 for `house` a single duplicate
is more than a percent of the population.

`tools/notion/outcome.py` already marked them. `mtx cohort` never read the
mark, so every percentile in the corpus was computed over a population that
double-counted eight records.
"""

from __future__ import annotations

from mtx.cohort import mark_duplicates


def row(key, sha, seconds, **kw):
    return {"recording_key": key, "sha256": sha, "duration_s": seconds, **kw}


def test_two_masters_of_one_recording_leave_one_primary():
    rows = [row("rec-1", "a", 210.0), row("rec-1", "b", 235.0)]

    assert mark_duplicates(rows) == 2
    assert [r["recording_primary"] for r in rows] == [False, True],         "the longer cut is the primary"
    assert all(r["recording_duplicates"] == 2 for r in rows)


def test_the_longest_wins_because_it_is_the_album_cut():
    """A radio edit is the same recording, shortened for airplay."""
    rows = [row("rec-1", "edit", 180.0), row("rec-1", "album", 300.0)]

    mark_duplicates(rows)

    assert next(r for r in rows if r["recording_primary"])["sha256"] == "album"


def test_a_tie_is_broken_so_a_re_run_picks_the_same_one():
    """Two runs disagreeing about which copy is real is its own defect."""
    first = [row("rec-1", "zzz", 200.0), row("rec-1", "aaa", 200.0)]
    second = [row("rec-1", "aaa", 200.0), row("rec-1", "zzz", 200.0)]

    mark_duplicates(first)
    mark_duplicates(second)

    assert (next(r for r in first if r["recording_primary"])["sha256"]
            == next(r for r in second if r["recording_primary"])["sha256"])


def test_rows_with_no_recording_key_are_all_primary():
    """An unidentified track is not a duplicate of every other one."""
    rows = [row(None, "a", 200.0), row(None, "b", 200.0)]

    assert mark_duplicates(rows) == 0
    assert all(r["recording_primary"] for r in rows)


def test_an_isrc_stands_in_when_there_is_no_recording_mbid():
    rows = [row("USUM72409273", "a", 200.0), row("USUM72409273", "b", 190.0)]

    assert mark_duplicates(rows) == 2


def test_three_copies_leave_exactly_one():
    rows = [row("rec-1", "a", 200.0), row("rec-1", "b", 210.0),
            row("rec-1", "c", 190.0)]

    assert mark_duplicates(rows) == 3
    assert sum(r["recording_primary"] for r in rows) == 1
    assert all(r["recording_duplicates"] == 3 for r in rows)


def test_distinct_recordings_are_left_alone():
    rows = [row("rec-1", "a", 200.0), row("rec-2", "b", 200.0)]

    assert mark_duplicates(rows) == 0
    assert all(r["recording_primary"] for r in rows)
    assert all(r["recording_duplicates"] == 1 for r in rows)
