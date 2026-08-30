"""Song form: what the letters are allowed to merge, and what the rules may claim.

Form is the one place in the tool where a measurement (section boundaries,
loudness, vocal presence) is turned into an inference (verse, chorus, bridge),
and the two failure modes are opposite.  The clustering can merge things that
are not the same part of a song; the labelling can put a confident name on a
part it has no evidence for.  These tests are about the guards against each,
so none of them decode audio: the section features are given directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from mtx.metrics.form import _cluster, _label, _parts
from mtx.util import Collector


def _sections(n: int) -> list[dict]:
    return [{"index": i, "start_s": float(i), "end_s": float(i + 1)}
            for i in range(n)]


def _build(letters: list[int], lufs: list[float], vocal: list[bool | None]):
    """Parts straight from a letter sequence, as `analyse` would build them."""
    sections = _sections(len(letters))
    for i, s in enumerate(sections):
        s["lufs_i"] = lufs[i]
        s["vocal_present"] = vocal[i]
    parts = _parts(sections, letters, vocal)
    return sections, parts


# --------------------------------------------------------------- the clustering

def test_a_singing_section_never_merges_with_a_silent_one():
    """An instrumental hook and the last chorus over it are not one part.

    Their timbre can sit a short cosine distance apart -- the hook dominates
    both -- and merging them costs a chorus.  Vocal presence is measured where
    the distance is a guess, so it wins.
    """
    X = np.array([[1.0, 0.0], [1.0, 0.02], [0.0, 1.0]])   # 0 and 1 nearly identical
    assert _cluster(X, 0.8)[0] == _cluster(X, 0.8)[1], "same timbre, so they merge"

    letters = _cluster(X, 0.8, [False, True, True])
    assert letters[0] != letters[1], "one sings and the other does not"


def test_without_stems_the_clustering_is_untouched():
    """No vocals stem means no constraint, not a different answer."""
    X = np.array([[1.0, 0.0], [1.0, 0.02], [0.0, 1.0]])
    assert _cluster(X, 0.8, [None, None, None]) == _cluster(X, 0.8)
    assert _cluster(X, 0.8, None) == _cluster(X, 0.8)


def test_a_section_with_no_vocal_reading_constrains_nothing():
    """A half-known track must not be split on the strength of the gaps."""
    X = np.array([[1.0, 0.0], [1.0, 0.02], [0.0, 1.0]])
    assert _cluster(X, 0.8, [None, True, True]) == _cluster(X, 0.8)


# ----------------------------------------------------------------- the labelling

def test_every_part_is_named():
    """A part the rules cannot place is still named, and says which it is."""
    sections, parts = _build([0, 1, 2, 1, 2, 3],
                             [-20, -10, -6, -10, -6, -9],
                             [False, True, True, True, True, True])
    _label(sections, parts, Collector())
    labels = [p["label"] for p in parts]
    assert "unlabelled" not in labels
    assert all(p["label"] for p in parts)
    assert all(p["label_evidence"] for p in parts), "a name owes its reasons"


def test_the_loudest_part_in_a_track_is_not_called_a_bridge():
    """The one part a bridge characteristically is not is the biggest moment.

    An unrepeated part louder than the chorus is something these rules cannot
    name; saying so beats a confident wrong answer.
    """
    # A B C B C X A: X repeats nowhere, sits in the second half, is not the
    # last part, and is louder than either part called chorus.
    sections, parts = _build([0, 1, 2, 1, 2, 3, 0],
                             [-20, -10, -6, -10, -6, -4, -20],
                             [False, True, True, True, True, True, False])
    _label(sections, parts, Collector())
    assert parts[2]["label"] == "chorus", "the loud repeat is the chorus"
    assert parts[5]["label"] == "section"
    assert parts[5]["label"] != "bridge"


def test_a_quiet_unrepeated_part_is_still_a_bridge():
    """The guard must not cost the label its ordinary case."""
    sections, parts = _build([0, 1, 2, 1, 2, 3, 0],
                             [-20, -10, -6, -10, -6, -12, -20],
                             [False, True, True, True, True, True, False])
    _label(sections, parts, Collector())
    assert parts[5]["label"] == "bridge"


def test_an_unnameable_part_is_flagged_not_swallowed():
    """`chorus_count` counts only what the rules named, so it has to say so."""
    collector = Collector()
    sections, parts = _build([0, 1, 2, 1, 2, 3, 0],
                             [-20, -10, -6, -10, -6, -4, -20],
                             [False, True, True, True, True, True, False])
    _label(sections, parts, collector)
    notes = [n for n in collector.notes if n["metric"] == "form.labels"]
    assert notes, "a part with no rule behind it is not a silent outcome"
    assert "loudest part" in notes[0]["reason"]
    assert notes[0]["confidence"] == "low"


def test_a_track_whose_letters_never_repeat_says_so():
    collector = Collector()
    sections, parts = _build([0, 1, 2, 3], [-20, -10, -6, -9],
                             [False, True, True, True])
    _label(sections, parts, collector)
    reasons = " ".join(n["reason"] for n in collector.notes)
    assert "no part letter repeats" in reasons


@pytest.mark.parametrize("label", ["intro", "outro", "chorus", "verse"])
def test_the_confident_labels_stay_confident(label):
    """The guards are on the last rung; the evidenced ones keep their rank."""
    sections, parts = _build([0, 1, 2, 1, 2, 3],
                             [-20, -10, -6, -10, -6, -12],
                             [False, True, True, True, True, False])
    _label(sections, parts, Collector())
    named = {p["label"]: p["label_confidence"] for p in parts}
    if label in named:
        assert named[label] == "medium"
