"""What counts as "already transcribed", which is not what it looks like.

A track whose file tags carry a real lyric sheet keeps `lyrics.source` at
`file:tag`: a sheet written by the label beats a transcription of a separated
vocal stem, and it should. The transcript is still made, still stored, and
still what the word-level alignment and prosody measurements read.

The resume check asked whether the transcript had *won*, not whether it
existed -- so those tracks looked untouched on every pass and were
re-transcribed from scratch, thirty seconds each, producing a transcript
identical to the one already on disk. 95 of 1,321 tracks in this corpus, on
every run, forever, growing as more tagged FLACs arrive.

The bug is invisible from the log, which reports each one as a fresh `ok`.
"""

from __future__ import annotations

import os
import sys

import pytest

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "tools")
sys.path.insert(0, TOOLS)

transcribe = pytest.importorskip("transcribe")


def doc(**lyrics):
    return {"lyrics": lyrics}


def test_a_tag_lyric_alongside_a_transcript_counts_as_done():
    """The regression: 95 tracks re-derived an identical transcript every run."""
    d = doc(source="file:tag",
            transcript={"available": True, "source": "transcript",
                        "text": "one\ntwo", "lines": 2})

    assert transcribe.already_done(d) is True


def test_a_chosen_transcript_counts_as_done():
    """The case the old check did catch, which must keep working."""
    d = doc(source="transcript",
            transcript={"available": True, "source": "transcript"})

    assert transcribe.already_done(d) is True


def test_a_transcript_that_heard_nothing_counts_as_done():
    """An empty transcript is a finding about the track, not a missing one."""
    d = doc(source="file:tag",
            transcript={"available": True, "rejected_as_lyric":
                        "3 words is under the 20-word floor"})

    assert transcribe.already_done(d) is True


def test_a_tag_lyric_with_no_transcript_is_not_done():
    """The whole point of the pass: tagged tracks still need aligning."""
    d = doc(source="file:tag", transcript={"available": False,
                                           "reason": "not requested"})

    assert transcribe.already_done(d) is False


def test_a_failed_transcript_is_not_done():
    """An out-of-memory track must be picked up by the next run, not skipped."""
    d = doc(source="file:tag",
            transcript={"available": False,
                        "reason": "cuda: RuntimeError('out of memory')"})

    assert transcribe.already_done(d) is False


def test_an_untouched_document_is_not_done():
    assert transcribe.already_done({}) is False
    assert transcribe.already_done({"lyrics": {}}) is False


def test_a_lyric_from_somewhere_else_does_not_fake_a_transcript():
    """`available` alone is not enough -- a tag lyric sets it too.

    The block has to say it came from a transcription, or a tagged track with
    no transcript would be marked done and never aligned at all: a silent
    hole rather than a repeated cost, which is the worse of the two.
    """
    d = doc(source="file:tag",
            transcript={"available": True, "source": "file:tag"})

    assert transcribe.already_done(d) is False
