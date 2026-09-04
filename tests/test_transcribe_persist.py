"""Where an amended transcript gets written, and when that is not the index.

`transcribe.py` reads the *index* (`analysis.json`) rather than the merged
document on purpose: it keeps the `split` manifest and the part files
untouched, so writing back rewrites one 3 MB index instead of re-emitting 17
MB of timelines that did not change.

That only holds while `lyrics` actually lives in the index, which today it
does. A long enough transcript would push it out to a part, and an inline
write would then silently collapse the split instead: `load_analysis` pops the
`split` manifest, so dumping the merged document over the index inlines every
timeline again and abandons the part files on disk.

The value read back stays correct, which is why this survives a spot check.
What it costs is the `want=` optimisation that made `mtx cohort` fast -- with
no manifest there is nothing to skip, and reaching forty scalars means parsing
every timeline in the corpus again.

The second test here is the one that matters: it writes the way the code used
to, shows the manifest disappearing, then does the same amendment through the
guard -- so the guard cannot be removed without a red test explaining what it
was for.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "tools")
sys.path.insert(0, TOOLS)

transcribe = pytest.importorskip("transcribe")

from mtx.split import load_analysis, write_analysis      # noqa: E402


def build(folder: str, lyric_text: str) -> dict:
    """A document whose `lyrics` is large enough to be moved out to a part.

    `write_analysis` moves whole top-level sections, largest first, until the
    index fits under the cap -- so a realistic transcript is not enough to
    trigger this today.  The word-level timings are: a four-minute song is a
    few hundred words, each with a start and an end, and one long enough track
    puts `lyrics` at the top of that ordering.  This builds that document
    directly rather than waiting for the corpus to grow one.
    """
    words = [{"word": f"w{i}", "start_s": i * 0.4, "end_s": i * 0.4 + 0.3}
             for i in range(2000)]
    doc = {
        "file": {"path_absolute": os.path.join(folder, "x.flac")},
        "headline": {"lufs_i": -8.9},
        "lyrics": {"source": "transcript", "text": lyric_text,
                   "transcript": {"available": True, "source": "transcript",
                                  "text": lyric_text, "words": words}},
        "processing": {"timeline": [round(i * 0.01, 4) for i in range(4000)]},
    }
    write_analysis(doc, folder, max_bytes=8192)
    return doc


def test_lyrics_in_a_part_survive_an_amendment(tmp_path):
    """The guard: re-split the whole document rather than write past it."""
    folder = str(tmp_path)
    build(folder, "the original words")

    index = json.load(open(os.path.join(folder, "analysis.json"), encoding="utf-8"))
    assert index["lyrics"].get("mtx_moved"), \
        "this test is meaningless unless lyrics really was moved to a part"

    doc = load_analysis(os.path.join(folder, "analysis.json"))
    doc["lyrics"]["text"] = "the amended words"
    doc["lyrics"]["transcript"]["text"] = "the amended words"
    transcribe._save(folder, os.path.join(folder, "analysis.json"), doc,
                     whole=True)

    index_path = os.path.join(folder, "analysis.json")
    back = load_analysis(index_path)
    assert back["lyrics"]["text"] == "the amended words"
    assert back["processing"]["timeline"][:3] == [0.0, 0.01, 0.02], \
        "the sections that did not change must still be there"

    raw = json.load(open(index_path, encoding="utf-8"))
    named = {e["file"] for e in (raw.get("split") or {}).get("parts", [])}
    assert {n for n in os.listdir(folder) if ".part" in n} == named, \
        "no part may be left behind looking like current data"


def test_an_inline_write_would_have_de_split_the_document(tmp_path):
    """The bug the guard exists for, written down so it stays fixed.

    `load_analysis` *pops* the `split` manifest, so dumping the merged
    document over the index does not revert the change -- it silently
    collapses the split: every timeline that was moved out is inlined again,
    the manifest naming the parts is gone, and the part files stay on disk
    where nothing will ever read them.

    The reading stays correct, which is exactly why it survives a spot check.
    What it costs is the `want=` optimisation: with no manifest there is
    nothing to skip, so `mtx cohort` goes back to parsing every timeline in
    the corpus to reach forty scalars.
    """
    folder = str(tmp_path)
    build(folder, "the original words")
    path = os.path.join(folder, "analysis.json")
    parts = sorted(n for n in os.listdir(folder) if ".part" in n)
    assert parts, "expected a split document"

    # Exactly what the old code did: amend the merged document and dump it
    # over the index, leaving the manifest and the parts as they were.
    doc = load_analysis(path)
    doc["lyrics"]["text"] = "the amended words"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)

    raw = json.load(open(path, encoding="utf-8"))
    assert "split" not in raw, "the manifest was lost -- that is the bug"
    assert sorted(n for n in os.listdir(folder) if ".part" in n) == parts, \
        "and the orphaned parts are still sitting there"

    # Now the same amendment through the guard.  It does not promise the
    # document stays split -- that is `write_analysis`'s call, made against the
    # real cap, and a document this small genuinely belongs in one file.  What
    # it promises is that the folder ends up equal to the result: whatever
    # layout it chooses, the manifest matches it and no part is left behind
    # pretending to be current data.
    build(folder, "the original words")
    doc = load_analysis(path)
    doc["lyrics"]["text"] = "the amended words"
    transcribe._save(folder, path, doc, whole=True)

    raw = json.load(open(path, encoding="utf-8"))
    named = {e["file"] for e in (raw.get("split") or {}).get("parts", [])}
    on_disk = {n for n in os.listdir(folder) if ".part" in n}
    assert on_disk == named, \
        f"the folder must match the manifest; orphans: {sorted(on_disk - named)}"
    assert load_analysis(path)["lyrics"]["text"] == "the amended words"
    assert load_analysis(path)["processing"]["timeline"][:2] == [0.0, 0.01]


def test_an_inline_index_is_rewritten_in_place(tmp_path):
    """The common case: one small atomic write, parts untouched."""
    folder = str(tmp_path)
    build(folder, "the original words")
    path = os.path.join(folder, "analysis.json")
    parts_before = sorted(n for n in os.listdir(folder) if ".part" in n)
    assert parts_before, "expected a split document"

    index = json.load(open(path, encoding="utf-8"))
    index["lyrics"] = {"source": "transcript", "text": "written inline"}
    transcribe._save(folder, path, index, whole=False)

    assert sorted(n for n in os.listdir(folder) if ".part" in n) == parts_before
    raw = json.load(open(path, encoding="utf-8"))
    assert raw["split"], "the manifest must survive an inline write"
    assert raw["lyrics"]["text"] == "written inline"


def test_a_failed_write_leaves_no_temp_file(tmp_path):
    """A half-written analysis is worse than an unwritten one."""
    folder = str(tmp_path)
    build(folder, "words")
    path = os.path.join(folder, "analysis.json")
    before = open(path, encoding="utf-8").read()

    # A set is not JSON-serialisable: json.dump raises part way through.
    doomed = {"lyrics": {"text": {"not", "serialisable"}}}
    with pytest.raises(TypeError):
        transcribe._save(folder, path, doomed, whole=False)

    assert not [n for n in os.listdir(folder) if n.endswith(".tmp")]
    assert open(path, encoding="utf-8").read() == before, \
        "the original must be untouched when the write fails"


def test_a_failure_leaves_a_reason_behind(tmp_path):
    """A failed run must not be indistinguishable from one nobody asked for.

    78 tracks in this corpus had died with a CUDA out-of-memory part way
    through decoding.  Because a failure wrote nothing, each analysis came out
    byte-identical to a track that had simply never been transcribed -- so the
    audit filed them as "no lyric from any source", which is a finding about
    the music, when the true finding was about the job.
    """
    folder = str(tmp_path)
    build(folder, "words")
    path = os.path.join(folder, "analysis.json")
    doc = load_analysis(path)
    doc["lyrics"] = {"available": False,
                     "transcript": {"available": False,
                                    "reason": "not requested"}}

    transcribe._record_attempt(
        folder, path, doc, True,
        "cuda/float16: RuntimeError('CUDA failed with error out of memory')",
        {"attempted": ["cuda/float16", "cuda/int8_float16", "cpu/int8"]})

    note = load_analysis(path)["lyrics"]["transcript"]
    assert "out of memory" in note["reason"]
    assert note["attempted_devices"][0] == "cuda/float16"
    assert note["attempted_utc"].endswith("Z")


def test_a_recorded_failure_is_still_retried(tmp_path):
    """The note must not be mistaken for a result: these tracks need re-running."""
    folder = str(tmp_path)
    build(folder, "words")
    path = os.path.join(folder, "analysis.json")
    doc = load_analysis(path)
    doc["lyrics"] = {"available": False, "transcript": {"available": False}}

    transcribe._record_attempt(folder, path, doc, True, "out of memory", {})

    assert transcribe.already_done(load_analysis(path)) is False, \
        "a recorded failure that counted as done would strand the track"


def test_recording_a_failure_leaves_the_rest_of_the_document_alone(tmp_path):
    """It is a note in the margin, not an edit to the measurements."""
    folder = str(tmp_path)
    build(folder, "words")
    path = os.path.join(folder, "analysis.json")
    doc = load_analysis(path)
    doc["lyrics"] = {"available": False, "transcript": {"available": False}}

    transcribe._record_attempt(folder, path, doc, True, "out of memory", {})

    back = load_analysis(path)
    assert back["processing"]["timeline"][:2] == [0.0, 0.01]
    assert back["headline"]["lufs_i"] == -8.9
