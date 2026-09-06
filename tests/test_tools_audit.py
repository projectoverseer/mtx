"""The gate has to fail when it should, and pass when it should.

`tools/audit.py` is the only thing standing between a defect and the evidence
base, and it is the kind of code that rots quietly: a check whose reader stops
matching what the writer produces reports "clean" forever.  So each check here
is fed a corpus built to trip exactly it.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "tools")
sys.path.insert(0, TOOLS)

audit = pytest.importorskip("audit")


def track(root, artist, album, title, *, online=None, row=None, run=None,
          analysis=None):
    """One analysed folder, shaped the way `mtx scan` shapes them."""
    folder = os.path.join(root, artist, album, title)
    os.makedirs(folder, exist_ok=True)
    doc = {"file": {"sha256": title}}
    doc.update(analysis or {})
    with open(os.path.join(folder, "analysis.json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    with open(os.path.join(folder, "corpus_row.json"), "w", encoding="utf-8") as fh:
        json.dump(row or {"LUFS-I": -9.0, "Title": title}, fh)
    with open(os.path.join(folder, "mtx_source.json"), "w", encoding="utf-8") as fh:
        json.dump({"run": run or {"schema_version": "1.3.0", "profile": "full",
                                  "tool_version": "0.4.0", "stems": True}}, fh)
    if online is not None:
        with open(os.path.join(folder, "online.json"), "w", encoding="utf-8") as fh:
            json.dump(online, fh)
    return folder


def enriched(**over):
    """A clean `online.json`, so a test only has to state its own defect."""
    doc = {
        "query": {"artist": "Real Artist", "album": "The Album",
                  "date": "2020-01-01", "duration_s": 200.0},
        "musicbrainz": {
            "available": True,
            "match": {"score": 1.0, "duration_delta_s": 0.0},
            "artists": [{"name": "Real Artist", "mbid": "mbid-1"}],
            "release": {"title": "The Album", "status": "Official"},
            "release_group": {"title": "The Album", "primary_type": "Album",
                              "secondary_types": []},
        },
        "identity": {"isrc": "X", "recording_mbid": "rec-1",
                     "discogs_release_id": 1},
        "popularity": {"lastfm_playcount": 100},
        "credits": {"producer": [{"name": "P"}]},
        "genres": {"primary": "pop", "umbrella": "pop",
                   "ranked": [{"name": "pop", "confidence": 1.0}]},
        "descriptive_tags": [],
        "cross_checks": {"release_date": {"earliest": "2020-01-01"}},
    }
    for path, value in over.items():
        cur = doc
        parts = path.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value
    return doc


def find(rep, name):
    return next(f for f in rep.findings if f.check == name)


def build(tmp_path, count=3, **kw):
    root = str(tmp_path)
    for i in range(count):
        track(root, "Real Artist", "The Album", f"t{i}", online=enriched(**kw))
    return root


def identity_file(root, folder="Real Artist", name="Real Artist"):
    with open(os.path.join(root, "artists.json"), "w", encoding="utf-8") as fh:
        json.dump({"artists": {folder: {"folder": folder, "name": name,
                                        "notion_name": name, "mbid": "mbid-1",
                                        "source": "musicbrainz", "tracks": 3,
                                        "votes": 3, "renamed": False}}}, fh)


# ---------------------------------------------------------------------------

def test_a_clean_corpus_raises_no_errors(tmp_path):
    root = build(tmp_path)
    identity_file(root)
    rep = audit.run(root)
    assert rep.errors() == [], [f.check for f in rep.errors()]


def test_an_unenriched_track_is_an_error(tmp_path):
    root = build(tmp_path)
    identity_file(root)
    track(root, "Real Artist", "The Album", "bare")     # no online.json
    rep = audit.run(root)
    assert find(rep, "coverage.not_enriched").hits
    assert "coverage.not_enriched" in {f.check for f in rep.errors()}


def test_a_bootleg_release_is_an_error(tmp_path):
    root = build(tmp_path, **{"musicbrainz.release": {
        "title": "Best", "status": "Bootleg"}})
    identity_file(root)
    rep = audit.run(root)
    assert find(rep, "release.bootleg").hits


def test_a_compilation_that_is_not_the_album_tag_warns(tmp_path):
    root = build(tmp_path, **{
        "musicbrainz.release": {"title": "Big Hits 99", "status": "Official"},
        "musicbrainz.release_group": {"title": "Big Hits 99",
                                      "primary_type": "Album",
                                      "secondary_types": ["Compilation"]}})
    identity_file(root)
    rep = audit.run(root)
    assert find(rep, "release.repackage_source").hits


def test_the_album_the_file_claims_is_not_flagged_as_a_repackage(tmp_path):
    """A soundtrack the file is actually filed under is the right release."""
    root = build(tmp_path, **{
        "query": {"artist": "Real Artist", "album": "Big Hits 99",
                  "date": "2020-01-01", "duration_s": 200.0},
        "musicbrainz.release_group": {"title": "Big Hits 99",
                                      "primary_type": "Album",
                                      "secondary_types": ["Compilation"]}})
    identity_file(root)
    rep = audit.run(root)
    assert not find(rep, "release.repackage_source").hits


def test_a_credit_matching_neither_folder_nor_tag_is_an_error(tmp_path):
    """The `OLIVIA` case: the match scores 1.00 and names somebody else."""
    root = build(tmp_path, **{
        "musicbrainz.artists": [{"name": "Someone Else", "mbid": "x"}]})
    identity_file(root)
    rep = audit.run(root)
    assert find(rep, "identity.credit_mismatch").hits


def test_a_feature_credit_is_not_a_mismatch(tmp_path):
    root = build(tmp_path, **{
        "musicbrainz.artists": [{"name": "Real Artist", "mbid": "mbid-1"},
                                {"name": "A Guest", "mbid": "y"}]})
    identity_file(root)
    rep = audit.run(root)
    assert not find(rep, "identity.credit_mismatch").hits


def test_a_missing_artists_file_is_an_error(tmp_path):
    root = build(tmp_path)                              # no artists.json
    rep = audit.run(root)
    assert find(rep, "identity.stale").hits


def test_two_spellings_of_one_genre_are_an_error(tmp_path):
    root = str(tmp_path)
    for i, name in enumerate(("Hip-Hop", "hip hop")):
        track(root, "Real Artist", "The Album", f"t{i}",
              online=enriched(**{"genres": {
                  "primary": name, "umbrella": "hip hop",
                  "ranked": [{"name": name, "confidence": 1.0}]}}))
    identity_file(root)
    rep = audit.run(root)
    assert find(rep, "vocab.case_collision").hits


def test_a_shelf_label_filed_as_a_genre_is_an_error(tmp_path):
    root = build(tmp_path, **{"genres": {
        "primary": "best of 2016", "umbrella": "pop",
        "ranked": [{"name": "best of 2016", "confidence": 1.0}]}})
    identity_file(root)
    rep = audit.run(root)
    assert find(rep, "vocab.not_a_genre").hits


def test_an_empty_file_measures_and_is_caught(tmp_path):
    root = build(tmp_path)
    identity_file(root)
    track(root, "Real Artist", "The Album", "silent", online=enriched(),
          row={"LUFS-I": -70.0})
    rep = audit.run(root)
    assert find(rep, "audio.near_silent").hits


def test_a_date_a_decade_from_the_file_tag_warns(tmp_path):
    root = build(tmp_path, **{
        "cross_checks": {"release_date": {"earliest": "2010-01-01"}}})
    identity_file(root)
    rep = audit.run(root)
    assert find(rep, "release.date_conflict").hits


def test_one_recording_in_two_folders_is_counted_twice(tmp_path):
    root = build(tmp_path)
    identity_file(root)
    track(root, "Real Artist", "Greatest Hits", "again", online=enriched())
    rep = audit.run(root)
    assert find(rep, "corpus.duplicate_recording").hits


def test_a_mixed_schema_corpus_warns(tmp_path):
    root = build(tmp_path)
    identity_file(root)
    track(root, "Real Artist", "The Album", "old", online=enriched(),
          run={"schema_version": "1.2.0", "profile": "full",
               "tool_version": "0.3.0", "stems": True})
    rep = audit.run(root)
    assert find(rep, "analysis.stale_schema").hits


def test_an_empty_root_is_reported_not_silently_clean(tmp_path):
    with pytest.raises(ValueError):
        audit.run(str(tmp_path))


def test_the_report_survives_being_written_as_json(tmp_path):
    root = build(tmp_path)
    identity_file(root)
    rep = audit.run(root)
    doc = {"facts": rep.facts, "findings": [f.as_dict() for f in rep.findings]}
    json.dumps(doc)             # a report that cannot be saved is not a report
    assert doc["facts"]["tracks"] == 3


# --- the deep pass -----------------------------------------------------------
#
# `--deep` reads the analysis rather than the summaries, and every test above
# writes an analysis with no lyrics in it -- so the deep checks short-circuited
# on `available` and none of the code past that line was ever executed by a
# test.  It was also never executed against the corpus, because the one shape
# it read was wrong and the run died on the first track.  A crash in a mode
# nobody runs looks exactly like a mode nobody runs.


def lyric_analysis(text="a line\nanother line", *, source="transcript",
                   lines=2, characters=20, **over):
    """A lyrics block shaped the way `mtx.metrics.lyrics` really writes one.

    The counts are bare integers.  That is the whole point of this fixture:
    the audit read them as `{"count": n}`, which nothing has ever emitted.
    """
    doc = {"lyrics": {"available": True, "source": source, "text": text,
                      "statistics": {"lines": lines, "characters": characters,
                                     "words": len(text.split())}}}
    doc["lyrics"].update(over)
    return doc


def test_the_deep_pass_survives_a_real_statistics_block(tmp_path):
    """The regression: `--deep` died before it checked a single track."""
    root = str(tmp_path)
    track(root, "Real Artist", "The Album", "t0", online=enriched(),
          analysis=lyric_analysis())
    identity_file(root)

    rep = audit.run(root, deep=True)

    assert not [f for f in rep.findings if f.check == "audit.crashed"]
    assert len(find(rep, "lyrics.credit_not_lyric").hits) == 0


def test_a_songwriter_credit_read_as_a_lyric_is_caught(tmp_path):
    """Two lines and 40 characters under `file:tag` is a credit, not a song."""
    root = str(tmp_path)
    track(root, "Real Artist", "The Album", "t0", online=enriched(),
          analysis=lyric_analysis("Some Person", source="file:tag",
                                  lines=1, characters=11))
    identity_file(root)

    rep = audit.run(root, deep=True)

    assert len(find(rep, "lyrics.credit_not_lyric").hits) == 1


def test_a_real_tag_lyric_is_not_mistaken_for_a_credit(tmp_path):
    """167 tracks in this corpus carry a genuine sheet; none may be flagged."""
    root = str(tmp_path)
    track(root, "Real Artist", "The Album", "t0", online=enriched(),
          analysis=lyric_analysis(source="file:tag", lines=38,
                                  characters=1467))
    identity_file(root)

    rep = audit.run(root, deep=True)

    assert len(find(rep, "lyrics.credit_not_lyric").hits) == 0


def test_a_failed_transcription_is_told_apart_from_an_absent_one(tmp_path):
    """The check that a clean audit needed and did not have.

    78 tracks had died with a CUDA out-of-memory part way through decoding.
    A failure writes nothing, so each one looked exactly like a track nobody
    had asked to transcribe -- and both landed in the same `info` finding.
    """
    root = str(tmp_path)
    track(root, "Real Artist", "The Album", "broke", online=enriched(),
          analysis={"lyrics": {"available": False, "transcript": {
              "available": False,
              "reason": "cuda: RuntimeError('CUDA failed with error out of "
                        "memory')"}}})
    track(root, "Real Artist", "The Album", "never", online=enriched(),
          analysis={"lyrics": {"available": False, "transcript": {
              "available": False,
              "reason": "not requested; pass --transcribe to run one"}}})
    identity_file(root)

    rep = audit.run(root, deep=True)

    assert len(find(rep, "lyrics.transcript_failed").hits) == 1
    assert len(find(rep, "lyrics.absent").hits) == 1
    assert find(rep, "lyrics.transcript_failed").severity == "warn", \
        "a broken run is not an observation about the music"


def test_the_deep_pass_reads_lyrics_that_live_in_a_part(tmp_path):
    """The audit must read the document, not the index that stands in for it."""
    from mtx.split import write_analysis                # noqa: PLC0415

    root = str(tmp_path)
    folder = track(root, "Real Artist", "The Album", "t0", online=enriched())
    identity_file(root)
    doc = dict(lyric_analysis(source="file:tag", lines=1, characters=11))
    doc["file"] = {"sha256": "t0"}
    doc["lyrics"]["text"] = "Some Person"
    doc["padding"] = {"timeline": [round(i * 0.01, 4) for i in range(4000)]}
    doc["lyrics"]["words"] = [{"word": f"w{i}", "start_s": i * 0.4}
                              for i in range(2000)]
    write_analysis(doc, folder, max_bytes=8192)

    index = json.load(open(os.path.join(folder, "analysis.json"),
                           encoding="utf-8"))
    assert index["lyrics"].get("mtx_moved"), \
        "this test is meaningless unless lyrics really was moved to a part"

    rep = audit.run(root, deep=True)

    assert len(find(rep, "lyrics.credit_not_lyric").hits) == 1, \
        "reading the bare index would have seen a moved marker and no lyric"


def test_a_transcript_of_paragraphs_is_flagged(tmp_path):
    """346 words over 4 lines: the words are fine, the line structure is not."""
    root = str(tmp_path)
    track(root, "Real Artist", "The Album", "t0", online=enriched(),
          analysis=lyric_analysis("a " * 346, lines=4, characters=700))
    identity_file(root)

    rep = audit.run(root, deep=True)

    hit = find(rep, "lyrics.line_structure").hits
    assert len(hit) == 1 and hit[0]["words_per_line"] > 20


def test_a_repetitive_song_is_not_flagged(tmp_path):
    """`Around the World` repeats one phrase 144 times and is transcribed right.

    The check must key on segmentation, never on repetition: a repetition
    threshold would flag the most accurate transcript in the corpus, and flag
    it for being exactly what a hit chorus is.
    """
    root = str(tmp_path)
    text = "around the world " * 144
    track(root, "Real Artist", "The Album", "t0", online=enriched(),
          analysis=lyric_analysis(text, lines=72, characters=len(text)))
    identity_file(root)

    rep = audit.run(root, deep=True)

    assert find(rep, "lyrics.line_structure").hits == []


def test_a_tag_lyric_is_not_judged_on_line_length(tmp_path):
    """A sheet is laid out by whoever typed it; that is not a defect."""
    root = str(tmp_path)
    track(root, "Real Artist", "The Album", "t0", online=enriched(),
          analysis=lyric_analysis("a " * 346, source="file:tag", lines=4,
                                  characters=700))
    identity_file(root)

    rep = audit.run(root, deep=True)

    assert find(rep, "lyrics.line_structure").hits == []


def test_an_unreachable_notion_is_a_finding_not_a_traceback(tmp_path, monkeypatch):
    """A DNS blip must not read like corpus corruption.

    The live checks died in a raw urllib traceback, at the exact moment the
    corpus was fine and the network was not.  It has to keep failing the gate
    -- "could not look" must never be published as "looked and found nothing
    wrong" -- while saying which of the two things actually broke.
    """
    root = build(tmp_path)
    identity_file(root)
    monkeypatch.setattr(audit, "check_notion",
                        lambda *a, **k: (_ for _ in ()).throw(
                            OSError("[Errno 11001] getaddrinfo failed")))

    rep = audit.run(root, notion=True)

    hits = find(rep, "notion.unreachable").hits
    assert len(hits) == 1
    assert "getaddrinfo" in hits[0]["error"]
    assert "notion.unreachable" in {f.check for f in rep.errors()},         "the gate must still fail closed"


def test_a_reachable_notion_reports_no_such_finding(tmp_path, monkeypatch):
    root = build(tmp_path)
    identity_file(root)
    monkeypatch.setattr(audit, "check_notion", lambda *a, **k: None)

    rep = audit.run(root, notion=True)

    assert find(rep, "notion.unreachable").hits == []


def test_a_compilation_of_an_older_song_is_not_a_date_conflict(tmp_path):
    """A 2015 soundtrack carrying a 2003 recording resolved exactly right.

    48 of the 51 rows this check reported were this shape, and a finding that
    is right 6% of the time is one people learn to scroll past -- which costs
    more than the check earns.
    """
    root = build(tmp_path, **{
        "query.date": "2015-10-30",
        "cross_checks.release_date": {"consensus": "2003-10-06"},
        "musicbrainz.first_release_date": "2003-10-06"})
    identity_file(root)

    rep = audit.run(root)

    assert find(rep, "release.date_conflict").hits == []


def test_a_reissue_matched_instead_of_the_original_still_flags(tmp_path):
    """The case the check exists for: a date agreeing with nothing."""
    root = build(tmp_path, **{
        "query.date": "1979-05-01",
        "cross_checks.release_date": {"consensus": "2011-06-01"},
        "musicbrainz.first_release_date": "1979-05-01"})
    identity_file(root)

    rep = audit.run(root)

    assert len(find(rep, "release.date_conflict").hits) == 3
