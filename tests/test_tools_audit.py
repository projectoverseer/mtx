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


def track(root, artist, album, title, *, online=None, row=None, run=None):
    """One analysed folder, shaped the way `mtx scan` shapes them."""
    folder = os.path.join(root, artist, album, title)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "analysis.json"), "w", encoding="utf-8") as fh:
        json.dump({"file": {"sha256": title}}, fh)
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
