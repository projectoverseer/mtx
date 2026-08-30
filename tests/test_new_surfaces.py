"""Declared metadata, version identity, coverage, export, cohort and lyrics.

None of these are DSP, so none of them need audio: they are tested against
hand-built documents, which also makes the contract each one keeps explicit.
"""

from __future__ import annotations

import json

import pytest

from mtx import coverage as m_coverage
from mtx import declared as m_declared


# ------------------------------------------------------------------- declared
@pytest.mark.parametrize("title,markers,is_primary", [
    ("Lovely", [], True),
    ("Lovely (Radio Edit)", ["radio_edit"], False),
    ("Bad Guy - Sped Up", ["sped_up"], False),
    ("Levitating (feat. DaBaby) [Clean]", ["clean"], False),
    ("Blinding Lights (Live)", ["live"], False),
    ("Song (Chris Lake Remix)", ["remix"], False),
    ("Song - 2011 Remaster", ["remaster"], False),
])
def test_version_markers_are_read_from_the_title(title, markers, is_primary):
    got = m_declared.version_identity({"named": {"title": title, "artist": "X"},
                                       "all": {}})
    assert got["markers"] == markers, got
    assert got["is_primary_version"] is is_primary


def test_two_versions_of_one_song_share_a_work_key():
    a = m_declared.work_key("Billie Eilish", "lovely")
    b = m_declared.work_key("Billie Eilish", "lovely (Radio Edit)")
    c = m_declared.work_key("Billie Eilish", "Lovely - Sped Up")
    d = m_declared.work_key("Billie Eilish", "bad guy")
    assert a == b == c, (a, b, c)
    assert a != d
    # Accents, punctuation and a feat. clause do not make a different work.
    assert (m_declared.work_key("ROSÉ", "APT.")
            == m_declared.work_key("Rose", "Apt (feat. Bruno Mars)"))


def test_a_declared_sidecar_is_labelled_and_never_merged(tmp_path):
    from mtx.util import Collector

    audio = tmp_path / "song.flac"
    audio.write_bytes(b"")
    (tmp_path / "declared.json").write_text(json.dumps({
        "lyrics": "a line\nanother line",
        "genre": "alt-pop",
        "writers": [{"name": "Someone", "share_pct": 50}],
        "not_a_real_field": 1,
    }), encoding="utf-8")

    collector = Collector()
    got = m_declared.load(str(audio), collector)
    assert got["available"] and got["source"] == "declared"
    for name, entry in got["fields"].items():
        assert entry["source"] == "declared", name
    assert got["unknown_fields"] == ["not_a_real_field"]
    assert any("not_a_real_field" in w for w in collector.warnings)
    assert m_declared.declared_value(got, "genre") == "alt-pop"
    assert m_declared.declared_value(got, "isrc") is None


def test_a_missing_sidecar_says_where_it_looked(tmp_path):
    got = m_declared.load(str(tmp_path / "nope.flac"))
    assert got["available"] is False
    assert got["searched"]
    assert "lyrics" in got["schema"]


# ------------------------------------------------------------------- coverage
def test_the_coverage_mask_marks_nulls_and_unavailable_blocks():
    res = {
        "headline": {"lufs_i": -9.0, "dr14": None, "key": "A minor"},
        "structure": {"available": True, "section_count": 4,
                      "tempo": {"available": True, "bpm": 120.0,
                                "confidence": "low"}},
        "harmony": {"available": False, "reason": "skipped by --profile quick"},
        "params": {"ignored": True},
        "run": {"ignored": True},
        "warnings": ["a warning"],
        "confidence_notes": [{"metric": "structure.tempo", "confidence": "low",
                              "reason": "because"}],
    }
    mask = m_coverage.build(res)
    f = mask["features"]

    assert f["headline.lufs_i"]["present"] is True
    assert f["headline.dr14"]["present"] is False
    assert f["harmony"]["present"] is False
    assert f["harmony"]["reason"] == "skipped by --profile quick"
    # Confidence flows down from the enclosing block.
    assert f["structure.tempo.bpm"]["confidence"] == "low"
    # params and run are provenance, not measurement.
    assert not any(k.startswith(("params", "run")) for k in f)
    assert mask["present_count"] < mask["feature_count"]
    assert mask["by_group"]["headline"]["features"] == 3
    assert mask["declared_confidences"]["structure.tempo"] == "low"


def test_coverage_counts_an_empty_series_as_absent():
    mask = m_coverage.build({"structure": {"available": True,
                                           "boundaries_s": [],
                                           "beat_times_s": [1.0, 2.0]}})
    f = mask["features"]
    assert f["structure.boundaries_s"]["present"] is False
    assert f["structure.beat_times_s"]["present"] is True
    assert f["structure.beat_times_s"]["length"] == 2


# --------------------------------------------------------------------- export
def _fake_analysis() -> dict:
    return {
        "file": {"sha256": "a" * 64},
        "audio": {"duration_s": 100.0},
        "tags": {"named": {"title": "T", "artist": "A"},
                 "all": {"junk": "x" * 5000}},
        "headline": {"lufs_i": -9.0, "key": "A minor"},
        "structure": {"available": True, "sections": [
            {"index": 0, "start_s": 0.0, "end_s": 50.0, "duration_s": 50.0,
             "lufs_i": -12.0, "band_energy_pct": {"sub": 10.0}},
            {"index": 1, "start_s": 50.0, "end_s": 100.0, "duration_s": 50.0,
             "lufs_i": -8.0, "band_energy_pct": {"sub": 12.0}}]},
        "form": {"available": True, "sections": [
            {"index": 0, "letter": "A", "label": "verse",
             "label_confidence": "medium", "vocal_present": True},
            {"index": 1, "letter": "B", "label": "chorus",
             "label_confidence": "medium", "vocal_present": True}]},
        "rhythm": {"pulse_rate": {"per_section": [
            {"section_index": 0, "onsets_per_beat": 1.0, "relative_pulse_rate": 1.0},
            {"section_index": 1, "onsets_per_beat": 2.0, "relative_pulse_rate": 2.0}]}},
        "stems": {"masking": {"per_section": [
            {"index": 0, "masking_index_db": {"drums_into_vocals": -3.0},
             "vocal_minus_instrumental_lu": 2.0},
            {"index": 1, "masking_index_db": {"drums_into_vocals": -1.0},
             "vocal_minus_instrumental_lu": 1.0}]}},
    }


def test_the_section_table_joins_everything_indexed_by_section(tmp_path):
    from mtx.export import section_rows, track_row

    res = _fake_analysis()
    rows = section_rows(res, str(tmp_path / "analysis.json"))
    assert len(rows) == 2
    a, b = rows
    assert a["section.index"] == 0 and b["section.index"] == 1
    assert a["form.label"] == "verse" and b["form.label"] == "chorus"
    assert b["rhythm.relative_pulse_rate"] == 2.0
    assert b["masking.drums_into_vocals_db"] == -1.0
    assert a["tags.title"] == "T"
    # Bulk arrays never reach a flat table.
    assert not any(k.endswith(".all") for k in track_row(res, "x"))


def test_the_track_table_flattens_scalars_under_dotted_paths():
    from mtx.export import flatten

    flat = flatten(_fake_analysis())
    assert flat["headline.lufs_i"] == -9.0
    assert flat["tags.named.artist"] == "A"
    assert "structure.sections" not in flat
    assert "params" not in flat


# --------------------------------------------------------------------- cohort
def test_cohort_labels_prefer_declared_then_online_then_tag(tmp_path):
    from mtx.cohort import labels_for

    res = {"tags": {"named": {"artist": "A", "title": "T", "genre": "rock",
                              "date": "1994-05-01"}}}
    folder = tmp_path / "t"
    folder.mkdir()
    got = labels_for(res, str(folder))
    assert (got["genre"], got["genre_source"]) == ("rock", "file:tag")
    assert (got["year"], got["year_source"]) == (1994, "file:tag")

    (folder / "online.json").write_text(json.dumps(
        {"genre": {"umbrella": "Grunge"}, "release": {"date": "1994-09-27"}}),
        encoding="utf-8")
    got = labels_for(res, str(folder))
    assert (got["genre"], got["genre_source"]) == ("grunge", "online")

    (folder / "declared.json").write_text(json.dumps(
        {"cohort": {"genre": "Alt-Pop", "year": 2026}}), encoding="utf-8")
    got = labels_for(res, str(folder))
    assert (got["genre"], got["genre_source"]) == ("alt-pop", "declared")
    assert (got["year"], got["year_source"]) == (2026, "declared")


def test_percentile_and_z_are_the_stated_definitions():
    from mtx.cohort import _percentile_of, _z

    pool = [1.0, 2.0, 3.0, 4.0]
    assert _percentile_of(0.5, pool) == 0.0
    assert _percentile_of(5.0, pool) == 100.0
    # Ties count as half, as the parameter block says.
    assert _percentile_of(2.0, pool) == pytest.approx(37.5)
    assert _z(2.5, pool) == pytest.approx(0.0)
    assert _z(2.5, [1.0]) is None


def test_corpus_hygiene_calls_out_a_one_artist_corpus():
    from mtx.cohort import _hygiene

    rows = [{"artist": "Billie Eilish"} for _ in range(21)]
    rows += [{"artist": f"Other {i}"} for i in range(43)]
    got = _hygiene(rows, {"all": list(range(64))})
    assert got["clean"] is False
    assert got["largest_artist"] == "Billie Eilish"
    assert got["largest_artist_share"] == pytest.approx(21 / 64)
    joined = " ".join(got["problems"])
    assert "stratification" in joined
    assert "64 tracks is below" in joined


# --------------------------------------------------------------------- lyrics
def test_the_syllable_counter_declines_outside_english():
    from mtx.metrics.lyrics import _text_stats

    en = _text_stats("the quick brown fox\njumped over it", "en", None)
    assert en["syllables"]["total"] > 0
    assert en["readability_flesch"] is not None

    vi = _text_stats("Anh khong the noi\nnhung loi yeu thuong", "vi", None)
    assert vi["syllables"]["available"] is False
    assert "English heuristic" in vi["syllables"]["reason"]
    assert vi["readability_flesch"] is None


def test_language_detection_uses_script_then_stopwords():
    from mtx.metrics.lyrics import detect_language

    en = detect_language("i know you want it but i can't give you all of that")
    assert en["language"] == "en" and en["basis"] == "stop-word frequency"
    ko = detect_language("아무도 모르게 우리 둘만의 이야기")
    assert ko["language"] == "ko" and ko["basis"] == "script"
    assert detect_language("")["available"] is False


def test_repetition_and_title_occurrences_are_counted():
    from mtx.metrics.lyrics import _text_stats

    text = ("lovely is the word\n" * 3) + "and nothing else at all\n"
    got = _text_stats(text, "en", "Lovely")
    assert got["lines"] == 4
    assert got["repeated_line_pct"] == pytest.approx(75.0)
    assert got["compression_ratio"] > 1.0
    # The longest repeated phrase runs across the line break and
    # occurs twice; the *most* repeated one is the line itself.
    assert got["longest_repeated_ngram"]["occurrences"] == 2
    assert got["most_repeated_ngram"]["occurrences"] == 3
    assert got["most_repeated_ngram"]["text"] == "lovely is the word"
    assert got["title_in_lyric"]["occurrences"] == 3
    assert got["title_in_lyric"]["first_line_index"] == 0


def test_rhyme_scheme_splits_perfect_from_slant():
    from mtx.metrics.lyrics import _rhymes

    got = _rhymes(["i saw the light", "it was so bright",
                   "then came the day", "and went away"])
    assert got["scheme"][:2] == "aa"
    assert got["scheme"][2] == got["scheme"][3]
    assert got["perfect"] >= 2
    assert got["confidence"] == "low"


def test_no_lyric_from_any_source_says_so_rather_than_guessing():
    from mtx.metrics import lyrics as m_lyrics
    from mtx.util import Collector

    got = m_lyrics.analyse({"named": {}, "all": {}}, None, None, None, Collector())
    assert got["available"] is False
    assert "no lyric" in got["reason"]
    assert got["source"] is None
    assert "coverage_note" in got


def test_a_declared_lyric_outranks_a_tag():
    from mtx.metrics import lyrics as m_lyrics
    from mtx.util import Collector

    tags = {"named": {"title": "T"}, "all": {"LYRICS": "tagged words here"}}
    got = m_lyrics.analyse(tags, None, None, None, Collector())
    assert got["source"] == "file:tag"

    dec = {"available": True, "fields": {
        "lyrics": {"value": "declared words here", "source": "declared"}}}
    got = m_lyrics.analyse(tags, dec, None, None, Collector())
    assert got["source"] == "declared"
    assert got["is_inference"] is False
