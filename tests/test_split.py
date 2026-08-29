"""Splitting analysis.json into uploadable parts, and putting it back together.

The contract is short: every file written is under the cap, every file is valid
JSON on its own, and rejoining returns the document that went in -- not a
rounded, reordered or truncated version of it.
"""

from __future__ import annotations

import json
import os

import pytest

from mtx.split import (DEFAULT_PART_BYTES, NOTION_UPLOAD_LIMIT, join,
                       load_analysis, write_analysis)


def _doc(n_points: int = 4000) -> dict:
    """A result-shaped document: small sections plus two heavy timelines."""
    return {
        "headline": {"lufs_i": -8.5, "dr14": 6.0},
        "run": {"tool_version": "0.0.0", "schema_version": "1.1.0"},
        "warnings": ["loudness: something worth saying"],
        "tags": {"named": {"title": "Ténèbres", "artist": "ROSÉ"}},
        "processing": {
            "note": "a small key next to a large one",
            "timeline": [{"t": i / 10.0, "db": -i * 0.001} for i in range(n_points)],
        },
        "spectrum": {
            "ltas": [round(-i * 0.01, 4) for i in range(n_points)],
            "tilt": {"slope_db_per_oct": -3.1},
        },
    }


def _files(out) -> list[str]:
    return sorted(p.name for p in out.iterdir())


def test_a_document_under_the_cap_is_written_whole(tmp_path):
    doc = _doc()
    written = write_analysis(doc, str(tmp_path), max_bytes=DEFAULT_PART_BYTES)
    assert list(written) == ["analysis.json"]
    assert _files(tmp_path) == ["analysis.json"]
    assert json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8")) == doc


def test_no_split_writes_one_file_however_large(tmp_path):
    doc = _doc()
    written = write_analysis(doc, str(tmp_path), max_bytes=None)
    assert list(written) == ["analysis.json"]
    assert load_analysis(written["analysis.json"]) == doc


@pytest.mark.parametrize("cap", [200_000, 50_000, 20_000])
def test_every_part_fits_under_the_cap_and_rejoins_exactly(tmp_path, cap):
    doc = _doc()
    written = write_analysis(doc, str(tmp_path), max_bytes=cap)
    assert len(written) > 1, "an oversize document has to produce parts"
    for name, path in written.items():
        assert os.path.getsize(path) <= cap, f"{name} is over the cap"
        with open(path, encoding="utf-8") as f:
            json.load(f)  # every part stands on its own as JSON
    assert load_analysis(str(tmp_path / "analysis.json")) == doc


def test_the_index_keeps_the_headline_and_names_its_parts(tmp_path):
    doc = _doc()
    write_analysis(doc, str(tmp_path), max_bytes=50_000)
    index = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
    # What a reader needs first stays inline; the heavy sections are pointers.
    assert index["headline"] == doc["headline"]
    assert index["warnings"] == doc["warnings"]
    assert index["split"]["part_max_bytes"] == 50_000
    assert index["split"]["whole_bytes"] > 50_000
    for entry in index["split"]["parts"]:
        assert (tmp_path / entry["file"]).is_file()
    for section in index["split"]["sections_in_parts"]:
        assert index[section]["mtx_moved"] is True
        assert index[section]["parts"]


def test_a_missing_part_is_an_error_not_a_short_document(tmp_path):
    doc = _doc()
    written = write_analysis(doc, str(tmp_path), max_bytes=50_000)
    victim = [p for n, p in written.items() if n != "analysis.json"][0]
    os.remove(victim)
    with pytest.raises(FileNotFoundError):
        load_analysis(str(tmp_path / "analysis.json"))


def test_join_writes_the_whole_document_next_to_the_index(tmp_path):
    doc = _doc()
    write_analysis(doc, str(tmp_path), max_bytes=50_000)
    out = join(str(tmp_path / "analysis.json"))
    assert os.path.basename(out) == "analysis.full.json"
    assert json.loads(open(out, encoding="utf-8").read()) == doc
    # A directory works as well as the index file itself.
    assert json.loads(open(join(str(tmp_path)), encoding="utf-8").read()) == doc


def test_the_default_cap_is_under_the_notion_upload_limit():
    assert DEFAULT_PART_BYTES < NOTION_UPLOAD_LIMIT


def test_a_long_list_is_cut_into_absolute_slices(tmp_path):
    doc = {"spectrum": {"ltas": list(range(20000))}, "headline": {"lufs_i": -8.0}}
    write_analysis(doc, str(tmp_path), max_bytes=20_000)
    index = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
    slices = [e["slice"] for e in index["split"]["parts"] if e["slice"]]
    assert slices, "a list too long for one part is chunked"
    assert slices[0][0] == 0
    for a, b in zip(slices, slices[1:]):
        assert a[1] == b[0], "slices are consecutive and absolute"
    assert load_analysis(str(tmp_path / "analysis.json")) == doc
