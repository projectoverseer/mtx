"""Chart outcomes coming in, and the title cleaning that finds a play count.

The corpus describes what a record sounds like in extraordinary detail and
almost nothing about whether it worked. Its one outcome column is a Last.fm
play count, which counts scrobbling listeners -- a proxy for one kind of
enthusiast, biased by era, genre and platform in ways nothing here corrects
for. A chart peak is the closest thing to a hard outcome that exists, and no
free database carries it, so it has to be supplied.

`declared.json` had no place to put it: `schema.py` read
`declared.outcome.billboard_peak` and `declare.py` never offered the key, so
the column could not be filled by anyone following the tool.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "tools")
sys.path.insert(0, TOOLS)

charts = pytest.importorskip("charts")
declare = pytest.importorskip("declare")

from mtx.online.match import bare_title, search_title    # noqa: E402


# --- the sidecar has somewhere to put an outcome -----------------------------

def test_the_declared_template_offers_the_outcome_keys():
    """The schema read these for the life of the column; nothing wrote them."""
    outcome = declare.TEMPLATE["outcome"]

    for key in ("billboard_peak", "weeks_on_chart", "certification", "chart"):
        assert key in outcome


# --- bulk ingest -------------------------------------------------------------

def track(root, artist, album, title, credit=None):
    """`artist` names the folder; `credit` is what the file is tagged with."""
    folder = os.path.join(root, artist, album, title)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "analysis.json"), "w", encoding="utf-8") as fh:
        json.dump({"file": {"sha256": f"sha-{title}"},
                   "tags": {"named": {"artist": credit or artist,
                                      "title": title}}}, fh)
    return folder


def test_a_row_lands_in_the_declared_sidecar(tmp_path):
    root = str(tmp_path)
    folder = track(root, "Adele", "21", "Rolling in the Deep")
    index = charts.index_corpus(root)

    found, how = charts.lookup(index, {"artist": "Adele",
                                       "title": "Rolling in the Deep"})
    assert found == [folder] and how == "artist+title"

    charts.apply_row(folder, {"billboard_peak": "1", "weeks_on_chart": "65",
                              "certification": "9x platinum",
                              "chart": "Billboard Hot 100"}, False)

    doc = json.load(open(os.path.join(folder, "declared.json"), encoding="utf-8"))
    assert doc["outcome"]["billboard_peak"] == 1
    assert doc["outcome"]["weeks_on_chart"] == 65
    assert doc["outcome"]["certification"] == "9x platinum"


def test_a_hand_written_value_is_not_overwritten_by_a_bulk_load(tmp_path):
    """Somebody's own work outranks a CSV unless they say otherwise."""
    root = str(tmp_path)
    folder = track(root, "Adele", "21", "Rolling in the Deep")
    with open(os.path.join(folder, "declared.json"), "w", encoding="utf-8") as fh:
        json.dump({"outcome": {"billboard_peak": 3}}, fh)

    charts.apply_row(folder, {"billboard_peak": "1"}, False)
    doc = json.load(open(os.path.join(folder, "declared.json"), encoding="utf-8"))
    assert doc["outcome"]["billboard_peak"] == 3

    charts.apply_row(folder, {"billboard_peak": "1"}, True)
    doc = json.load(open(os.path.join(folder, "declared.json"), encoding="utf-8"))
    assert doc["outcome"]["billboard_peak"] == 1


def test_a_featured_credit_is_indexed_under_the_lead_artist(tmp_path):
    """A chart lists `Daft Punk`; the file is tagged with the whole credit."""
    root = str(tmp_path)
    folder = track(root, "Daft Punk", "RAM", "Get Lucky",
                   credit="Daft Punk ft. Pharrell Williams")
    index = charts.index_corpus(root)

    found, how = charts.lookup(index, {"artist": "Daft Punk",
                                       "title": "Get Lucky"})
    assert found == [folder]


def test_a_bracketed_suffix_still_matches_the_chart_title(tmp_path):
    """`Get Lucky` on the chart, `Get Lucky (Radio Edit - ...)` on disk."""
    root = str(tmp_path)
    folder = track(root, "Daft Punk", "Singles",
                   "Get Lucky (Radio Edit - feat. Nile Rodgers)")
    index = charts.index_corpus(root)

    found, how = charts.lookup(index, {"artist": "Daft Punk",
                                       "title": "Get Lucky"})
    assert found == [folder] and how == "artist+title prefix"


def test_two_candidates_are_refused_rather_than_guessed(tmp_path):
    """A chart peak on the wrong record is worse than one left off."""
    root = str(tmp_path)
    track(root, "Daft Punk", "Singles", "Get Lucky (Radio Edit)")
    track(root, "Daft Punk", "RAM", "Get Lucky (Album Version)")
    index = charts.index_corpus(root)

    found, how = charts.lookup(index, {"artist": "Daft Punk",
                                       "title": "Get Lucky"})
    assert found == []
    assert "ambiguous" in how


def test_one_folder_under_two_artist_keys_is_not_ambiguous_with_itself(tmp_path):
    """Counting keys instead of folders called a single track ambiguous."""
    root = str(tmp_path)
    folder = track(root, "Daft Punk", "RAM", "Get Lucky (Album Version)",
                   credit="Daft Punk ft. Pharrell Williams")
    index = charts.index_corpus(root)

    found, _how = charts.lookup(index, {"artist": "Daft Punk",
                                        "title": "Get Lucky"})
    assert found == [folder]


def test_a_sha256_beats_a_title(tmp_path):
    root = str(tmp_path)
    right = track(root, "Adele", "21", "Rolling in the Deep")
    track(root, "Adele", "Live", "Rolling in the Deep (Live)")
    index = charts.index_corpus(root)

    found, how = charts.lookup(index, {"sha256": "sha-Rolling in the Deep",
                                       "artist": "Adele", "title": "Nonsense"})
    assert found == [right] and how == "sha256"


# --- the title that finds a play count ---------------------------------------

def test_bare_title_strips_an_annotation_no_cleaner_recognises():
    """`(Debut Single 2015)` is not packaging by any word list, and not the name."""
    assert search_title("My Everything (Debut Single 2015)") == \
        "My Everything (Debut Single 2015)", "the careful cleaner keeps it"
    assert bare_title("My Everything (Debut Single 2015)") == "My Everything"


def test_bare_title_never_returns_nothing():
    """A title that is entirely bracketed still has to be asked about."""
    assert bare_title("(Untitled)") == "(Untitled)"
    assert bare_title("") == ""


def test_bare_title_leaves_a_plain_title_alone():
    assert bare_title("Rolling in the Deep") == "Rolling in the Deep"
