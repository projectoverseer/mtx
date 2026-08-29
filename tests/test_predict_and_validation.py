"""The prediction sheet, the calibration check, and the DR14 validation record.

The property under test in the first group is that a prediction sheet cannot
leak the answer: a peeked prediction is worth nothing, so a sheet that carries
the measured values anywhere in it is a bug, not a cosmetic issue.
"""

from __future__ import annotations

import json

import pytest

from mtx.digest import BLANK, PREDICT_FIELDS, _headline
from mtx.predict import (check, parse_predictions, read_actuals,
                         render_predict_sheet)
from mtx.validation import add_entry, summary


# ------------------------------------------------------------------ the sheet
def test_every_predictable_label_is_printed_in_the_headline(res):
    """One table behind both, so a renamed field cannot silently stop scoring."""
    block = _headline(res)
    for label, _key, _unit, _nd in PREDICT_FIELDS:
        assert f"\n{label}" in block, f"{label} is not a HEADLINE row"


def test_the_sheet_carries_no_measured_value(res):
    sheet = render_predict_sheet(res)
    body = sheet.split("## PREDICTIONS", 1)[1].split("## FLAGS", 1)[0]
    assert BLANK in body
    for label, key, _unit, _nd in PREDICT_FIELDS:
        value = res["headline"][key]
        if value is None:
            continue
        assert str(value) not in body, f"{label} leaked its value into the sheet"


def test_the_sheet_keeps_method_and_flags_but_not_detail_or_corpus_row(res):
    sheet = render_predict_sheet(res)
    assert "## METHOD" in sheet and "## FLAGS" in sheet
    assert "## DETAIL" not in sheet, "DETAIL restates the headline numbers"
    assert "## CORPUS ROW" not in sheet, "the corpus row restates them too"


# ------------------------------------------------------------------ the check
FILLED = """
LUFS-I           = -8.0  +/- 1.0   conf 70%
DR14             = 11  +/- 2   conf 60%
PSR min          = 5.0 +/- 2.0 conf 40
Tempo            = ____  +/- ____   conf ____%
Nonsense field   = 3 +/- 1 conf 50%
"""


def test_parsing_takes_filled_lines_and_ignores_the_rest():
    preds = parse_predictions(FILLED)
    assert set(preds) == {"LUFS-I", "DR14", "PSR min"}
    assert preds["LUFS-I"] == {"value": -8.0, "range": 1.0, "confidence": 70.0}
    assert preds["PSR min"]["confidence"] == 40.0


def test_check_reports_signed_error_absolute_error_and_the_interval(res):
    result = check(parse_predictions(FILLED),
                   {label: res["headline"][key] for label, key, _, _ in PREDICT_FIELDS})
    rows = {r["field"]: r for r in result["rows"]}
    assert rows["LUFS-I"]["error"] == pytest.approx(1.37)
    assert rows["LUFS-I"]["abs_error"] == pytest.approx(1.37)
    assert rows["LUFS-I"]["interval_held"] is False
    assert rows["PSR min"]["interval_held"] is True
    assert result["intervals_stated"] == 3 and result["intervals_held"] == 1


def test_the_most_confident_miss_comes_first(res):
    """|error| x confidence is the order the errors are worth reading in."""
    result = check(parse_predictions(FILLED),
                   {label: res["headline"][key] for label, key, _, _ in PREDICT_FIELDS})
    assert result["rows"][0]["field"] == "DR14"  # 3 DR out at 60% beats the rest


def test_reading_actuals_back_from_a_digest_matches_the_json(res, tmp_path):
    from mtx.digest import render_digest
    d = tmp_path / "digest.md"
    d.write_text(render_digest(res), encoding="utf-8")
    j = tmp_path / "analysis.json"
    j.write_text(json.dumps(res), encoding="utf-8")
    from_digest = read_actuals(str(d))
    from_json = read_actuals(str(j))
    for label, key, _unit, nd in PREDICT_FIELDS:
        if from_json[label] is None:
            continue
        assert from_digest[label] == pytest.approx(from_json[label], abs=10 ** -nd)


# ------------------------------------------------- the DR14 validation record
def test_an_empty_record_is_not_validated(tmp_path):
    rec = summary(str(tmp_path / "none.json"))
    assert rec["validated"] is False and rec["tracks_checked"] == 0


def test_one_agreeing_track_validates_and_a_disagreeing_one_does_not(tmp_path):
    store = str(tmp_path / "dr14.json")
    add_entry({"title": "a", "sha256": "1", "published_dr": 9.0,
               "measured_dr": 9.4, "delta": 0.4}, store)
    rec = summary(store)
    assert rec["validated"] is True and rec["tracks_checked"] == 1
    add_entry({"title": "b", "sha256": "2", "published_dr": 12.0,
               "measured_dr": 6.0, "delta": -6.0}, store)
    rec = summary(store)
    assert rec["validated"] is False
    assert rec["tracks_checked"] == 2 and rec["tracks_within_tolerance"] == 1
    assert rec["max_abs_delta_dr"] == 6.0


def test_rechecking_the_same_file_replaces_its_entry(tmp_path):
    store = str(tmp_path / "dr14.json")
    add_entry({"title": "a", "sha256": "1", "published_dr": 9.0,
               "measured_dr": 3.0, "delta": -6.0}, store)
    add_entry({"title": "a", "sha256": "1", "published_dr": 9.0,
               "measured_dr": 9.1, "delta": 0.1}, store)
    rec = summary(store)
    assert rec["tracks_checked"] == 1 and rec["validated"] is True


def test_an_unreadable_record_does_not_claim_validation(tmp_path):
    store = tmp_path / "broken.json"
    store.write_text("{not json", encoding="utf-8")
    rec = summary(str(store))
    assert rec["validated"] is False and rec["tracks_checked"] == 0


def test_the_flags_line_changes_with_the_record(res, tmp_path):
    from mtx.digest import _flags
    assert "[unverified] DR14" in _flags(res)
    res["loudness"]["dr14"]["validation"] = {
        "validated_against_published_reference": True,
        "record": {"tracks_checked": 2, "max_abs_delta_dr": 0.4,
                   "tolerance_dr": 1.0, "tracks_within_tolerance": 2}}
    assert "[validated against 2 track(s)]" in _flags(res)
