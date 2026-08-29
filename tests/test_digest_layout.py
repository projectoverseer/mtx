"""Digest layout: the stems section, section selection and the budget.

Contract tests, not DSP tests: the `res` and `res_with_stems` fixtures in
conftest stand in for an analysed file, and what is under test is what the
digest is allowed to drop, rename or leave out.
"""

from __future__ import annotations

import json

import pytest

from mtx.digest import (SIZE_BUDGET_BYTES, STEMS_BUDGET_BONUS, corpus_row_dict,
                        render_digest, resolve_sections, run_provenance)


def test_stems_reach_the_digest_when_they_were_measured(res_with_stems):
    out = render_digest(res_with_stems)
    assert "## STEMS" in out
    body = out.split("## STEMS", 1)[1].split("## CORPUS ROW", 1)[0]
    for stem in ("drums", "bass", "other", "vocals"):
        assert stem in body
    for column in ("lvl_vs_mix", "lufs_delta", "LUFS-I", "crest10s", "tilt(R2)",
                   "side/mid", "corr", "sub%", "air%"):
        assert column in body


def test_stems_name_their_model_and_carry_the_caveat(res_with_stems):
    out = render_digest(res_with_stems)
    body = out.split("## STEMS", 1)[1].split("## CORPUS ROW", 1)[0]
    assert "model: htdemucs" in body, "stems from different models must not be comparable by accident"
    assert "stems: htdemucs" in out.split("## HEADLINE")[0]
    assert "FLAGS" in body and "separation artefacts" in body


def test_no_stems_section_without_stems(res):
    assert "## STEMS" not in render_digest(res)


def test_a_stems_run_gets_a_bigger_budget(res_with_stems):
    """The stem table exists nowhere else in the paste-able output."""
    out = render_digest(res_with_stems)
    assert len(out.encode("utf-8")) <= SIZE_BUDGET_BYTES + STEMS_BUDGET_BONUS
    assert "## STEMS" in out


def test_section_order_survives_the_stems_section(res_with_stems):
    out = render_digest(res_with_stems)
    order = ["## HEADLINE", "## FLAGS", "## DETAIL", "## STEMS", "## CORPUS ROW",
             "## METHOD"]
    positions = [out.index(s) for s in order]
    assert positions == sorted(positions)


def test_sections_selects_by_group_and_by_block_name():
    assert "stereo detail" in resolve_sections(["stereo"])
    assert "side/mid per third-octave" in resolve_sections(["stereo"])
    assert resolve_sections(["Band Energy"]) == {"band energy"}
    assert "sections" in resolve_sections(["structure"])


def test_an_unknown_section_name_is_an_error_not_a_silent_drop():
    with pytest.raises(ValueError) as exc:
        resolve_sections(["stereoo"])
    assert "stereoo" in str(exc.value)
    assert "groups:" in str(exc.value)


def test_digest_budget_is_a_default_not_a_law(res):
    default = render_digest(res)
    assert "Dropped to stay under" not in default
    small = render_digest(res, budget=5120)
    assert len(small.encode("utf-8")) < len(default.encode("utf-8"))
    assert "Dropped to stay under the 5 KB digest budget" in small


def test_a_budget_that_cannot_be_met_says_so(res):
    """The never-dropped sections are a floor; meeting a cap below it would
    mean dropping provenance, so the digest says the cap was not met."""
    tiny = render_digest(res, budget=1024)
    assert "could not be met" in tiny
    assert "## CORPUS ROW" in tiny and "## METHOD" in tiny


def test_corpus_row_carries_the_fields_the_csv_already_computed(res):
    out = render_digest(res)
    block = out.split("## CORPUS ROW", 1)[1]
    for field in ("PSR min:", "PSR median:", "DR14:", "Crest (loudest 10s):",
                  "mtx run:"):
        assert field in block


def test_run_provenance_names_version_schema_profile_and_hash(res):
    line = run_provenance(res)
    assert "mtx 0.2.0" in line and "schema 1.1.0" in line
    assert "profile full" in line and "sha256 aaaaaaaaaaaaaaaa" in line


def test_corpus_row_json_is_typed_and_never_guesses(res):
    row = corpus_row_dict(res)
    assert row["LUFS-I"] == -9.37 and isinstance(row["LUFS-I"], float)
    assert row["PSR min"] == 5.5 and row["DR14"] == 8.0
    assert row["Crest (loudest 10s)"] == 9.4
    assert row["Title"] is None and row["Engineers"] is None
    assert row["mtx run"].startswith("mtx 0.2.0")
    # It has to survive a JSON round trip to be pasteable at all.
    assert json.loads(json.dumps(row))["_units"]["PSR min"] == "dB"


def test_corpus_row_json_names_the_stem_model_in_provenance(res_with_stems):
    assert "stems htdemucs" in corpus_row_dict(res_with_stems)["mtx run"]
