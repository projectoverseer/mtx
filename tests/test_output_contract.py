"""The output contract: digest size, reproducibility, and no fabricated numbers."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import soundfile as sf

from mtx import RUN_VOLATILE_FIELDS
from mtx.analyze import analyze_file, write_outputs
from mtx.digest import SIZE_BUDGET_BYTES, render_digest

SR = 44100


@pytest.fixture(scope="module")
def analysed(tmp_path_factory):
    """A short synthetic track analysed once for the whole module."""
    d = tmp_path_factory.mktemp("contract")
    t = np.arange(int(20 * SR)) / SR
    rng = np.random.default_rng(0)
    x = (0.4 * np.sin(2 * np.pi * 110 * t) + 0.1 * rng.standard_normal(t.size))
    y = (0.4 * np.sin(2 * np.pi * 110 * t) + 0.1 * rng.standard_normal(t.size))
    sig = np.clip(np.stack([x, y], axis=1) * 1.5, -0.7, 0.7)
    path = d / "contract.flac"
    sf.write(path, sig, SR, subtype="PCM_24")
    return str(path), analyze_file(str(path), profile="quick"), d


def test_digest_fits_the_budget(analysed):
    _, res, _ = analysed
    digest = render_digest(res)
    assert len(digest.encode("utf-8")) <= SIZE_BUDGET_BYTES


def test_digest_has_the_five_required_sections_in_order(analysed):
    _, res, _ = analysed
    digest = render_digest(res)
    order = ["## HEADLINE", "## FLAGS", "## DETAIL", "## CORPUS ROW", "## METHOD"]
    positions = [digest.index(s) for s in order]
    assert positions == sorted(positions), "digest sections are out of order"


def test_corpus_row_never_guesses(analysed):
    _, res, _ = analysed
    digest = render_digest(res)
    block = digest.split("## CORPUS ROW")[1]
    # The file has no tags, so these fields must be present and empty.
    for field in ("Title:", "Artist:", "Year:", "Genre:", "Engineers:"):
        line = next(l for l in block.splitlines() if l.startswith(field))
        assert line.strip() == field.rstrip(":") + ":", f"{field} was filled in by guesswork"


def test_missing_metrics_are_null_and_explained(analysed):
    _, res, _ = analysed
    # quick profile skips the 16x true peak; it must be null AND explained.
    assert res["loudness"]["true_peak"]["overall_dbtp_16x"] is None
    assert res["loudness"]["true_peak"]["overs"]["skipped"] is True
    assert any("quick" in w for w in res["warnings"])


def test_json_round_trips_without_nan_or_infinity(analysed):
    path, res, d = analysed
    out = d / "outdir"
    write_outputs(res, str(out), json_only=True)
    text = (out / "analysis.json").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    loaded = json.loads(text)
    assert loaded["run"]["schema_version"]


def test_two_runs_are_byte_identical_apart_from_the_volatile_fields(analysed):
    path, _, d = analysed
    a = analyze_file(path, profile="quick")
    b = analyze_file(path, profile="quick")
    for res in (a, b):
        for field in RUN_VOLATILE_FIELDS:
            section, key = field.split(".")
            res[section].pop(key, None)
    from mtx.util import jsonable
    sa = json.dumps(jsonable(a), sort_keys=True, allow_nan=False)
    sb = json.dumps(jsonable(b), sort_keys=True, allow_nan=False)
    assert sa == sb


def test_every_headline_number_is_a_number_or_none(analysed):
    _, res, _ = analysed
    for key, value in res["headline"].items():
        if value is None or isinstance(value, str):
            continue
        assert math.isfinite(float(value)), f"headline.{key} is not finite"


def test_params_block_is_emitted_with_the_result(analysed):
    _, res, _ = analysed
    for group in ("loudness", "true_peak", "dr14", "flat_top", "spectrum",
                  "stereo", "structure", "processing", "forensics"):
        assert group in res["params"], f"params.{group} is missing"
    assert res["params"]["profile"]["profile"] == "quick"
    assert res["params"]["profile"]["skipped_in_quick"]
