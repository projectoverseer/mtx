"""Fabricated result dictionaries shared by the layout and prediction tests.

These tests are about what the digest is allowed to print, drop or leave out,
so they run against a hand-built result rather than an analysed file: the DSP
is tested elsewhere, and a fixture makes the layout contract explicit.
"""

from __future__ import annotations

import pytest

THIRD_OCTAVE = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400,
                500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
                6300, 8000, 10000, 12500, 16000, 20000]


def stereo_block() -> dict:
    """Enough of a stereo block to make two DETAIL blocks render.

    The budget tests need something real to drop; an all-unavailable fixture
    would assert nothing about the assembly.
    """
    return {
        "available": True,
        "side_minus_mid_db": -7.0,
        "side_minus_mid_below_120hz_db": -30.0,
        "mono_crossover_hz": 120.0,
        "side_minus_mid_per_third_octave": [
            {"centre_hz": c, "side_minus_mid_db": -12.0} for c in THIRD_OCTAVE],
        "mono_sum_damage": {"broadband_loss_db": -0.42,
                            "per_third_octave": [
                                {"centre_hz": c, "mono_sum_loss_db": -1.1}
                                for c in THIRD_OCTAVE]},
        "correlation": {"overall": 0.8, "min": 0.4, "p5": 0.5, "median": 0.82,
                        "pct_time_below_0": 0.0, "pct_time_below_0_3": 1.2,
                        "most_negative_windows": [{"start": "1:02.000",
                                                   "correlation": 0.31}]},
        "channel_balance": {"rms_l_minus_r_db": 0.1, "lufs_l_minus_r": 0.05},
        "inter_channel_time_offset": {"lag_samples": 0, "lag_us": 0.0,
                                      "correlation_at_lag": 0.81},
        "goniometer": {"fraction_energy_outside_45_deg": 0.12},
    }


def stem_entry(lufs: float, tilt: float) -> dict:
    return {
        "source": "separated",
        "loudness": {"integrated_lufs": lufs},
        "dynamics": {"crest": {"whole_file_db": 12.0,
                               "loudest_window": {"crest_db": 10.5}}},
        "spectrum": {"available": True,
                     "tilt": {"slope_db_per_oct": tilt, "r2": 0.8},
                     "sub_band_pct": 18.2, "air_band_pct": 1.4},
        "stereo": {"available": True, "side_minus_mid_db": -11.4,
                   "correlation": {"overall": 0.81}},
        "level_vs_mix": {"rms_db": -6.2, "lufs_delta": lufs + 4.8},
        "warnings": [],
    }


@pytest.fixture
def res() -> dict:
    """A result dictionary with every field the digest reads."""
    return {
        "file": {"filename": "x.flac", "sha256": "a" * 64},
        "audio": {"sample_rate_hz": 44100, "channels": 2, "subtype": "PCM_24",
                  "duration_s": 200.0},
        "tags": {"named": {"title": None, "artist": None}},
        "run": {"tool_version": "0.2.0", "schema_version": "1.1.0",
                "profile": "full"},
        "warnings": [], "confidence_notes": [],
        "loudness": {"true_peak": {},
                     "dr14": {"validation": {
                         "validated_against_published_reference": False,
                         "record": {}}}},
        "stereo": stereo_block(),
        "spectrum": {"available": False},
        "dynamics": {},
        "structure": {"available": False},
        "processing": {},
        "stems": {"requested": False},
        "headline": {
            "lufs_i": -9.37, "lra_lu": 5.0, "true_peak_dbtp_16x": -0.3,
            "sample_peak_dbfs": -1.0, "plr_db": 8.7, "psr_min_db": 5.5,
            "psr_min_time": "1:02.000", "psr_median_db": 9.1, "dr14": 8,
            "crest_whole_db": 11.0, "crest_loudest_10s_db": 9.4,
            "spectral_tilt_db_per_oct": -3.0, "spectral_tilt_r2": 0.7,
            "air_band_pct": 1.1, "sub_band_pct": 12.0,
            "side_minus_mid_db": -7.0, "side_minus_mid_below_120hz_db": -30.0,
            "mono_crossover_hz": 120.0, "correlation_mean": 0.8,
            "correlation_min": 0.4, "flat_top_sample_count": 12,
            "flat_top_longest_run_ms": 0.3, "hf_cutoff_hz": 20000.0,
            "effective_bit_depth": 24, "tempo_bpm": 124.0, "key": "A minor",
            "section_count": 9, "duration_s": 200.0,
        },
    }


@pytest.fixture
def res_with_stems(res: dict) -> dict:
    res["stems"] = {
        "requested": True, "available": True, "model": "htdemucs",
        "source": "separated",
        "caveat": "every number below is measured on a separated stem, not on "
                  "the mix; separation artefacts are part of the measurement",
        "stems": {"drums": stem_entry(-16.4, -2.1), "bass": stem_entry(-19.7, -8.8),
                  "other": stem_entry(-14.3, -5.2), "vocals": stem_entry(-19.4, -4.4)},
    }
    return res
