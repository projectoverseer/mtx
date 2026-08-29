"""Every tunable in one place.

The values here are echoed verbatim into `params` in analysis.json, so a number
in the output can always be traced back to the settings that produced it.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------- band edges
# 8 analysis bands, used for band energy, per-band crest, per-band envelopes,
# the spectral timeline and the transient density map.
BANDS: tuple[tuple[str, float, float], ...] = (
    ("sub", 20.0, 60.0),
    ("bass", 60.0, 120.0),
    ("low_bass", 120.0, 250.0),
    ("low_mid", 250.0, 500.0),
    ("mid", 500.0, 2000.0),
    ("high_mid", 2000.0, 6000.0),
    ("presence", 6000.0, 12000.0),
    ("air", 12000.0, 20000.0),
)

# ISO 266 preferred third-octave centres, 20 Hz .. 20 kHz.
THIRD_OCTAVE_CENTRES: tuple[float, ...] = (
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
    800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
    12500, 16000, 20000,
)

# Zwicker Bark band edges (24 critical bands), Hz.
BARK_EDGES: tuple[float, ...] = (
    20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000,
    2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500,
)

# Lossy-codec shelf frequencies that a detected cutoff is matched against.
CODEC_SHELVES_HZ: tuple[float, ...] = (
    11025, 15000, 15500, 16000, 16500, 17000, 18000, 19000, 19500, 20000,
    20500, 22050,
)

PARAMS: dict[str, Any] = {
    "general": {
        "random_seed": 0,
        "decode_dtype": "float32 (decode) / float64 (maths)",
        "midside_convention": "mid = (L+R)/2, side = (L-R)/2",
        "db_floor_dbfs": -200.0,
        "band_analysis_sr_rule": "min(native_sr, 48000); actual value in audio.band_analysis_sr_hz",
        "librosa_sr_hz": 22050,
        "librosa_hop_length": 512,
        "librosa_n_fft": 2048,
    },
    "loudness": {
        "standard": "ITU-R BS.1770-4 / EBU R128",
        "weighting": "K (RLB high-pass + high-shelf)",
        "block_ms": 400,
        "block_overlap_pct": 75,
        "absolute_gate_lufs": -70.0,
        "relative_gate_lu": -10.0,
        "shortterm_block_s": 3.0,
        "lra_block_s": 3.0,
        "lra_absolute_gate_lufs": -70.0,
        "lra_relative_gate_lu": -20.0,
        "lra_percentiles": [10, 95],
        "cross_check": "ffmpeg -af ebur128",
        "cross_check_tolerance_lu": 0.2,
        "digest_timeline_grid_s": 5.0,
    },
    "true_peak": {
        "standard": "ITU-R BS.1770-4 Annex 2",
        "oversampling_factors": [4, 16],
        "resampler": "scipy.signal.resample_poly (Kaiser window, beta 5.0)",
        "cross_check_tolerance_db": 0.3,
        "over_thresholds_dbtp": [0.0, -0.3, -1.0],
    },
    "psr": {
        "window_s": 3.0,
        "hop_s": 1.0,
        "definition": "short-term true peak (4x) minus short-term LUFS over the same window",
    },
    "dr14": {
        "algorithm": "TT Dynamic Range Meter (offline DR)",
        "block_s": 3.0,
        "block_rms": "sqrt(2 * mean(x^2))",
        "loudest_fraction": 0.2,
        "peak_rank": 2,
        "formula": "20*log10(peak2 / rms_top20) per channel, then mean over channels",
    },
    "streaming_targets_lufs": {
        "spotify_youtube_tidal_amazon": -14.0,
        "apple": -16.0,
    },
    "crest": {
        "loudest_window_s": 10.0,
        "timeline_hop_s": 1.0,
        "definition": "20*log10(sample_peak / rms) over the window",
    },
    "flat_top": {
        "threshold_rule": "thr = max(abs(x)) * 0.99999, computed per channel",
        "run_length_bins": [1, 2, 3, 6, 11, 21],
        "longest_runs_reported": 10,
        "lf_context_window_ms": 20.0,
        "slope_window_ms": 2.0,
        "ceiling_density_db": [0.1, 0.5, 1.0, 3.0, 6.0],
    },
    "spectrum": {
        "ltas_broadband": {
            "method": "Welch",
            "nperseg": 16384,
            "window": "hann",
            "overlap_pct": 50,
        },
        "ltas_lowfreq": {
            "method": "Welch",
            "nperseg": 131072,
            "window": "hann",
            "overlap_pct": 50,
            "section_target_s": 90.0,
            "section_rule": "densest sustained material by 20-2000 Hz RMS on a 1 s grid",
        },
        "bands_hz": [[n, lo, hi] for n, lo, hi in BANDS],
        "third_octave_centres_hz": list(THIRD_OCTAVE_CENTRES),
        "bark_edges_hz": list(BARK_EDGES),
        "tilt_fit_range_hz": [100.0, 10000.0],
        "tilt_piecewise_hz": [[30, 120], [120, 1000], [1000, 6000], [6000, 20000]],
        "bass_peak_search_hz": [20.0, 200.0],
        "bass_peak_prominence_db": 3.0,
        "resonance_prominence_db": 4.0,
        "resonance_max_bandwidth_oct": 0.5,
        "descriptor_timeline_hop_s": 1.0,
        "band_timeline_hop_ms": 100.0,
    },
    "forensics": {
        "hf_cutoff": {
            "frame_s": 5.0,
            "nfft": 16384,
            "search_range_hz": [8000.0, "nyquist"],
            "collapse_depth_db": 25.0,
            "smoothing_octaves": 0.08333333333333333,
            "knee_tolerance_db": 3.0,
            "steep_slope_db_per_oct": -30.0,
            "floor_offset_db": 6.0,
            "shelf_candidates_hz": list(CODEC_SHELVES_HZ),
        },
        "spectral_hole": {"neighbour_octaves": 0.5, "min_depth_db": 8.0},
        "effective_bit_depth": {"method": "trailing-zero-bit histogram over integer samples"},
        "noise_floor": {"quietest_fraction": 0.01, "frame_ms": 400.0},
        "silence": {"digital_black_dbfs": -90.0, "fade_detect_db": 20.0},
        "hum": {"mains_hz": [50.0, 60.0], "harmonics": 5, "local_median_octaves": 0.5},
        "rumble_hz": [1.0, 30.0],
        "tape_bias_hz": [15000.0, 25000.0],
        "wow_flutter": {"frame_s": 0.5, "max_frames": 240,
                        "method": "librosa.estimate_tuning per frame, cents"},
    },
    "stereo": {
        "correlation_window_s": 1.0,
        "itd_search_ms": 5.0,
        "mono_crossover_threshold_db": -20.0,
        "goniometer_bins_deg": 15,
    },
    "structure": {
        "features": "MFCC(20) + chroma_cqt(12) + RMS + spectral_contrast(7), z-scored, stacked",
        "ssm": "cosine self-similarity, median-filtered",
        "ssm_downsample_frames": 4,
        "ssm_frame_hop_s": 4 * 512 / 22050.0,
        "novelty_kernel_s": 8.0,
        "min_section_s": 4.0,
        "peak_pick": {
            "pre_max_s": 4.0,
            "post_max_s": 4.0,
            "pre_avg_s": 8.0,
            "post_avg_s": 8.0,
            "delta": 0.05,
            "wait_s": 4.0,
        },
        "arrangement_gap": {"min_ms": 200.0, "drop_db": 20.0},
        "tempo_drift_window_s": 30.0,
        "key_profiles": "Krumhansl-Schmuckler major/minor",
        "key_low_confidence_margin": 0.05,
    },
    "processing": {
        "saturation": {
            "frame_ms": 50.0,
            "hf_band_hz": [5000.0, 10000.0],
            "regression": "least squares of HF dB on broadband dB",
        },
        "pumping": {
            "lf_band_hz": [20.0, 120.0],
            "mid_band_hz": [500.0, 6000.0],
            "envelope_hop_ms": 5.0,
            "lag_range_ms": [-200.0, 200.0],
        },
        "modulation": {
            "envelope_hop_ms": 5.0,
            "fft_window_s": 8.0,
            "rates": ["beat", "half_beat", "quarter_beat"],
        },
        "multiband_timeline_hop_ms": 10.0,
        "hpss": {"kernel_size": 31, "power": 2.0, "hop_length": 1024},
        "reverb": {
            "octave_bands_hz": [63, 125, 250, 500, 1000, 2000, 4000, 8000],
            "decay_min_ms": 150.0,
            "schroeder": "reverse integration",
            "t20_range_db": [-5.0, -25.0],
            "t30_range_db": [-5.0, -35.0],
            "early_late_split_ms": 80.0,
        },
        "vocal_band_hz": [1000.0, 4000.0],
    },
    "compare": {
        "level_match": "gain both files to equal LUFS-I before any comparison",
        "null_test": {
            "align_search_s": 10.0,
            "min_correlation": 0.5,
            "resample_to": "higher of the two sample rates",
        },
    },
    "stems": {
        "model": "htdemucs",
        "stems": ["drums", "bass", "other", "vocals"],
        "note": "every stem-derived number carries source=separated",
    },
}

QUICK_SKIPS: tuple[str, ...] = (
    "loudness.true_peak_16x",
    "loudness.intersample_overs",
    "dynamics.onsets",
    "stereo.goniometer",
    "spectrum.resonances",
    "spectrum.descriptor_timeline",
    "forensics.cutoff_stability",
    "structure.sections",
    "structure.tempo",
    "structure.key",
    "processing.reverb",
    "processing.modulation_spectrum",
    "processing.hpss",
    "processing.multiband_timeline",
    "processing.transient_density",
    "spectrum.ltas_lowfreq",
    "forensics.wow_flutter",
    "stems",
)


def profile_params(profile: str) -> dict[str, Any]:
    """What a profile switches off.  `full` runs everything."""
    return {
        "profile": profile,
        "skipped_in_quick": list(QUICK_SKIPS) if profile == "quick" else [],
    }
