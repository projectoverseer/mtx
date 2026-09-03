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
        "tuning_report_cents": 15.0,
        "tuning_report_cents_note": "a deviation from A440 at least this large "
                                    "is reported as a note. Below it, the "
                                    "figure is still in structure.key and is "
                                    "mostly the estimator's own scatter",
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
    "harmony": {
        "chroma": "librosa chroma_cqt at 22.05 kHz / hop 512, L1-normalised per "
                  "frame, aggregated to the beat grid by median when "
                  "structure.tempo supplies one",
        "qualities": ["maj", "min", "dim", "aug", "sus2", "sus4", "maj7",
                      "min7", "dom7", "min6", "maj6", "hdim7", "dim7"],
        "template": "Pearson correlation between the chroma frame and the "
                    "binary chord-tone mask (both mean-removed, then "
                    "L2-normalised), so energy on a tone the chord does not "
                    "contain counts against it",
        "template_note": "a plain cosine against uncentred masks cannot work: a "
                         "four-tone mask that contains a triad scores at least "
                         "as high as the triad whenever the fourth tone has any "
                         "energy at all, so every chord came back as a seventh",
        "complexity_penalty": 0.18,
        "complexity_penalty_note": "subtracted from the score once per chord "
                                   "tone beyond the third, so a seventh has to "
                                   "earn its place against the triad inside it",
        "quality_prior": {"sus2": 0.03, "sus4": 0.03, "dim": 0.05,
                          "aug": 0.08, "dim7": 0.08, "hdim7": 0.06},
        "quality_prior_note": "a suspended or diminished chord is rarer than a "
                              "triad; without this the recogniser reaches for "
                              "one whenever a passing tone lands on the second "
                              "or the fourth",
        "no_chord_score": 0.22,
        "emission_gain": 30.0,
        "key_from_chords": {
            "method": "the key whose scale explains the most chord time, plus "
                      "the evidence that says which of the two relative keys "
                      "sharing that scale is the tonic: time on the tonic "
                      "chord, and whether the track opens and closes on it",
            "tonic_weight": 0.4,
            "first_chord_weight": 0.10,
            "final_chord_weight": 0.0,
            "final_chord_weight_note": "measured to hurt on the reference set "
                                       "and left at zero: pop records fade out "
                                       "rather than cadence, so the last chord "
                                       "the recogniser sees is often the "
                                       "quietest and least reliable one",
            "measured_accuracy": "3 of 7 published keys before the tonic "
                                 "evidence, 4 of 7 after; structure.key (mean "
                                 "chroma) got 5 of 7 on the same seven tracks. "
                                 "This is a second opinion, not a better one, "
                                 "and its failure mode is the relative key.",
            "note": "a scale fit cannot separate a major key from its relative "
                    "minor -- they contain the same seven notes -- so without "
                    "the tonic evidence this estimate lands on the wrong one of "
                    "the pair about half the time",
        },
        "viterbi_self_transition": 0.86,
        "loop_candidate_bars": [1, 2, 4, 8, 16],
        "loop_match_threshold": 0.75,
        "modulation": {"window_s": 20.0, "hop_s": 5.0, "min_hold_s": 10.0,
                       "min_margin": 0.02},
        "pedal_min_chords": 3,
        "cadence_degrees": {"authentic": [7, 0], "plagal": [5, 0],
                            "deceptive": [7, 9], "half": [0, 7]},
    },
    "melody": {
        "f0": {"algorithm": "librosa.pyin", "sr_hz": 22050,
               "frame_length": 2048, "hop_length": 256,
               "vocal_fmin_hz": 65.406, "vocal_fmax_hz": 2093.005,
               "bass_fmin_hz": 27.5, "bass_fmax_hz": 523.251},
        "note_segmentation": {"median_filter_frames": 5,
                              "split_semitones": 0.6,
                              "min_note_ms": 60.0,
                              "unvoiced_gap_ms": 100.0},
        "phrase_gap_ms": 350.0,
        "stable_frame_cents": 30.0,
        "vibrato": {"min_note_ms": 300.0, "rate_range_hz": [3.0, 9.0],
                    "min_depth_cents": 20.0},
        "quantisation": {"transition_target_fraction": 0.8,
                         "grid_tolerance_cents": 5.0,
                         "fast_transition_ms": 40.0},
        "self_similarity_ngram": 4,
        "octave_outlier_semitones": 14.0,
        "range_rule": "the reported range is taken over the notes within "
                      "octave_outlier_semitones of the duration-weighted median "
                      "pitch. The raw extremes are reported too, separately: "
                      "measured against four reference vocals they were 40-58 "
                      "semitones wide, because a monophonic pitch tracker "
                      "running on a separated stem makes octave errors on 7-12% "
                      "of note time, and one of them sets the maximum.",
        "note": "pitch is measured on a separated stem; every number inherits "
                "source=separated",
    },
    "rhythm": {
        "meters": [2, 3, 4, 6],
        "octave_check": {
            "midpoint_ratio_double": 0.80,
            "alternation_ratio_half": 0.50,
            "method": "onset strength sampled on the beat grid itself: the "
                      "energy halfway between beats relative to the energy on "
                      "them (a high ratio means the real beat is twice as "
                      "fast), and the weaker of the two alternating beat "
                      "phases relative to the stronger (a low ratio means "
                      "every other beat is empty and the real beat is half as "
                      "fast)",
            "why_not_autocorrelation": "a periodic pulse train correlates with "
                                       "itself just as well at twice its "
                                       "period as at its own, so an "
                                       "autocorrelation comparison is biased "
                                       "toward the slower reading and reports "
                                       "half tempo on a plain click track",
            "note": "librosa's beat tracker picks a metrical level, not the "
                    "only one: on the reference set it reported half the "
                    "published tempo for one track and double it for another. "
                    "The ambiguity is measured and reported rather than "
                    "resolved, because every bar-relative number here inherits "
                    "whichever level was chosen.",
        },
        "downbeat_accent": "z-scored sum of the onset-strength value, the "
                           "20-120 Hz band energy and the chroma change at each "
                           "beat; meter and phase are the pair that maximises "
                           "the mean accent on the downbeat",
        "grid_subdivision": 4,
        "onset_hop_length": 128,
        "onset_hop_note": "onsets are re-detected at hop 128 (5.8 ms) rather than reusing the shared hop-512 envelope: a 23 ms quantisation would be larger than most of the microtiming it is meant to measure",
        "swing": {"search_fraction": [0.30, 0.75], "min_events": 8},
        "syncopation": "Longuet-Higgins & Lee metric weights over a 16-step bar",
        "microtiming_max_deviation_ms": 120.0,
        "pulse_rate_window_bars": 4,
        "programmed_tightness_ms": 8.0,
    },
    "form": {
        "cluster": {
            "features": "per-section mean chroma(12) and MFCC(20) plus the "
                        "measured section vector (LUFS, crest, tilt, "
                        "side-minus-mid, onset rate, 8 band percentages), each "
                        "z-scored across the sections of this track",
            "distance": "cosine",
            "merge_threshold": 0.8,
            "merge_threshold_note": "chosen so a track keeps a "
                                    "song-like alphabet: across "
                                    "eight reference tracks of "
                                    "16-31 sections this yields "
                                    "3-7 letters, where 0.22 "
                                    "yielded 11-25. It was tuned "
                                    "for how many letters come "
                                    "out, not for whether they "
                                    "are the right ones",
            "cannot_link": "with a vocals stem, a section that sings and "
                           "a section that does not are never merged, "
                           "whatever the distance between them says. "
                           "Vocal presence is measured where the distance "
                           "is a guess, and merging an instrumental hook "
                           "with the last chorus over it costs a chorus",
        },
        "vocal_presence_db_below_p95": 12.0,
        "implausible_part_count": 14,
        "implausible_part_count_note": "above this many parts the letter "
                                       "sequence is reporting texture changes "
                                       "rather than song form, and says so",
        "use_allin1": False,
        "use_allin1_note": "when true and the optional allin1 model is "
                           "installed, its segmentation is reported alongside "
                           "the measured one as a second opinion",
        "loopability_window_s": 8.0,
        "labels": ["intro", "verse", "pre-chorus", "chorus", "post-chorus",
                   "bridge", "drop", "instrumental", "outro"],
        "rule_note": "functional labels are an inference over measured evidence "
                     "(letter clustering, loudness rank, repeat count, vocal "
                     "presence, position); the measured boundaries and letters "
                     "are never replaced by them",
    },
    "arrangement": {
        "presence_db_below_stem_p95": 18.0,
        "density_hop_s": 0.5,
        "sub_bass_split_hz": 60.0,
        "eight_o_eight": {"min_decay_ms": 400.0, "sub_share_min_pct": 45.0},
        "glide": {"max_gap_ms": 60.0, "min_semitones": 2.0},
        "drum_hit_window_ms": 80.0,
        "layer_salience_floor_db": -12.0,
        "call_response_lag_s": [0.2, 2.0],
    },
    "masking": {
        "bands": "third-octave centres, 20 Hz - 20 kHz",
        "pair_metric": "energy-weighted level of the masker inside the target's "
                       "own band-energy distribution, in dB",
        "overlap_metric": "cosine and Bhattacharyya coefficient between the two "
                          "L1-normalised band-energy vectors",
        "sibilance_band_hz": [5000.0, 10000.0],
        "sibilance_reference_band_hz": [1000.0, 4000.0],
        "sibilance_frame_ms": 25.0,
        "delay_throw_subdivisions": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
        "highpass_probe_hz": [20.0, 500.0],
        "highpass_plateau_hz": [200.0, 500.0],
    },
    "lyrics": {
        "sources_in_priority_order": ["declared", "file:tag", "transcript"],
        # Matched exactly against the lower-cased tag key, never as a
        # substring: "composerlyricist" contains "lyric" and is a credit, not
        # a lyric.  Apple-style tagging puts that key on most commercial
        # files, so a substring test silently measures a list of songwriter
        # names as though it were the song's words.
        "tag_keys": ["lyrics", "lyric", "unsyncedlyrics", "unsynced lyrics",
                     "unsyncedlyric", "uslt", "sylt", "©lyr", "lyr",
                     "lyrics:description", "wm/lyrics"],
        "transcript": {
            "backends": ["whisperx", "whisper_timestamped", "faster_whisper"],
            # `small` is the smallest model that reliably hears a sung lyric
            # over a mix; `base` is roughly twice as fast and mishears chorus
            # lines often enough to change a rhyme count.
            "model": "small",
            "device": "auto",
            # Silero's voice-activity filter is trained on speech and rejects
            # a sung vocal over a full mix: on `Heat Waves` it cut a 366-word
            # transcript down to five.  It helps on an isolated vocal stem and
            # destroys a mix transcript, so it is off.
            "vad": False,
            # Below this a transcript is not a short lyric, it is an
            # instrumental or a vocal the model could not hear.  Kept as
            # evidence, refused as the record's lyric.
            "min_words": 20,
            "note": "optional; a transcript is an inference and is never merged "
                    "into a declared or tagged lyric",
        },
        "language_detection": "unicode-script share plus stop-word frequency "
                              "over a fixed table",
        "syllable_counter_languages": ["en"],
        "rhyme": {"languages": ["en"],
                  "slant_rule": "same vowel nucleus with a different coda, or "
                                "the same coda with a different nucleus"},
        "ngram_max": 8,
        "sentiment": {"backend": "vaderSentiment (optional)",
                      "note": "no lexicon ships with mtx; without one the "
                              "valence arc is null with a reason"},
        "lexicons": {"concreteness": None, "valence": None},
    },
    "embedding": {
        "backends_in_priority_order": ["laion_clap", "openl3", "transformers:MERT"],
        "note": "an embedding is a fingerprint, not a measurement: it is stored "
                "in its own block with the model name and version, and no "
                "measured field is ever derived from it",
        "section_embeddings": True,
    },
    "delivery": {
        "encodes": [
            {"name": "aac_256", "codec": "aac", "bitrate": "256k", "suffix": ".m4a"},
            {"name": "opus_128", "codec": "libopus", "bitrate": "128k", "suffix": ".opus"},
        ],
        "small_speaker_band_hz": [400.0, 8000.0],
        "excerpt_s": [15.0, 30.0],
        "hf_damage_band_hz": [10000.0, 20000.0],
        "requires": "ffmpeg; without it every rendering is null with a reason",
    },
    "version": {
        "markers": "title, subtitle, version and comment tags matched against a "
                   "fixed vocabulary (radio edit, clean, explicit, remix, live, "
                   "acoustic, instrumental, extended, sped up, slowed, "
                   "remaster, demo, edit)",
        "work_key": "case-folded, accent-stripped, punctuation-stripped artist "
                    "and title with every version marker removed",
    },
    "declared": {
        "sidecar": "declared.json beside the audio file, or passed with "
                   "--declared",
        "rule": "a declared value is passed through with source=declared and is "
                "never merged into a measured or database-sourced field",
    },
    "cohort": {
        "year_window": 2,
        "min_cohort_size": 12,
        "min_corpus_for_statistics": 200,
        "max_single_artist_share": 0.15,
        # A record belongs to every genre that got a real vote, not only to the
        # one that won.  Filed under its winner alone, a club record sits in
        # `electronic` (116 tracks) and never in `house` (199), which is the
        # cohort someone mixing a club record actually wants.  The floor keeps
        # a single stray tag from inventing membership.
        "secondary_genre_confidence": 0.34,
        "max_genres_per_track": 6,
        "percentile_rule": "share of the cohort strictly below the value, plus "
                           "half the ties",
        "label_priority": ["declared", "online", "file:tag"],
        "note": "computed by `mtx cohort` into its own file; `analyze` never "
                "reads the folder its output goes into, because a per-track "
                "measurement must not depend on what else is beside it",
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
        "device": "auto",
        "segment": None,
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
    "harmony",
    "rhythm",
    "form",
    "melody.vibrato",
    "delivery",
    "lyrics.transcript",
    "embedding",
)
# `stems` is deliberately absent: separation is opt-in through --stems at
# either profile, never a profile switch.  The per-stem measurements inherit
# whichever profile the run used.


def profile_params(profile: str) -> dict[str, Any]:
    """What a profile switches off.  `full` runs everything."""
    return {
        "profile": profile,
        "skipped_in_quick": list(QUICK_SKIPS) if profile == "quick" else [],
    }
