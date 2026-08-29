# `analysis.json` schema

`schema_version` **1.1.0**.

Changes since 1.0.0, all additive: `processing.multiband_timeline.
band_envelope_correlation` gains `offdiagonal`, `least_correlated_pairs` and
`most_correlated_pair`; `loudness.dr14.validation` gains `record` and its
`validated_against_published_reference` is no longer always `false`.

One line per field. Units are in the field name wherever they exist. Any field
may be `null`; a `null` always means "could not be computed", and the reason is
in `warnings[]` — it never means zero.

Conventions used throughout:

- `*_db` / `*_dbfs` / `*_dbtp` — decibels, relative to full scale, true peak,
  or as a ratio between two quantities named in the key.
- `*_s` — seconds; `*_ms` — milliseconds; `*_hz` — hertz.
- `*_time` — a `M:SS.mmm` string next to the matching `*_time_s` number.
- Timelines come as parallel arrays: `times_s` plus one array of values.
- Any block with `available: false` carries a `reason`.
- Any inferred quantity carries `confidence` (`high`/`medium`/`low`) and, below
  high, a `confidence_reason`.

---

## `run`

| Field | Description |
| --- | --- |
| `tool_version` | mtx version that produced the file. |
| `schema_version` | Version of this schema. |
| `generated_utc` | ISO 8601 timestamp. **Volatile**: excluded from the reproducibility guarantee. |
| `elapsed_seconds` | Wall-clock time of the run. **Volatile**. |
| `profile` | `quick` or `full`. |
| `stems_requested` | Whether `--stems` was passed. |
| `random_seed` | Seed set for `random` and `numpy.random` before any work. |
| `versions` | Python, platform, and the version of every library used, plus the `ffmpeg`/`ffprobe` banner lines. |
| `reproducibility` | Plain-language statement of what is guaranteed identical between runs. |

## `params`

A verbatim copy of the parameter block that produced this run: `general`,
`loudness`, `true_peak`, `psr`, `dr14`, `streaming_targets_lufs`, `crest`,
`flat_top`, `spectrum`, `forensics`, `stereo`, `structure`, `processing`,
`compare`, `stems`, and `profile` (which lists `skipped_in_quick`). Every metric
in this document is controlled by one of these entries; see the metric table in
`README.md` for which.

## `audio`

| Field | Description |
| --- | --- |
| `sample_rate_hz`, `channels`, `frames`, `duration_s` | As decoded. |
| `subtype`, `format`, `endian` | libsndfile's description of the container. |
| `band_analysis_sr_hz` | Rate used for band-split work: `min(native, 48000)`. |
| `librosa_sr_hz` | Rate used for librosa features (22050). |

## `file`

| Field | Description |
| --- | --- |
| `path_absolute` | Absolute path. **Volatile**: excluded from the reproducibility guarantee. |
| `filename`, `size_bytes` | Name and size on disk. |
| `sha256` | SHA-256 of the file bytes. |
| `decoded_md5` | MD5 of the decoded PCM in FLAC's byte layout, or `null` for float subtypes. |
| `decoded_md5_unavailable_reason` | Why `decoded_md5` is null, when it is. |
| `flac_streaminfo_md5` | The MD5 the FLAC encoder stored, if any. |
| `flac_md5_verified` | `true` if the two agree, `false` if they do not (also a warning), `null` if not applicable. |

## `container`

| Field | Description |
| --- | --- |
| `format`, `subtype`, `endian`, `sample_rate_hz`, `channels`, `frames`, `duration_s` | Container facts. |
| `bit_depth_container` | Bits the container offers (not necessarily the bits used — see `forensics.effective_bit_depth`). |
| `flac_streaminfo` | Parsed STREAMINFO: block and frame sizes, bits per sample, total samples, stored MD5. |
| `flac_compression_level_inferred` | Level guessed from an encoder string, usually `null`. |
| `flac_compression_level_note` | States that FLAC does not record the compression level. |
| `encoder_string`, `vendor_string` | From the tags and the Vorbis comment vendor field. |
| `ffprobe_raw` | The complete `ffprobe -show_format -show_streams` JSON, verbatim. `null` if ffprobe is unavailable. |

## `tags`

| Field | Description |
| --- | --- |
| `all` | Every tag mutagen found, key → string, sorted. |
| `named` | Normalised lookups: `title`, `artist`, `album`, `albumartist`, `date`, `genre`, `label`, `catalognumber`, `isrc`, `barcode`, `composer`, `comment`, `encoder`, `encoder_settings`, `tracknumber`, `discnumber`. Absent tags are `null`. |
| `musicbrainz` | Every `musicbrainz*`/`acoustid*` tag. |
| `replaygain` | Every `replaygain*` tag — what the distributor measured. |
| `apple_digital_master_markers` | Which "Mastered for iTunes" / Apple Digital Master markers were present. |
| `vendor_string` | Vorbis comment vendor string. |
| `cover_art` | `present`, `count`, `width_px`, `height_px`, `mime`, `bytes`. |

## `forensics`

### `hf_cutoff`

| Field | Description |
| --- | --- |
| `cutoff_hz` | The knee of the high-frequency collapse, or `null` when the spectrum runs to Nyquist. |
| `full_bandwidth` | `true` when no shelf was found. |
| `reason` | Present when `cutoff_hz` is null. |
| `level_at_cutoff_db`, `level_above_cutoff_db`, `collapse_depth_db` | Levels either side of the knee and the depth of the drop. |
| `rolloff_slope_db_per_oct`, `rolloff_fit_r2` | Slope across the transition and the quality of that fit. |
| `steep_region_start_hz`, `trend_slope_db_per_oct`, `knee_tolerance_db`, `steep_slope_threshold_db_per_oct` | The intermediate quantities the knee was derived from. |
| `codec_shelf_match` | `nearest_shelf_hz`, `distance_hz`, `sharpness_db_per_oct` — the match is reported with its distance so it can be judged rather than trusted. |
| `fraction_of_frames_above_cutoff_below_floor` | Fraction of 5 s frames whose above-cutoff band sits below the collapse threshold. |
| `measured_on`, `definition`, `params` | Method and parameters. |
| `reference_db` | Median level over 1–5 kHz, the reference the collapse is measured against. |

### `cutoff_stability`

Per-5-second cutoff timeline (`times_s`, `cutoff_hz`), `frames_measured`,
`frames_full_bandwidth`, `mean_hz`, `std_hz`, `min_hz`, `max_hz`. A moving
cutoff is joint-stereo/VBR behaviour; a rock-steady one is a fixed filter or
CBR. `note` says so without drawing the conclusion for you.

### Other forensics blocks

| Field | Description |
| --- | --- |
| `spectral_holes[]` | `centre_hz`, `depth_db`, `prominence_db` for narrow bands with anomalously low energy. |
| `effective_bit_depth` | `effective_bits`, `container_bits`, `significant_bit_histogram`, `nonzero_samples`, `container_holds_fewer_bits_than_it_offers`, `method`. |
| `upsampling` | `checked`, `cutoff_hz`, `nearest_original_rate_hz`, `distance_hz`, `mirror_correlation`, `suspected_upsampled`, `confidence`, `method`. Only runs above 48 kHz. |
| `noise_floor` | `level_dbfs`, `third_octave[]`, `slope_above_10k_db_per_oct`, `frames_used`, `quietest_fraction`, `interpretation_note`, `confidence`. |
| `silence` | `leading_black_ms`, `trailing_black_ms`, `start_kind`/`end_kind` (`hard cut`/`fade`/`indeterminate`), `fade_in_ms`, `fade_out_ms`, `fade_rule`, `digital_black_threshold_dbfs`. |
| `analog_signatures.mains_hum` | Per mains frequency (`50hz`, `60hz`): `harmonics[]` with `excess_db`, plus `max_excess_db` and `mean_excess_db`. |
| `analog_signatures.rumble` | `band_hz`, `level_db_rel_total`, `method`. |
| `analog_signatures.elliptical_eq` | `mono_crossover_hz`, `side_minus_mid_below_120hz_db`, `detected`, `confidence`. |
| `analog_signatures.tape_bias` | `peaks[]` above 15 kHz with `frequency_hz`, `excess_db`, `prominence_db`. |
| `analog_signatures.wow_flutter` | `cents_std`, `cents_detrended_std`, `slow_drift_cents_per_min`, `cents_range`, `times_s`, `cents`, `frames`, `frame_stride_s`, `confidence`. |

## `loudness`

| Field | Description |
| --- | --- |
| `integrated_lufs` | Integrated loudness, BS.1770-4 with both gates. |
| `lra_lu` | Loudness range, EBU Tech 3342. |
| `gated_block_fraction` | Fraction of 400 ms blocks that survived gating. |
| `momentary`, `shortterm` | `block_ms`/`block_s`, `hop_*`, `times_s`, `lufs`, `max_lufs`, `percentiles_lufs` (P10/P25/P50/P75/P90/P95). |
| `cross_check.ffmpeg_ebur128` | `available`, `integrated_lufs`, `lra_lu`, `true_peak_dbtp`, `threshold_lufs`, `reason`. |
| `cross_check.pyloudnorm` | `available`, `integrated_lufs`, `version`, `note`. |
| `cross_check.delta_lufs_mtx_minus_ffmpeg` etc. | Signed differences, plus `tolerance_lu`. Exceeding the tolerance also raises a warning. |
| `sample_peak` | `overall_dbfs`, `per_channel_dbfs`, `per_channel_linear`. |
| `true_peak.overall_dbtp_4x` / `_16x` | Both oversampling factors, never averaged. |
| `true_peak.per_channel_dbtp_4x` / `_16x` | The same, per channel. |
| `true_peak.delta_16x_minus_4x_db` | How far the two methods disagree. |
| `true_peak.delta_truepeak16x_minus_samplepeak_db` | Inter-sample headroom the limiter left. |
| `true_peak.overs` | `thresholds_dbtp`, `counts_in_order`, `counts` (named), `highest_peak_dbtp`, `highest_peak_time(_s)`, `over_definition`, `scanned_fraction_of_file`, `pruning`. `skipped: true` under `--profile quick`. |
| `plr_db`, `plr_definition`, `plr_true_peak_source` | PLR and which true-peak figure it used (`16x`, or `4x` in quick mode). |
| `psr` | `window_s`, `hop_s`, `times_s`, `psr_db`, `shortterm_true_peak_dbtp`, `shortterm_lufs`, `min_db`, `min_time(_s)`, `p10_db`, `median_db`, `max_db`, `definition`. |
| `streaming_preview.<platform>` | `target_lufs`, `gain_db`, `true_peak_after_dbtp`, `gain_is_positive`, `note`. |
| `dr14` | `dr`, `dr_unrounded`, `dr_alt_sample_peak2_unrounded`, `per_channel[]` (blocks, `rms_top20_dbfs`, both peak-2 definitions, both DR values), `blocks_used`, `peak2_definition`, `validation`. |
| `dr14.validation` | `validated_against_published_reference`, `status`, `reason`, `self_checked_synthetically`, `record`. Read this before quoting DR. |
| `dr14.validation.record` | What `mtx validate-dr` has stored on this machine: `store_path`, `tracks_checked`, `tracks_within_tolerance`, `tolerance_dr`, `max_abs_delta_dr`, `mean_delta_dr`, `entries[]` (`title`, `artist`, `published_dr`, `measured_dr`, `delta`, `sha256`, `checked_utc`, `tool_version`), `validated`. Empty until a track with a published rating has been checked; the record is per machine, so it is part of what `mtx run` provenance is standing in for. |

## `dynamics`

| Field | Description |
| --- | --- |
| `crest` | `whole_file_db`, `loudest_window` (`window_s`, `crest_db`, `start(_s)`), `timeline_times_s`, `timeline_db`, `timeline_percentiles_db`, `definition`. |
| `per_band_crest` | `bands[]` with `band`, `low_hz`, `high_hz`, `crest_db`, `rms_dbfs`, `peak_dbfs`; plus `spread_db` and `measured_on`. |
| `flat_top.per_channel[]` | `channel_peak_dbfs`, `threshold_dbfs`, `threshold_rule`, `flat_sample_count`, `flat_sample_fraction`, `event_count`, `run_length_histogram` (1, 2, 3-5, 6-10, 11-20, 21+), `longest_run_samples`, `longest_run_ms`, `longest_runs[]` (ten, with timestamps), `ceiling_density`. |
| `flat_top.ceiling_density` | Fraction of samples within 0.1/0.5/1/3/6 dB of that channel's ceiling — the threshold-free version of "how squashed is the top". |
| `flat_top.clip_then_normalise` | `flat_value_dbfs`, `flat_runs_of_3_or_more`, `ceiling_below_full_scale`, `detected`, `method`, `confidence`. |
| `flat_top.low_frequency_association` | `correlation`, `lf_at_events_db`, `lf_track_db`, `delta_db`, `frames_with_events`, `window_ms`. |
| `flat_top.limiter_vs_clipper` | `entry_slope_db_per_ms`, `exit_slope_db_per_ms` (median and percentiles), `fraction_of_events_with_steep_edges`, `inference`, `inference_is_an_inference`, `confidence`, `method`. |
| `onsets` | `onset_count`, `onset_rate_per_s`, `median_onset_strength`, `onset_strength_percentiles`, `median_attack_slope_db_per_ms`, `onset_times_s`, `method`. |
| `dc_offset[]` | Per channel: `dc_offset`, `dc_offset_dbfs`, `max_1s_window_dc`, `max_1s_window_time(_s)`. |

## `spectrum`

| Field | Description |
| --- | --- |
| `ltas.broadband` | `frequencies_hz`, `psd_db` keyed by signal (`mid`, `side`, `mono`, `ch0`, `ch1`, …), `resolution_hz`, `params`. |
| `ltas_lowfreq` | The 131072-point pass: `computed`, `frequencies_hz` and `psd_db_mid` up to 500 Hz, `resolution_hz`, `section_start(_s)`, `section_end(_s)`, `section_duration_s`, `params`. `reason` when skipped. |
| `band_energy.tables.<signal>[]` | Per band: `band`, `low_hz`, `high_hz`, `power`, `db`, `pct`. |
| `air_band_pct`, `sub_band_pct` | 12–20 kHz and 20–60 kHz as a percentage of 20 Hz–20 kHz energy on mono. |
| `third_octave.mid` / `.side` | Per band: `centre_hz`, `low_hz`, `high_hz`, `power`, `db`, `pct`, `db_rel_loudest`. |
| `third_octave.side_minus_mid_db[]` | `centre_hz`, `side_minus_mid_db`. |
| `bark.mid` / `.side` | The same shape over 24 Zwicker critical bands, with `bark_band`. |
| `tilt` | `slope_db_per_oct`, `r2`, `fit_range_hz`, `measured_on`, `piecewise[]` (`low_hz`, `high_hz`, `slope_db_per_oct`, `r2`). A low `r2` means "tilt" describes this spectrum poorly. |
| `bass_fundamentals` | `peaks[]` (`rank`, `frequency_hz`, `level_db_rel_strongest`, `prominence_db`, `nearest_note`, `cents_from_note`, `q`), `single_note`, `single_note_rule`, `resolution_hz`. |
| `resonances[]` | `frequency_hz`, `prominence_db`, `q`, `frame_presence_fraction`, `level_db`. |
| `descriptors` | `whole_track` and `timeline` for `centroid_hz`, `spread_hz`, `skewness`, `kurtosis`, `flatness`, `rolloff85/95/99_hz`, `zcr`; plus `times_s`, `percentiles`, `hop_s`, `window`. |
| `band_timeline` | `hop_ms`, `times_s`, `bands.<band>` in dB at 100 ms. |

## `stereo`

`available: false` with a `reason` for mono files.

| Field | Description |
| --- | --- |
| `convention` | The mid/side definition in words. |
| `side_minus_mid_db` | Overall, `10·log10(P_side/P_mid)`. `-200` is the floor and means "no side energy at all". |
| `side_minus_mid_below_120hz_db` | The same over 20–120 Hz. |
| `side_minus_mid_per_third_octave[]` | `centre_hz`, `side_minus_mid_db`. |
| `side_minus_mid_per_band[]` | Per analysis band, with the band correlation and the method. |
| `mono_crossover_hz`, `mono_crossover_rule` | Highest third-octave centre below which side/mid stays under the threshold. |
| `correlation` | `overall`, `times_s`, `values`, `min`, `p5`, `median`, `pct_time_below_0`, `pct_time_below_0_3`, `most_negative_windows[]` (three, with timestamps), `window_s`. |
| `channel_balance` | `rms_l_dbfs`, `rms_r_dbfs`, `rms_l_minus_r_db`, `lufs_l`, `lufs_r`, `lufs_l_minus_r`. |
| `inter_channel_time_offset` | `lag_samples`, `lag_us`, `correlation_at_lag`, `correlation_at_zero`, `search_ms`, `excerpt_s`, `note`. |
| `width_timeline` | `hop_s`, `times_s`, `side_minus_mid_db` per second. |
| `mono_sum_damage` | `broadband_loss_db`, `per_third_octave[]` (`centre_hz`, `mono_sum_loss_db`), `definition`. |
| `goniometer` | `bin_deg`, `bin_edges_deg`, `energy_fraction_per_bin`, `fraction_energy_outside_45_deg`, `definition`. |

## `structure`

| Field | Description |
| --- | --- |
| `section_count`, `boundaries_s`, `method` | Segmentation result and how it was produced. |
| `sections[]` | `index`, `start(_s)`, `end(_s)`, `duration_s`, `lufs_i`, `shortterm_max_lufs`, `crest_db`, `tilt_db_per_oct`, `tilt_r2`, `band_energy_pct`, `side_minus_mid_db`, `onset_rate_per_s`, `delta_vs_previous_lufs`, `delta_vs_track_lufs`. |
| `loudest_section_index`, `quietest_section_index`, `widest_section_index` | Indices into `sections[]`. |
| `biggest_jump` | `db`, `at_section`, `time(_s)`. |
| `arrangement_gaps[]` | `band`, `start(_s)`, `duration_ms`, `band_level_db_rel_track`; `arrangement_gap_rule` states the criterion. |
| `tempo` | `bpm`, `bpm_beat_track_raw`, `bpm_source`, `bpm_grid_fit_r2`, `beat_count`, `beat_times_s`, `inter_beat_interval_stability`, `bpm_per_window`, `bpm_drift_std`, `drift_window_s`, `confidence`, `drift_note`. |
| `key` | `key`, `correlation`, `runner_up`, `runner_up_correlation`, `margin`, `confidence`, `tuning_cents`, `implied_a4_hz`, `method`. |

## `processing`

Everything in this section is an inference. Each block carries `method`,
`confidence`, `confidence_reason` and, where it helps, a `reading` that states
what the number means without stating what it implies about the record.

| Field | Description |
| --- | --- |
| `saturation_proxy` | `slope_db_per_db`, `r2`, `frames_used`, `per_section[]`, `params`, `reading`. Slope above 1 means the material gets brighter as it gets louder. |
| `bus_compression` | `most_negative_correlation`, `most_negative_lag_ms`, `most_positive_correlation`, `most_positive_lag_ms`, `zero_lag_correlation`, `dip_depth_db`, `estimated_release_ms`, `release_method`. "Most negative" is literal: on material with no ducking the minimum over the lag range can itself be positive. |
| `modulation_spectrum` | `beat_rate_hz`, `envelope_rate_hz`, `bands.<band>` with `beat_depth_db`, `half_beat_depth_db`, `quarter_beat_depth_db`, `dip_phase_fraction_of_beat`, `beat_profile_depth_db`. |
| `multiband_timeline` | `hop_ms`, `times_s`, `rms_db.<band>`, `crest_db.<band>`, `band_envelope_correlation` (`bands`, `matrix`, `mean_offdiagonal`, `offdiagonal` with `min`/`median`/`max`/`mean`/`pairs`, `least_correlated_pairs[]` and `most_correlated_pair` as `{bands, r}`, `reading`). The summary and the extreme pairs are what the digest renders; the full matrix stays here. |
| `hpss` | `percussive_to_harmonic_db`, `percussive_energy_fraction`, `per_band_percussive_to_harmonic_db`, `vocal_band_proxy` (`band_hz`, `times_s`, `values_db`, `confidence`). |
| `reverb` | `per_octave_band[]` (`centre_hz`, `events_used`, `t20_s`, `t30_s`, `t20_iqr_s`, `early_to_late_db`), `tail_stereo_correlation`, `tail_correlation_method`, `params`. |
| `transient_density` | `hop_s`, `bands.<band>` (onsets per second) and `rate_per_s.<band>`, plus `method`. |

## `stems`

`requested: false` unless `--stems` was passed. When present: `model`,
`cache_dir`, `source: "separated"`, `caveat`, and `stems.<name>` for each of
drums/bass/other/vocals, each holding the full `loudness`, `dynamics`,
`spectrum` and `stereo` blocks for that stem plus `level_vs_mix`
(`rms_db`, `lufs_delta`) and its own `warnings`. Every number under `stems` is
measured on a separated signal, not on the mix.

## `headline`

The ~28 numbers the digest table is built from, all also present in their home
sections: `lufs_i`, `lra_lu`, `true_peak_dbtp_16x`, `sample_peak_dbfs`,
`plr_db`, `psr_min_db`, `psr_min_time`, `psr_median_db`, `dr14`,
`crest_whole_db`, `crest_loudest_10s_db`, `spectral_tilt_db_per_oct`,
`spectral_tilt_r2`, `air_band_pct`, `sub_band_pct`, `side_minus_mid_db`,
`side_minus_mid_below_120hz_db`, `mono_crossover_hz`, `correlation_mean`,
`correlation_min`, `flat_top_sample_count`, `flat_top_longest_run_ms`,
`hf_cutoff_hz`, `effective_bit_depth`, `tempo_bpm`, `key`, `section_count`,
`duration_s`.

## `warnings[]` and `confidence_notes[]`

`warnings[]` is a list of `"<where>: <message>"` strings, in emission order:
anything that could not be computed, every cross-method disagreement beyond
tolerance, and every explicit statement about what a run did or did not do.

`confidence_notes[]` is a list of `{metric, confidence, reason}` for every
metric reported below full confidence.

Both are rendered into the digest's `FLAGS` section, before the detail.

---

## `split` and the part files

Present only when the document was too large for one file. `analysis.json` is
then an index: every section small enough stays inline, the rest are replaced
by `{mtx_moved: true, parts: [...], note}` and live in `analysis.partNN.json`.
`mtx join` (or `mtx.split.load_analysis`) rebuilds the whole document; nothing
is dropped, rounded or summarised by the split.

| Field | Description |
| --- | --- |
| `split.whole_bytes` | Size the single file would have been. |
| `split.part_max_bytes` | The per-file cap the split was made to fit under (`--max-part-size`, default 4.5 MB). |
| `split.part_count` | Number of part files. |
| `split.sections_in_parts[]` | Top-level keys that were moved out. |
| `split.parts[]` | `file`, `path`, `slice`, `bytes` — one row per part, in the order they merge. |
| `split.rejoin` | The command that puts it back together. |
| `split.oversize_parts` | Only when a single indivisible value is still over the cap. Nothing was truncated. |

Each part file holds `mtx_part` (`stem`, `index`, `of`, `path`, `slice`,
`index_file`, `note`) and `data`. `path` is the key sequence the fragment
belongs at; `slice` is `[start, stop]`, absolute, when the fragment is a chunk
of a list at that path, and `null` when it is a (possibly partial) object.
Merging in part order restores the document: object fragments set their keys,
list chunks concatenate.

`comparison.json` is split by the same rule and carries the same block.

`schema_version` does not move for this: no measured field changed, and a
rejoined document is identical to the one a `--no-split` run writes. The split
describes how the document was carried, not what was measured.

---

## `corpus_row.json` (written next to `digest.md`)

The `CORPUS ROW` block as typed JSON, so a measurement reaches an archive
without a transcription step. Keys are the property names a corpus database
uses rather than mtx's internal field names: `Title`, `Artist`, `Year`,
`Genre`, `Engineers`, `LUFS-I`, `True peak`, `LRA`, `PLR`, `PSR min`,
`PSR median`, `DR14`, `Crest (loudest 10s)`, `Tonal tilt notes`,
`Width/mono notes`, `mtx run`. Numbers stay numbers; `_units` states what each
is in; `_source` carries the filename, the full `sha256` and the timestamp of
the PSR minimum. Anything mtx cannot measure is `null`, never guessed --
`Engineers` always is, and no session field (calibration, lessons, verdict)
appears at all.

## `online.json` (`mtx enrich`)

`online.schema_version` **1.0.0**. A **sidecar**, written beside
`analysis.json` and never merged into it: `mtx analyze` guarantees
byte-identical output for the same input, and a section whose content depends
on what MusicBrainz looked like this morning cannot live inside that
guarantee. Absent unless `mtx enrich` has been run.

| Field | Description |
| --- | --- |
| `schema_version` | Version of this section. |
| `queried_utc` | ISO 8601 timestamp of the lookup. |
| `query` | What the file claimed about itself: `isrc`, `title`, `artist`, `album`, `barcode`, `date`, `genre_tag`, `duration_s`. These are the inputs every provider was matched against. |
| `providers_requested[]` | Providers asked, in order. |
| `providers_available[]` | Providers that returned a match above the score floor. |
| `match_confidence` | Mean match score across the providers that matched. `0.0` when none did. |
| `errors[]` | Every failure, prefixed by provider. A provider that raised is caught here rather than losing the run. |
| `cache` | `hit` / `miss` / `error` / `skipped` request counts for this track. |
| `elapsed_seconds` | Wall-clock time of the lookup. |

### `genres`

| Field | Description |
| --- | --- |
| `available` | `false` when no source returned a usable label; `primary` is then `null` rather than guessed. |
| `primary` | Highest-scoring genre. |
| `umbrella` | Its coarse bucket (`pop`, `hip hop`, `r&b/soul`, `electronic`, `rock`, `jazz`, `latin`, …), decided head-final: `ambient pop` is pop, `pop rock` is rock. |
| `ranked[]` | `{name, score, confidence, umbrella, sources[]}`, descending. `confidence` is the score relative to the winner, so "how much weaker is the second guess" reads off directly. `sources[]` names every provider level that voted for it. |
| `umbrella_ranked[]` | `{name, score}` — the ranked list collapsed into buckets, for filtering. |
| `agreement` | Fraction of contributing sources that backed the winner. |
| `source_count` | Number of sources that returned any genre. |
| `by_source` | The raw normalised votes per source, before weighting. |

Scoring: within a source each vote is scaled against **that source's own top
vote**, then multiplied by the source's trust weight (a genre attached to this
recording outranks one attached to the artist's whole career), and the products
are summed. Scaling against the source's sum instead would hand the win to a
shop returning one coarse word over a database returning nine precise ones.
Normalisation repairs spelling only — `Hip-Hop/Rap` becomes `hip hop` — and
never merges two genres a listener can tell apart.

### `descriptive_tags[]`

`{name, score, sources[]}`. Mood and context words that are *not* genres —
`dark`, `nocturnal`, `party`. Kept separate so they cannot pollute a genre
filter; years, chart names, review-site handles and personal shelf labels are
dropped.

### `cross_checks`

| Field | Description |
| --- | --- |
| `tempo` | `local_bpm` and `local_confidence` from `structure.tempo`, against `published_bpm` from `published_source`. `verdict` is `agree` (within 2 %), `octave` (half or double — a metrical-level disagreement, not a tempo one), `triplet` (3:2), `disagree`, or `unavailable`. `resolved_bpm` and `resolved_confidence` are the conclusion: agreement promotes a low-confidence estimate to `high`; an `octave` verdict resolves to the published value at `medium` and keeps the other reading in `alternate_bpm`, because the relationship is certain while which level to call "the tempo" is not; a real disagreement keeps the local value at `low`. |
| `duration` | `local_s`, `providers_s`, `deltas_s`, `max_abs_delta_s`, and `verdict` (`exact` ≤ 2 s, `close` ≤ 5 s, `differs`). This is what makes a match verifiable rather than assumed. |
| `release_date` | `sources` (tag, MusicBrainz release and release group, Deezer, Apple), `earliest`, and `agree`. |

### `credits`

`{role: [{name, sources[]}]}`, merged across MusicBrainz relations, Discogs
sleeve credits, Deezer contributors and the file's own tags. A name backed by
more than one source is a confirmed credit; the `sources[]` list is what makes
that visible. Songwriting comes from the MusicBrainz *work* behind the
recording, which is the one place composer and lyricist are reliably recorded.

### `identity` and `popularity`

`identity` carries the stable handles: `isrc`, `recording_mbid`,
`release_mbid`, `release_group_mbid`, `work_mbid`, `iswcs[]`, `deezer_id`,
`itunes_id`, `discogs_release_id`, `label`. `popularity` carries
`deezer_rank`, `deezer_album_fans`, and — with a Last.fm key —
`lastfm_listeners`, `lastfm_playcount`, `lastfm_artist_listeners`.

### Per-provider blocks

`musicbrainz`, `deezer`, `itunes`, `lastfm`, `discogs` each keep their own raw
result: `available`, `errors[]`, `requests`, the `match` breakdown
(`score`, `duration_delta_s`, `title_score`, `artist_score`, `matched_by`), and
`candidates[]` — the rows that were considered and rejected, with their scores.
A wrong match is therefore auditable rather than invisible.

---

## `comparison.json` (`mtx compare`)

| Field | Description |
| --- | --- |
| `file_a`, `file_b` | `path`, `filename`, `sha256`, `lufs_i`. |
| `level_match` | `rule`, `gain_applied_to_b_db`, `matched_lufs_i`. Applied before anything is compared. |
| `headline_comparison[]` | `metric`, `a`, `b`, `delta_b_minus_a`, `delta_level_matched`, `level_dependent`. |
| `third_octave_difference_db.mid` / `.side` | `centre_hz`, `a_db`, `b_db`, `b_minus_a_db`. |
| `per_band_side_mid_difference[]` | `band`, `a_side_minus_mid_db`, `b_side_minus_mid_db`, `delta_db`. |
| `correlation_difference`, `psr_difference`, `crest_difference` | `a`, `b`, `delta_b_minus_a`. |
| `null_test` | `performed`; when true: `resampled_to_hz`, `lag_samples`, `lag_ms`, `alignment_correlation`, `gain_applied_to_b_db`, `residual_dbfs`, `reference_a_dbfs`, `residual_rel_reference_db`, `residual_per_third_octave[]`, `residual_timeline_dbfs`. When false, `reason` — including the refusal when the aligned correlation is below 0.5. |
