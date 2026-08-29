# mtx — master extractor

`mtx` reads a lossless audio file on your machine and writes an exhaustive,
reproducible measurement dump: a large `analysis.json` that stays local, and a
compact `digest.md` (hard cap ~12 KB) you can paste into an analysis workflow
somewhere else.

**This tool measures. It does not interpret, score, grade or recommend.** There
is no "good", "bad" or "too loud" anywhere in its output. Where a number is an
inference rather than a measurement, it says so and carries a confidence.

Everything runs offline. No API keys, no network calls, no paid services, and
your input file is never modified or moved.

---

## Install

```bash
git clone https://github.com/projectoverseer/mtx
cd mtx
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .
```

Python 3.11+ on macOS, Linux or Windows. `ffmpeg`/`ffprobe` on your `PATH` are
optional but recommended: without them you lose the container dump and the
independent loudness cross-check, and the tool says so in `FLAGS` rather than
quietly carrying on.

Optional extras:

```bash
pip install -e ".[plots]"   # matplotlib, for --plots
pip install -e ".[stems]"   # demucs + torch, for --stems (large, slow on CPU)
pip install -e ".[dev]"     # pytest
```

Check the install:

```bash
mtx selftest
```

That generates synthetic signals with known answers and prints a pass/fail line
with the measured value for each. It exits non-zero if anything fails.

---

## Usage

```
mtx analyze <file> [--out DIR] [--profile quick|full] [--plots] [--stems] [--json-only]
mtx batch <dir> [--out DIR] [--recursive] [--csv summary.csv]
mtx compare <fileA> <fileB> [--out DIR] [--null-test]
mtx selftest
mtx --version
```

- `analyze` — the main path. `--profile full` is the default; `quick` skips the
  expensive DSP (see the profile table below).
- `batch` — one JSON per file plus a single CSV of headline metrics, one row per
  track. This is how you bootstrap a reference library from records you own.
- `compare` — two files, **level-matched first**, with an optional null test.
- Default output directory: `./mtx_out/<basename>/`.
- Progress goes to stderr; stdout carries only the output path.
- Exit codes: `0` success, `1` unreadable input, `2` a self-test assertion failed.

### Output

| File | What it is |
| --- | --- |
| `analysis.json` | Everything, with the full parameter block. Large; stays on your machine. |
| `digest.md` | `HEADLINE` / `FLAGS` / `DETAIL` / `CORPUS ROW` / `METHOD`. Hard cap ~12 KB. |
| `plots/*.png` | Only with `--plots`. For your own eyes; never referenced by the digest. |

`FLAGS` comes **before** the detail: warnings, method disagreements and every
low-confidence metric are the first thing you read.

A worked example is in [`samples/digest.md`](samples/digest.md).

---

## The five properties this tool is built around

**Reproducible.** Two runs over the same file on the same machine and library
set produce byte-identical JSON, apart from `run.generated_utc`,
`run.elapsed_seconds` and `file.path_absolute`. Seeds are fixed; the tool
version, schema version, Python version and the version of every library used
are recorded in `run.versions`.

**Parameter provenance.** Every metric group carries the parameters that
produced it, and the whole set is echoed in a top-level `params` block.

**Self-verifying.** Where two independent methods exist, both are computed and
both are reported, with the delta:

| Quantity | Method A | Method B | Tolerance |
| --- | --- | --- | --- |
| Integrated loudness | mtx's own BS.1770-4 K-weighting | `ffmpeg -af ebur128` (and `pyloudnorm` as a third opinion) | 0.2 LU |
| True peak | 4x oversampling | 16x oversampling | 0.3 dB |
| DR14 | second-highest per-block peak (TT DR) | second largest distinct sample magnitude | reported side by side |
| Mid/side spectra | derived from L/R auto- and cross-spectra | direct Welch of the mid signal | asserted exact in tests |

Disagreement beyond tolerance becomes a warning. Nothing is averaged away.

**Fail loudly.** A metric that cannot be computed is `null` plus a reason in
`warnings[]`. No defaults are substituted and nothing is silently skipped.

**No baked-in judgement.** Detectors use thresholds internally, but the
underlying continuous measurement is always reported next to the boolean, so
the threshold can be second-guessed later.

---

## Known traps, and what this tool does instead

1. **The fixed -0.1 dBFS clipping threshold.** Useless on a master whose
   ceiling sits below it. `mtx` derives the threshold per channel as
   `max(|x|) * 0.99999`. There is a regression test for exactly this
   (a sine hard-clipped at a -3 dBFS ceiling) in `mtx selftest`.
2. **Trusting one reported true peak.** Both 4x and 16x are computed and
   reported, and disagreement is flagged.
3. **Default FFT resolution on the low end.** A second Welch pass at
   `nperseg=131072` runs over an automatically chosen ~90 s body section, and
   the chosen time range is reported. `mtx selftest` asserts that 62 Hz and
   70 Hz are resolved as two separate peaks.
4. **Averaging over the whole track.** Every headline metric also exists per
   section, and there are per-second timelines for the rest.
5. **A derived metric without its inputs.** PLR is always printed next to the
   true peak and LUFS-I it came from.
6. **Silently coercing channel counts or rates.** 1, 2 and >2 channels are
   handled explicitly, 44.1 k to 192 k are supported, and what was done is
   stated in `audio` and in `warnings[]`.

### Two things the tool tells you it has *not* verified

- **DR14 is not validated against a published DR rating.** `mtx` ships no
  copyrighted reference track, so the implementation is only checked against
  analytically known synthetic cases (a continuous sine must give DR 0.0).
  Every run says so in `loudness.dr14.validation` and in `FLAGS`. Compare one
  track whose rating you already know before trusting the number.
- **The specification's own sine test is self-contradictory** — it asks for
  LUFS-I ≈ -20.0 *and* a sample peak of -20.0 dBFS from the same 1 kHz sine,
  which differ by the 3.01 dB crest of a sine. The self-test asserts both
  readings separately and prints a note saying why.

---

## Metric catalogue

Units are in the key name or an adjacent field. "Parameter" is the entry in the
`params` block that controls the metric.

### File, container, provenance (`file`, `container`, `tags`)

| Metric | Unit | Method | Parameter |
| --- | --- | --- | --- |
| `file.sha256` | hex | SHA-256 of the file bytes | — |
| `file.decoded_md5` | hex | MD5 of the decoded PCM in FLAC's own byte layout | — |
| `file.flac_md5_verified` | bool | decoded MD5 vs the FLAC STREAMINFO MD5 | — |
| `container.*` | — | libsndfile + a direct STREAMINFO parse | — |
| `container.ffprobe_raw` | — | `ffprobe -show_format -show_streams`, verbatim | — |
| `tags.named/musicbrainz/replaygain` | — | `mutagen`, with ISRC, UPC, ReplayGain and Apple Digital Master markers | — |
| `tags.cover_art` | px | embedded picture, dimensions from the tag or the PNG/JPEG header | — |

### Source forensics (`forensics`)

| Metric | Unit | Method | Parameter |
| --- | --- | --- | --- |
| `hf_cutoff.cutoff_hz` | Hz | knee of the HF collapse: where the 1/12-octave-smoothed LTAS departs from its own fitted trend and stays down | `forensics.hf_cutoff` |
| `hf_cutoff.rolloff_slope_db_per_oct` | dB/oct | least squares over the transition | `forensics.hf_cutoff` |
| `hf_cutoff.collapse_depth_db` | dB | level at the knee minus the median above it | `collapse_depth_db` |
| `hf_cutoff.codec_shelf_match` | Hz | nearest of 11025…22050 Hz, with the distance and the slope | `shelf_candidates_hz` |
| `hf_cutoff.fraction_of_frames_above_cutoff_below_floor` | 0–1 | per 5 s frame | `frame_s` |
| `cutoff_stability` | Hz | cutoff per 5 s frame, with mean/std/min/max | `frame_s` |
| `spectral_holes[]` | Hz, dB | negative peaks against a half-octave running mean | `spectral_hole` |
| `effective_bit_depth` | bits | 32 − trailing zero bits of the left-justified int32 sample, max over non-zero samples | `effective_bit_depth` |
| `upsampling` | Hz | cutoff proximity to 22.05/24/44.1/48 kHz plus a mirror-image correlation | — |
| `noise_floor` | dBFS, dB/oct | quietest 1 % of 400 ms frames, third-octave spectrum and the slope above 10 kHz | `noise_floor` |
| `silence` | ms | leading/trailing digital black, hard cut vs fade, fade length | `silence` |
| `analog_signatures.mains_hum` | dB | 50/60 Hz and 5 harmonics vs the local half-octave median | `hum` |
| `analog_signatures.rumble` | dB | energy below 30 Hz relative to 20 Hz–20 kHz | `rumble_hz` |
| `analog_signatures.elliptical_eq` | Hz, dB | bass mono-ness, from the stereo mono-crossover | `stereo.mono_crossover_threshold_db` |
| `analog_signatures.tape_bias` | Hz, dB | narrowband peaks above 15 kHz | `tape_bias_hz` |
| `analog_signatures.wow_flutter` | cents | frame-wise `librosa.estimate_tuning`, std, detrended std, slow drift | `wow_flutter` |

### Loudness, peak, dynamics (`loudness`)

| Metric | Unit | Method | Parameter |
| --- | --- | --- | --- |
| `integrated_lufs` | LUFS | BS.1770-4, 400 ms blocks at 75 % overlap, gates -70 LUFS / -10 LU | `loudness` |
| `lra_lu` | LU | EBU Tech 3342, 3 s blocks, gates -70 / -20 LU, P95 − P10 | `loudness.lra_*` |
| `momentary`, `shortterm` | LUFS | full timelines plus P10/P25/P50/P75/P90/P95 and max | `block_ms`, `shortterm_block_s` |
| `cross_check.*` | LU | ffmpeg ebur128 and pyloudnorm, with deltas | `cross_check_tolerance_lu` |
| `true_peak.overall_dbtp_4x/16x` | dBTP | `resample_poly` (Kaiser β 5.0) at both factors | `true_peak.oversampling_factors` |
| `true_peak.delta_truepeak16x_minus_samplepeak_db` | dB | how much inter-sample energy the limiter left | — |
| `true_peak.overs` | count | contiguous excursions above 0.0 / -0.3 / -1.0 dBTP at 16x, with the timestamp of the highest | `over_thresholds_dbtp` |
| `plr_db` | dB | true peak − LUFS-I | — |
| `psr` | dB | per 3 s window: short-term true peak − short-term LUFS; min/P10/median/max **and the timestamp of the minimum** | `psr` |
| `streaming_preview` | dB | gain to -14 and -16 LUFS, resulting true peak, and whether the gain is positive | `streaming_targets_lufs` |
| `dr14` | dB | TT offline DR: 3 s blocks, RMS `sqrt(2·mean(x²))`, loudest 20 %, second-highest block peak | `dr14` |

### Dynamics and limiting fingerprints (`dynamics`)

| Metric | Unit | Method | Parameter |
| --- | --- | --- | --- |
| `crest.whole_file_db` | dB | sample peak − RMS | `crest` |
| `crest.loudest_window` | dB, s | the highest-RMS 10 s window, with its timestamp | `crest.loudest_window_s` |
| `crest.timeline_db` | dB | 1 s grid | `crest.timeline_hop_s` |
| `per_band_crest` | dB | crest independently in each of the 8 bands, plus the spread | `spectrum.bands_hz` |
| `flat_top.per_channel` | count, ms | threshold `max(|x|)·0.99999`, run-length histogram, longest run, ten longest with timestamps | `flat_top` |
| `flat_top.clip_then_normalise` | dBFS | flat runs of 3+ whose flat value sits below full scale | `flat_top` |
| `flat_top.low_frequency_association` | dB | sub-120 Hz level in ±20 ms around each event vs the track mean | `lf_context_window_ms` |
| `flat_top.per_channel[].ceiling_density` | fraction | samples within 0.1/0.5/1/3/6 dB of that channel's ceiling | `ceiling_density_db` |
| `flat_top.limiter_vs_clipper` | dB/ms | mean slope 2 ms before entry and after exit of each run — **inferred** | `slope_window_ms` |
| `onsets` | per s, dB/ms | `librosa.onset`, rate, median strength, median attack slope of the 100 strongest | `general.librosa_*` |
| `dc_offset` | — | per channel, plus the worst 1 s window | — |

### Spectrum (`spectrum`)

| Metric | Unit | Method | Parameter |
| --- | --- | --- | --- |
| `ltas.broadband` | dB | Welch, Hann, 50 % overlap, `nperseg=16384`, for mid/side/mono/each channel | `ltas_broadband` |
| `ltas_lowfreq` | dB | Welch `nperseg=131072` over an auto-selected ~90 s body section; the range used is reported | `ltas_lowfreq` |
| `band_energy.tables` | %, dB | 8 bands, on mid, side and each channel | `bands_hz` |
| `third_octave` | dB | ISO centres 20 Hz–20 kHz, relative to the loudest band, mid and side | `third_octave_centres_hz` |
| `bark` | dB | 24 Zwicker critical bands | `bark_edges_hz` |
| `tilt` | dB/oct | least squares 100 Hz–10 kHz **with R²**, plus 4 piecewise slopes | `tilt_fit_range_hz`, `tilt_piecewise_hz` |
| `bass_fundamentals` | Hz, dB, cents, Q | peak picking below 200 Hz on the high-resolution LTAS, with the nearest note and deviation | `bass_peak_*` |
| `resonances` | Hz, dB, Q, fraction | narrow peaks against a running mean, with the fraction of frames they appear in | `resonance_*` |
| `descriptors` | Hz, — | centroid, spread, skew, kurtosis, flatness, rolloff 85/95/99, ZCR: whole track and 1 s timeline | `descriptor_timeline_hop_s` |
| `band_timeline` | dB | per-band energy at 100 ms | `band_timeline_hop_ms` |

### Stereo field (`stereo`)

Convention, stated in the output as well: `mid = (L+R)/2`, `side = (L-R)/2`.

| Metric | Unit | Method | Parameter |
| --- | --- | --- | --- |
| `side_minus_mid_db` | dB | `10·log10(P_side / P_mid)` | — |
| `side_minus_mid_per_third_octave` | dB | from the L/R auto- and cross-spectra | `spectrum.third_octave_centres_hz` |
| `mono_crossover_hz` | Hz | highest third-octave centre below which side/mid stays under -20 dB | `mono_crossover_threshold_db` |
| `correlation` | — | overall, 1 s timeline, per band, min/P5/median, % of time below 0 and 0.3, three most negative windows | `correlation_window_s` |
| `channel_balance` | dB, LUFS | L vs R RMS and integrated loudness | — |
| `inter_channel_time_offset` | samples, µs | cross-correlation over ±5 ms, with the correlation at that lag | `itd_search_ms` |
| `width_timeline` | dB | side/mid per second (per-section values live in `structure`) | — |
| `mono_sum_damage` | dB | `10·log10(P_mid/(P_mid+P_side))` per third-octave | — |
| `goniometer` | °, fraction | energy-weighted histogram of `atan2(side, mid)` in 15° bins, and the energy outside ±45° | `goniometer_bins_deg` |

### Structure, tempo, key (`structure`)

| Metric | Unit | Method | Parameter |
| --- | --- | --- | --- |
| `sections[]` | s, LUFS, dB | MFCC+chroma+RMS+spectral-contrast stack, cosine SSM, Foote novelty, peak-picked, segments under 4 s merged | `structure` |
| `sections[].*` | — | per section: LUFS-I, short-term max, crest, tilt, 8-band table, side/mid, onset rate, delta vs previous and vs the track | — |
| `biggest_jump` | dB, s | largest section-to-section change, with its timestamp | — |
| `arrangement_gaps[]` | ms, dB | a band more than 20 dB below its own track RMS for at least 200 ms | `arrangement_gap` |
| `tempo.bpm` | BPM | `librosa.beat.beat_track`, refined by regressing beat time on beat index | `general.librosa_*` |
| `tempo.bpm_drift_std` | BPM | per 30 s window | `tempo_drift_window_s` |
| `key` | — | mean chroma-CQT against Krumhansl-Schmuckler profiles, with the runner-up and a margin | `key_low_confidence_margin` |
| `key.tuning_cents`, `implied_a4_hz` | cents, Hz | `librosa.estimate_tuning` | — |

### Processing forensics (`processing`) — every value here is an inference

| Metric | Unit | Method | Parameter |
| --- | --- | --- | --- |
| `saturation_proxy.slope_db_per_db` | dB/dB | least squares of 5–10 kHz frame level on broadband frame level, 50 ms frames, with R² and per section | `saturation` |
| `bus_compression` | —, ms, dB | cross-correlation of the sub-120 Hz and 500 Hz–6 kHz dB envelopes over ±200 ms; dip depth and a 1/e release estimate | `pumping` |
| `modulation_spectrum` | dB | FFT of each band's 5 ms envelope; depth at the beat, half-beat and quarter-beat rate, plus the dip phase against the beat grid | `modulation` |
| `multiband_timeline` | dB | per-band RMS and crest at 10 ms, plus the band-envelope correlation matrix | `multiband_timeline_hop_ms` |
| `hpss` | dB | `librosa.decompose.hpss`; percussive-to-harmonic overall and per band | `hpss` |
| `hpss.vocal_band_proxy` | dB | harmonic energy in 1–4 kHz relative to total, per second | `vocal_band_hz` |
| `reverb` | s, dB | Schroeder reverse integration after strong onsets, per octave band: T20, T30, early-to-late, tail L/R correlation | `reverb` |
| `transient_density` | per s | per-band envelope rises of 6 dB within 20 ms | — |

### Stems (`stems`, only with `--stems`)

`demucs` (htdemucs, 4 stems) runs locally and the loudness, dynamics, spectrum
and stereo metric sets are computed on each stem, plus its level relative to the
mix in dB and LUFS. Separated stems are cached under `~/.cache/mtx/stems` so
re-runs are cheap. Every stem-derived number carries `source: "separated"`,
because separation artefacts are real and a stem measurement is not a mix
measurement.

---

## `compare`

1. **Level-match first.** Both files are gained to equal LUFS-I before anything
   is compared, and the gain applied is reported. Comparing unmatched is the
   single most reliable way to reach a wrong conclusion in this field.
2. Side-by-side table of every headline metric, with the delta and — for
   level-dependent metrics — the level-matched delta.
3. Per-third-octave spectral difference (B − A), for mid and side separately.
4. Per-band side/mid, correlation, PSR and crest differences.
5. `--null-test`: finds the offset by cross-correlation, resamples if the rates
   differ, gain-matches, inverts and sums. Reports the residual in dBFS overall,
   per third-octave and as a timeline. **It refuses, with a clear message, if
   the correlation after alignment is below 0.5** — the two files are then not
   plausibly the same performance and the residual would mean nothing.

---

## Profiles and performance

`--profile quick` skips:

`loudness.true_peak_16x` · `loudness.intersample_overs` · `dynamics.onsets` ·
`stereo.goniometer` · `spectrum.resonances` · `spectrum.descriptor_timeline` ·
`forensics.cutoff_stability` · `structure.sections` · `structure.tempo` ·
`structure.key` · `processing.*` (reverb, modulation, HPSS, multiband,
transients) · `spectrum.ltas_lowfreq` · `forensics.wow_flutter` · `stems`

In quick mode PLR and the streaming preview fall back to the 4x true peak, and
`loudness.plr_true_peak_source` says so.

Measured on a 2019-era Windows laptop (Python 3.14, single-threaded numpy),
44.1 kHz / 24-bit stereo, `--stems` excluded:

| Track length | `--profile quick` | `--profile full` |
| --- | --- | --- |
| 1:15 | 6.5 s | 17 s |
| 4:20 | 20 s | 51 s |

Notes on how that is achieved, since the numbers are otherwise surprising:

- The 16x true-peak pass is **pruned exactly**, not approximately. An
  interpolated sample is a weighted sum of the input samples in its support, so
  it cannot exceed the largest of them times the filter's per-phase L1 gain
  (+7.01 dB for this filter). Stretches whose bound falls below both the file's
  own sample peak and the lowest reporting threshold cannot contain the maximum
  or an over, and are skipped. `mtx selftest` asserts that the pruned scan
  returns bit-identical results to a full scan.
- The 4x pass keeps a 1 ms max-envelope, and the PSR timeline is a rolling
  maximum over it, so no window is ever oversampled twice.
- Mid and side spectra are derived from the L/R auto- and cross-spectra rather
  than computed separately. This is an identity, not an approximation, and
  `tests/test_midside.py` asserts it against a direct Welch.
- The band split, the long-term spectra and the librosa features (onset
  envelope, chroma-CQT) are each computed once per run and shared.

Memory is proportional to duration: the file is decoded once into float32 in
chunked reads, and band-split work runs at `min(native_sr, 48000)` because every
analysis band tops out at 20 kHz. Forensics deliberately run at the file's own
rate, so that a resampling rolloff is never mistaken for a codec shelf. A file
that decodes to more than 1 GB gets a warning and then proceeds.

---

## Robustness

Handled explicitly, with a warning rather than a crash: files with no tags,
files with corrupt tags, leading silence longer than the analysis window, very
short files (under 10 s — metrics that need 3 s or 10 s windows return `null`),
mono and multichannel files, sample rates from 44.1 k to 192 k, and a missing
`ffmpeg` (the Python path still runs, and `FLAGS` records that
cross-validation was unavailable).

---

## Development

```bash
pip install -e ".[dev]"
pytest          # unit tests for band splitting, mid/side maths and the output contract
mtx selftest    # the synthetic-signal suite
```

`SCHEMA.md` documents every field of `analysis.json`.

## Licence

MIT.
