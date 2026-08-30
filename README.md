# mtx — master extractor

`mtx` reads a lossless audio file on your machine and writes an exhaustive,
reproducible measurement dump: a large `analysis.json` that stays local, and a
compact `digest.md` (~12 KB by default, `--digest-budget` to change it) you can
paste into an analysis workflow somewhere else, plus a `corpus_row.json` for the
archive.

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
mtx analyze <file> [--out DIR] [--profile quick|full] [--plots] [--stems]
                   [--blind] [--sections A,B,C] [--digest-budget 20k] [--json-only]
                   [--max-part-size 4.5m] [--no-split]
mtx scan [PATH] [--out DIR] [--library-root DIR] [-j N] [--force] [--recheck]
                [--dry-run] [--profile quick|full] [--json-only] [--no-summary]
mtx batch <dir> [--out DIR] [--recursive] [--csv summary.csv]
                [--csv-schema internal|corpus]
mtx compare <fileA> <fileB> [--out DIR] [--null-test]
mtx enrich <DIR> [--providers A,B,C] [--cache DIR] [--offline] [--refresh]
mtx join <analysis.json|DIR> [--out FILE]
mtx predict --check <predictions> <digest.md|analysis.json>
mtx validate-dr <file> --published <DR> [--source TEXT] [--show]
mtx selftest
mtx --version
```

- `analyze` — the main path. `--profile full` is the default; `quick` skips the
  expensive DSP (see the profile table below).
- `scan` — the way to measure more than one file. Takes an album, an artist or a
  whole library, measures in parallel, skips what it has already measured, and
  survives being interrupted. See *Scanning a library* below.
- `batch` — one JSON per file plus a single CSV of headline metrics, one row per
  track. This is how you bootstrap a reference library from records you own.
  `--csv-schema corpus` names the columns after the properties a corpus
  database is likely to already have (`LUFS-I`, `True peak`, `PSR min`, `DR14`,
  `Crest (loudest 10s)`, `mtx run`, …) so the CSV imports as a populated table
  instead of a mapping exercise. CSV values are rounded to 3 decimals — the
  unrounded number stays in `analysis.json`. Bootstrap a corpus with the full
  profile: `quick` skips the 16x true peak, which leaves the `True peak` column
  empty in every row, and `batch` says so before it starts.
- `enrich` — the one command that uses the network, and it is off unless you
  run it. Looks each analysed folder up in the public music databases and
  writes `online.json` beside `analysis.json`. See *Enrichment* below.
- `compare` — two files, **level-matched first**, with an optional null test.
- `join` — puts a split `analysis.json` back together (see *Uploading the
  analysis* below). Reads the index or the directory holding it.
- `predict` — scores a filled-in prediction sheet against the measurements.
  Arithmetic only: signed error, absolute error, and whether the stated
  interval held. It never says whether a prediction was a good one.
- `validate-dr` — records this implementation's DR14 against a published rating
  for a track you own. See *DR14 and the validation record* below.
- Default output directory: `./mtx_out/<artist - title>/`, taken from the
  embedded tags; the filename is used when the file carries no title tag.
- Progress goes to stderr; stdout carries only the output path.
- Exit codes: `0` success, `1` unreadable input, `2` a self-test assertion failed.

### Scanning a library

`mtx scan` measures whatever the path covers. The scope is the only thing that
changes between these three:

```text
mtx scan "E:\Music"                        # the whole library
mtx scan "E:\Music\Ed Sheeran"             # one artist
mtx scan "E:\Music\Ed Sheeran\÷"           # one album
mtx scan                                   # whatever the shell is sitting in
```

Results land in a mirror of the library tree, so a track measured as part of a
library scan and the same track measured on its own are the same folder:

```text
E:\Music\Ed Sheeran\÷\04. Shape of You.flac
E:\mtx_out\Ed Sheeran\÷\04. Shape of You\analysis.json
                                          \digest.md
                                          \corpus_row.json
                                          \mtx_source.json
```

That mapping needs to know where the library starts, so say it once:

```text
mtx scan "E:\Music" --out "E:\mtx_out" --dry-run
```

The root is recorded (in the user config directory, not in the music folder,
which is never written to) and every later `mtx scan` from any level underneath
resolves to the same tree with no flags at all. Until a root is registered,
`mtx scan` refuses to guess one rather than strand a first scan's results in a
tree the next one will not look in.

**Nothing is measured twice.** Each output folder carries `mtx_source.json`, a
receipt naming the file it came from — size, modification time, profile, schema
version. A scan reads those, not the audio, so a library that is already
measured is re-checked in under a second. A track is measured again only when
its source changed, the profile changed, or the schema version moved:

```text
[mtx] 597 file(s) found: 585 already measured, 12 to do
```

Because the receipt is written per track, an interrupted scan is resumable:
stop it with Ctrl-C, run it again, and it picks up where it left off. There is
no progress file to lose.

- `--force` re-measures everything.
- `--recheck` decides staleness by hashing each source rather than trusting its
  modification time. Slower, and the right choice after copying a library
  between drives, which rewrites every mtime.
- `--dry-run` lists what would be measured and why, and stops.
- `-j N` sets the parallelism budget (see *Profiles and performance*).

`summary.csv` is rewritten over the scanned subtree at the end of every run,
covering every track under it that has a result — including ones measured on an
earlier run — with the same column names `--csv-schema corpus` uses.

### Predicting before measuring

`--blind` writes the digest without printing it, and prints the path of a
prediction sheet instead:

```
$ mtx analyze track.flac --blind
[mtx] blind mode: digest.md was written and is NOT printed; commit the prediction first, then read it
mtx_out/track/predict.md

$ $EDITOR mtx_out/track/predict.md          # fill in value, +/- range, confidence
$ mtx predict --check mtx_out/track/predict.md mtx_out/track/digest.md
```

The sheet carries the field list, the units, `FLAGS` and `METHOD` — knowing
*how* a number is derived is fair information for a prediction — and none of
the values, including the ones `DETAIL` and `CORPUS ROW` would otherwise
restate. Score against `analysis.json` instead of `digest.md` if you want the
unrounded values.

### Choosing what the digest spends its budget on

The digest has a size cap and a fixed drop order, which means a
stereo-focused session can lose the stereo detail it needed while carrying a
reverb block it did not. Two ways out, neither of them the default:

```
mtx analyze track.flac --sections stereo,forensics,structure
mtx analyze track.flac --digest-budget 20k
```

`--sections` takes groups (`loudness`, `dynamics`, `spectrum`, `stereo`,
`forensics`, `structure`, `processing`) or exact block names; an unrecognised
name is an error rather than a silent no-op. A `--stems` run raises the cap by
4 KB on its own, because the stem table exists nowhere else in the paste-able
output.

### Output

| File | What it is |
| --- | --- |
| `analysis.json` | Everything, with the full parameter block. Large. Past the part size limit it becomes an index plus `analysis.partNN.json`. |
| `analysis.partNN.json` | Only when the analysis is over the limit. One fragment each, listed in the index's `split` block. |
| `digest.md` | `HEADLINE` / `FLAGS` / `DETAIL` / `STEMS` (with `--stems`) / `CORPUS ROW` / `METHOD`. ~12 KB by default. |
| `corpus_row.json` | The corpus row as typed JSON, keyed by property name, for import rather than retyping. |
| `predict.md` | Only with `--blind`. The headline as a form to fill in before reading the digest. |
| `plots/*.png` | Only with `--plots`. For your own eyes; never referenced by the digest. |
| `online.json` | Only after `mtx enrich`. Genre vote, credits, identifiers and cross-checks from the public databases. Never merged into `analysis.json`. |

`FLAGS` comes **before** the detail: warnings, method disagreements and every
low-confidence metric are the first thing you read.

A worked example is in [`samples/digest.md`](samples/digest.md).

### Uploading the analysis

A full-profile analysis of a four-minute track is comfortably past 5 MB, which
is the per-file cap on an upload to Notion and to most other places a
measurement archive ends up. So `analysis.json` is written whole while it fits
under the limit, and as an index plus numbered parts when it does not:

```
analysis.json            the index: the headline, run, params, warnings and
                         every other section small enough to stay inline,
                         plus `split` -- the manifest naming the parts
analysis.part01.json     one fragment each, with the path it belongs at
analysis.part02.json
```

Every file in the set is valid JSON on its own and every one is under the
limit, so the whole set uploads. Nothing is dropped, rounded or summarised:
the split is a transport detail.

```
mtx analyze track.flac --max-part-size 2m    # smaller parts
mtx analyze track.flac --no-split            # one file, however large
mtx join mtx_out/track/                      # -> analysis.full.json
```

The default is 4.5 MB, under the 5 MB limit with room for the part header.
`--max-part-size` and `--no-split` apply to `analyze`, `batch` and `compare`
alike; `comparison.json` is split by the same rule. `mtx predict --check` reads
a split `analysis.json` directly — the headline stays in the index.

---

### Enrichment: what the file does not know about itself

`mtx analyze` never touches the network. `mtx enrich` does, deliberately and
separately, because a purchased download keeps the ISRC and throws away almost
everything else: who mixed it, who wrote it, what a listener would call it.

```
mtx enrich ./mtx_out                      # a whole corpus
mtx enrich ./mtx_out/"Artist - Title"     # one track, --print to see it
mtx enrich ./mtx_out --providers all      # add the two that need credentials
mtx enrich ./mtx_out --offline            # answer only from the cache
```

Keyless by default: **MusicBrainz** (community-voted genres at recording,
release-group and artist level, plus engineer and songwriter credits),
**Deezer** (exact ISRC addressing, a published BPM, popularity), **Apple /
iTunes** (a third, independent genre taxonomy). Two more switch on when their
credentials are present: **Last.fm** (`LASTFM_API_KEY`) for listener tags and
real play counts, **Discogs** (`DISCOGS_TOKEN`) for sleeve credits and a
genre/style split.

Three things make the result trustworthy rather than merely present:

**A database row is not accepted just because the ISRC matched.** Labels reuse
an ISRC across a radio edit and the album cut. `bad guy`'s returns three
MusicBrainz recordings and lists the 175 s radio edit first; the file is 194 s.
Every candidate is scored against what mtx already measured — duration
loudest, since that is the one field the analysis knows exactly — and the
losing candidates stay in the output with their scores, so a wrong match is
auditable instead of invisible.

**The genre vote does not reward coarseness.** Each source is scaled against
its own top vote, not against the sum of its votes. Sharing the total would
punish exactly the sources worth having: MusicBrainz spreads nine genres over a
record, so each would land near a ninth, while a shop returning the single word
`Alternative` would collect its full weight and win. Every genre carries the
sources that voted for it, and a coarse `umbrella` is offered *alongside* the
ranked list, never instead of it — so a query can filter on `pop` and still
read `avant-garde pop`.

**Disagreement is the output, not an error to be smoothed away.** Where an
outside number can be compared with one mtx derived, both are kept:

| Check | What it settles |
| --- | --- |
| `cross_checks.tempo` | mtx estimates tempo from an onset envelope and marks it low-confidence on most of a pop corpus. A published BPM that agrees promotes it to `high`; one that is exactly double is reported as `octave` — the beat tracker locked to a different metrical level, not a different tempo — and a real disagreement leaves the local value alone at `low`. |
| `cross_checks.duration` | `exact` / `close` / `differs` against every provider, which is what makes the match itself verifiable. |
| `cross_checks.release_date` | The tag, the release, the release group and two shops, plus the earliest of them and whether they agree. |

Nothing measured is ever overwritten. `online.json` is a **sidecar**, not a
section of `analysis.json`, because `mtx analyze` promises byte-identical
output for the same input and a section built from whatever MusicBrainz looked
like this morning cannot live inside that promise.

Responses are cached under `.mtx_cache/`, so a second pass over an enriched
corpus makes no requests and `--offline` works with the network unplugged.
Per-host rate limits are honoured — MusicBrainz's one-request-per-second above
all — and the whole subpackage is stdlib-only, so enrichment adds no dependency
to a tool whose point is reproducible local measurement.

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

- **DR14 starts out unvalidated against a published DR rating.** `mtx` ships no
  copyrighted reference track, so out of the box the implementation is only
  checked against analytically known synthetic cases (a continuous sine must
  give DR 0.0). Every run says so in `loudness.dr14.validation` and in `FLAGS`.

  This is fixable once, permanently, on your own machine. Measure a track whose
  published DR rating you already know:

  ```
  mtx validate-dr "Some Track.flac" --published 12 --source "dr.loudness-war.info"
  mtx validate-dr --show      # the record, at any time
  ```

  The pair is stored (default: the platform config directory, override with
  `MTX_DR14_VALIDATION`), and from then on `FLAGS` and `METHOD` report what the
  record says — `[validated against N track(s)]` with the worst disagreement,
  or `[disputed]` if a recorded rating is more than 1 DR out. The record holds
  measured value, published value and the difference; it draws no conclusion
  beyond that.
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

### Harmony (`harmony`)

The chord track. Not a learned model: binary chord-tone templates matched by
Pearson correlation against beat-synchronous chroma, smoothed by a Viterbi pass
with one self-transition probability, so every number is reproducible from
`params.harmony` alone and adds no dependency.

| Metric | Unit | Method | Parameter |
| --- | --- | --- | --- |
| `chords[]` | s | 13 qualities x 12 roots plus a no-chord state, merged into segments with a per-segment match score | `harmony` |
| `chords[].inversion`, `slash_label` | — | the chord root against a low-register (C1, two-octave) chroma | — |
| `harmonic_rhythm` | per bar, per s | chord changes, with the caveat that a bar comes from `structure.tempo` | — |
| `degrees` | % | roman-numeral reduction against `structure.key`; diatonic vs borrowed chord time | — |
| `loop` | bars | shortest period in 1/2/4/8/16 bars whose chord sets repeat above the threshold | `loop_candidate_bars` |
| `cadences` | count | V-I, IV-I, V-vi, I-V degree transitions | `cadence_degrees` |
| `pedal_points[]` | s | the bass note holds while the chord root moves | `pedal_min_chords` |
| `modulation` | — | sliding-window Krumhansl-Schmuckler; always `confidence: low` | `modulation` |
| `key_from_chords` | — | the key whose scale explains the most chord time, plus tonic evidence | `key_from_chords` |
| `key_cross_check` | — | the chord-track key against `structure.key`; a disagreement is a FLAG | — |

**Measured accuracy.** Against seven published, human-transcribed chord charts
the recogniser spends **82% of chord time** on a chord whose root and triad
quality appear in the chart (86% at root level). `key_from_chords` got **4 of 7**
published keys; `structure.key` got 5 of 7 on the same tracks. So the chord
track is a genuine second opinion on the key and not a better one, its failure
mode is the relative major/minor, and the block says so with a confidence.

### Rhythm (`rhythm`)

A tempo is not a groove: without a downbeat there is no bar.

| Metric | Unit | Method | Parameter |
| --- | --- | --- | --- |
| `downbeats` | — | meter and phase that maximise the mean downbeat accent, where accent is the z-scored sum of onset strength, 20–120 Hz energy and chroma change at each beat | `meters` |
| `tempo_octave` | ratio | onset strength at the midpoints between beats, and the weaker of the two alternating beat phases | `octave_check` |
| `swing` | ratio | median position of the off-beat onset nearest each beat midpoint; 0.5 is straight, 0.667 is a triplet shuffle | `swing` |
| `grid.deviation` | ms | onset-to-nearest-grid distance at 5.8 ms onset resolution, with a programmed-grid inference | `grid_subdivision` |
| `syncopation` | per bar | Longuet-Higgins & Lee weights over a 16-step bar | — |
| `beat_position_profile` | dB | kick- and snare-band level per position in the bar; four-on-the-floor and backbeat are only claimed once a kick pattern exists | — |
| `pulse_rate` | per beat | onsets per beat per section, and half/double-time switches | — |

**A caveat the block reports on itself.** `structure.tempo` picks one metrical
level, and on the seven reference tracks it reported half the published tempo
once and double it once. `tempo_octave` measures the ambiguity but does not
resolve it: on that set it raised no false alarm on the four correct tempos and
detected none of the three wrong ones, because the classes overlap. Read the
two ratios, and read `bar_count` and `changes_per_bar` knowing what they are
divided by.

### Song form (`form`)

Two stages, kept apart. Sections are clustered into **letters** by cosine
distance over their measured vectors and consecutive same-letter sections are
merged into **parts** — that is the measurement. Function names (verse, chorus,
bridge) are an **inference** over it by the rules in `params.form`, and every
label carries the evidence that produced it and a confidence.

Gives what people actually ask a record: time to the first chorus in seconds
and as a fraction, intro length, time to vocal entry, chorus count and share,
whether the second chorus is arranged up from the first, ending type, and
loopability. An optional `allin1` model is reported beside the measurement,
never merged into it.

### Delivery conditions (`delivery`)

What the master does once it is distributed — all local, all offline, and one
of the two things an unfinished mix can actually use.

| Metric | Unit | Method | Parameter |
| --- | --- | --- | --- |
| `encode` | LUFS, dBTP, dB | ffmpeg encode to AAC 256 and Opus 128, decoded back and re-measured; new true-peak overs and HF damage | `encodes` |
| `small_speaker` | %, LU | what survives a 400 Hz – 8 kHz band-pass | `small_speaker_band_hz` |
| `mono_fold` | LU, dB | the mono sum, loudness-weighted and per octave | — |
| `excerpts` | LUFS, dBTP | the first 15 s, the first 30 s, and the chorus as a 15 s clip | `excerpt_s` |

### Lyrics (`lyrics`)

A **declared** lyric beats a **tag** beats a **transcript**, and the source
travels with the text. Language is detected before anything English-specific
runs: the syllable counter and the readability score decline rather than
produce a meaningless number. Shape (counts, type-token ratio, repetition,
compression ratio, longest and most repeated n-gram, pronouns, title
occurrences) is measured; valence and concreteness need a lexicon that does not
ship with mtx and report `available: false` with what to install.

### Declared metadata (`declared`) and version identity (`version`)

For your own unreleased work the splits, the publisher and the lyric are not
missing — they are unentered. A `declared.json` sidecar supplies them, and every
value is reported with `source: "declared"` and never merged into a measured
field or into `online.*`. Version identity is derived from tags alone: two files
that agree on `work_key` and differ on `markers` are two versions of one song.

### Coverage (`coverage`)

One uniform mask over the whole document: which of the N features are present,
and how far each is trusted, so a consumer does not have to rediscover that by
walking the document itself.

### Stems (`stems`, only with `--stems`, rendered as `## STEMS` in the digest)

`demucs` (htdemucs, 4 stems) runs locally and the loudness, dynamics, spectrum
and stereo metric sets are computed on each stem, plus its level relative to the
mix in dB and LUFS. Separated stems are cached under `~/.cache/mtx/stems` so
re-runs are cheap. Every stem-derived number carries `source: "separated"`,
because separation artefacts are real and a stem measurement is not a mix
measurement. `--stems-model htdemucs_6s` splits guitar and piano out of `other`
at no new dependency.

Separation is also the gate on four measurements that only exist once there is
more than one signal, all computed from a single load of the stems:

- **`masking`** — every other stem number in this tool is measured in isolation
  or against the mix. This measures each stem *against another stem*, which is
  the whole of mix engineering: a per-band masking matrix, spectral overlap per
  pair, masking release across sections, and the vocal-to-instrumental balance
  per section. Plus, from the vocal stem alone: sibilance behaviour as a dB/dB
  slope (de-esser evidence), the high-pass corner, reverb send and pre-delay,
  and tempo-synced delay throws.
- **`melody`** — `librosa.pyin` on the vocal and bass stems. Read
  `range.p5_p95_semitones`: checked against two published vocal ranges the
  duration-weighted percentiles landed within a semitone of both, while the raw
  extremes came out 40–58 semitones wide, because a monophonic tracker on a
  separated stem makes octave errors on 7–12% of note time and one of them sets
  the maximum. Those outliers are counted and reported rather than hidden.
  Also intervals, phrases, vibrato, chromaticism, contour per section,
  sung-vs-rapped, and a **pitch-quantisation signature** — grid deviation and
  note-to-note transition time, reported as forensics and never as a verdict
  about a singer.
- **`arrangement`** — entry and exit per stem in seconds and bars, concurrent
  source count over time, drum-machine evidence, 808 behaviour and glide,
  vocal stacking, lead-versus-backing balance and call-and-response.
- **`microtiming`** — "the drums are dragging" as the median onset deviation
  from the beat grid, per stem. Read `median_minus_common_mode_ms`: the beat
  tracker and the onset detector each carry a constant lag which is identical
  for every stem and cancels between them.

---

## `cohort` — where a track sits among comparable records

`-7.77 LUFS` means nothing on its own. `mtx cohort <folder>` reads a folder of
analysed folders and writes `cohort.json` and `cohort.md` **beside** them: per
metric, the percentile and z-score within a `(genre, year)` cohort, within the
whole corpus and within the same artist's other tracks, plus a distance to the
cohort centroid and nearest neighbours.

**It is deliberately not part of `analyze`.** A per-track measurement must not
depend on what else happens to be in the folder — that would break
reproducibility, which is property one. The absolute numbers are never touched.

Cohort labels come from `enrich` for published records and from a declared
sidecar for an unreleased one, which is the useful direction: a mix in progress
can be positioned against the released records it is competing with, provided
you state what it should be compared to. The most specific cohort with enough
members wins, and the fallback is recorded.

The **corpus hygiene** report is part of the output, not a footnote: it names a
corpus too small or too dominated by one artist for its percentiles to mean
anything. `typicality.mean_abs_z` is a distance from the cohort centre, not a
rating.

## `export` — flat tables at track and section level

`mtx export <folder>` writes `mtx_tracks.csv` (one row per track, every scalar
under its dotted path) and `mtx_sections.csv` (one row per track x section,
joining the measured section vector to the form label, the per-section masking
indices, pulse rate, melodic contour and arrangement density). Parquet too,
when `pyarrow` is installed. The per-section vectors are the most valuable part
of the dump and were previously the hardest to get at.

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
transients) · `spectrum.ltas_lowfreq` · `forensics.wow_flutter`

`--stems` is not a profile switch: it is opt-in at either profile, and the
per-stem measurements inherit whichever profile the run used.

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

### Where a full run actually goes

Profiled on a 3:54 track, 44.1 kHz / 24-bit stereo, full profile, no stems:

| Stage | Time | Share |
| --- | --- | --- |
| loudness, true peak, DR | 16.6 s | 33% |
| processing forensics | 9.0 s | 18% |
| structure, tempo, key | 6.9 s | 14% |
| spectrum | 5.8 s | 11% |
| stereo field | 4.7 s | 9% |
| source forensics | 4.6 s | 9% |
| dynamics | 1.8 s | 4% |
| file, container, decode | 1.2 s | 2% |

Inside that, the four heaviest leaves are the true-peak oversampling
(`resample_poly`: 12.9 s on one thread, 10.5 s of it the 16x pass), the HPSS
median filters (5.4 s), the zero-phase band filters (`sosfiltfilt`, 5.5 s
across 21 calls) and roughly twenty thousand short FFTs from the per-frame
Welch loops.

The 16x pass is also where the pruning described above stops helping. It is
still exact, but on a master that runs into a limiter the bound it tests is
cleared nearly everywhere: on the track above it scanned **98.5%** of the file.
Pruning earns its keep on dynamic material and is close to free on modern pop,
which is the material this tool is usually pointed at — so that pass is
threaded rather than relied on to skip work.

### Two layers of parallelism, which never multiply

Between files there is no shared state, so `mtx scan` runs **one process per
physical core**. Within one file only some of the work can be threaded, because
only some of it lets go of the GIL — measured here, on scipy 1.18 / numpy 2.5:

| Primitive | Used by | GIL | 4 threads |
| --- | --- | --- | --- |
| `resample_poly` / `upfirdn` | true peak | released | 3.8x |
| `ndimage.median_filter` | HPSS | released | 3.1x |
| `lfilter` | K-weighting | released | 2.7x |
| `sosfilt` / `sosfiltfilt` | band split | **held** | 1.2x |
| `welch`, `rfft` | every spectrum | **held** | 1.1x |

So threads are spent on the true-peak scan and nowhere else; the GIL-bound
majority of the work is why the between-files layer is the one that matters.
`-j N` is a single budget covering both: processes are taken first, and threads
only pick up the slack when fewer files remain than there is room to run. On a
6-core machine `-j 5` means five processes with one thread each while there is a
queue, and one process with five threads for the last file.

Threading the true-peak scan changes no output. The oversampling of each chunk
is pure, and the results are folded back into the running scan in chunk order,
so an inter-sample over that straddles a chunk boundary is still counted once.
`mtx selftest` asserts that a scan on one thread and on four are bit-identical,
envelope included, and the whole 309,611-value `analysis.json` of a real track
is unchanged at every thread count.

Memory is the other limit: about 0.5 GB of resident memory per worker on a
44.1 kHz track (measured), and proportionally more at higher rates, so the
default stops at 8 workers however many cores are present.

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

`GAPS.md` documents what the tool does **not** measure — an audit of the musical
content that never reaches the dump (harmony, melody, groove, song form,
instrument identity, inter-stem masking, lyric meaning), what each gap would
cost, which of them must stay outside `src/mtx/` to keep the tool's five
properties intact, and which need the song to have been *released* before their
data exists at all. Ten of the twelve are available on an unreleased master.
Read it before proposing a new metric module.

## Licence

MIT.
