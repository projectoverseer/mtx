# What `mtx` does not measure

A gap analysis, written after auditing the 0.2.0 output of a 64-track corpus
(`analysis.json` with all four parts joined, `online.json`, `digest.md`, and the
193 Notion properties that `mtx_enrich.py` builds from them).

The question that produced it: *if you wanted to model what a record is, from
the record alone, what is not in the dump yet?* Everything below is **intrinsic
to the song** — no sales, streams, chart positions, playlist adds or awards.
Those are outside-the-song, they change weekly, and they do not belong in a
measurement tool.

This is a list of measurements that are absent. It is not a work order. Nothing
here is more important than the five properties in `README.md`, and a gap that
cannot be closed reproducibly, with its parameters recorded and its confidence
stated, is better left open.

---

## The rule these gaps have to respect

> **This tool measures. It does not interpret, score, grade or recommend.**

That rule survives all of it. Several gaps below are attractive precisely
because someone wants to predict something with them. Resist the follow-on: a
scoring or prediction layer consumes `analysis.json` from outside, the way
`mtx_enrich.py` does. It does not live in `src/mtx/`. The moment a
genre-conditional threshold or a "this chorus arrives late" judgement is baked
into a metric module, the tool stops being trustworthy for the mastering work it
is already good at.

Corollary for every item here: report the continuous measurement next to any
boolean, carry a `confidence` where the value is an inference, and put the
parameters in `params`.

---

## What is already complete — do not re-add it

The signal chain is covered more thoroughly than in any commercial tool I am
aware of, and a newcomer should read the metric catalogue in `README.md` before
proposing anything:

Loudness (BS.1770-4 with ffmpeg and pyloudnorm cross-checks), true peak at 4x
and 16x, PSR/PLR/DR14, crest whole and per-band, flat-top forensics with the
derived per-channel threshold and the limiter-vs-clipper slope inference, LTAS
broadband and the second low-frequency pass, tilt with piecewise slopes, third-
octave mid/side, mono-sum damage, goniometer, inter-channel time offset,
correlation timeline, noise floor, HF cutoff and its stability, effective bit
depth, hum/rumble/wow-flutter/tape-bias, spectral holes, silence and fade
geometry, reverb T20 per octave, bus-compression pumping, saturation proxy,
modulation spectrum, transient density, arrangement gaps, per-section metrics,
four htdemucs stems each run through the same battery, and the whole
`online/` identity, credit and genre-voting layer.

The gaps are not in the signal chain. They are in **music**.

---

## Offline and online

Every gap below is tagged with where its data can come from. The distinction is
not "does it need a network call" — it is **whether a song has to have been
released for the data to exist at all**:

- **Offline** — obtainable from the audio itself. A master bounced out of a DAW
  an hour ago, never released, with no ISRC and no database entry anywhere,
  yields exactly the same measurement as a thirty-year-old catalogue record.
  This is `mtx analyze` territory: no network, no keys, no dependence on anyone
  else having catalogued the work.
- **Online** — exists only because the song was published. Someone else had to
  create the record: a distributor issued the ISRC, an editor typed the credits
  into MusicBrainz, listeners tagged it, a service counted plays. An unreleased
  master has none of it and cannot be made to have it. This is `mtx enrich`
  territory.

**Ten of the twelve gaps are fully offline.** That is not a coincidence — the
musical content of a record is in the record. Harmony, melody, groove, form,
instrumentation, masking, lyric meaning, timbre and delivery behaviour are all
recoverable from a file nobody has ever heard.

| # | Gap | Offline / online |
| --- | --- | --- |
| 1 | Harmony (chords) | Offline |
| 2 | Melody (F0 on the vocal stem) | Offline |
| 3 | Rhythm beyond BPM | Offline |
| 4 | Song form labels | Offline |
| 5 | Instrument ID, arrangement density | Offline |
| 6 | Inter-stem masking | Offline |
| 7 | Lyrics | Offline (transcription); **declared** for the text itself |
| 8 | Embeddings and trained taggers | Offline (local inference) |
| 9 | Corpus-relative layer | Offline maths, **online cohort labels** |
| 10 | Rights, credit graph, version identity | **Online** — or declared |
| 11 | Delivery-condition renderings | Offline |
| 12 | Export shape, corpus hygiene | Offline |

### What an unreleased master does not get, today

For context on the two mixed rows: a master that has never been published
receives **none** of the current `online.json`. `mtx enrich` will match nothing,
and every one of these is empty:

ISRC, UPC, all MBIDs, ISWC, label, release type and date, album context, the
voted genre with its umbrella and agreement score, the descriptive tags,
database-confirmed credits, `Explicit`, popularity ranks, and — importantly —
all three `cross_checks` (tempo against a published BPM, duration against the
distributed runtime, release-date agreement). Those cross-checks are
self-verification of the kind the tool is built around, and they are structurally
unavailable before release. On an unreleased file, `structure.tempo.confidence`
is the last word rather than the first.

That is the correct behaviour, not a bug: the fields are `null` with a reason,
nothing is guessed. But it means anything downstream that consumes `online.*`
must degrade cleanly rather than assume it is there.

### The design implication: declared metadata

Rows 7, 9 and 10 are the interesting ones, because for your **own** unreleased
work the data is not missing — you simply happen to be the source of it rather
than a database.

You know the lyric because you wrote it. You know the writer splits, the
publisher, the producer credits and whether this is the radio edit, because you
made those decisions. You know the intended genre and the target release year.
For a published record all of that is *discoverable*; for an unreleased one it is
*declarable*, and no amount of DSP will produce it.

So the gap for unreleased work is not a measurement gap at all — it is an input
gap. Closing it means an optional sidecar (`declared.json` beside
`analysis.json`, say) that `analyze` reads and passes through **clearly labelled
as declared**, never blended into measured or database-sourced fields. The
`sources` arrays in `online.credits` are already the right pattern: a claim
carries where it came from, and `"declared"` would be one more origin alongside
`"file:tag"` and `"musicbrainz"`.

This matters most for §9. Cohort comparison — where a track sits among
comparable records — is the feature that makes an unreleased mix legible, and it
is precisely the one that needs a genre and a year the unreleased track cannot
look up. The reference corpus it is compared *against* is built from published
records and gets its labels online; the candidate track declares its own.

---

## The gaps

Each entry states what is missing, who it is missing for, where it would live,
what it costs, and whether the data is offline or online in the sense defined
above.

### 1. Harmony — completely absent

**Offline.** Chord recognition runs on the waveform. An unreleased master is
no harder than a catalogue record.

**Missing.** Chords. There is no chord data anywhere in the schema. What exists
is one global key from a mean chroma-CQT correlated against Krumhansl-Schmuckler
profiles (`structure.key`), and on real material it is weak — on the reference
track it returns G major over G minor with a margin of 0.047 and
`confidence: "low"`, which is the detector telling you honestly that it cannot
separate a major from its parallel minor.

Nothing downstream can ask a harmonic question. Not "what progression is this",
not "how often does the harmony change", not "does it modulate".

Worth having: chord sequence over time, harmonic rhythm (changes per bar), loop
length and whether the song is a repeating 2/4/8-bar loop, degree reduction
relative to the detected key, share of diatonic vs borrowed chords, modulation
points, cadence types, inversions and slash chords (compare the chord root with
the measured bass fundamental — `spectrum.bass_fundamentals` already exists),
pedal points, chord-vocabulary size.

**For whom.** The songwriter, entirely. This is the half of the record `mtx`
currently cannot see at all.

**Where.** A new `metrics/harmony.py`, with `structure.key` staying where it is
and gaining a cross-check against the chord track the way loudness cross-checks
against ffmpeg — a key inferred from the chord sequence is an independent second
opinion on the KS estimate, and the disagreement is the interesting number.

**How.** Chordino (via `vamp`), `chord-extractor`, BTC, or `madmom`'s deep
chroma + CRF recogniser. All are inferences: they need a confidence and they
need to say so, like `processing/` does.

**Cost.** Medium. New dependency, likely an optional extra
(`pip install -e ".[harmony]"`) so the base install stays light.

### 2. Melody — absent, and the hardest part is already done

**Offline.** F0 comes off a stem the tool already separates locally.

**Missing.** F0. The vocal stem is separated and then measured only for level,
tilt, crest, centroid and onset rate — every number in `stems.stems.vocals` is a
mix-engineering number. The pitch content of the melody is thrown away.

Worth having, all from the isolated vocal: range in semitones, tessitura, median
pitch, the highest sung note **and its timestamp**, interval histogram,
stepwise-vs-leap ratio, contour per section, phrase lengths and breath
positions, notes per second, melodic self-similarity (how much the hook repeats
itself), chromaticism against the detected key, vibrato rate and depth, average
note duration, melisma index, and sung-vs-spoken/rapped classification.

And one that belongs in `forensics/` rather than melody: **pitch-quantisation
signature**. An F0 track that snaps to the semitone grid with near-zero
transition time is a measurable artefact of hard pitch correction, and it is
exactly the kind of thing the rest of the forensics module already does for
codecs, clipping and mains hum. State it as the measured transition statistics
plus an inference, never as a verdict about the singer.

**For whom.** Songwriter, producer, and the forensics reader.

**Where.** `metrics/stems.py` gains a pitch block per stem where pitch is
meaningful (vocals, bass), or a new `metrics/melody.py` that consumes stem
paths.

**How.** CREPE or `librosa.pyin` on `stems.stems.vocals.path`. pyin adds no
dependency at all.

**Cost.** Low — this is the best value-per-line item in the document. The
separation, which is the expensive part, has already run and is cached in
`~/.cache/mtx/stems`.

### 3. Rhythm beyond BPM

**Offline.** Downbeats, meter and microtiming are all in the signal. Note the
contrast with the existing `cross_checks.tempo`, which compares against a
*published* BPM and is therefore online-only — on an unreleased file the beat
tracker's own confidence is all there is.

**Missing.** `structure.tempo` gives BPM, beat times, drift, per-window tempo
and a grid-fit R². That is a tempo, not a groove.

Worth having: **downbeats** and time signature (without downbeats there is no
bar, and without a bar half the musical questions cannot be phrased), bar count,
swing ratio, syncopation index, microtiming deviation **per stem** against the
beat grid — is the bass late, is the snare pushing — quantisation tightness as a
programmed-vs-played measurement, backbeat/four-on-the-floor/half-time
detection, and half-time↔double-time switches.

Microtiming per stem is the strongest item in this section and, again, the
separation already exists. A producer's "the drums are dragging" is
`median onset deviation from the grid, in ms, per stem`.

**For whom.** Producer, and any structural work in §4 (labels need downbeats).

**Where.** `metrics/structure.py` for downbeats and meter; per-stem timing in
`metrics/stems.py`.

**How.** `madmom` (RNNDownBeatProcessor) or the downbeat output of the model in
§4. Microtiming needs only existing onsets plus a beat grid.

**Cost.** Medium.

### 4. Song form — there are segments, but no structure

**Offline.** Segmentation and labelling are both audio tasks.

**Missing.** `structure.sections` is 22 novelty-derived boundaries with LUFS,
tilt, width, crest, onset rate and band energy each. As raw material that is
excellent and better than most tools produce. But the sections are
**unlabeled**: nothing says intro, verse, pre-chorus, chorus, post-chorus,
bridge, drop, outro.

Without labels the following cannot be computed, and all of them are things
people actually ask a record: time to first chorus (in seconds and as a fraction
of duration), intro length, time to vocal entry, time to the title being sung,
number of choruses, chorus share of total duration, whether the second chorus is
arranged up from the first (compare their existing per-section vectors — the
data is already there, only the labels are missing), beat-switch count, ending
type (cold stop vs fade vs repeat-out — `forensics.silence` already distinguishes
`"fade"` from `"hard cut"`, so this is half-built), and loopability (similarity
of the last seconds to the first).

**For whom.** Everyone. This is the gap that most limits what any downstream
consumer can ask.

**Where.** `metrics/structure.py`, as a labelling pass over the existing
boundaries rather than a replacement for them. Keep the novelty segmentation:
it is reproducible and parameterised, and a learned labeller is neither.

**How.** `all-in-one` (Kim & Nam) returns functional segments *and* downbeats in
one pass, which closes half of §3 at the same time. Being a learned model, its
output belongs next to the measured boundaries with a confidence, not instead of
them.

**Cost.** Medium; a large optional dependency.

### 5. Instrument identification and arrangement density

**Offline.** Separation and tagging are local inference.

**Missing.** Four stems means guitars, keys, synths, strings, horns and pads all
collapse into `other`. There is no answer to "what is playing", and therefore
none to "when does it come in".

Worth having: instrument tags over time, an entry/exit map per section (which
element enters at which bar), concurrent-source count as an arrangement-density
curve, acoustic-vs-electronic drum classification, 808 presence and glide,
sub-bass type, vocal-layer count (harmony-stack detection), lead-vs-backing
ratio, ad-lib density, call-and-response.

**For whom.** Producer and arranger. "Arrangement" as a working concept barely
exists in the current dump; `structure.arrangement_gaps` is the only trace of
it, and it is defined on frequency bands, not on instruments.

**Where.** `metrics/stems.py`.

**How.** Cheapest first step by a wide margin: demucs 6-stem, which adds guitar
and piano and needs no new dependency, only a model choice and a re-run.
Beyond that, PANNs / MTG-Jamendo / AST tagging over time.

**Cost.** Low for 6-stem, medium for tagging.

### 6. Inter-stem masking — the mixing metric that is missing

**Offline**, and emphatically so — this is the item that most rewards being run
on unreleased work, because it is diagnostic of a mix in progress rather than
descriptive of a finished record. It is the one gap here that belongs in a
revision loop.

**Missing.** Every stem is measured in isolation and against the mix
(`level_vs_mix.lufs_delta`). No stem is ever measured **against another stem**.

That is the whole of mix engineering. A per-band masking matrix — how much
energy each stem contributes in the bands where the vocal lives, spectral
overlap between pairs, masking release across sections, and the vocal-to-
instrumental balance *per section* rather than only whole-track — is computable
today from data already on disk, needs no new dependency, and as far as I know
no consumer-facing tool does it.

**For whom.** The mixing engineer, more directly than anything else in this
document.

**Where.** `metrics/stems.py`, a `masking` block. Related and equally cheap,
all from the vocal stem: sibilance-band behaviour as de-esser evidence, vocal
reverb send and pre-delay from the stem tail (the machinery is in
`processing.reverb`), tempo-synced delay-throw detection, vocal high-pass corner.

**Cost.** Low. Pure DSP over existing signals — the best ratio in the list after
§2.

### 7. Lyrics — counted, not read, and missing on more than half the corpus

**Offline for alignment and analysis; declared for the text.** Transcription
from the vocal stem is pure local inference and works on an unreleased master.
But a transcript is not a lyric sheet — it is a guess at one, and mishears.
For your own unreleased work you hold the authoritative text, so the right input
is a declared lyric (or a tag you wrote) with the transcript supplying only the
timings. Lyric *lookup* from a service, which this tool does not do, would be
online and unavailable pre-release in any case.

**Missing, in three separate ways.**

*Acquisition.* Lyrics come from file tags only. Across the 64-track reference
corpus, 29 folders have a `lyrics` tag and 35 do not. Any lyric feature is
therefore absent on 55% of the corpus, and its presence correlates with how the
file was tagged rather than with anything about the song.

*Alignment.* There is no time-aligned lyric anywhere. Nothing can be located in
time — not the hook, not the title, not the first line.

*Semantics.* `lyric_stats()` in `mtx_enrich.py` (not in `mtx` itself — worth
noting that lyric analysis currently lives downstream) computes word count,
unique words, type-token ratio, lines, words per line, syllables per line and a
repeated-line percentage. That is shape. There is no meaning.

Worth having: sentiment and valence arc across the song, pronoun distribution,
concreteness and imagery, tense, rhyme scheme and density, internal vs end
rhyme, perfect vs slant, longest repeated n-gram, title-in-lyric count and the
timestamp of its first occurrence, compression ratio as a repetition measure,
readability, language and code-switching, explicit-content markers independent
of the `Explicit` flag the databases return.

The syllable counter is also an English heuristic and says so — on the Vietnamese
track in the corpus its output is meaningless. Whatever replaces it should
detect language first and decline rather than produce a number.

**Where.** Acquisition and alignment are two different things: a transcript from
the vocal stem (WhisperX / `whisper-timestamped`) gives *both* alignment and
coverage on the 35 untagged tracks, but it is a transcription, not the lyric,
and must be labelled as such — an inference with a confidence. Text statistics
belong in a new `metrics/lyrics.py` or, arguably, stay downstream where they are.

Note the payoff of alignment: word timings against the beat grid give syllables
per second as **delivery rate**, which is a rhythmic measurement, not a text one.

**Cost.** Medium; transcription is slow, and it is a network-free but heavy
optional extra.

### 8. Learned embeddings and trained taggers

**Offline.** Model weights are downloaded once; inference is local and the input
is the audio. Nothing here needs the song to exist publicly.

**Missing.** There is no embedding vector. 193 hand-engineered scalars are ideal
for interpretability and poor at capturing timbre similarity — two records can
agree on every column here and sound nothing alike.

One CLAP / MERT / MusicFM / Discogs-EffNet vector per track (and per section)
would give nearest-neighbour search across a corpus — "which five records sound
most like this one" — which is a useful capability on its own, independent of
any model anyone might train later.

In the same pass, Essentia's MTG mood and theme classifiers reproduce
danceability, energy, valence, acousticness, speechiness, instrumentalness and
liveness locally. Those are the features every published piece of work in this
area was built on, and Spotify has since withdrawn them from its API — computing
them locally removes a dependency on a service that already proved it can
disappear.

**Caution.** An embedding is opaque, and opacity is against the grain of this
tool. Store it as its own block with the model name and version in
`run.versions`, never mixed in with measured quantities, and never used to
derive a measured value. It is a fingerprint, not a measurement.

**Cost.** Medium; large optional dependency, and the first thing in this
document that puts a model's opinion into the output.

### 9. No corpus-relative layer

**Offline maths over an online-labelled cohort.** The percentile arithmetic is
local and needs no network. What it needs is a *cohort*, and cohorts are defined
by genre and year — which for the published reference records come from `enrich`
(online), and for the unreleased candidate must be declared. So an unreleased
mix can be positioned against a corpus of released records you own, which is
exactly the useful direction, provided you state what it should be compared to.

**Missing.** `-7.77 LUFS` and `tilt -4.79 dB/oct` mean nothing on their own. A
consumer has no way to ask where a value sits among comparable records.

Worth having: each metric as a percentile within a `(genre, release year)`
cohort, z-scores against the corpus and against the same artist's other tracks,
distance to a genre centroid, and a typicality/novelty score.

**Where — and this one matters.** *Not in `analyze`.* A per-track measurement
must not depend on what else is in the folder; that would break reproducibility,
which is property one. This is a corpus-level command — an extension of `batch`,
or a new `mtx cohort` that reads a directory of analyses and writes a separate
file of relative positions. Keep the absolute numbers untouched.

**Cost.** Low. No new DSP, no new dependency. Cheapest useful addition in the
document, and it makes every existing column more legible without adding one.

### 10. Rights, credit graph and version identity

**Online for anyone else's catalogue; declared for your own.** Splits,
publisher, PRO, sample clearances, the collaborator graph and sibling-version
counts exist in databases only after release. For a record you made, none of it
is unknown — it is unentered. This is the row that most clearly needs the
declared-metadata sidecar, and version identity is the piece worth doing first:
it is what lets two bounces of the same song be recognised as two mixes rather
than two songs.

**Missing.** `online/` does well on credits and identity — ISRC, MBIDs, ISWC when
MusicBrainz has it, roles reconciled across sources with `sources` arrays so a
two-source agreement is visible. What is still absent, and is intrinsic to the
song rather than a success metric:

Writer splits and shares, publisher and PRO, sample and interpolation status,
cover-vs-original-vs-remix, work↔recording links beyond the single work MBID,
writer/producer **team size** and the collaborator graph, featured-artist count
and role, recording date vs release date, and **version identity** — radio edit,
clean, sped-up, remix, and how many sibling versions exist. That last one is a
real hole: two rows for the same song currently have no way to say they are two
versions of it, so no comparison between mixes of the same song is expressible.

**Where.** `online/`, plus a `version` block in the schema that `analyze` can
populate from tags.

**Cost.** Low for version identity, high for splits and the credit graph — most
of that data is not in any free source.

### 11. Delivery-condition renderings

**Offline.** Encoding to AAC or Opus, band-passing, folding to mono and slicing
an excerpt are all local operations on a local file. This is the other gap with
real pre-release value: it answers what a master will do once it is distributed,
before it is.

**Missing.** `loudness.streaming_preview` covers the -14/-16 LUFS targets, which
is the right instinct, applied once. The other conditions a record actually
meets are not measured:

- **Lossy encode.** Encode to AAC 256 and Opus 128, re-measure, report the new
  true-peak overs and the HF damage. `mtx` already has the entire forensics
  apparatus for detecting codec damage; here it would be applied to the file's
  own future rather than its past. This is a deliverable a mastering engineer
  would use directly.
- **Small-speaker survival.** How much of the track survives a 400 Hz – 8 kHz
  band-pass, which is roughly what most listening hardware reproduces.
- **Mono fold-down at listening level**, beyond the existing broadband
  `mono_sum_damage`.
- **Short excerpt.** What the first 15 s and 30 s contain on their own, and what
  the chorus contains as a 15 s clip.

**Where.** `metrics/loudness.py` for the encode pass; the band-pass and excerpt
measures reuse existing machinery over a filtered or sliced signal.

**Cost.** Low to medium. The encode pass needs ffmpeg, which is already an
optional dependency with a stated fallback.

### 12. Export shape and corpus hygiene

**Offline.** Tooling over analyses already on disk.

Not a measurement gap, but it will bite before any of the above does.

- **No tidy tabular export.** `batch --csv` writes headline metrics per track.
  Everything richer reaches a human only as Notion properties, which is a
  browsing surface and not a dataset. What is needed is a parquet or CSV at
  **track level and at track×section level** — the per-section vectors are the
  most valuable part of the dump and are currently the hardest to get at.
- **No uniform missingness/confidence mask.** Confidences are computed and
  reported per metric group; there is no single vector saying which of N
  features are present and how far each is trusted. Any consumer has to
  rediscover that by walking the document.
- **The reference corpus is not a corpus.** 64 tracks, of which 21 are Billie
  Eilish. It is a fine test set for the DSP and a poor basis for anything
  statistical: with n=64 and 193 columns, any pattern found is a pattern about
  Billie Eilish. If it is ever used for corpus-relative work (§9), the cohorts
  need to be artist-stratified, and it needs to be much larger.

---

## If the list has to be ordered

By value per unit of work, given that stem separation is already paid for:

1. **§6 inter-stem masking** — pure DSP over signals already on disk, no new
   dependency, and it is the mixing-engineer metric nothing else provides.
2. **§2 melody from the vocal stem** — pyin adds nothing to the dependency tree;
   the expensive half already ran and is cached.
3. **§9 corpus-relative layer** — no DSP at all, and it makes all 193 existing
   columns interpretable.
4. **§4 form labels** (which delivers §3's downbeats in the same pass) — the
   biggest unlock in what can be asked, at the cost of a large dependency.
5. **§1 harmony** — the songwriter's half of the record, currently invisible.

§8 embeddings ranks last of the substantial items and should stay last: it is
the only one that puts an opaque model's opinion into the output, and it earns
its place only once a corpus is large enough for similarity search to return
something other than the same artist.

**If the priority is unreleased work rather than catalogue analysis**, the order
changes at the top: **§6 masking** and **§11 delivery-condition renderings**
first, then **§9** with declared cohort labels. Those three are what an
unfinished mix can actually use — the first two say something about the record
in front of you and are answerable while it is still changeable, and the third
tells you where it sits relative to the released records you are competing with.
Everything in §10 stays unavailable until you enter it yourself, and nothing in
`online.json` arrives at all.

---

## Deliberately not proposed

- **Any success, quality or "hit" score.** Not in `mtx`, at any point. If such a
  model is built, it reads `analysis.json` from outside, exactly as
  `mtx_enrich.py` does today.
- **Sales, streaming, chart, playlist or award data.** Outside the song, and
  time-varying; the tool would start being wrong the moment it finished writing.
- **Genre-conditional thresholds inside a metric.** Cohort context is §9's job,
  in a separate command, over a separate file.
- **Anything requiring a paid API, or a network call inside `analyze`.**
  `enrich` is and stays the only command that touches the network.
