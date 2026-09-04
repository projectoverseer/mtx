# Closing the gaps

`GAPS.md` lists what `mtx` does not measure. This file is the other half: what
is missing from **your corpus**, what each missing thing would buy, what it
costs to get, and which of them need something only you can supply.

The goal it is all pointed at: decide what to do next on a song, from what
released records actually did, not from what a mixing book says they should
have done.

---

## The checking step

There wasn't one. There is now.

```
python tools/audit.py "E:/Music/_mtx_out"            # 4 seconds
python tools/audit.py "E:/Music/_mtx_out" --notion   # and the live tables
python tools/audit.py "E:/Music/_mtx_out" --deep     # + every analysis.json
```

Twenty-eight named checks. Each carries a severity, a count, the offending
rows, and what to do about it. It exits non-zero on an `error`, which is how
`tools/pipeline.py` gates the push: a wrong row is worse than a missing one,
because the missing one gets noticed.

`--deep` is worth running weekly, and worth distrusting until it has run
once: it read `lyrics.statistics.lines` as `{"count": n}` when it is a bare
`int`, so every `--deep` run died on the first track from the day it was
written until 2026-09-04. No test caught it, because every fixture wrote an
analysis with no lyrics in it and the deep checks short-circuited before
reaching the line that crashed. A crash in a mode nobody runs looks exactly
like a mode nobody runs. Its first completed run found 67 tracks whose
"lyric" is a songwriter credit.

It writes nothing. An audit that repairs things cannot be trusted to report
honestly on the next run.

**Why it matters more than it sounds.** Every defect it now checks for was
found in data that had already been published and looked fine. None of them
raised an exception. None scored badly. The pipeline reported success on all
of them:

| What was wrong | How it looked |
| --- | --- |
| An Olivia Dean track credited to an unrelated artist called `OLIVIA` | match score **1.00** |
| `Scar Tissue` dated from a German bootleg compilation | a release date, correctly typed |
| 35% of tracks choosing a release from an arbitrary quarter of the candidates | a release |
| Every cohort in the corpus built from the shop's own genre tag | cohorts |
| 63 of 80 sampled lyrics being songwriter credits | `source: "file:tag"` |
| `best of 2016` filed as a genre | a genre |
| 264 artist values for 55 artists | a categorical column |
| 78 tracks killed by an out-of-memory mid-decode | an analysis byte-identical to one nobody asked to transcribe |
| 95 tracks re-transcribing an identical transcript every run, forever | a fresh `ok` in the log, every time |

That is the failure mode that matters here. The corpus is the evidence base.
Evidence that is confidently wrong is worse than evidence that is missing.

**And a check can be wrong in the same way.** The obvious next check after
finding a hallucinated, endlessly-repeating transcript is a repetition
threshold. Measured against the corpus first, it turns out the most
repetitive transcript here -- Daft Punk's *Around the World*, a distinct-word
ratio of 0.017 -- is completely correct; the song repeats one phrase 144
times. That check would have flagged the most accurate transcriptions in the
corpus, and flagged them for being what a hit chorus is. What separates
cleanly is words per line: a median of 7.9, p99 of 16.0, and a tail to 86.5
where whisper returned four segments for a whole song. `lyrics.line_structure`
measures that instead, at `info`, because the words are still right and only
the per-line figures are not.

---

## The daily workflow

```
python tools/pipeline.py
```

That is the whole thing. It runs, in the order their dependencies require:

```
scan      measure the audio                      no network, hours
enrich    look every track up                    network, minutes
identity  one canonical name + MBID per artist   needs enrich
outcome   normalise plays within each artist     needs enrich + identity
cohort    percentiles within genre and era       needs enrich
audit     stop if the corpus is wrong            gate
push      Notion                                 gated by audit
```

Every stage skips work already done, so a day with 20 new tracks costs
whatever those 20 tracks cost and nothing more. Keys come from
`E:\Music\_mtx_out\mtx.env`, which lives with the music rather than with the
code, so it never lands in git. It is already written.

Useful variants:

```
python tools/pipeline.py --from enrich      # audio already measured
python tools/pipeline.py --only audit       # just check
python tools/pipeline.py --dry-run          # print the commands
python tools/pipeline.py --transcribe       # see "Lyrics" below
```

### Watching one while it runs

The long stages print one line per track, which is a fine record and a poor
progress report: nothing in the scroll says how far along it is, how fast it
is going, or -- the one that actually matters overnight -- whether it is still
moving at all. A stalled job and a slow job look identical in a `tail`.

```powershell
.\tools\watch.ps1                 # live, refreshes every 10s
.\tools\watch.ps1 -Once           # one snapshot, then exit
.\tools\watch.ps1 -Failures       # what did not work, instead of what just did
.\tools\watch.ps1 -Log $env:TEMP\scan.log -Interval 30
```

It reads the log and nothing else, so Ctrl-C stops the watching and never the
job, and running six of them costs nothing. `-Once` exits 0 when it found a
job to report on and 1 when there was no log, so it works in a script as well
as on screen.

Redirect a long run's output somewhere it can find:

```powershell
python tools/transcribe.py "E:/Music/_mtx_out" 2>&1 | Tee-Object $env:TEMP\transcribe.log
```

---

## The gaps

### 1. `mtx cohort` — percentiles. **Closed, and it has now run.**

1,321 tracks, 55 artists, **779 cohorts**, in 2m27s.

**What it is.** A corpus-level command that reads every analysis and writes a
separate `cohort.json` of *relative* positions. It never touches the per-track
measurements, on purpose: `mtx analyze` promises the same output for the same
input, and a number that changed because a neighbour arrived would break that.

**What it buys.** This is the difference between a measurement and an answer.

> `-9.45 LUFS, PSR min 4.7 dB, tilt -5.2 dB/oct`

is a fact about your file that tells you nothing. What you wanted was:

> `-9.45 LUFS is the 31st percentile of house records released since 2022
> (n=63, median -8.6). Your PSR min of 4.7 dB is the 8th percentile — the
> cohort's median is 6.9, and the five records closest to yours on every
> other axis are …`

Every metric now carries `cohort_percentile`, `cohort_median`, `cohort_z`,
plus the same against the whole corpus and against that artist's own
catalogue. In Notion: `Cohort`, `Cohort size`, `Cohort is fallback`,
`LUFS-I pct`, `PSR min pct`, `Tilt pct`, and eighteen more.

**Two things I had to fix before it could run at all.**

`labels_for` read `online["genre"]` and `online["release"]`. Enrichment writes
`online["genres"]` and nests the release under the provider that said it.
Neither key exists in any `online.json` in your corpus, so the genre and the
year both fell through to the file's own tag — on all 1,321 tracks. The tag is
the exact string the genre vote exists to replace. Both sides had passing
tests, each written against its own invented fixture. There is now a contract
test that runs `enrich` and feeds its real output to `labels_for`.

A record now belongs to **every** genre that cleared a confidence floor in the
vote, not only the winner. Filed under its winner alone, a club record sits in
`electronic` (116 tracks) and never in `house` (199) — and `house` is the
cohort somebody mixing a club record is asking about.

It also got about twenty times faster on the way. `load_analysis` was reading
every part file — 17 MB a track, 18 GB for the corpus — to find 36 scalars.
It now takes the sections the caller names and leaves the two big timeline
arrays on disk, which is the difference between a 45-minute daily step and a
two-minute one.

Worked example, which is your original question answered from the corpus:

| cohort | n | LUFS-I median | true peak | PSR min | PLR |
| --- | --- | --- | --- | --- | --- |
| `house` | 78 | **−8.91** (p10 −11.29, p90 −7.13) | +0.75 dBTP | 6.71 dB | 9.49 dB |
| `dance-pop` | 113 | −7.91 | +0.95 dBTP | 6.09 dB | 8.88 dB |
| `electronic` | 172 | −9.05 | +0.39 dBTP | 6.60 dB | 9.71 dB |

Sliced by era as well: `house | 2021-2025` holds 50 records. And every track
carries its five nearest neighbours as a precomputed A/B list.

**Cost:** ~2.5 minutes per run, no network, no keys. It is in the pipeline.

**You do:** nothing.

---

### 2. `--embed` — "which released records sound like mine". *Optional, recommended second.*

**What it is.** One learned vector per track (and per section) from an audio
model — CLAP, OpenL3 or MERT. It is stored in its own block, never mixed with
measurements, and no measured value is ever derived from it.

**What it buys.** Nearest-neighbour search that hears timbre. The 176 numbers
in the Notion table are excellent for "is my low end wide" and poor at "does
this sound like the records it is competing with" — two tracks can agree on
every column and sound nothing alike. The `A/B references` column already
works without it, using distance in z-space over the measured metrics; with
embeddings it gets substantially better, because it starts comparing *sound*
rather than *statistics about sound*.

**What it does not buy.** Anything explainable. A vector cannot tell you why.
It hands you five records; the measured columns tell you how you differ from
them.

**Cost:** a one-off pass over 1,321 tracks. CLAP on your GTX 1650 is roughly
5–15 s/track → **2–5 hours**, once, then ~4 minutes a day for new tracks.
Install is `pip install laion-clap`.

**You do:** say go, and let a long job run overnight.

---

### 3. `--transcribe` — lyrics. *The largest empty column, and the one your goal needs most.*

**What it is.** Whisper over the vocal (or the mix), producing a time-aligned
transcript that the existing lyric battery then measures.

**What it buys.** Right now, everything about the *writing* is unmeasurable.
The corpus can tell you how a hit is mixed and mastered and arranged; it can
tell you nothing about how a hit is written. Transcription turns on: rhyme
scheme and density, perfect-versus-slant split, repeated-line percentage,
longest repeated n-gram (the hook), whether and when the title appears,
pronoun balance (`I` / `you` / `we` — one of the more reliable correlates of
pop reach), readability, explicit terms, and — via word timings against the
beat grid — **syllables per second, which is a delivery rate you can compare
across a cohort.**

For "judge my song against the corpus and tell me what is handicapping it",
this is the single biggest addition available.

**The state of it.** Three things blocked it; all three are now fixed:

- It refused to run without a vocal stem, and your stems were pruned after
  scanning. It now falls back to the full mix, labelled `input: "full mix"`
  with a caveat, because a mix transcript mishears more and that has to be
  visible to whoever reads it.
- The device was hardcoded to CPU int8 while your CUDA card sat idle. It now
  picks the GPU and falls back rather than failing.
- The model was hardcoded to `base`; it is now `params.lyrics.transcript.model`,
  defaulting to `small` — the smallest that reliably hears a sung lyric over a
  mix.

**It works now, and here are the real numbers.** Whisper `small` on your
GTX 1650: **30 s for a 239 s track**, about 8x realtime. So **≈13 hours** for
the 1,321-track backfill, resumable at any point, and ~10 minutes a day after
that. On `Heat Waves` it produced 366 words over 51 lines — 37.3% of them
repeats, 9.3 syllables a line, rhyme density 1.35 with 20 perfect and 49
slant, delivery 2.76 syllables/second, first word at 0:03.1.

Two things had to be worked around, both recorded in `.models/README.md`:

- The HuggingFace download from this machine resets constantly — `WinError
  10054`, through `huggingface_hub` *and* `curl`, while the HF API itself
  answers `200`. The 483 MB `model.bin` came down first try; the small JSON
  files each took several attempts. The weights now live in
  `.models/faster-whisper-small`, pointed at by `MTX_WHISPER_MODEL` in
  `mtx.env`. If you ever need to re-fetch, try
  `HF_ENDPOINT=https://hf-mirror.com`.
- Backfilling via `mtx scan --force` would re-run demucs and cost **240
  hours**. `tools/transcribe.py` patches the `lyrics` block into an existing
  `analysis.json` instead, recording the amendment under `run.amendments`, so
  the job is the 13 hours transcription actually costs.

**There is also a live defect here.** 63 of 80 sampled tracks have a "lyric"
that is one line of 15–141 characters, labelled `source: "file:tag"`. They are
songwriter credits: a substring match read `composerlyricist` as a lyric, and
Apple writes that key on most commercial files. The matcher was fixed earlier;
these analyses predate the fix and still hold the credit, presented as the
song's words with a source attached. `audit.py --deep` now reports them, and
the Notion schema suppresses them with a shape test, but the fix on disk is
either a re-scan or a transcript.

**You do:** nothing to set it up — it is running. Stop it any time with
Ctrl-C; it is resumable and skips what it has done. To publish the lyric
columns once it finishes: `python tools/pipeline.py --from identity`.

---

### 4. `declared.json` — the sidecar for your own unreleased work. *You must write these.*

**What it is.** A small JSON file beside a track holding the facts **you** are
the source of. Write one with:

```
python tools/declare.py "E:/Music/_mtx_out/Me/Demo/01. New Song"
python tools/declare.py "E:/Music/_mtx_out" --gaps    # who needs one
```

**Why it exists, and why it is not optional for your own songs.** `mtx enrich`
answers "what is this record" by asking databases. That only works for records
that have been *released*: a distributor issued the ISRC, an editor typed the
credits in, listeners tagged it. Your unreleased mix has none of that and
cannot be made to have it. So for your own work the facts are not missing —
you are simply the source of them rather than a database.

Concretely: without a sidecar your new master reaches `mtx cohort` with no
genre and no year, so it joins **no cohort**, so there is nothing to compare
it against — which is exactly the question you ran it to answer. Two fields do
almost all the work:

```json
{"cohort": {"genre": "house", "year": 2026}}
```

That is the difference between "your PSR min is 4.7 dB" and "your PSR min is
4.7 dB, the 8th percentile of house since 2024". Use `python tools/vocab.py
<root>` to pick a genre with enough records behind it to mean something —
`house` has 199, `future house` would have four.

The other fields (`version`, `work_key`, `sibling_versions`) are what let two
bounces of one song be recognised as two mixes rather than two songs, so
`mtx compare` and the corpus can talk about your v3 versus your v7.

A declared value never overwrites a measurement. It travels with
`source: "declared"` beside whatever the analysis found, so the two can
disagree in the open.

**Good news:** for the 1,321 *released* tracks, `--gaps` reports **zero**
tracks that need one. Every one of them has a voted genre and a resolved year.
The sidecar is purely for your own material.

**You do:** run `tools/declare.py` on each of your own mixes and fill in
`cohort.genre` and `cohort.year`. Two minutes each.

---

### 5. Chart outcomes — Billboard peak, weeks, certifications. *Only you can supply these.*

**What it is.** `Billboard peak`, `Weeks on chart` and `Certification` are
already Notion columns, read from `declared.outcome.*`. Nothing fills them.

**What it buys.** A ground truth that is *settled*. Play counts drift — Heat
Waves took 59 weeks to reach #1 — so a playcount observed today is a reading,
not a property, which is why they live in an append-only Observations log.
A chart peak from 2021 is finished and will never change. It is the only
outcome variable in the system that is safe to regress against.

**The good news:** you may not need it. The corpus already contains a natural
experiment that needs no chart data at all — **172 singles against 1,140 album
cuts by the same artists**. Same artist, same era, same producers, same
mastering chain, very different outcomes. Every confound that normally wrecks
this comparison is held constant by construction, and what varies is the song
and the mix. That comparison got materially more trustworthy today, because
`Is single` now asks the whole packaging history rather than the type of
whichever release happened to be picked.

**You do:** optionally, a CSV of `sha256,billboard_peak,weeks_on_chart,cert`
for the tracks you care about. Low priority.

---

### 6. Discogs, Last.fm, and the rest of the online layer. *Mostly closed.*

| | Before today | Now |
| --- | --- | --- |
| Discogs matched | 0 / 1321 | **896**, remaining 425 mostly digital-only |
| Missing playcounts | 91 | **2** |
| Release chosen from a truncated list | 465 | **0** |
| Release chosen from a bootleg | 24 | **0** |
| Dated to the day | 956 | **1,318** |
| Dated only to the year | 365 | **0** |
| Known singles (the contrast set) | 172 | **381** vs 935 album cuts |

The 91 missing play counts were one bug with three faces.
`primary_artist("Ariana Grande (ft. Pharell Willians)")` returned the whole
string — there is no space before the `ft.`, so a word-boundary split left it
whole and Last.fm was asked about an artist who does not exist.
`primary_artist("Tyler, The Creator")` returned `"Tyler"` — who *does* exist,
and came back with a play count 7,000× too small. And Last.fm's autocorrect
answers almost any string with *something*, which arrives looking exactly like
a measurement: `blazed` returned a track with **2 plays**.

Fixed: bracketed feature credits are stripped, a comma before a determiner is
not a separator, the name MusicBrainz resolved is tried first, and a row
credited to somebody else is refused. `blazed` now reads 142,792.
`EARFQUAKE` went from 5,475 to 40,209,881.

**You do:** nothing. Keys are in `mtx.env`.

---

### 7. Six-stem separation. *Low priority.*

`htdemucs_6s` adds guitar and piano to the existing four stems. It buys
instrument-level masking (is the guitar eating the vocal, or is it the synth)
at the cost of re-running separation over the whole corpus — the expensive
half of a scan, and the reason a full scan takes ~11 minutes a track.

**You do:** nothing, unless you specifically want per-instrument masking.

---

## What I need from you, in priority order

1. **A green light on transcription**, plus a working HuggingFace route (VPN,
   or `HF_ENDPOINT=https://hf-mirror.com` in `mtx.env`). Biggest single
   unlock: it is the only thing that makes the *songwriting* measurable.
2. **`declared.json` for each of your own mixes** — two fields, two minutes,
   and without it your own track can be compared to nothing.
3. **A green light on embeddings** (one overnight job).
4. **Keep buying breadth.** Your largest single artist is 10.4% of the corpus,
   which is healthy. The thin spots are cohorts: `house` 199, `country` 41,
   `latin` 35, `jazz` 2. If you intend to write in a genre, buy 40 records in
   it before trusting a percentile from it.
5. **Optional:** chart peaks as a CSV.
6. **Rotate the Last.fm, Discogs and Notion tokens** at some point — they have
   been pasted into a chat window.

---

## Standing rule

`mtx` measures. It does not interpret, score, grade or recommend. Everything
in `tools/` is the layer that interprets, and it stays outside `src/mtx/` so
the measurement half remains trustworthy for the mastering work it is already
good at. A cohort percentile is a position, not a verdict: sitting at the
cohort median is neither good nor bad, and the corpus can only ever tell you
what released records did — never what yours should do.
