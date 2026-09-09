# `tools/notion` — push a measured corpus into Notion

This is not part of `mtx`. It sits outside the package on purpose.

> **`mtx` measures. It does not interpret, score, grade or recommend.**

That rule is what makes the tool trustworthy for mastering work, and it does
not survive contact with a question like "what should I master this to". So
everything that compares, ranks or judges lives here, consuming `analysis.json`
from outside — the arrangement `GAPS.md` prescribes for exactly this.

Two thresholds *are* applied here, and both are recorded rather than assumed:
the trait rules in `schema.py` (stamped on every row as `TRAIT_VERSION`), and
the lyric shape test that works around a corpus written before the
`composerlyricist` bug was fixed. Neither is allowed anywhere near
`src/mtx/`.

---

## The order to run things in

Do not run these by hand. `tools/pipeline.py` runs them in this order, skips
what is already done, and refuses to push a corpus the audit failed:

```bash
python tools/pipeline.py                    # everything
python tools/pipeline.py --from enrich      # audio already measured
python tools/pipeline.py --only audit       # just check
```

The order is not arbitrary. Each stage needs what the one before it wrote:

| stage | reads | writes | needs |
| --- | --- | --- | --- |
| `mtx scan` | the FLACs | `analysis.json` | — |
| `tools/enrich_fast.py` | `analysis.json` | `online.json` | scan |
| `tools/transcribe.py` | `analysis.json`, the FLAC | amends `analysis.json` | scan |
| `tools/identity.py` | `online.json` | `artists.json` | enrich |
| `tools/notion/outcome.py` | `online.json` | `outcome.json` | enrich |
| `mtx cohort` | `analysis.json`, `online.json` | `cohort.json` | enrich |
| `tools/audit.py` | all of it | `audit.json` | — |
| `tools/notion/push.py` | all of it | Notion | everything above |

Skip `outcome.py` and the within-artist columns are simply empty — the loader
says so rather than inventing them. Skip `identity.py` and the `Artist` column
falls back to whatever the library folder happened to be named. Skip `cohort`
and every percentile column is empty, which is most of what makes the table
answer a question rather than report a number.

## The audit is a gate, not a report

```bash
python tools/audit.py <root>            # 4 seconds
python tools/audit.py <root> --notion   # and the live tables
python tools/audit.py <root> --deep     # + every analysis.json, minutes
```

It exits non-zero on an `error`, and `pipeline.py` stops there rather than
publishing. Every check exists because the defect it looks for was found in
data that had already been published and looked fine: a track credited to the
wrong artist at match score 1.00, a song dated from a bootleg compilation, 264
artist values for 55 artists, `best of 2016` filed as a genre. None of them
raised. The corpus is the evidence base, and evidence that is confidently
wrong is worse than evidence that is missing.

Re-run the pipeline with `--refresh` to take a fresh popularity snapshot.
That appends to the Observations log; it does not correct it.

## Run it

```bash
# See what would be sent, without a token and without touching the network.
python tools/notion/push.py "E:\Music\_mtx_out" --dry-run --limit 5

# For real. NOTION_TOKEN in the environment, or --token.
python tools/notion/push.py "E:\Music\_mtx_out" --parent <page_id>
```

`--parent` is the id of a Notion page your integration has been shared with.
Two databases are created under it and reused on every later run:

| Database | Shape | Grows by |
| --- | --- | --- |
| **Corpus** | one page per analysed folder | one row per track |
| **Corpus Observations** | append-only log | one row per figure per lookup |

The run is **idempotent** on tracks — matched by `file.sha256`, recorded in
`<root>/.notion_state.json` — and **additive** on observations. Interrupt it
and run it again; it resumes.

Useful flags: `--limit N`, `--force` (re-push known tracks), `--no-body`
(properties only, much faster), `--skip-observations`.

`--archive-db "Masters"` retires a superseded database once the push has
succeeded with zero failures — never before. An archived database is
recoverable from Notion's trash; a window where the old data is gone and the
new is not yet there is not worth risking.

---

## What goes where, and why

`analysis.json` holds ~4,261 fields; `mtx export` flattens ~2,091 of them.
Neither can be a Notion property list, because a database query returns every
property of every matched row — a 2,000-column database makes one query a
megabyte. But nothing is discarded for that reason. The data is ranked by
**retrieval cost**, not thrown away.

| Tier | What | Where | Cost |
| --- | --- | --- | --- |
| 1 | ~150 fields worth filtering, sorting or benchmarking on | Notion properties | one query returns 100 rows |
| 2 | the full flattened row, grouped | page body, code blocks | one extra request |
| 3 | section timeline, chord track, confidence notes | page body | same request as tier 2 |
| 4 | LTAS, F0 contour, beat times, correlation timeline | `analysis.json` on disk | a local file read |

The tier-1 test is one question: *would you ever compare this across tracks?*
"What should I master an EDM club track to" needs LUFS-I, true peak and PSR
from a whole cohort at once, so those are properties. Vocal reverb pre-delay
matters only after you have chosen a track, so it is not.

### Three things here that no flat export produces

`mtx export` drops JSON lists, and three of them carry the most useful numbers
in the document:

- **`delivery.encode.renderings[]`** — what the master does after AAC 256 and
  Opus 128. On Calvin Harris's *Blessings* the Opus decode lands 0.97 dB above
  the source true peak. This is the "will it survive distribution" data, and
  it is invisible to every query until lifted out by hand.
- **`stems.masking.per_section[]`** — vocal against instrumental, per section.
  The chorus figure is the mix number the corpus exists to compare.
- **`online.genres.ranked[]`** — every genre vote, not only the winner.
  "house" is often the third vote on a record a listener would call house, so
  storing just `primary` makes a genre query miss it.

---

## What Notion does to categorical values

Three constraints, all found the hard way, all worth knowing before you design
a column:

**Commas are rejected outright** — in `select` *and* `multi_select` option
names, tested in both. `Tyler, The Creator` cannot be stored as itself in a
facet; the loader substitutes a semicolon. The exact string lives in
`Artist canonical` (rich text has no such limit) and the real join key is
`Artist MBID`.

**Option names are case-insensitive, and the first casing wins permanently.**
Writing `Angine De Poitrine` to a column that already has the option
`Angine de Poitrine` silently reuses the existing one. Renaming the option by
id does not work either. The only route is to delete the option from the
database schema — which clears it from every page holding it — and then write
the value again so it is recreated with the casing you want.

The consequence is that **correcting the source data is not sufficient**. A
column whose first write was dirty stays dirty until the option is removed.

**Options are never garbage-collected.** An option survives in the dropdown
for ever once created, even after the last page stops using it. Re-pushing
1,321 clean rows left all 264 broken artist values sitting in the filter menu,
still looking like categories and still matching nothing. `--prune-options`
writes back only the options in use.

---

## Traits are tri-state, never boolean

`Traits` is a multi-select carrying `four-on-the-floor` when the trait is
measured true and `no-four-on-the-floor` when measured false. A trait that
could not be measured contributes **nothing**.

That distinction is the entire point. `rhythm.beat_position_profile.inference.
four_on_the_floor` is null on 63 of 72 electronic tracks tested — it populates
only where a kick pattern was detected at all. A checkbox would render those
63 as "no" and silently drop 87% of the corpus out of a club-music query, with
no error anywhere. Filtering for `four-on-the-floor` finds what was measured;
filtering for `no-four-on-the-floor` finds what was measured to be otherwise;
neither pretends the unmeasured majority is an answer.

Every threshold is printed in the POINTERS block of each page, alongside the
list of traits that could not be measured on that track.

---

## Popularity is an observation, not a property

There is no `Popularity` column, and that is deliberate.

Heat Waves took 59 weeks to reach number one. Cruel Summer was a 2019 album
cut that topped the chart in 2023. A single stored scalar cannot tell either
record apart from one that debuted at the top and fell away — today they read
the same. Nor can the difference be recovered later: Deezer rank and Last.fm
playcount are **current-value endpoints with no history**, so a figure not
captured this month is gone for good.

So time-varying figures go to **Corpus Observations**, one row per
`(track, metric, value, observed_at, source)`, never updated. Re-run
`mtx enrich --refresh` on a schedule and push again: new rows, old ones
untouched. What that buys:

- staleness is **visible** rather than silent — every value says when it was true
- velocity, time-to-peak and slow-burn-versus-fast-decay become computable
- the log accrues forward, which is the only direction it can

The Tracks database keeps `Latest …` caches of the newest observation next to a
`Last observed` date, so a query does not have to join for the common case. It
is a cache, and the date says how old it is.

Settled outcomes behave differently and *are* properties: `Billboard peak`,
`Weeks on chart`, `Certification`. Once a record's chart run is over those
stop moving, so they belong on the row. Fill them from `declared.json`.

---

## Files

| File | Does |
| --- | --- |
| `schema.py` | the ~150 tier-1 properties, the derived readers that reach into lists, and the trait rules |
| `rows.py` | one analysed folder → Notion properties + body blocks + observation rows |
| `client.py` | stdlib-only Notion client: pinned API version, ~2.8 req/s throttle, retries 429 and 5xx |
| `push.py` | CLI, database creation, resume state, `--archive-db` |
| `outcome.py` | within-artist playcount z, terciles, single-vs-album cut |
| `../enrich_fast.py` | parallel `mtx enrich` over a corpus, ~3.2x faster |
| `../pipeline.py` | every stage, in order, gated by the audit |
| `../audit.py` | 25 checks over the corpus and the live tables; exits non-zero |
| `../identity.py` | one canonical artist name and MBID per library folder |
| `../transcribe.py` | add a lyric to analyses that already exist |
| `../vocab.py` | the genre vocabulary, with the size of each cohort |
| `../declare.py` | the `declared.json` an unreleased mix needs to join a cohort |

No third-party dependency. `mtx`'s `online/` subpackage is stdlib-only for the
same reason, and a loader that needs a dependency tree to move JSON would be a
poor advertisement for a tool whose point is reproducible local measurement.

---

## Before the first real run

1. **Enrich the corpus.** Without it there is no genre, no release date, no
   ISRC and no popularity — roughly a third of the tier-1 properties stay
   empty, and cohorts are not expressible at all. Use `enrich_fast.py`; it
   calls the same `enrich()` and writes the same `online.json`, it just stops
   waiting on one host at a time.
2. **Set `LASTFM_API_KEY`.** It is free, and it is the only provider that
   returns playcount. Without it the Observations log gets Deezer rank only.
3. **Transcribe.** Analyses written before the `lyrics.py` fix carry a
   songwriter credit where the lyric should be, and `audit.py --deep` counts
   them. The loader's shape test hides the damage in Notion; clearing it on
   disk means `tools/transcribe.py`, which replaces the block in place for
   thirty seconds a track rather than the eleven minutes a re-scan costs.
4. **Dry-run first.** `--dry-run --limit 5`, then read a payload. It costs
   nothing and it is the cheapest way to catch a schema mistake before 1,321
   pages carry it.
