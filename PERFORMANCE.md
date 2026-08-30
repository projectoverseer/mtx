# What limits `mtx scan`, measured

Written after instrumenting a real library run on one laptop rather than
reasoning about the code. Every number below was measured on the machine
described in *The bench*; nothing here is extrapolated from a datasheet, and
where a figure is derived rather than observed it says so.

The short version, for anyone who only wants the conclusion:

> **The DSP pool saturates at or below the physical core count, and adding
> workers past that point buys exactly nothing.** Separation is the cheaper
> half and it is not the bottleneck. Anyone tuning a scan should stop
> adjusting `--jobs` and start looking at the phase structure.

---

## The bench

| | |
|---|---|
| CPU | Intel i7-9750H, **6 physical / 12 logical**, base 2.6 GHz, 45 W mobile |
| RAM | 31.7 GB |
| GPU | NVIDIA GTX 1650 Mobile, **4095 MiB**, driver 591.74, 30 W enforced limit |
| OS | Windows 10 Enterprise 19045, power plan Balanced |
| Library | FLAC, mixed bitrate, `E:\Music` |
| Corpus | 59 Ed Sheeran masters, 13 253 s of audio, mean 225 s a track |

Two environment facts that matter for reproducing any of this:

- The stem cache was moved off `C:` with `MTX_STEMS_CACHE`. At ~165 MB a track
  a library scan will fill a system disk, and the failure arrives halfway
  through a long run.
- The machine is **muxless Optimus** — the panel is wired to the Intel UHD 630
  and the GTX 1650 drives no display. This is why NVIDIA Control Panel reports
  "you are not currently using a display attached to an NVIDIA GPU". It has no
  bearing on CUDA, which works normally.

---

## The chip is not throttled, and one Windows counter will tell you it is

`\Processor Information(_Total)\% Processor Performance` reads a flat **30 %**
on this machine under every load, which works out to 777 MHz against a 2592 MHz
base and looks exactly like a severely power-limited laptop.

It is wrong. Two independent readings contradict it:

- `\Processor Information(_Total)\Processor Frequency` reads **2592**.
- Single-threaded FP64 matmul measures **39.0 GFLOPS**, which is squarely in
  the expected 35–50 band for this part at turbo with AVX2 FMA.

A chip actually running at 777 MHz could not produce 39 GFLOPS. **Do not trust
`% Processor Performance` on this platform**; benchmark the arithmetic instead.

Balanced is also not a problem here: on AC it carries `PROCTHROTTLEMAX = 100 %`
with active cooling, so it already runs the part flat out under sustained load.
Switching to High Performance changes nothing that matters for a batch job.

---

## Finding 1 — the DSP pool is saturated at six workers

Twelve identical tracks, stems pre-cached so nothing but DSP is timed, one
thread per worker, `ProcessPoolExecutor` exactly as `scan.py` builds it.

| workers | wall | throughput (audio-s / wall-s) | CPU s per track |
|---|---|---|---|
| 6 | 886.6 s | 2.754 | 420.2 |
| 9 | 891.6 s | 2.739 | 547.9 |
| 12 | 886.0 s | **2.756** | 823.8 |

**Throughput is flat across a 2× change in worker count** — 0.6 % spread, which
is noise. Doubling from 6 to 12 produced no additional work per unit time.

The third column is the mechanism. Per-track CPU time rises 420 → 548 → 824 s,
very nearly linearly with `k`, while wall clock does not move. Every worker
added past six takes its share of a fixed throughput by making the others
proportionally slower. That is the signature of a bandwidth-saturated pool, not
a core-starved one.

This confirms, and slightly overstates, the reasoning already written in
`parallel.physical_cores()`:

> two hyperthreads on one core share the load/store units and the cache these
> passes are bound by, so the second thread buys perhaps a fifth of a core, not
> a whole one.

Measured on this machine the second hyperthread buys **nothing at all** — not a
fifth of a core, zero. The docstring's estimate is conservative in the right
direction, and the sizing decision it justifies is correct.

Three consequences worth writing down:

- **`--jobs` above the physical core count is never worth setting.** It cannot
  raise throughput and it inflates per-file service time, which makes the ETA
  noisier and a `Ctrl-C` more expensive.
- **More RAM cannot help.** 31.7 GB was never near exhausted; the limit is
  bandwidth *to* memory, not the amount of it.
- **Oversubscription — two or three processes per core — is strictly worse.**
  `k=12` already *is* two per core on this part and it is worth 0 %. Three
  would add context switching and cache pressure for the same flat line. The
  k=18 and k=24 points were cancelled rather than spend an hour confirming a
  flat line stays flat.

### A caveat on the knee, and a methodology warning

A follow-up sweep at `k = 5, 4, 3` was run on a **different, six-track subset**
of the same library. Its first point reads:

| workers | wall | throughput | CPU s per track |
|---|---|---|---|
| 5 | 539.0 s | 2.204 | 325.2 |

That 2.204 is **not** directly comparable to the 2.754 at `k=6` above, because
the corpus changed at the same time as the worker count. Per-track DSP cost
varies with content, so throughput normalised by audio-seconds still moves when
the tracks move.

**This was a design error, recorded so the next person does not repeat it: vary
one thing at a time.** The `6 / 9 / 12` comparison is clean — identical tracks,
identical work. Any future scaling point must reuse that same fixed list.

---

## Finding 2 — one separation leaves the card two thirds idle

`separate_first()` originally ran separations strictly one at a time, justified
in its docstring as a memory constraint:

> several separations on one consumer GPU is an out-of-memory error rather than
> several times the speed

The first half is true at some concurrency; the second is not true at two or
three. Fifteen fresh (uncached) tracks per point, `nvidia-smi` sampled at 1 Hz
throughout:

| streams | wall | throughput | mean GPU util | peak VRAM | speedup |
|---|---|---|---|---|---|
| 1 | 397.1 s | 7.24 | 68 % | 875 MiB | 1.00× |
| 2 | 370.8 s | 9.86 | 87 % | 1750 MiB | **1.36×** |
| 3 | 328.2 s | 10.90 | **94 %** | 2625 MiB | **1.51×** |

A single demucs stream leaves the SM **68 %** busy. The missing third is spent
decoding the input and writing four uncompressed WAVs with the device idle —
sampling during a run shows utilisation cycling 0 % → 76 % → 0 %. Overlapping
streams fills those gaps, and by three streams utilisation is 94 % and there is
very little left to fill.

**VRAM scales exactly linearly at 875 MiB a stream.** Three fit a 4 GiB card
with 1.4 GiB spare; four would want ~3500 MiB of 4095 and sit close enough to
the ceiling that an OOM — which costs a step down the segment ladder *and* the
separation that hit it — becomes likely. Hence `MAX_STREAMS = 4` with a
`VRAM_RESERVE_MIB = 1100` reserve, which yields exactly 3 on this card.

This is implemented: `separation_streams()` in `metrics/stems.py` derives the
count from `torch.cuda.get_device_properties`, `--stems-jobs` overrides it, and
`separate_first()` runs them through a `ThreadPoolExecutor`.

---

## Finding 3 — the phases are sequential, and that is now the ceiling

This is the important one, and it reorders every other optimisation.

`scan.py` runs separation to completion and only then starts the DSP pool. On
this corpus, per 59 tracks:

| phase | wall | during which |
|---|---|---|
| separation, 1 stream | ~30.5 min | 6 cores mostly idle |
| separation, 3 streams | ~20.3 min | 6 cores mostly idle |
| **DSP at `-j 6`** | **~76 min** | GPU completely idle |

*(Separation figures derived from the measured 7.24 and 10.90 audio-s/wall-s
over the corpus's 13 253 audio-seconds. The DSP figure is the observed wall
clock of the real 59-track scan, which reported `2.9 s of audio per second of
wall clock` — a good cross-check on the 2.754 measured in isolation.)*

**DSP costs 2.5× what separation costs.** That single ratio demotes most of the
obvious GPU work:

- Parallel separation is worth **1.51× on 22 % of the run** — real, but it
  cannot touch the other 78 %.
- The ~6 s of torch import and model load re-paid per file lives entirely
  inside the phase that is *not* the bottleneck.
- Once the phases are overlapped, separation hides behind DSP completely and
  further GPU work is worth **nothing**.

### The remaining win: overlap the phases

Nothing forces separation to finish before measurement starts. A track's stems
only have to exist before *that track* is measured. Feeding the process pool
from a separation thread as each track's stems land would let the idle cores
during separation do DSP.

Estimated, not measured — the assumption is stated so it can be checked:

- Total CPU work per 59 tracks ≈ 456 core-min of DSP (76 min × 6) plus roughly
  60 core-min of separation-side decode and WAV writing (~2 cores over 30 min,
  from sampled CPU utilisation during the separation phase).
- At 6 cores that is ~86 min against the 106 min the sequential design spends:
  a **~19 % saving**, bounded by total CPU work rather than by either phase.

The estimate is soft because the pool is already bandwidth-saturated, so
demucs's CPU-side work will not compose perfectly additively with DSP. Treat
~19 % as an optimistic bound and measure before believing it.

**This has not been implemented.** It restructures the result-reporting loop in
`run_scan`, which carries careful service-time ETA logic (see the `eta_seconds`
docstring), and landing it without a measurement harness in place risks
breaking the ledger and ETA behaviour for a gain that is real but not dramatic.
It is the correct next piece of work and it should be done deliberately.

---

## What this means at library scale

Extrapolating the measured 106 min per 59 tracks to 100 000 masters, at this
corpus's mean of 225 s a track:

| configuration | est. continuous runtime |
|---|---|
| sequential phases, serial separation (before this work) | ~125 days |
| sequential phases, 3 separation streams (**current**) | ~113 days |
| overlapped phases (not implemented) | ~89–98 days |

Storage matters as much as time at that scale, and neither number is small:

- **Output** ~17 MB a track → ~1.7 TB per 100 000.
- **Stem cache** ~165 MB a track → **~15 TB** if nothing is ever evicted.

The stem cache is never pruned after a track is measured. At library scale it
is not a cache, it is an unbounded write. `separate_first()` also separates the
*entire* todo list before a single measurement is taken, so a cold 100 000-track
run would try to materialise all 15 TB up front. **Bounding and evicting the
stem cache is a prerequisite for any run of this size**, and is a larger
problem than any throughput tuning in this document.

---

## Reproducing this

The scaling sweep, GPU concurrency test and clock benchmark were written as
throwaway scripts, deliberately not committed — they hardcode paths into one
person's library and would rot. What matters is the method:

1. **Pre-warm the stem cache** for a fixed track list before any DSP timing, so
   the sweep measures the pool and not demucs.
2. **Hold the track list fixed across every `k`.** Vary one thing at a time.
   (See the caveat under Finding 1 for what happens when you do not.)
3. **Report throughput as audio-seconds per wall-second**, not seconds per
   track — it is the only figure comparable across corpora of different length.
4. **Report per-track CPU time alongside it.** Flat throughput with rising CPU
   time is the saturation signature; without the second column a flat line is
   ambiguous between "saturated" and "measurement broken".
5. **Sample `nvidia-smi` at ~1 Hz for the whole point**, never spot-check. A
   handful of samples during separation suggested ~35 % mean utilisation; the
   real figure over 15 tracks is 68 %. Spot checks land in the gaps.
6. **Do not let the benchmark discard real work.** The GPU concurrency test was
   pointed at uncached tracks the library actually needed, so its separations
   were banked to the cache rather than thrown away.

### Two traps on Windows specifically

- `% Processor Performance` lies (see above). Benchmark arithmetic.
- PowerShell parses `[` and `]` in a `-Path` as a **wildcard character class**,
  so a real folder named `Play (Deluxe) [E]` matches nothing and the cmdlet
  returns **zero rows with no error** — which looks exactly like data loss.
  Use `-LiteralPath`, or walk the tree from a shell that does not do this.

---

## Changes this document produced

| change | file | status |
|---|---|---|
| `MTX_STEMS_CACHE` to relocate the stem cache off the system disk | `metrics/stems.py` | landed |
| `_mtx_out` / `_mtx_stems` added to `SKIP_DIRS` so a scan cannot measure its own separated stems as masters | `scan.py` | landed |
| `separation_streams()`, VRAM-derived, `--stems-jobs` to override | `metrics/stems.py`, `cli.py` | landed |
| `separate_first()` runs several streams through a thread pool | `scan.py` | landed |
| Overlap the separation and DSP phases | `scan.py` | **open**, ~19 % est. |
| Bound and evict the stem cache | `metrics/stems.py` | **open**, blocks library-scale runs |
| Keep one demucs process alive instead of re-importing torch per file | `metrics/stems.py` | **open**, ~6 s/track inside the non-bottleneck phase |
