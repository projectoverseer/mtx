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

| component | |
| --- | --- |
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
| --- | --- | --- | --- |
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
| --- | --- | --- | --- |
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
| --- | --- | --- | --- | --- | --- |
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
| --- | --- | --- |
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

**This has now been implemented**, and the estimate above was wrong in an
instructive way. `scan.drive` runs the separation threads and the measuring
pool at the same time; a track is submitted the moment *its own* stems exist.
The card and the cores do overlap, and the separation phase does disappear.

What the estimate missed is that the two halves compete for something other
than cores. A demucs process holds **2.5 GB of host memory** while it works,
and the byte budget below cannot see it, so every concurrent separation costs
about half a measuring lane. Overlapping the phases is not free; it is paid
for in RAM, on a machine where RAM is what runs out first.

That is also why the pipelined default is **one separation stream**, not the
three that Finding 2 recommends. The 1.51x from overlapping three streams
belonged entirely to the old design, where separation was a phase the cores
sat through. Once it hides under the measuring, a second stream buys nothing
the pipeline needs and costs memory the measuring wants — which is exactly
what Finding 2 predicted would happen ("once the phases are overlapped,
further GPU work is worth nothing"), arriving sooner than expected.

---

## Finding 4a — measure the machine before you believe anything about the code

Read this before Finding 4, which it partly invalidates.

Every memory number in this document was collected on a machine that had
`AMPLibraryAgent.exe` — the Apple Music library agent, part of the Apple
Music app for Windows — running in the background:

| | when the scan was diagnosed |
| --- | --- |
| commit charged by that one process | **45.4 GB** |
| CPU it had burned since 20:13 the previous evening | **27.3 hours** |
| wall clock over that period | 11.8 hours |
| cores it therefore held, continuously | **2.3 of 6** |
| commit free for everything else | **0.8 GB** |
| commit free one second after stopping it | **47.1 GB** |

It had started at 20:13, minutes before the overnight scan, and it was still
growing at 07:00. It indexes music files, and a library scan reading 1 274 of
them is exactly what wakes it.

So the overnight run was not measuring what it looked like it was measuring. It
was sharing the machine with a process taking a third of the cores and all but
0.8 GB of the commit limit, and the `MemoryError`s were raised against *that*,
not against six workers' own footprint. Allocations of **54 MiB** were failing.

**What this invalidates.** Finding 4 attributes Coldplay's collapse to 192 kHz
masters. The two are confounded: Coldplay ran last, at about 06:00, by which
point the agent's commit had grown to its maximum. The 192 kHz footprint is
real and measured, and it is larger, but the claim that it *alone* caused the
collapse is not supported and should not be repeated.

**What survives.** The per-track footprints (measured directly, one process at
a time), the sample-rate distribution, the shape of the cost model, and the
argument that admission belongs on bytes rather than on job count. Scheduling
by size is right whether or not it was what bit that night.

**Method, for next time.** Before attributing a slowdown to the code, check:

```powershell
# commit, not just working set: this is what MemoryError is raised against
$o = Get-CimInstance Win32_OperatingSystem
'{0:N1} GB commit free of {1:N1} GB' -f ($o.FreeVirtualMemory/1MB), ($o.TotalVirtualMemory/1MB)

# who is holding it
Get-Process | Sort-Object -Descending PrivateMemorySize64 |
    Select-Object -First 5 Name,
        @{n='CommitGB';e={[math]::Round($_.PrivateMemorySize64/1GB,2)}},
        @{n='CPUh';e={[math]::Round($_.CPU/3600,1)}}
```

Working set is what `GetProcessMemoryInfo` reports and what Task Manager shows
by default, and it is *not* the quantity that fails. A process can hold 45 GB
of commit against a 6.9 GB working set, which is exactly what this one did —
invisible in the obvious place to look, and fatal.

Before a long unattended run on Windows, stop it:

```powershell
Stop-Process -Name AMPLibraryAgent -Force -ErrorAction SilentlyContinue
```

It is a COM-activated background agent and relaunches on demand, so this costs
nothing but the indexing pass it was in the middle of.

---

## Finding 4 — the pool is bounded by memory, not by cores, and it fails hard

The overnight run that produced this section measured 192 tracks in 7 h 17 m and
lost most of that time to a limit nothing in the code was watching.

Nine workers were killed outright (`BrokenProcessPool`) and fourteen more died
with `MemoryError` — every one of them on Coldplay, whose masters are 192 kHz:

```
9  x worker lost: BrokenProcessPool: terminated abruptly
1  x MemoryError: Unable to allocate 1.08 GiB, shape (72350720, 2), float64  loudness.py:198
1  x MemoryError: Unable to allocate 877. MiB, shape (57458958, 2), float64  loudness.py:198
1  x MemoryError: Unable to allocate 794. MiB, shape (52052480, 2), float64  _signaltools.py
...
```

The per-artist throughput tells the rest of the story:

| artist | tracks | wall | audio-s / wall-s | rate |
| --- | --- | --- | --- | --- |
| Ariana Grande (44.1 kHz) | 121 | 3 h 34 m | 1.8 | healthy |
| Adele (44.1 kHz) | 24 | 1 h 03 m | 1.7 | healthy |
| Calvin Harris (44.1 kHz) | 24 | 49 m | 1.8 | healthy |
| **Coldplay (192 kHz)** | **7 of 30** | **49 m** | **0.5** | **collapsed** |

The collapse is worse than it looks. A single process measuring one 192 kHz
track end to end, with warm stems, sustains **0.49 audio-s/wall-s**. Six
workers on the same content managed **0.5 in total** — the pool as a whole did
no more than one worker alone would have.

*How much of that was the 192 kHz masters and how much was the background
agent of Finding 4a is not separable from this run.* Coldplay ran last, when
the agent was at its largest. The footprints below are measured one process at
a time and stand on their own; the attribution of this particular collapse does
not.

### What a worker actually holds

Peak working set, measured with `GetProcessMemoryInfo`, one track, `threads=1`:

| track | peak RSS |
| --- | --- |
| 44.1 kHz stereo, 285 s, with stems | **4.68 GB** |
| 192 kHz stereo, 283 s, with stems | **5.75 GB** |

`audio.py`'s docstring claims "peak memory is the size of the decoded float32
signal and not a multiple of it". That is true of *decoding* and untrue of the
run: `AudioSource` then caches `mono`, `mid`, `side`, `band_x`, `band_mid`,
`band_side` and an int32 copy, all but the first in float64, and a stems run
holds **five of these objects at once** — the master and four stems.

Six lanes × 4.68 GB is 28 GB on a 34 GB machine. It fits, barely, which is why
44.1 kHz artists ran at full speed. Six lanes × the 192 kHz footprint does not,
and nothing in the code noticed.

### Why worker count is the wrong unit

`-j 6` is a statement about cores. Memory demand is a statement about frames ×
channels, and those two disagree by a factor of four across this library:

| rate | files | share of files | share of all frames |
| --- | --- | --- | --- |
| 44.1 kHz | 743 | 58.3 % | 46.5 % |
| 48 kHz | 353 | 27.7 % | 23.3 % |
| 88.2–192 kHz | 178 | **14.0 %** | **30.2 %** |

Hi-res is 14 % of the files and 30 % of the samples, and it is not confined to
one corner: Queen (40 files), Mariah Carey (29), Taylor Swift (26), Coldplay
(21), Elvis Presley (14), Harry Styles (12), Amy Winehouse (11), Bruno Mars (9).
Any library scan will walk into it.

So admission is now by size. Each job carries `decoded_bytes`, `drive` holds a
byte budget rather than a job count, and a run of long or high-rate tracks
narrows the pool by itself:

| track | modelled | measured | lanes admitted |
| --- | --- | --- | --- |
| 44.1 kHz, 285 s, stems | 4.73 GB | 4.68 GB | **5** |
| 48 kHz, 303 s, stems | 5.13 GB | — | **5** |
| 192 kHz, 283 s, stems | 7.07 GB | 5.75 GB | **3** |
| 192 kHz, 435 s, stems (longest in the library) | 10.87 GB | — | **2** |

The model is built from what `audio.py` keeps rather than fitted to a curve,
which is why it lands within 1 % on the common case. It is deliberately
conservative on hi-res: the estimate counts every cached view as live for the
whole run, and Windows trims working sets, so the modelled 7.07 GB against a
measured 5.75 GB is headroom, not error.

The budget is `total RAM − 3 GB − 3 GB per separation stream`, held back as
amounts rather than a share. Both are roughly constant — the OS and the file
cache the decoding leans on, and a demucs process measured at 2.47 GB — so taking
a proportion would punish a small machine, where the reserve matters most. On
this bench, with the pipeline's one stream, that is 28.1 GB.

Note what that costs. **Six lanes of 44.1 kHz fitted before the pipeline and
five fit after it**, because a concurrent separation holds about half a lane's
worth of memory. That is the price of overlapping the phases on this machine,
and it is smaller than the phase the overlap removes.

---

## What the rebuilt pipeline measures

One artist, nine tracks, 1 935 s of audio, stems already cached so this is the
measuring half alone, on a machine with the Finding 4a agent stopped:

| quantity | value |
| --- | --- |
| wall clock | 16 m 16 s |
| throughput | **2.0 audio-s per wall-s** |
| sum of per-track service times | 4 580 s |
| effective lanes (service / wall) | 4.7 of 6 |
| per-lane rate | **0.42 audio-s per wall-s** |
| stems freed as it went | 1.2 GB |

The same code measuring the same kind of track in **one** process sustains
**0.90 audio-s per wall-s**. So six workers return 2.8x, not 6x, which is
Finding 1 again from a different direction: the pool is bandwidth-bound well
before it is core-bound, and the last workers are nearly free to add and nearly
worthless. The 4.7 effective lanes is the tail of a nine-track run draining
narrower than the pool; a library run does not pay that.

For comparison, the overnight run this work started from managed **1.55
audio-s per wall-s** — but it was sharing the machine, so that number belongs
to Finding 4a as much as to the old design, and the two cannot be separated
after the fact.

### The band-view change is bit-for-bit

`band_mid`/`band_side` aliasing `mid`/`side` below the cap is an arithmetic
identity, not an approximation, and it is checked rather than argued: one
track measured before the change, re-measured after it with `--force`, and the
two `analysis.json` files compared field by field.

    numeric fields compared, differing: 1
      .run.generated_utc
          before '2026-08-30T23:45:47+00:00'
          after  '2026-08-31T01:26:49+00:00'

The timestamp. Nothing else in the file moved.

---

## Where a track's time actually goes

cProfile over one full-profile 44.1 kHz track with warm stems, 285 s of audio,
one thread — cumulative seconds, so the entries nest:

| what | cumulative | share |
| --- | --- | --- |
| `stems.analyse` (four stems, every metric) | 274 s | 56 % |
| `delivery.analyse` (lossy encode passes) | 105 s | 22 % |
| `loudness.analyse` × 5 (master + 4 stems) | 111 s | 23 % |
| `resample_poly` / `upfirdn` (true-peak oversampling) | 85 s | 18 % |
| `melody.track_f0` / librosa `pyin` | 71 s | 15 % |

There is no redundant work of consequence in here. The one uncached view —
`AudioSource.channel()`, rebuilt 50 times a track — costs about 5 GB of
repeated float64 conversion, which at this machine's bandwidth is well under a
second against a 486 s run. Caching it would trade ~1 GB of extra resident
memory for that. It is not worth it, and it is written down here so nobody
measures it a third time.

**The DSP cost is inherent.** Per-worker throughput is ~0.48 audio-s/wall-s on
both 44.1 kHz and 192 kHz content, and six workers saturate memory bandwidth at
~2.75–2.9. That is the ceiling this machine has; the work below was about
reaching it, not raising it.

---

## Four things the pipeline got wrong first, and what they cost

All four were found by running it, not by reading it. All four are worth
knowing about because the same shapes will recur in any producer/consumer
stage added here.

### The card must never wait on the memory gate

The first version had the separation threads submit their own track to the
pool. A stage thread that finished separating and found no memory free would
block *holding its separation slot*, so the card stopped — waiting on a core.
That is precisely the coupling the pipeline exists to remove, reintroduced one
layer down.

`drive` now puts a feeder thread between them. Separation threads separate,
push onto a queue and go straight back to the card; the feeder alone waits for
memory. The card runs on to the next track, the next album, the next artist,
and stops only when the *disk* lookahead says it is far enough ahead.

### The budget has to count the separations too

The first pipelined run of nine tracks failed **all nine** with `MemoryError` —
including allocations as small as 4 MiB, which is what an exhausted commit
looks like. The budget had admitted five measuring lanes against 28 GB while
three demucs processes held 7.5 GB it knew nothing about.

| quantity | bytes |
| --- | --- |
| measured demucs peak, one stream | **2.47 GB** |
| reserve now held back per stream | 3 GB |
| base reserve (OS, file cache, parent) | 3 GB |

With one stream that leaves 28.1 GB, five lanes of a 4.7 GB track, and a
29.6 GB peak on a 34.1 GB machine.

### The budget was sized against memory the machine did not have

This is the one that cost a second run, and it is the most instructive
failure in this document, because the mechanism worked perfectly and the
number it was given was wrong.

`memory_budget` subtracted its reserves from `total_memory_bytes()` — the RAM
the machine *has*. What matters is the RAM that is *free*, and on this bench
the two differ by the 8-10 GB the OS and the desktop are already holding.
Worse, `available_memory_bytes()` existed in the same file and was used only
to print a warning, whose threshold (`free < budget * 0.75`) was slack enough
that 26 GB free against a 28 GB budget did not trip it.

| | |
| --- | --- |
| budget authorised | **28.1 GB** |
| memory actually free | **26.0 GB** |
| already held by OS and apps | 8.0 GB |

The scan was of a library whose 192 kHz albums carry 6.6-9.6 GB a track. The
gate admitted five lanes, about 26 GB of decoded audio, into 26 GB of free
memory. Windows killed a worker four minutes in.

**What it looked like** matters as much as what it was. Not an error: the run
went to 0% CPU, 0% GPU, 27 GB resident, and sat there. Nothing in the console
said "out of memory" — the only clue was three `BrokenProcessPool` lines that
PowerShell had not yet flushed to the terminal.

### One dead worker must not end the run

A worker killed by the OS does not fail its own track politely. It breaks the
executor, so that track and **every track still to come** raise
`BrokenProcessPool` on submission. One bad minute at track 4 was therefore
about to fail all 821, and the run neither recovered nor exited.

`drive` now takes a `restart` callback. A broken pool is rebuilt once per
break — not once per track it took down, which the generation counter in
`send()` is there to distinguish — the tracks caught in it go back on the
queue, and the budget tightens by a quarter on the theory that a pool only
breaks because it was too wide. A track that breaks the pool twice is reported
as a failure rather than retried forever.

### Workers that memory cannot run are not free

Sizing the budget correctly exposed a smaller error in the same arithmetic: a
measuring worker holds **824 MB** of interpreter, numpy and scipy before it
decodes a single sample, and the cost model counted that as zero. Six workers
start 4.9 GB down.

So starting more workers than memory can ever run at once is not merely
useless — each one takes `WORKER_RESERVE` from the lanes that do run.
`workers_that_fit` now walks the count down until it is consistent with the
budget its own overhead produces, sized on the **typical** track rather than
the largest (a library's biggest master would size the pool at one; the
admission gate is what handles outliers). On the library here — median 2.80 GB,
p90 4.04 GB, max 9.65 GB — that is 5 workers rather than 6. Scoped to one
192 kHz album, where the typical track is 6.0 GB, it is 2:

```
jobs: 2 process(es), not 6: 6 would want 41 GB for a typical track
      and there is not that much
memory: 16 GB for decoded audio; a typical track wants 6.0 GB so 2 measure
        at once, and the largest wants 9.0 GB so that one measures 1 at a time
```

### A failed track keeps its stems

`--prune-stems` originally dropped a track's stems on any completion. A track
that failed therefore lost minutes of GPU to save 165 MB of disk for the few
seconds until the next run reached it. It now drops them only on success,
which is the rule the standalone `tools/prune_stems.py` already followed (it
requires a `corpus_row.json` before deleting anything).

---

## What this means at library scale

Extrapolating the measured 106 min per 59 tracks to 100 000 masters, at this
corpus's mean of 225 s a track:

| configuration | est. continuous runtime |
| --- | --- |
| sequential phases, serial separation (before this work) | ~125 days |
| sequential phases, 3 separation streams | ~113 days |
| overlapped phases, five lanes (**current**) | ~101 days |

The overlap does not deliver the whole of its arithmetic, because it is
paid for in the memory a concurrent separation holds: the phase
disappears, and a measuring lane goes with it. On a machine with more RAM
the same code keeps six lanes and the gain is larger; that is the single
upgrade this workload responds to.

Storage matters as much as time at that scale, and neither number is small:

- **Output** ~17 MB a track → ~1.7 TB per 100 000.
- **Stem cache** ~165 MB a track → **~15 TB** if nothing is ever evicted.

### The stem cache is the binding constraint, and it bites long before 100 000

This is not a scale-out worry to be handled later. On the bench library:

| quantity | |
| --- | --- |
| tracks | 1 274 |
| stem cache if all are separated | **210.2 GB** |
| output | 21.7 GB |
| free disk | **63.7 GB** |

**A single `mtx scan E:\Music --stems` cannot complete.** It would separate its
way to a full disk somewhere around a third of the way in and die, having
written measurements for none of the tracks it had already separated, because
`separate_first()` completes the entire todo list before the pool measures
anything.

Two independent defects combine to produce that:

1. **Separation is unbounded and up front.** The whole todo list is separated
   before a single measurement, so peak cache is the size of the *run*, not of
   any working set.
2. **Nothing is ever evicted.** A track's stems are dead weight the moment its
   `corpus_row.json` is written, and they stay on disk forever.

The fix in the code is to overlap the phases (which caps in-flight separations
at the stream count) and evict a track's stems once it is measured. Until then,
`tools/scan_library.ps1` works around it from outside: it scans one artist at a
time and runs `tools/prune_stems.py` between artists, so peak cache is the
largest single artist — 137 tracks, ~22.6 GB on this library — rather than all
1 274. Both are resumable, because `mtx scan` already skips tracks that have a
receipt.

`prune_stems.py` reads the sha256 in each `mtx_source.json` receipt, which is
the same hash the cache key is the first 24 characters of, so it never has to
touch the audio. It requires a `corpus_row.json` beside the receipt before it
will delete anything — a run interrupted between writing the two must not lose
stems for a track that still has to be measured.

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
| --- | --- | --- |
| `MTX_STEMS_CACHE` to relocate the stem cache off the system disk | `metrics/stems.py` | landed |
| `_mtx_out` / `_mtx_stems` added to `SKIP_DIRS` so a scan cannot measure its own separated stems as masters | `scan.py` | landed |
| `separation_streams()`, VRAM-derived, `--stems-jobs` to override | `metrics/stems.py`, `cli.py` | landed |
| `separate_first()` runs several streams through a thread pool | `scan.py` | landed |
| Overlap the separation and DSP phases | `scan.py` | landed |
| Size the memory budget against free RAM, not total | `parallel.py` | landed |
| Charge each worker the 824 MB it holds before decoding | `parallel.py` | landed |
| Start no more workers than memory can run (`workers_that_fit`) | `parallel.py`, `scan.py` | landed |
| Rebuild the pool when a worker is killed, instead of failing every remaining track | `scan.py` | landed |
| Bound and evict the stem cache | `metrics/stems.py` | **open**, blocks library-scale runs |
| Keep one demucs process alive instead of re-importing torch per file | `metrics/stems.py` | **open**, ~6 s/track inside the non-bottleneck phase |
