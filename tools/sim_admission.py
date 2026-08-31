"""Does the feeder's strict FIFO admission idle workers on this library?

The pipeline admits tracks to the measuring pool in queue order.  A track that
does not fit beside what is running blocks *every track behind it*, including
ones that would fit -- classic head-of-line blocking.  It costs nothing on a
library of uniform tracks and everything on one where a 30-track 192 kHz album
sits contiguously in directory order.

Discrete-event simulation over the real library, comparing:

  fifo      admit the head when it fits             (what ships today)
  bestfit   admit the largest waiting job that fits (work-conserving)

Service time is audio_seconds / RATE per lane, which ignores memory-bandwidth
contention and so flatters both policies equally.
"""
import heapq
import os
import sys

sys.path.insert(0, r"E:\Git\projectoverseer\mtx\src")

RATE = 0.45          # audio-seconds per wall-second, one lane
BUDGET = 15_000_000_000
PROCS = 5
LOOKAHEAD = PROCS + 2


def library():
    import soundfile as sf
    from mtx.scan import decoded_bytes
    exts = {".flac", ".wav", ".m4a", ".mp3", ".aiff", ".aif", ".ogg", ".opus"}
    jobs = []
    for dp, dn, fn in os.walk(r"E:\Music"):
        if "_mtx_out" in dp or "_mtx_stems" in dp:
            continue
        for f in sorted(fn):
            if os.path.splitext(f)[1].lower() not in exts:
                continue
            p = os.path.join(dp, f)
            try:
                i = sf.info(p)
            except Exception:
                continue
            b = decoded_bytes(int(i.frames), int(i.channels),
                              int(i.samplerate), float(i.duration), True)
            jobs.append((b, float(i.duration)))
    return jobs


def run(jobs, policy):
    """Return (makespan, mean lanes in use, worker-seconds idle)."""
    pending = list(jobs)
    waiting = []
    running = []                      # heap of (finish_time, bytes)
    now = last = 0.0
    in_flight = 0
    lane_seconds = 0.0
    done = 0
    total = len(jobs)

    def advance(to):
        nonlocal lane_seconds, last
        lane_seconds += len(running) * (to - last)
        last = to

    while done < total:
        while pending and len(waiting) + len(running) < LOOKAHEAD:
            waiting.append(pending.pop(0))

        pick = None
        if waiting and len(running) < PROCS:
            if policy == "fifo":
                b, d = waiting[0]
                if not in_flight or in_flight + b <= BUDGET:
                    pick = waiting.pop(0)
            else:
                fits = [t for t in waiting
                        if not in_flight or in_flight + t[0] <= BUDGET]
                if fits:
                    pick = max(fits, key=lambda t: t[0])
                    waiting.remove(pick)

        if pick is not None:
            b, d = pick
            advance(now)
            in_flight += b
            heapq.heappush(running, (now + d / RATE, b))
            continue

        if not running:
            break
        t, b = heapq.heappop(running)
        advance(t)
        now = t
        in_flight -= b
        done += 1

    return now, (lane_seconds / now if now else 0.0), PROCS * now - lane_seconds


if __name__ == "__main__":
    jobs = library()
    audio = sum(d for _, d in jobs)
    print(f"{len(jobs)} tracks, {audio / 3600:.1f} h of audio")
    print(f"budget {BUDGET / 1e9:.0f} GB, {PROCS} workers, {RATE} audio-s/wall-s per lane\n")
    base = None
    for policy in ("fifo", "bestfit"):
        span, lanes, idle = run(jobs, policy)
        if base is None:
            base = span
        print(f"{policy:8} makespan {span / 3600:6.2f} h   "
              f"mean lanes {lanes:4.2f}/{PROCS}   "
              f"idle worker-hours {idle / 3600:6.1f}   "
              f"{audio / span:4.2f} audio-s/wall-s   "
              f"{100 * (base - span) / base:+5.1f}%")


def sweep():
    """Where does the throughput actually come from: order, lookahead, or RAM?"""
    global BUDGET, PROCS, LOOKAHEAD
    jobs = library()
    audio = sum(d for _, d in jobs)

    print("\n--- deeper lookahead (more choice for best-fit) ---")
    BUDGET, PROCS = 15_000_000_000, 5
    for la in (7, 12, 24, 48):
        LOOKAHEAD = la
        span, lanes, idle = run(jobs, "bestfit")
        print(f"  lookahead {la:3}  {audio / span:4.2f} audio-s/wall-s  "
              f"mean lanes {lanes:4.2f}")

    print("\n--- more memory (free RAM after OS, workers and one stream) ---")
    LOOKAHEAD = 7
    for gb, ram in ((15, 34), (21, 40), (27, 48), (39, 64), (63, 96)):
        BUDGET = gb * 1_000_000_000
        PROCS = min(6, max(1, int(gb // 2.8)))
        span, lanes, idle = run(jobs, "bestfit")
        print(f"  {ram:3} GB machine -> budget {gb:2} GB, {PROCS} workers: "
              f"{audio / span:4.2f} audio-s/wall-s  mean lanes {lanes:4.2f}/{PROCS}"
              f"  ({audio / span / 1.84:.2f}x)")

    print("\n--- if a track held one stem at a time instead of four ---")
    lighter = [(int(b - 0.55 * (b - b / 4)), d) for b, d in jobs]
    BUDGET, PROCS, LOOKAHEAD = 15_000_000_000, 5, 7
    span, lanes, idle = run(lighter, "bestfit")
    print(f"  same 34 GB machine: {audio / span:4.2f} audio-s/wall-s  "
          f"mean lanes {lanes:4.2f}/{PROCS}  ({audio / span / 1.84:.2f}x)")


sweep()
