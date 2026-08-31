"""What a finished (or running) scan actually achieved, from its own receipts.

Every measured track writes an `mtx_source.json` carrying its own wall time.
Joined against the source file's duration and sample rate, that is a cost
model measured on the real machine under real contention -- which is worth
more than any single benchmark track, and is the only way to tell a slow run
from an expensive one.

The distinction matters because they look identical from outside. A scan
working through a block of 192 kHz masters is memory-limited to two or three
lanes and leaves the machine cool and quiet; so does a scan that is broken.
The numbers below separate them.

    python tools/scan_report.py "E:\\Music\\_mtx_out"
    python tools/scan_report.py "E:\\Music\\_mtx_out" --since 2026-08-31
    python tools/scan_report.py "E:\\Music\\_mtx_out" --library "E:\\Music"

`--library` additionally prices what is left to do, using the rates just
measured rather than assumed ones.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import statistics as st
import sys

MB_PER_TRACK = 16.1          # measured: output bytes per finished track


def read_receipts(out_dir: str, since: dt.datetime | None):
    import soundfile as sf
    rows = []
    pattern = os.path.join(out_dir, "**", "mtx_source.json")
    for p in glob.iglob(pattern, recursive=True):
        try:
            d = json.load(open(p, encoding="utf-8"))
            run, src = d["run"], d["source"]
            done = dt.datetime.fromisoformat(run["completed_utc"])
            if since and done < since:
                continue
            info = sf.info(src["path"])
        except Exception:
            continue
        rows.append({
            "path": src["path"],
            "elapsed": float(run["elapsed_seconds"]),
            "done": done,
            "dur": float(info.duration),
            "sr": int(info.samplerate),
        })
    return rows


def lanes_over_time(rows):
    """Mean concurrent lanes, by sampling the timeline rather than by pairing.

    Pairwise overlap counting inflates badly when receipts span several
    separate runs, so the occupancy is integrated over the wall clock and
    divided by the time the machine was actually working.
    """
    if not rows:
        return 0.0, 0
    events = []
    for r in rows:
        start = r["done"] - dt.timedelta(seconds=r["elapsed"])
        events.append((start, 1))
        events.append((r["done"], -1))
    events.sort()
    live = peak = 0
    busy = 0.0
    area = 0.0
    prev = events[0][0]
    for t, delta in events:
        if live > 0:
            span = (t - prev).total_seconds()
            area += live * span
            busy += span
        prev = t
        live += delta
        peak = max(peak, live)
    return (area / busy if busy else 0.0), peak


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="the scan's output tree, e.g. E:\\Music\\_mtx_out")
    ap.add_argument("--since", help="only receipts completed on/after this date")
    ap.add_argument("--library", help="also price what is still to do")
    args = ap.parse_args()

    since = None
    if args.since:
        since = dt.datetime.fromisoformat(args.since)
        if since.tzinfo is None:
            since = since.replace(tzinfo=dt.timezone.utc)

    rows = read_receipts(args.out_dir, since)
    if not rows:
        print("no receipts found (has anything finished yet?)")
        return 1

    print(f"{len(rows)} finished track(s)"
          + (f" since {args.since}" if args.since else ""))

    audio = sum(r["dur"] for r in rows)
    lane_s = sum(r["elapsed"] for r in rows)
    start = min(r["done"] - dt.timedelta(seconds=r["elapsed"]) for r in rows)
    end = max(r["done"] for r in rows)
    wall = (end - start).total_seconds()
    mean_lanes, peak = lanes_over_time(rows)

    print(f"  audio measured   {audio / 3600:8.1f} h")
    print(f"  wall elapsed     {wall / 3600:8.1f} h")
    print(f"  throughput       {audio / wall:8.2f} audio-s/wall-s")
    print(f"  lanes in use     {mean_lanes:8.2f} mean, {peak} peak")
    print(f"  output written   {len(rows) * MB_PER_TRACK / 1000:8.2f} GB")
    if wall:
        print(f"  rate             {len(rows) * MB_PER_TRACK / 1000 / (wall / 25200):8.2f}"
              f" GB per 7 h at this pace")

    # Per-lane cost, which is what tells an expensive track from a slow one.
    print(f"\n{'rate':>8} {'n':>5} {'audio-s/wall-s':>16} {'median s/track':>16}"
          f" {'share of compute':>18}")
    by_sr: dict[int, list] = {}
    for r in rows:
        by_sr.setdefault(r["sr"], []).append(r)
    rates = {}
    for sr in sorted(by_sr):
        g = by_sr[sr]
        rate = st.median(x["dur"] / x["elapsed"] for x in g)
        rates[sr] = rate
        share = 100 * sum(x["elapsed"] for x in g) / lane_s
        print(f"{sr:>8} {len(g):>5} {rate:>16.3f} "
              f"{st.median(x['elapsed'] for x in g):>16.0f} {share:>17.0f}%")

    if not args.library:
        return 0

    import soundfile as sf
    exts = {".flac", ".wav", ".m4a", ".mp3", ".aiff", ".aif", ".ogg", ".opus",
            ".w64", ".caf", ".aac", ".wv", ".ape"}
    seen = {os.path.normcase(r["path"]) for r in rows}
    default = st.median(rates.values()) if rates else 0.4

    def rate_for(sr):
        if sr in rates:
            return rates[sr]
        near = min(rates, key=lambda k: abs(k - sr)) if rates else None
        return rates[near] if near else default

    todo, todo_audio, todo_lane_s, hires = 0, 0.0, 0.0, 0
    for dp, _dn, fn in os.walk(args.library):
        if "_mtx_out" in dp or "_mtx_stems" in dp:
            continue
        for f in fn:
            if os.path.splitext(f)[1].lower() not in exts:
                continue
            p = os.path.join(dp, f)
            if os.path.normcase(p) in seen:
                continue
            try:
                i = sf.info(p)
            except Exception:
                continue
            todo += 1
            todo_audio += float(i.duration)
            todo_lane_s += float(i.duration) / rate_for(int(i.samplerate))
            if i.samplerate > 48000:
                hires += 1

    if not todo:
        print("\nnothing left to measure.")
        return 0

    print(f"\n{todo} track(s) left, {todo_audio / 3600:.1f} h of audio, "
          f"{hires} hi-res ({100 * hires / todo:.0f}%)")
    print(f"  {todo_lane_s / 3600:.0f} lane-hours of work at the rates above\n")
    print(f"{'lanes':>6} {'wall to finish':>16} {'in 7 h':>26}")
    for lanes in (2, 3, 4, 5, 6):
        w = todo_lane_s / lanes
        n = 25200 / (w / todo)
        print(f"{lanes:>6} {w / 3600:>13.0f} h "
              f"{n:>14.0f} tracks = {n * MB_PER_TRACK / 1000:>4.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
