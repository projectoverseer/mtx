"""Is this machine running at the speed it usually does?

A scan cannot tell a clamped CPU from expensive audio.  Both present as low
utilisation and long tracks, and one morning here was lost to exactly that
confusion: an i7-9750H pinned at 778 MHz -- 30% of nominal, on AC, cooler and
slower than the same machine had been on battery an hour earlier -- while the
scan reported honest ETAs of eighty hours and looked broken.

So the machine gets benchmarked before the library does.  The number is a
single-core FFT rate, which tracks clock speed closely and takes a second and
a half; it is compared against the best this machine has ever recorded, kept
in `.cpu_baseline.json` beside this file.  Self-calibrating on purpose: a
hardcoded threshold would be wrong on every machine but this one.

    python tools/bench_cpu.py              # measure, compare, update baseline
    python tools/bench_cpu.py --json       # machine-readable
    python tools/bench_cpu.py --reset      # forget the baseline

Exit status is 0 when the machine is at or near its own best, 4 when it is
clearly below it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        ".cpu_baseline.json")
SECONDS = 1.5
# Below this fraction of the machine's own best, something is wrong with the
# machine rather than with the work.  Generous enough to absorb a background
# task and thermal variation; a clamped chip lands near 0.3.
FLOOR = 0.65


def measure(seconds: float = SECONDS) -> float:
    """Single-core FFT throughput, in transforms per second."""
    import numpy as np
    n = 1 << 20
    x = np.random.randn(n).astype(np.float64)
    np.fft.rfft(x)                       # warm the plan cache
    t0 = time.perf_counter()
    reps = 0
    while time.perf_counter() - t0 < seconds:
        np.fft.rfft(x)
        reps += 1
    return reps / (time.perf_counter() - t0)


def load() -> dict:
    try:
        with open(BASELINE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save(data: dict) -> None:
    try:
        with open(BASELINE, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=1, sort_keys=True)
            f.write("\n")
    except OSError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--reset", action="store_true", help="forget the baseline")
    ap.add_argument("--seconds", type=float, default=SECONDS)
    args = ap.parse_args()

    if args.reset:
        try:
            os.remove(BASELINE)
            print("baseline forgotten")
        except OSError:
            print("no baseline to forget")
        return 0

    now = measure(args.seconds)
    data = load()
    best = float(data.get("best") or 0.0)
    ratio = (now / best) if best else 1.0
    fresh = now > best

    if fresh:
        data["best"] = now
        data["best_recorded"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        save(data)

    verdict = "ok"
    if best and ratio < FLOOR:
        verdict = "slow"

    if args.json:
        print(json.dumps({"now": round(now, 2), "best": round(max(best, now), 2),
                          "ratio": round(ratio, 3), "verdict": verdict},
                         sort_keys=True))
    else:
        print(f"single-core FFT: {now:.1f} transforms/s")
        if best:
            print(f"this machine's best: {best:.1f}  ({ratio:.0%})")
        if fresh:
            print("(new best, recorded)")
        if verdict == "slow":
            print(f"\n  This machine is at {ratio:.0%} of its own best.")
            print("  That is not the audio and not the scan. Usual causes, in")
            print("  the order worth trying:")
            print("    1. A charger the firmware does not recognise. Try the")
            print("       original adapter; check the boot screen for a warning.")
            print("    2. A stuck embedded controller. Shut down fully, unplug,")
            print("       hold the power button 20 s, plug in, boot.")
            print("    3. Thermal: a failed fan or dried paste clamps the clock")
            print("       within milliseconds, so the chassis stays cool.")
            print("  Re-run this after each step.")
    return 4 if verdict == "slow" else 0


if __name__ == "__main__":
    sys.exit(main())
