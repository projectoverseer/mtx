"""Generate the synthetic track used for the sample digest and smoke tests.

Deliberately loaded with things the tool is supposed to notice: a mono-ish bass
end, a hard ceiling below full scale, a beat-synchronous duck, an HF shelf in
the last section, and a fade at each end.  No copyrighted audio is involved.
"""

from __future__ import annotations

import argparse

import numpy as np
import soundfile as sf
from scipy import signal as sps

SR = 44100
BPM = 120.0


def build(seconds: float = 75.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    beat = 60.0 / BPM
    L = np.zeros(n)
    R = np.zeros(n)

    # Bass: a 62 Hz fundamental with its octave, centred (mono below 120 Hz).
    bass_env = np.zeros(n)
    for k in range(int(seconds / beat)):
        i = int(k * beat * SR)
        env = np.exp(-np.arange(min(int(0.45 * SR), n - i)) / (0.12 * SR))
        bass_env[i : i + env.size] += env
    bass = bass_env * (0.55 * np.sin(2 * np.pi * 62.0 * t)
                       + 0.18 * np.sin(2 * np.pi * 124.0 * t))
    L += bass
    R += bass

    # Kick: short low thump on every beat.
    for k in range(int(seconds / beat)):
        i = int(k * beat * SR)
        m = min(int(0.09 * SR), n - i)
        e = np.exp(-np.arange(m) / (0.02 * SR))
        f = np.linspace(110, 45, m)
        click = e * np.sin(2 * np.pi * np.cumsum(f) / SR)
        L[i : i + m] += 0.6 * click
        R[i : i + m] += 0.6 * click

    # Hats: wide, on the offbeat, decorrelated between channels.
    for k in range(int(seconds / (beat / 2))):
        i = int((k * beat / 2 + beat / 2) * SR)
        m = min(int(0.05 * SR), max(0, n - i))
        if m <= 0:
            continue
        e = np.exp(-np.arange(m) / (0.008 * SR))
        L[i : i + m] += 0.10 * e * rng.standard_normal(m)
        R[i : i + m] += 0.10 * e * rng.standard_normal(m)

    # Pad: a wide chord, different detune per channel.
    chord = (196.0, 233.08, 293.66, 392.0)
    for f0 in chord:
        L += 0.055 * np.sin(2 * np.pi * f0 * t)
        R += 0.055 * np.sin(2 * np.pi * f0 * 1.002 * t + 0.7)

    # Lead: enters at 25 s, leaves at 55 s -- gives the segmenter something.
    seg = (t > 25) & (t < 55)
    lead = 0.16 * np.sin(2 * np.pi * 587.33 * t) * seg
    L += lead * 0.8
    R += lead * 1.0

    # Sidechain duck on everything but the kick, synchronous with the beat.
    phase = (t % beat) / beat
    duck = 1.0 - 0.45 * np.exp(-phase / 0.18)
    L *= duck
    R *= duck

    # Last section: shelve the highs down, so a section-wise tilt change exists.
    late = t > 55
    sos = sps.butter(2, 6000 / (SR / 2), btype="high", output="sos")
    for ch in (L, R):
        hp = sps.sosfilt(sos, ch)
        ch[late] -= 0.7 * hp[late]

    # Fades at both ends.
    fi = int(1.5 * SR)
    fo = int(3.0 * SR)
    ramp_in = np.linspace(0, 1, fi) ** 2
    ramp_out = np.linspace(1, 0, fo) ** 2
    for ch in (L, R):
        ch[:fi] *= ramp_in
        ch[-fo:] *= ramp_out

    x = np.stack([L, R], axis=1)
    x /= np.max(np.abs(x)) / 0.999

    # Drive into a hard ceiling at -1.0 dBFS, then turn the whole thing down:
    # the clip-then-normalise case that a fixed -0.1 dBFS threshold misses.
    ceiling = 10 ** (-1.0 / 20.0)
    x = np.clip(x * 1.8, -ceiling, ceiling)
    x *= 10 ** (-0.7 / 20.0)

    # Leading and trailing digital black.
    pad = int(0.25 * SR)
    return np.vstack([np.zeros((pad, 2)), x, np.zeros((pad, 2))])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", default="mtx_testsignal.flac")
    ap.add_argument("--seconds", type=float, default=75.0)
    ap.add_argument("--subtype", default="PCM_24")
    args = ap.parse_args()
    x = build(args.seconds)
    sf.write(args.out, x, SR, subtype=args.subtype)
    print(f"wrote {args.out}: {x.shape[0] / SR:.2f} s, {SR} Hz, {args.subtype}")


if __name__ == "__main__":
    main()
