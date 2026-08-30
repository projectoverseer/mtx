"""Synthetic signals with answers that are known in advance.

Every assertion prints the measured value next to the expectation, so a pass is
as informative as a failure.  Exit code is non-zero if anything fails.
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Any, Callable

import numpy as np
import soundfile as sf
from scipy import signal as sps

from .dsp import (_rlb_coeffs, _shelf_coeffs, block_loudness, gated_integrated,
                  true_peak)
from .util import Collector, db_amp

SR = 48000

# BS.1770-4 Table 1 (48 kHz), against which the filter design is checked.
BS1770_SHELF_B = (1.53512485958697, -2.69169618940638, 1.19839281085285)
BS1770_SHELF_A = (1.0, -1.69065929318241, 0.73248077421585)
BS1770_RLB_A = (1.0, -1.99004745483398, 0.99007225036621)


class Suite:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.notes: list[str] = []

    def check(self, name: str, ok: bool, measured: Any, expected: str) -> bool:
        status = "PASS" if ok else "FAIL"
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        if self.verbose or not ok:
            print(f"{status}  {name}\n        measured: {measured}\n"
                  f"        expected: {expected}",
                  file=sys.stdout if ok else sys.stderr)
        return ok

    def near(self, name: str, measured: float | None, target: float,
             tol: float, unit: str = "") -> bool:
        ok = measured is not None and abs(float(measured) - target) <= tol
        m = "None" if measured is None else f"{float(measured):.4f}{unit}"
        return self.check(name, ok, m, f"{target}{unit} +/- {tol}{unit}")

    def note(self, text: str) -> None:
        self.notes.append(text)
        if self.verbose:
            print(f"NOTE  {text}")


def _sine(freq: float, seconds: float, amp: float, sr: int = SR,
          phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float64)


def _write(path: str, x: np.ndarray, sr: int, subtype: str) -> str:
    sf.write(path, x, sr, subtype=subtype)
    return path


# ------------------------------------------------------------------ the tests
def t_kweighting(s: Suite) -> None:
    b, a = _shelf_coeffs(48000)
    _, a2 = _rlb_coeffs(48000)
    s.check("K-weighting shelf matches BS.1770-4 Table 1 at 48 kHz",
            np.allclose(b, BS1770_SHELF_B, atol=1e-8) and
            np.allclose(a, BS1770_SHELF_A, atol=1e-8),
            f"b={np.round(b, 9).tolist()} a={np.round(a, 9).tolist()}",
            f"b={list(BS1770_SHELF_B)} a={list(BS1770_SHELF_A)}")
    s.check("K-weighting RLB stage matches BS.1770-4 Table 2 at 48 kHz",
            np.allclose(a2, BS1770_RLB_A, atol=1e-8),
            f"a={np.round(a2, 9).tolist()}", f"a={list(BS1770_RLB_A)}")


def t_sine_loudness(s: Suite) -> None:
    s.note("The specification lists both 'LUFS-I ~ -20.0' and 'sample peak "
           "-20.0 dBFS' for the same 1 kHz sine. A sine cannot satisfy both: its "
           "peak sits 3.01 dB above its RMS. Both readings are asserted "
           "separately below.")
    # Case A: mono sine at -20 dBFS RMS -> -20 LUFS.
    x = _sine(1000.0, 10.0, 10 ** (-20 / 20) * np.sqrt(2))[:, None]
    _, bl = block_loudness(x, SR, 0.4, 0.1)
    lufs, _ = gated_integrated(bl)
    s.near("1 kHz sine at -20 dBFS RMS -> LUFS-I", lufs, -20.0, 0.1, " LUFS")
    s.near("  its sample peak", db_amp(float(np.max(np.abs(x)))), -16.9897, 0.01, " dBFS")
    # Case B: mono sine whose sample peak is -20 dBFS.
    y = _sine(1000.0, 10.0, 10 ** (-20 / 20))[:, None]
    s.near("1 kHz sine peaking at -20 dBFS -> sample peak",
           db_amp(float(np.max(np.abs(y)))), -20.0, 0.001, " dBFS")
    _, bl2 = block_loudness(y, SR, 0.4, 0.1)
    lufs2, _ = gated_integrated(bl2)
    s.near("  its LUFS-I (3.01 dB below case A)", lufs2, -23.01, 0.1, " LUFS")
    # Correlation and side/mid for the same sine in both channels.
    st = np.repeat(x, 2, axis=1)
    L, R = st[:, 0], st[:, 1]
    corr = float(np.corrcoef(L, R)[0, 1])
    side = (L - R) / 2.0
    s.near("identical L and R -> correlation", corr, 1.0, 1e-9)
    s.check("identical L and R -> side/mid is -inf (side energy exactly zero)",
            float(np.sum(side ** 2)) == 0.0, f"side energy {float(np.sum(side ** 2))}",
            "0.0 exactly")


def t_intersample_peak(s: Suite) -> None:
    # A sine at fs/4 with a 45 degree phase offset peaks exactly between samples.
    x = _sine(SR / 4.0, 2.0, 1.0, phase=np.pi / 4)[:, None]
    sp = db_amp(float(np.max(np.abs(x))))
    tp4 = db_amp(float(true_peak(x, 4)[0]))
    tp16 = db_amp(float(true_peak(x, 16)[0]))
    s.check("inter-sample sine: true peak exceeds sample peak",
            tp4 > sp + 0.5, f"sample {sp:.3f} dBFS, 4x {tp4:.3f} dBTP",
            "4x true peak more than 0.5 dB above the sample peak")
    s.check("inter-sample sine: 16x estimate is at least the 4x estimate",
            tp16 >= tp4 - 1e-9, f"4x {tp4:.4f} dBTP, 16x {tp16:.4f} dBTP",
            "16x >= 4x")
    s.near("inter-sample sine: 16x true peak", tp16, 0.0, 0.15, " dBTP")


def t_pruned_scan_equivalence(s: Suite) -> None:
    """The pruned 16x scan must give exactly the full scan's answer."""
    from .dsp import true_peak_scan

    rng = np.random.default_rng(7)
    # Dynamic material plus a loud burst: the case pruning is supposed to skip.
    x = (rng.standard_normal((SR * 6, 2)) * 0.02).astype(np.float64)
    burst = slice(SR * 3, SR * 3 + 2000)
    x[burst] += np.repeat(_sine(SR / 4.0, 2000 / SR, 0.95, phase=np.pi / 4)[:, None],
                          2, axis=1)
    thr = [0.0, -0.3, -1.0]
    full = true_peak_scan(x, SR, 16, thresholds_dbtp=thr, env_hop_s=0.001)
    pruned = true_peak_scan(x, SR, 16, thresholds_dbtp=thr, env_hop_s=None)
    s.check("pruned 16x scan equals the full 16x scan (peak)",
            full["peak"] == pruned["peak"],
            f"full {full['peak']:.9f} vs pruned {pruned['peak']:.9f}",
            "bit-identical")
    s.check("pruned 16x scan equals the full 16x scan (per channel and overs)",
            np.array_equal(full["peak_per_channel"], pruned["peak_per_channel"])
            and full["over_counts"] == pruned["over_counts"]
            and full["peak_time_s"] == pruned["peak_time_s"],
            f"per-channel {pruned['peak_per_channel']}, overs {pruned['over_counts']}, "
            f"scanned {pruned['scanned_fraction']:.1%} of the file",
            "identical to the full scan")


def t_threaded_scan_equivalence(s: Suite) -> None:
    """Threading the oversampling must not change a single number.

    The scan folds chunk results into running state, and an excursion that
    straddles a chunk boundary is only counted once if the chunks arrive in
    order.  This is the regression test for that: the same signal scanned on
    one thread and on several has to give bit-identical answers, envelope
    included.
    """
    from .dsp import true_peak_scan

    rng = np.random.default_rng(11)
    # Long enough to cross several chunk boundaries, and hot enough that the
    # over counting has boundaries to get wrong.
    x = (rng.standard_normal((SR * 25, 2)) * 0.25).astype(np.float64)
    x += np.repeat(_sine(997.0, SR * 25 / SR, 0.72)[:, None], 2, axis=1)
    x = np.clip(x, -0.999, 0.999)
    thr = [0.0, -0.3, -1.0]
    for label, kw in (("4x with envelope", dict(env_hop_s=0.001)),
                      ("16x pruned", dict(env_hop_s=None, thresholds_dbtp=thr))):
        ov = 4 if "4x" in label else 16
        one = true_peak_scan(x, SR, ov, workers=1, chunk=1 << 16, **kw)
        many = true_peak_scan(x, SR, ov, workers=4, chunk=1 << 16, **kw)
        same = (one["peak"] == many["peak"]
                and one["peak_time_s"] == many["peak_time_s"]
                and one["over_counts"] == many["over_counts"]
                and np.array_equal(one["peak_per_channel"], many["peak_per_channel"])
                and np.array_equal(one["envelope"], many["envelope"]))
        s.check(f"{label} scan is identical on 1 and 4 threads", same,
                f"peak {many['peak']:.9f}, overs {many['over_counts']}, "
                f"envelope {many['envelope'].size} points",
                "bit-identical to the single-threaded scan")


def t_clipped_below_full_scale(s: Suite) -> None:
    """Trap #1: a hard-clipped sine whose ceiling sits below -0.1 dBFS."""
    from .metrics import dynamics as dyn

    ceiling = 10 ** (-3.0 / 20.0)
    x = np.clip(_sine(200.0, 5.0, 1.0), -ceiling, ceiling)
    with tempfile.TemporaryDirectory() as d:
        p = _write(os.path.join(d, "clipped.wav"), np.repeat(x[:, None], 2, axis=1),
                   SR, "PCM_24")
        from .audio import AudioSource
        col = Collector()
        src = AudioSource(p, col)
        ft = dyn._flat_top(src, col)
    peak_db = ft["per_channel"][0]["channel_peak_dbfs"]
    s.near("clipped sine ceiling sits below -0.1 dBFS", peak_db, -3.0, 0.02, " dBFS")
    s.check("REGRESSION (trap #1): flat-top detector finds clipping below -0.1 dBFS",
            ft["total_flat_samples"] > 1000 and ft["longest_run_samples"] > 5,
            f"{ft['total_flat_samples']} flat samples, longest run "
            f"{ft['longest_run_samples']} samples "
            f"({ft['longest_run_ms']:.2f} ms)",
            "thousands of flat samples and runs longer than 5 samples, despite "
            "the ceiling being 3 dB below full scale")
    s.check("clip-then-normalise signature reported",
            ft["clip_then_normalise"]["detected"] is True,
            ft["clip_then_normalise"], "detected = True")


def t_noise_stereo(s: Suite) -> None:
    rng = np.random.default_rng(0)
    L = rng.standard_normal(SR * 5) * 0.1
    R = rng.standard_normal(SR * 5) * 0.1
    corr = float(np.corrcoef(L, R)[0, 1])
    mid, side = (L + R) / 2, (L - R) / 2
    sm = 10 * np.log10(np.mean(side ** 2) / np.mean(mid ** 2))
    s.near("independent L/R noise -> correlation", corr, 0.0, 0.02)
    s.near("independent L/R noise -> side/mid", sm, 0.0, 0.2, " dB")


def t_bass_fundamentals(s: Suite) -> None:
    from .metrics.spectrum import _bass_fundamentals
    from .dsp import welch_psd

    for target in (62.0, 70.0):
        x = _sine(target, 8.0, 0.3, sr=44100)
        f, p = welch_psd(x, 44100, 131072)
        res = _bass_fundamentals(f, p, Collector())
        got = res["peaks"][0]["frequency_hz"] if res["peaks"] else None
        s.near(f"bass fundamental detector on a {target:g} Hz sine", got, target,
               1.0, " Hz")
    # The pair together: the 131072-point pass must resolve them as two peaks.
    x = (_sine(62.0, 8.0, 0.3, sr=44100) + _sine(70.0, 8.0, 0.3, sr=44100))
    f, p = welch_psd(x, 44100, 131072)
    res = _bass_fundamentals(f, p, Collector())
    freqs = sorted(pk["frequency_hz"] for pk in res["peaks"][:2])
    ok = (len(freqs) == 2 and abs(freqs[0] - 62.0) < 1.0 and abs(freqs[1] - 70.0) < 1.0)
    s.check("62 Hz and 70 Hz together are resolved as two peaks", ok, freqs,
            "[62.0, 70.0] within 1 Hz")


def t_brickwall(s: Suite) -> None:
    from .metrics.forensics import _find_cutoff, _smoothed_ltas
    from .audio import AudioSource

    rng = np.random.default_rng(1)
    x = rng.standard_normal(44100 * 8) * 0.1
    sos = sps.butter(16, 16000 / (44100 / 2), btype="low", output="sos")
    y = sps.sosfilt(sos, x)
    with tempfile.TemporaryDirectory() as d:
        p = _write(os.path.join(d, "brickwall.wav"),
                   np.repeat(y[:, None], 2, axis=1).astype(np.float32), 44100, "PCM_24")
        src = AudioSource(p, Collector())
        f, db = _smoothed_ltas(src.mono, src.sr, 16384)
        res = _find_cutoff(f, db, src.sr, 25.0)
    s.near("brickwall at 16 kHz -> detected cutoff", res.get("cutoff_hz"), 16000.0,
           200.0, " Hz")


def t_effective_bit_depth(s: Suite) -> None:
    from .metrics.forensics import _effective_bit_depth
    from .audio import AudioSource

    rng = np.random.default_rng(2)
    x = rng.standard_normal(44100 * 2) * 0.2
    q = np.round(x * (2 ** 15)) / (2 ** 15)  # 16-bit content
    with tempfile.TemporaryDirectory() as d:
        p24 = _write(os.path.join(d, "c24.wav"),
                     np.repeat(q[:, None], 2, axis=1), 44100, "PCM_24")
        col = Collector()
        r24 = _effective_bit_depth(AudioSource(p24, col), col)
        p16 = _write(os.path.join(d, "c16.wav"),
                     np.repeat(q[:, None], 2, axis=1), 44100, "PCM_16")
        r16 = _effective_bit_depth(AudioSource(p16, col), col)
    s.check("16-bit content in a 24-bit container -> effective bit depth 16",
            r24["effective_bits"] == 16 and r24["container_bits"] == 24,
            f"{r24['effective_bits']} effective of {r24['container_bits']} container",
            "16 of 24")
    s.check("the same content in a 16-bit container -> effective bit depth 16",
            r16["effective_bits"] == 16, f"{r16['effective_bits']}", "16")


def t_tempo(s: Suite) -> None:
    try:
        import librosa  # noqa: F401
    except ImportError:
        s.check("click track tempo", False, "librosa not installed",
                "librosa available")
        return
    from .metrics.structure import _tempo
    from .audio import AudioSource

    bpm = 120.0
    sr = 44100
    n = int(sr * 20)
    x = np.zeros(n)
    period = int(sr * 60.0 / bpm)
    click = np.exp(-np.arange(400) / 40.0) * np.sin(2 * np.pi * 1500 * np.arange(400) / sr)
    for i in range(0, n - 400, period):
        x[i : i + 400] += click
    with tempfile.TemporaryDirectory() as d:
        p = _write(os.path.join(d, "click.wav"),
                   np.repeat((x * 0.5)[:, None], 2, axis=1), sr, "PCM_16")
        import librosa as lb
        src = AudioSource(p, Collector())
        res = _tempo(src, lb, Collector())
    got = res.get("bpm")
    ok = got is not None and abs(got - bpm) / bpm <= 0.01
    s.check("120 BPM click track -> tempo within 1%", ok,
            f"{got:.3f} BPM" if got else "None", "120 BPM +/- 1.2")


def t_dr14(s: Suite) -> None:
    """DR of a continuous sine is analytically 0: block RMS equals the peak."""
    from .metrics.loudness import dr14
    from .audio import AudioSource

    with tempfile.TemporaryDirectory() as d:
        x = _sine(1000.0, 30.0, 0.5, sr=44100)
        p = _write(os.path.join(d, "sine.wav"),
                   np.repeat(x[:, None], 2, axis=1), 44100, "PCM_24")
        col = Collector()
        r = dr14(AudioSource(p, col), col)
    s.near("DR14 of a continuous sine (analytically 0)", r["dr_unrounded"], 0.0,
           0.1, " dB")
    s.check("DR14 validation status is reported as unverified",
            r["validation"]["validated_against_published_reference"] is False,
            r["validation"]["status"],
            "NOT VALIDATED against a published DR rating")


def t_midside(s: Suite) -> None:
    rng = np.random.default_rng(3)
    L = rng.standard_normal(1000)
    R = rng.standard_normal(1000)
    mid, side = (L + R) / 2, (L - R) / 2
    s.check("mid/side round-trips to L/R exactly",
            np.allclose(mid + side, L, atol=1e-12) and
            np.allclose(mid - side, R, atol=1e-12),
            "max error "
            f"{max(np.max(np.abs(mid + side - L)), np.max(np.abs(mid - side - R))):.2e}",
            "exact within 1e-12")


def t_short_file(s: Suite) -> None:
    """A file shorter than the analysis windows must degrade, not crash."""
    from .analyze import analyze_file

    with tempfile.TemporaryDirectory() as d:
        x = _sine(440.0, 1.5, 0.3, sr=44100)
        p = _write(os.path.join(d, "short.wav"),
                   np.repeat(x[:, None], 2, axis=1), 44100, "PCM_16")
        res = analyze_file(p, profile="quick")
    s.check("a 1.5 s file completes with warnings instead of crashing",
            res["headline"]["lra_lu"] is None and len(res["warnings"]) > 0,
            f"LRA={res['headline']['lra_lu']}, {len(res['warnings'])} warning(s)",
            "LRA is null and at least one warning is emitted")


TESTS: list[tuple[str, Callable[[Suite], None]]] = [
    ("K-weighting filter design", t_kweighting),
    ("sine loudness and peak conventions", t_sine_loudness),
    ("inter-sample peaks", t_intersample_peak),
    ("pruned vs full true-peak scan", t_pruned_scan_equivalence),
    ("single- vs multi-threaded true-peak scan", t_threaded_scan_equivalence),
    ("clipping below full scale (trap #1)", t_clipped_below_full_scale),
    ("independent stereo noise", t_noise_stereo),
    ("bass fundamental resolution", t_bass_fundamentals),
    ("brickwall cutoff detection", t_brickwall),
    ("effective bit depth", t_effective_bit_depth),
    ("tempo on a click track", t_tempo),
    ("DR14 on an analytic case", t_dr14),
    ("mid/side maths", t_midside),
    ("short-file degradation", t_short_file),
]


def run_selftest(verbose: bool = True) -> int:
    s = Suite(verbose=verbose)
    for name, fn in TESTS:
        if verbose:
            print(f"\n--- {name} ---")
        try:
            fn(s)
        except Exception as exc:  # a crashing test is a failing test
            s.check(f"{name} (raised)", False, repr(exc), "no exception")
    print(f"\n{s.passed} passed, {s.failed} failed")
    if s.notes:
        print("\nNotes:")
        for nte in s.notes:
            print(f"- {nte}")
    return 2 if s.failed else 0
