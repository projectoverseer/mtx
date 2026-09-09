"""Shared DSP primitives.

Everything here is deterministic and parameterised by explicit arguments; no
module-level state, no hidden defaults.  Each function documents the standard
or formula it implements so an output number can be traced back to a method.
"""

from __future__ import annotations

import math
from typing import Iterable, NamedTuple, Sequence

import numpy as np
from scipy import signal as sps

from .parallel import ordered_window
from .util import EPS, db_amp, db_pow

# ------------------------------------------------------------------ K-weighting
# ITU-R BS.1770-4 pre-filter, expressed as an analog prototype and bilinear
# transformed for the file's own sample rate (the tabulated coefficients in the
# recommendation are only valid at 48 kHz).
_SHELF = dict(G=3.999843853973347, Q=0.7071752369554196, fc=1681.974450955533)
_RLB = dict(G=0.0, Q=0.5003270373238773, fc=38.13547087602444)


def _shelf_coeffs(sr: float) -> tuple[np.ndarray, np.ndarray]:
    """Stage 1: the BS.1770 high-frequency shelf ("head effect" filter).

    Designed from the analog prototype with bilinear pre-warping, so it is
    correct at any sample rate; at 48 kHz it reproduces the coefficients
    tabulated in BS.1770-4 Table 1, which `mtx selftest` asserts.
    """
    G, Q, fc = _SHELF["G"], _SHELF["Q"], _SHELF["fc"]
    K = math.tan(math.pi * fc / sr)
    Vh = 10.0 ** (G / 20.0)
    Vb = Vh ** 0.4996667741545416
    a0 = 1.0 + K / Q + K * K
    b = np.array([
        (Vh + Vb * K / Q + K * K) / a0,
        2.0 * (K * K - Vh) / a0,
        (Vh - Vb * K / Q + K * K) / a0,
    ])
    a = np.array([
        1.0,
        2.0 * (K * K - 1.0) / a0,
        (1.0 - K / Q + K * K) / a0,
    ])
    return b, a


def _rlb_coeffs(sr: float) -> tuple[np.ndarray, np.ndarray]:
    """Stage 2: the BS.1770 RLB high-pass.  Unity gain in the pass band."""
    Q, fc = _RLB["Q"], _RLB["fc"]
    K = math.tan(math.pi * fc / sr)
    a0 = 1.0 + K / Q + K * K
    b = np.array([1.0, -2.0, 1.0])
    a = np.array([
        1.0,
        2.0 * (K * K - 1.0) / a0,
        (1.0 - K / Q + K * K) / a0,
    ])
    return b, a


def k_weight(x: np.ndarray, sr: float) -> np.ndarray:
    """Apply the BS.1770 K-weighting chain along axis 0.  Causal, as specified."""
    b1, a1 = _shelf_coeffs(sr)
    b2, a2 = _rlb_coeffs(sr)
    y = sps.lfilter(b1, a1, np.asarray(x, dtype=np.float64), axis=0)
    return sps.lfilter(b2, a2, y, axis=0)


def channel_weights(n_ch: int) -> np.ndarray:
    """BS.1770 channel weights G.  L/R/C = 1.0, surrounds = 1.41."""
    if n_ch <= 3:
        return np.ones(n_ch)
    w = np.ones(n_ch)
    w[3:] = 1.41
    return w


def block_loudness(x: np.ndarray, sr: float, block_s: float, hop_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-block loudness in LUFS.

    Returns (start_times_s, loudness_lufs).  x is (n, ch); the K-weighting is
    applied once over the whole signal, then mean-square is taken per block per
    channel and combined with the BS.1770 channel weights.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if x.ndim == 1:
        x = x[:, None]
    n, n_ch = x.shape
    bs = int(round(block_s * sr))
    hs = max(1, int(round(hop_s * sr)))
    if n < bs or bs <= 0:
        return np.zeros(0), np.zeros(0)
    y = k_weight(x, sr)
    starts = np.arange(0, n - bs + 1, hs)
    G = channel_weights(n_ch)
    # Cumulative sum of squares gives every block mean-square in one pass.
    cs = np.concatenate([np.zeros((1, n_ch)), np.cumsum(y * y, axis=0)], axis=0)
    z = (cs[starts + bs] - cs[starts]) / float(bs)  # (n_blocks, ch)
    power = np.maximum(z @ G, EPS)
    return starts / float(sr), -0.691 + 10.0 * np.log10(power)


def gated_integrated(loudness: np.ndarray, abs_gate: float = -70.0,
                     rel_gate_lu: float = -10.0) -> tuple[float | None, np.ndarray]:
    """BS.1770 two-stage gating.  Returns (integrated LUFS, gated mask)."""
    if loudness.size == 0:
        return None, np.zeros(0, dtype=bool)
    p = np.power(10.0, (loudness + 0.691) / 10.0)  # back to weighted power
    m1 = loudness > abs_gate
    if not np.any(m1):
        return None, m1
    rel = -0.691 + 10.0 * np.log10(np.mean(p[m1])) + rel_gate_lu
    m2 = m1 & (loudness > rel)
    if not np.any(m2):
        return None, m2
    return float(-0.691 + 10.0 * np.log10(np.mean(p[m2]))), m2


def loudness_range(shortterm: np.ndarray, abs_gate: float = -70.0,
                   rel_gate_lu: float = -20.0,
                   pcts: Sequence[float] = (10.0, 95.0)) -> float | None:
    """EBU Tech 3342 LRA from a short-term loudness series."""
    if shortterm.size == 0:
        return None
    p = np.power(10.0, (shortterm + 0.691) / 10.0)
    m1 = shortterm > abs_gate
    if not np.any(m1):
        return None
    rel = -0.691 + 10.0 * np.log10(np.mean(p[m1])) + rel_gate_lu
    kept = shortterm[m1 & (shortterm > rel)]
    if kept.size < 2:
        return None
    lo, hi = np.percentile(kept, pcts)
    return float(hi - lo)


# ------------------------------------------------------------------- true peak
_FILTER_CACHE: dict[int, tuple[np.ndarray, int, float]] = {}


def interpolation_filter(oversample: int) -> tuple[np.ndarray, int, float]:
    """The exact FIR `scipy.signal.resample_poly` builds, plus its L1 bound.

    Returns (filter, half_length, phase_gain).  `phase_gain` is
    max over phases of sum|h_phase|: no interpolated sample can exceed the
    largest input sample in its support times this number.  That inequality is
    what makes the candidate pruning below exact rather than a heuristic.
    """
    if oversample not in _FILTER_CACHE:
        half_len = 10 * oversample
        h_raw = sps.firwin(2 * half_len + 1, 1.0 / oversample,
                           window=("kaiser", 5.0))
        h = h_raw * oversample
        gain = max(float(np.abs(h[p::oversample]).sum()) for p in range(oversample))
        _FILTER_CACHE[oversample] = (h_raw, half_len, gain)
    return _FILTER_CACHE[oversample]


def _abs_max_axis(a: np.ndarray, axis: int) -> np.ndarray:
    """max(|a|) without materialising |a| (two reductions, one allocation)."""
    return np.maximum(a.max(axis=axis), -a.min(axis=axis))


class _Chunk(NamedTuple):
    """Everything one oversampled chunk contributes, with no shared state.

    Keeping this pure is what lets the oversampling run on a thread pool:
    `upfirdn` releases the GIL, so the expensive half scales, while the fold
    back into the running scan stays serial and in chunk order -- which is
    what an excursion straddling a chunk boundary needs to be counted once.
    """

    peaks: np.ndarray          # per-channel peak within this chunk
    top: float                 # highest interpolated magnitude, any channel
    top_index: int             # its position, in oversampled samples
    edges: list[int]           # rising edges above each threshold, internal
    first: list[bool]          # whether the chunk opens above each threshold
    last: list[bool]           # whether it closes above each threshold
    env: np.ndarray | None     # the max-envelope, when the caller wants it


def _scan_chunk(x: np.ndarray, oversample: int, lo: int, hi: int,
                drop_head: int, keep_len: int, thr_lin: list[float],
                keep_env: bool) -> _Chunk | None:
    """Oversample one chunk and reduce it to what the scan state needs.

    Channels are oversampled one at a time.  A reduction across the short axis
    of an (n, 2) array is an order of magnitude slower than the same reduction
    over two contiguous vectors, and at 16x this array has tens of millions of
    rows.
    """
    seg = x[lo:hi]
    a = drop_head * oversample
    b = a + keep_len * oversample
    env: np.ndarray | None = None
    peaks = np.zeros(seg.shape[1])
    for c in range(seg.shape[1]):
        col = np.ascontiguousarray(seg[:, c])
        up = sps.resample_poly(col, oversample, 1, window=("kaiser", 5.0))
        block = up[a:b]
        if block.size == 0:
            return None
        au = np.abs(block)
        peaks[c] = float(au.max())
        env = au if env is None else np.maximum(env, au, out=env)
    if env is None or env.size == 0:
        return None
    k = int(np.argmax(env))
    top = float(env[k])
    edges: list[int] = []
    first: list[bool] = []
    last: list[bool] = []
    for t in thr_lin:
        if top <= t:
            # Nothing in this chunk reaches the threshold; no scan needed.
            edges.append(0)
            first.append(False)
            last.append(False)
            continue
        above = env > t
        # An "over" is one contiguous excursion, not one sample.  Counting the
        # 0->1 transitions directly costs one pass over a boolean array; going
        # via diff() on an int8 copy costs three, over tens of millions of
        # samples per chunk.
        edges.append(int(np.count_nonzero(above[1:] & ~above[:-1])))
        first.append(bool(above[0]))
        last.append(bool(above[-1]))
    return _Chunk(peaks, top, k, edges, first, last, env if keep_env else None)


def _fold_chunk(state: dict, ch: _Chunk, start_frame: int, sr: float,
                oversample: int) -> None:
    """Merge one chunk's contribution into the running scan.  Order matters."""
    np.maximum(state["peaks"], ch.peaks, out=state["peaks"])
    if ch.top > state["best"]:
        state["best"] = ch.top
        state["best_time"] = (start_frame + ch.top_index / oversample) / float(sr)
    for j in range(len(ch.edges)):
        state["counts"][j] += ch.edges[j]
        if ch.first[j] and not state["prev_above"][j]:
            state["counts"][j] += 1
        state["prev_above"][j] = ch.last[j]


def true_peak_scan(x: np.ndarray, sr: float, oversample: int,
                   thresholds_dbtp: Sequence[float] = (),
                   env_hop_s: float | None = 0.001,
                   chunk: int = 1 << 18, overlap: int = 4096,
                   workers: int = 1) -> dict:
    """One oversampling pass; everything the true-peak metrics need.

    Returns the per-channel peak, the time of the overall peak, and the count of
    contiguous excursions above each threshold.  With `env_hop_s` set it also
    returns a max-envelope of the reconstructed waveform at that resolution,
    which is what makes the PSR timeline free: a short-term true peak is a
    rolling maximum over it, so the file is never oversampled twice.

    With `env_hop_s=None` no envelope is needed, and the scan runs only over the
    stretches that can possibly matter.  The pruning is exact, not a heuristic:
    an interpolated sample is a weighted sum of the input samples in its
    support, so it cannot exceed the largest of them times the filter's
    per-phase L1 gain.  Any stretch whose bound falls below both the file's own
    sample peak (a lower bound on the true peak) and the lowest reporting
    threshold cannot contain the maximum or an over, and is skipped.

    Oversampling runs in float32.  The quantity being measured is a peak near
    unity and float32 carries about seven significant digits there, four orders
    of magnitude finer than the 0.01 dB the value is reported to.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    n, n_ch = x.shape
    thr_lin = [10.0 ** (t / 20.0) for t in thresholds_dbtp]
    empty = {"peak_per_channel": np.zeros(n_ch), "peak": 0.0, "peak_time_s": None,
             "over_counts": [0] * len(thr_lin), "envelope": np.zeros(0),
             "envelope_hop_s": env_hop_s, "oversample": oversample,
             "pruned": False, "scanned_fraction": 0.0}
    if n == 0:
        return empty
    _, half_len, phase_gain = interpolation_filter(oversample)
    support = half_len // oversample + 2
    state = {"peaks": np.zeros(n_ch), "best": 0.0, "best_time": None,
             "counts": [0] * len(thr_lin), "prev_above": [False] * len(thr_lin)}

    def _run(task: tuple[int, int, int, int, int, int]) -> _Chunk | None:
        lo, hi, drop_head, keep_len, _start, _run_idx = task
        return _scan_chunk(x, oversample, lo, hi, drop_head, keep_len,
                           thr_lin, env_hop_s is not None)

    if env_hop_s is not None:
        hop_out = max(1, int(round(env_hop_s * sr * oversample)))
        env_parts: list[np.ndarray] = []
        carry = np.zeros(0, dtype=np.float32)
        tasks: list[tuple[int, int, int, int, int, int]] = []
        i = 0
        while i < n:
            lo = max(0, i - overlap)
            hi = min(n, i + chunk + overlap)
            tasks.append((lo, hi, i - lo, min(i + chunk, n) - i, i, 0))
            i += chunk
        for res, task in zip(ordered_window(_run, tasks, workers), tasks):
            if res is None:
                continue
            _fold_chunk(state, res, task[4], sr, oversample)
            env = res.env
            joined = np.concatenate([carry, env]) if carry.size else env
            m = (joined.size // hop_out) * hop_out
            if m:
                env_parts.append(joined[:m].reshape(-1, hop_out).max(axis=1))
            carry = joined[m:]
        if carry.size:
            env_parts.append(np.array([carry.max()]))
        envelope = np.concatenate(env_parts) if env_parts else np.zeros(0)
        scanned = 1.0
        pruned = False
    else:
        envelope = np.zeros(0)
        pruned = True
        # Per-channel sample peak is a lower bound on that channel's true peak.
        sample_peak_ch = _abs_max_axis(x, 0)
        floor_ch = sample_peak_ch.astype(np.float64)
        if thr_lin:
            floor_ch = np.minimum(floor_ch, min(thr_lin))
        block = 4096
        n_blocks = int(np.ceil(n / block))
        pad = n_blocks * block - n
        padded = np.pad(x, ((0, pad), (0, 0)), constant_values=0.0)
        bm = _abs_max_axis(padded.reshape(n_blocks, block, n_ch), 1)  # (blocks, ch)
        # A block's output also sees `support` samples of its neighbours.
        if n_blocks > 1:
            bm = np.maximum.reduce([bm,
                                    np.vstack([bm[1:], bm[-1:]]),
                                    np.vstack([bm[:1], bm[:-1]])])
        cand = np.any(bm * phase_gain >= floor_ch[None, :], axis=1)
        runs = []
        k = 0
        while k < n_blocks:
            if not cand[k]:
                k += 1
                continue
            j = k
            while j + 1 < n_blocks and cand[j + 1]:
                j += 1
            runs.append((k * block, min((j + 1) * block, n)))
            k = j + 1
        scanned_samples = 0
        tasks = []
        for r_idx, (s, e) in enumerate(runs):
            for i in range(s, e, chunk):
                seg_end = min(i + chunk, e)
                a = max(0, i - support - 8)
                b = min(n, seg_end + support + 8)
                tasks.append((a, b, i - a, seg_end - i, i, r_idx))
                scanned_samples += seg_end - i
        current_run = 0
        for res, task in zip(ordered_window(_run, tasks, workers), tasks):
            if task[5] != current_run:
                # Nothing between runs can be above any threshold, so an
                # excursion never spans a gap.
                state["prev_above"] = [False] * len(thr_lin)
                current_run = task[5]
            if res is None:
                continue
            _fold_chunk(state, res, task[4], sr, oversample)
        scanned = scanned_samples / float(n)
    return {
        "peak_per_channel": state["peaks"].astype(np.float64),
        "peak": state["best"],
        "peak_time_s": state["best_time"],
        "over_counts": state["counts"],
        "envelope": envelope.astype(np.float64),
        "envelope_hop_s": env_hop_s,
        "oversample": oversample,
        "pruned": pruned,
        "scanned_fraction": float(scanned),
        "phase_gain_bound_db": 20.0 * math.log10(phase_gain),
    }


def rolling_max(x: np.ndarray, window: int, hop: int) -> np.ndarray:
    """Maximum of `x` over each window, taken every `hop` samples."""
    x = np.ascontiguousarray(x)
    if window <= 0 or hop <= 0 or x.size < window:
        return np.zeros(0)
    n_win = 1 + (x.size - window) // hop
    out = np.empty(n_win)
    # Chunked so a long file never builds one huge (n_windows, window) view.
    block = max(1, 1_000_000 // window)
    for s in range(0, n_win, block):
        cnt = min(block, n_win - s)
        base = x[s * hop : s * hop + (cnt - 1) * hop + window]
        view = np.lib.stride_tricks.as_strided(
            base, shape=(cnt, window),
            strides=(base.strides[0] * hop, base.strides[0]), writeable=False)
        out[s : s + cnt] = view.max(axis=1)
    return out


def true_peak(x: np.ndarray, oversample: int, chunk: int = 1 << 19,
              overlap: int = 4096) -> np.ndarray:
    """Peak of the `oversample`x reconstructed waveform, per channel.

    Chunked with overlap so a long file never materialises an oversampled copy
    of itself: peak memory is chunk * oversample samples, not n * oversample.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if x.ndim == 1:
        x = x[:, None]
    n, n_ch = x.shape
    if n == 0:
        return np.zeros(n_ch)
    peaks = np.zeros(n_ch)
    i = 0
    while i < n:
        lo = max(0, i - overlap)
        hi = min(n, i + chunk + overlap)
        seg = x[lo:hi]
        up = sps.resample_poly(seg, oversample, 1, axis=0, window=("kaiser", 5.0))
        # Discard the region contributed by the overlap padding.
        a = (i - lo) * oversample
        b = a + (min(i + chunk, n) - i) * oversample
        block = up[a:b]
        if block.size:
            peaks = np.maximum(peaks, np.max(np.abs(block), axis=0))
        i += chunk
    return peaks


# ------------------------------------------------------------- band splitting
def band_filter(x: np.ndarray, sr: float, lo: float, hi: float,
                order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass.  Edges are clamped to the Nyquist."""
    nyq = sr / 2.0
    lo_c = max(float(lo), 0.1)
    hi_c = min(float(hi), nyq * 0.99)
    x = np.asarray(x, dtype=np.float64)
    if hi_c <= lo_c:
        return np.zeros_like(x)
    if lo_c <= 0.5:
        sos = sps.butter(order, hi_c / nyq, btype="low", output="sos")
    elif hi_c >= nyq * 0.985:
        sos = sps.butter(order, lo_c / nyq, btype="high", output="sos")
    else:
        sos = sps.butter(order, [lo_c / nyq, hi_c / nyq], btype="band", output="sos")
    pad = min(3 * (sos.shape[0] * 2), max(0, x.shape[0] - 1))
    return sps.sosfiltfilt(sos, x, axis=0, padlen=pad)


# -------------------------------------------------------------------- spectra
def welch_psd(x: np.ndarray, sr: float, nperseg: int,
              overlap_pct: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD, Hann window.  Falls back to the longest usable segment."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n == 0:
        return np.zeros(0), np.zeros(0)
    nps = int(min(nperseg, n))
    if nps < 16:
        return np.zeros(0), np.zeros(0)
    nov = int(nps * overlap_pct / 100.0)
    f, p = sps.welch(x, fs=sr, window="hann", nperseg=nps, noverlap=nov,
                     detrend=False, scaling="density", average="mean")
    return f, p


def band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    """Integrate a PSD between two frequencies (trapezoid over the bin grid)."""
    if freqs.size == 0:
        return 0.0
    m = (freqs >= lo) & (freqs < hi)
    if not np.any(m):
        return 0.0
    return float(np.trapezoid(psd[m], freqs[m])) if m.sum() > 1 else float(psd[m][0])


def third_octave_edges(centres: Iterable[float]) -> list[tuple[float, float, float]]:
    """(centre, low_edge, high_edge) for each third-octave centre."""
    k = 2 ** (1.0 / 6.0)
    return [(float(c), float(c) / k, float(c) * k) for c in centres]


def octave_band_edges(centres: Iterable[float]) -> list[tuple[float, float, float]]:
    k = 2 ** 0.5
    return [(float(c), float(c) / k, float(c) * k) for c in centres]


def spectrum_table(freqs: np.ndarray, psd: np.ndarray,
                   edges: Sequence[tuple[float, float, float]],
                   nyquist: float) -> list[dict]:
    """Band powers for a set of (centre, lo, hi) edges, in dB and percent."""
    rows = []
    total = float(np.trapezoid(psd, freqs)) if freqs.size > 1 else 0.0
    for centre, lo, hi in edges:
        if lo >= nyquist:
            rows.append({"centre_hz": centre, "low_hz": round(lo, 1),
                         "high_hz": round(hi, 1), "power": None,
                         "db": None, "pct": None,
                         "note": "band above Nyquist"})
            continue
        p = band_power(freqs, psd, lo, min(hi, nyquist))
        rows.append({"centre_hz": centre, "low_hz": round(lo, 1),
                     "high_hz": round(hi, 1), "power": p,
                     "db": db_pow(p) if p > 0 else None,
                     "pct": (100.0 * p / total) if total > 0 else None})
    return rows


def linear_fit_db_per_octave(freqs: np.ndarray, db: np.ndarray,
                             lo: float, hi: float) -> tuple[float | None, float | None]:
    """Least-squares slope in dB/octave over [lo, hi], with the fit R^2."""
    m = (freqs >= lo) & (freqs <= hi) & np.isfinite(db) & (freqs > 0)
    if m.sum() < 8:
        return None, None
    xo = np.log2(freqs[m])
    y = db[m]
    A = np.vstack([xo, np.ones_like(xo)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return float(coef[0]), r2


def log_smooth_indices(freqs: np.ndarray, octaves: float) -> tuple[np.ndarray, np.ndarray]:
    """Window start/stop bin for a constant-octave running mean."""
    k = 2 ** (octaves / 2.0)
    i_lo = np.searchsorted(freqs, freqs / k, side="left")
    i_hi = np.searchsorted(freqs, freqs * k, side="right")
    return i_lo, np.maximum(i_hi, i_lo + 1)


def log_smooth(freqs: np.ndarray, db: np.ndarray, octaves: float,
               idx: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Running mean of constant width in octaves, along the first axis.

    Pass `idx` from `log_smooth_indices` to smooth many spectra that share one
    frequency grid without recomputing the window bounds each time.
    """
    i_lo, i_hi = idx if idx is not None else log_smooth_indices(freqs, octaves)
    db = np.asarray(db)
    pad = ((1, 0),) + ((0, 0),) * (db.ndim - 1)
    cs = np.cumsum(np.pad(db, pad), axis=0)
    cnt = (i_hi - i_lo).astype(np.float64)
    if db.ndim > 1:
        cnt = cnt.reshape(-1, *([1] * (db.ndim - 1)))
    return (cs[i_hi] - cs[i_lo]) / np.maximum(cnt, 1)


def correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson correlation of two equal-length signals."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return None
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    if den <= 0:
        return None
    return float(np.dot(a, b) / den)


def crest_db(x: np.ndarray) -> float | None:
    """Sample peak minus RMS, in dB."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return None
    pk = float(np.max(np.abs(x)))
    r = float(np.sqrt(np.mean(x * x)))
    if pk <= 0 or r <= 0:
        return None
    return db_amp(pk) - db_amp(r)
