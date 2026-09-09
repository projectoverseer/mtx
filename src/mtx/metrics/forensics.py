"""4.2 Source forensics.

Whether the "lossless" file has a lossy or otherwise compromised ancestor is
settled here, before any tonal conclusion is drawn from it elsewhere.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal as sps

from ..audio import AudioSource
from ..dsp import (band_power, linear_fit_db_per_octave, log_smooth,
                   spectrum_table, third_octave_edges, welch_psd)
from ..params import CODEC_SHELVES_HZ, PARAMS, THIRD_OCTAVE_CENTRES
from ..util import Collector, db_amp, db_pow


SMOOTHING_OCTAVES = 1.0 / 12.0


def _smoothed_ltas(x: np.ndarray, sr: int, nfft: int) -> tuple[np.ndarray, np.ndarray]:
    """Welch LTAS smoothed over a constant width in octaves.

    1/12 octave, not wider: a wide log average straddles a brickwall knee and
    drags the curve down before the filter actually starts, which moves the
    reported cutoff several hundred Hz low.
    """
    f, p = welch_psd(x, sr, nfft)
    if f.size == 0:
        return f, p
    db = db_pow(np.maximum(p, 1e-30))
    return f, log_smooth(f, db, SMOOTHING_OCTAVES)


STEEP_SLOPE_DB_PER_OCT = -30.0
SEARCH_START_HZ = 8000.0
KNEE_TOLERANCE_DB = 3.0


def _loglin_fit(f: np.ndarray, db: np.ndarray, lo: float,
                hi: float) -> tuple[float, float] | None:
    """Least-squares line through (log2 f, dB) over [lo, hi]: (slope, intercept)."""
    m = (f >= lo) & (f <= hi) & (f > 0) & np.isfinite(db)
    if m.sum() < 8:
        return None
    x = np.log2(f[m])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, db[m], rcond=None)
    return float(coef[0]), float(coef[1])


def _find_cutoff(f: np.ndarray, db: np.ndarray, sr: int,
                 collapse_db: float) -> dict[str, Any]:
    """The knee of the high-frequency collapse, not its -N dB point.

    The reported cutoff is where the spectrum *starts* to fall away steeply --
    that is the frequency a codec's low-pass was set to, and it is what the
    shelf match needs.  A knee only counts if the level above it is at least
    `collapse_db` below the level at the knee, so a gently sloping but
    full-bandwidth spectrum is reported as full bandwidth rather than as a
    cutoff at some arbitrary frequency.
    """
    nyq = sr / 2.0
    ref_m = (f >= 1000.0) & (f <= min(5000.0, nyq * 0.9))
    ref = float(np.median(db[ref_m])) if np.any(ref_m) else None
    search = (f >= SEARCH_START_HZ) & (f < nyq * 0.999)
    if search.sum() < 16:
        return {"cutoff_hz": None, "reference_db": ref,
                "reason": "no usable search band below Nyquist",
                "full_bandwidth": None}

    # Local slope in dB/octave, over a half-span of 1/8 octave.
    lg = np.log2(np.maximum(f, 1e-9))
    idx = np.flatnonzero(search)
    span = 1.0 / 8.0
    lo_i = np.searchsorted(lg, lg[idx] - span)
    hi_i = np.minimum(np.searchsorted(lg, lg[idx] + span), len(f) - 1)
    dlg = lg[hi_i] - lg[lo_i]
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = np.where(dlg > 0, (db[hi_i] - db[lo_i]) / dlg, np.nan)

    fs = f[idx]
    steep = np.flatnonzero(slopes < STEEP_SLOPE_DB_PER_OCT)
    f_steep = None
    for k in steep:
        fk = float(fs[k])
        above = f > min(fk * 1.15, nyq * 0.999)
        if not np.any(above):
            continue
        if (db[idx[k]] - float(np.median(db[above]))) >= collapse_db:
            f_steep = fk
            break

    knee = None
    trend = None
    if f_steep is not None:
        # The slope window straddles the knee, so f_steep sits below it.  Fit the
        # spectrum's own trend from the region safely below the transition and
        # call the knee the point where the curve departs from that trend.
        fit_hi = f_steep / 1.3
        fit_lo = max(2000.0, f_steep / 4.0)
        if fit_hi <= fit_lo * 1.2:
            fit_lo = max(1000.0, fit_hi / 2.0)
        trend = _loglin_fit(f, db, fit_lo, fit_hi)
        if trend is not None:
            slope_t, icept = trend
            above_fit = np.flatnonzero((f >= fit_hi) & (f < nyq * 0.999))
            hold = max(2, int(np.searchsorted(lg, lg[above_fit[0]] + np.log2(2 ** (1 / 12)))
                              - above_fit[0])) if above_fit.size else 2
            pred = slope_t * lg + icept
            departed = db < (pred - KNEE_TOLERANCE_DB)
            for i in above_fit:
                if departed[i] and departed[i : i + hold].all():
                    knee = float(f[i])
                    break
        if knee is None:
            knee = f_steep
    if knee is None:
        return {"cutoff_hz": None, "reference_db": ref,
                "reason": "no steep collapse found between "
                          f"{SEARCH_START_HZ:.0f} Hz and Nyquist; the spectrum "
                          "runs to the top of the band",
                "full_bandwidth": True}

    tail = f > knee
    tail_level = float(np.median(db[tail])) if np.any(tail) else None
    slope, r2 = linear_fit_db_per_octave(f, db, knee, min(knee * 1.3, nyq * 0.99))
    if slope is None:
        # A brickwall can be narrower than the fit window; measure it directly.
        hi_f = min(knee * 1.1, nyq * 0.999)
        m = (f >= knee) & (f <= hi_f)
        if m.sum() >= 2 and hi_f > knee:
            slope = float((db[m][-1] - db[m][0]) / np.log2(hi_f / knee))
            r2 = None
    knee_level = float(db[np.argmin(np.abs(f - knee))])
    return {
        "cutoff_hz": round(knee, 1),
        "reference_db": ref,
        "level_at_cutoff_db": knee_level,
        "level_above_cutoff_db": tail_level,
        "collapse_depth_db": (knee_level - tail_level) if tail_level is not None else None,
        "rolloff_slope_db_per_oct": slope,
        "rolloff_fit_r2": r2,
        "steep_slope_threshold_db_per_oct": STEEP_SLOPE_DB_PER_OCT,
        "steep_region_start_hz": round(f_steep, 1) if f_steep else None,
        "trend_slope_db_per_oct": trend[0] if trend else None,
        "knee_tolerance_db": KNEE_TOLERANCE_DB,
        "definition": "the spectrum's own trend is fitted below the transition; the "
                      "cutoff is where the 1/12-octave-smoothed curve first departs "
                      f"from that trend by {KNEE_TOLERANCE_DB} dB and stays below "
                      "it, given a collapse of at least "
                      f"{collapse_db} dB above the knee",
        "full_bandwidth": False,
    }


def _frame_cutoffs(src: AudioSource, collapse_db: float, frame_s: float,
                   nfft: int, whole_cutoff_hz: float | None = None) -> dict[str, Any]:
    # Native rate throughout: resampling to the band rate would introduce a
    # rolloff of mtx's own making and it would be reported as a cutoff.
    sr = src.sr
    x = src.mono
    w = int(round(frame_s * sr))
    if x.size < w or w < nfft:
        return {"times_s": [], "cutoff_hz": [], "std_hz": None,
                "frames_above_cutoff_below_floor": None,
                "reason": "file shorter than one cutoff frame"}
    times, cuts = [], []
    hits = tot = 0
    for i in range(x.size // w):
        seg = x[i * w : (i + 1) * w]
        if float(np.sqrt(np.mean(seg * seg))) < 1e-6:
            continue  # near-silent frame carries no cutoff information
        f_raw, p_raw = welch_psd(seg, sr, nfft)
        if f_raw.size == 0:
            continue
        db = log_smooth(f_raw, db_pow(np.maximum(p_raw, 1e-30)), SMOOTHING_OCTAVES)
        res = _find_cutoff(f_raw, db, sr, collapse_db)
        times.append(i * frame_s)
        cuts.append(res.get("cutoff_hz"))
        # Same frame, same spectrum: how far the above-cutoff band sits below
        # the 1-5 kHz reference.
        if whole_cutoff_hz:
            ref = band_power(f_raw, p_raw, 1000.0, min(5000.0, sr / 2.0))
            tail = band_power(f_raw, p_raw, whole_cutoff_hz, sr / 2.0 * 0.99)
            if ref > 0:
                tot += 1
                if db_pow(max(tail, 1e-30) / ref) < -collapse_db:
                    hits += 1
    vals = np.array([c for c in cuts if c is not None], dtype=float)
    return {
        "frames_above_cutoff_below_floor": (hits / tot) if tot else None,
        "frame_s": frame_s,
        "times_s": times,
        "cutoff_hz": cuts,
        "frames_measured": len(cuts),
        "frames_full_bandwidth": int(sum(1 for c in cuts if c is None)),
        "mean_hz": float(np.mean(vals)) if vals.size else None,
        "std_hz": float(np.std(vals)) if vals.size else None,
        "min_hz": float(np.min(vals)) if vals.size else None,
        "max_hz": float(np.max(vals)) if vals.size else None,
        "note": "a cutoff that moves frame to frame is joint-stereo/VBR behaviour; "
                "a rock-steady one is a fixed filter or CBR",
    }


def _spectral_holes(f: np.ndarray, db: np.ndarray, min_depth: float,
                    octaves: float) -> list[dict[str, Any]]:
    if f.size < 32:
        return []
    smooth = log_smooth(f, db, octaves * 2)
    deficit = smooth - db
    idx, props = sps.find_peaks(deficit, prominence=min_depth)
    rows = []
    for i, p in zip(idx, props["prominences"]):
        if f[i] < 30.0:
            continue
        rows.append({"centre_hz": round(float(f[i]), 1),
                     "depth_db": float(deficit[i]),
                     "prominence_db": float(p)})
    rows.sort(key=lambda r: -r["depth_db"])
    return rows[:20]


def _effective_bit_depth(src: AudioSource, collector: Collector) -> dict[str, Any]:
    ints = src.int_samples()
    if ints is None:
        collector.warn("forensics.effective_bit_depth",
                       f"subtype {src.subtype} is not integer PCM; effective bit "
                       "depth is undefined for float content")
        return {"effective_bits": None, "container_bits": None,
                "reason": f"non-integer subtype {src.subtype}"}
    v = ints.reshape(-1)
    nz = v[v != 0]
    if nz.size == 0:
        collector.warn("forensics.effective_bit_depth", "all samples are zero")
        return {"effective_bits": 0, "container_bits": None,
                "reason": "file is digital silence"}
    # |v| & -|v| isolates the lowest set bit.  int64 first, because negating
    # the most negative int32 overflows; and once, not twice -- the cast is
    # over every non-zero sample in the file.
    mag = np.abs(nz.astype(np.int64))
    lsb = mag & -mag
    tz = np.log2(lsb).astype(np.int32)
    sig_bits = 32 - tz
    hist = np.bincount(np.clip(sig_bits, 0, 32), minlength=33)
    container = {"PCM_S8": 8, "PCM_U8": 8, "PCM_16": 16,
                 "PCM_24": 24, "PCM_32": 32}.get(src.subtype)
    eff = int(sig_bits.max())
    return {
        "effective_bits": eff,
        "container_bits": container,
        "method": "32 - (trailing zero bits of the left-justified int32 sample), "
                  "maximum over all non-zero samples",
        "significant_bit_histogram": {str(i): int(c) for i, c in enumerate(hist) if c},
        "nonzero_samples": int(nz.size),
        "container_holds_fewer_bits_than_it_offers":
            bool(container is not None and eff < container),
    }


def _noise_floor(src: AudioSource, collector: Collector) -> dict[str, Any]:
    P = PARAMS["forensics"]["noise_floor"]
    sr = src.sr
    x = src.mono
    w = max(1, int(round(P["frame_ms"] / 1000.0 * sr)))
    n = (x.size // w) * w
    if n < w * 20:
        collector.warn("forensics.noise_floor",
                       "fewer than 20 analysis frames; noise floor is null")
        return {"available": False, "reason": "file too short"}
    fr = x[:n].reshape(-1, w)
    rms = np.sqrt(np.mean(fr * fr, axis=1))
    k = max(1, int(round(P["quietest_fraction"] * rms.size)))
    order = np.argsort(rms)[:k]
    quiet = fr[order].reshape(-1)
    level = float(np.sqrt(np.mean(quiet * quiet)))
    f, p = welch_psd(quiet, sr, min(8192, quiet.size))
    rows = []
    slope = None
    if f.size:
        rows = spectrum_table(f, p, third_octave_edges(THIRD_OCTAVE_CENTRES), sr / 2.0)
        db = db_pow(np.maximum(p, 1e-30))
        slope, _ = linear_fit_db_per_octave(f, db, 10000.0, min(20000.0, sr / 2.0 * 0.98))
    return {
        "available": True,
        "quietest_fraction": P["quietest_fraction"],
        "frames_used": int(k),
        "frame_ms": P["frame_ms"],
        "level_dbfs": db_amp(level) if level > 0 else None,
        "third_octave": rows,
        "slope_above_10k_db_per_oct": slope,
        "interpretation_note":
            "a floor rising above 10 kHz is consistent with noise-shaped dither; "
            "a flat floor with TPDF; an unusually low one with truncation or a "
            "fade to digital black. The slope and level are reported so the "
            "reading can be checked.",
        "confidence": "medium",
        "confidence_reason": "the quietest frames of a dense master may still "
                             "contain programme material, not only the dither floor",
    }


def _silence(src: AudioSource, collector: Collector) -> dict[str, Any]:
    P = PARAMS["forensics"]["silence"]
    sr = src.sr
    x = np.abs(src.mono)
    if x.size == 0:
        return {"available": False, "reason": "empty file"}
    thr = 10.0 ** (P["digital_black_dbfs"] / 20.0)
    nz = np.flatnonzero(x > thr)
    if nz.size == 0:
        return {"available": True, "leading_black_ms": 1000.0 * x.size / sr,
                "trailing_black_ms": 1000.0 * x.size / sr,
                "note": "no sample above the digital-black threshold"}
    lead = int(nz[0])
    trail = int(x.size - 1 - nz[-1])
    # Fade shape from the 10 ms RMS envelope at each end.
    w = max(1, int(round(0.010 * sr)))
    n = (x.size // w) * w
    env = np.sqrt(np.mean(src.mono[:n].reshape(-1, w) ** 2, axis=1))
    env_db = db_amp(np.maximum(env, 1e-20))
    track_db = db_amp(float(np.sqrt(np.mean(src.mono ** 2))))

    def _fade(seq_db: np.ndarray) -> tuple[float | None, str]:
        hi = track_db - 6.0
        lo = track_db - PARAMS["forensics"]["silence"]["fade_detect_db"]
        i_lo = np.flatnonzero(seq_db > lo)
        i_hi = np.flatnonzero(seq_db > hi)
        if i_lo.size == 0 or i_hi.size == 0:
            return None, "indeterminate"
        length_ms = float((i_hi[0] - i_lo[0]) * 10.0)
        return length_ms, ("hard cut" if length_ms <= 20.0 else "fade")

    fade_in_ms, start_kind = _fade(env_db)
    fade_out_ms, end_kind = _fade(env_db[::-1])
    return {
        "available": True,
        "digital_black_threshold_dbfs": P["digital_black_dbfs"],
        "leading_black_ms": 1000.0 * lead / sr,
        "trailing_black_ms": 1000.0 * trail / sr,
        "start_kind": start_kind, "fade_in_ms": fade_in_ms,
        "end_kind": end_kind, "fade_out_ms": fade_out_ms,
        "fade_rule": f"time from {PARAMS['forensics']['silence']['fade_detect_db']} dB "
                     "below the track RMS to 6 dB below it, on a 10 ms envelope; "
                     "20 ms or less is called a hard cut",
    }


def _upsampling(src: AudioSource, cutoff: dict[str, Any],
                f: np.ndarray, db: np.ndarray) -> dict[str, Any]:
    if src.sr <= 48000:
        return {"checked": False,
                "reason": "file is at or below 48 kHz; no upsampling check applies"}
    c = cutoff.get("cutoff_hz")
    candidates = {"44100": 22050.0, "48000": 24000.0, "88200": 44100.0, "96000": 48000.0}
    best, best_d = None, None
    if c is not None:
        for name, hz in candidates.items():
            d = abs(c - hz)
            if best_d is None or d < best_d:
                best, best_d = name, d
    # Mirrored imaging: correlate the spectrum just below the cutoff against the
    # spectrum just above it, reflected.  Real content is not a mirror image.
    mirror_corr = None
    if c is not None and c > 0:
        lo = (f > c * 0.7) & (f < c)
        hi = (f > c) & (f < min(c * 1.3, src.sr / 2.0 * 0.99))
        k = min(int(lo.sum()), int(hi.sum()))
        if k >= 16:
            a = db[lo][-k:][::-1]
            b = db[hi][:k]
            if np.std(a) > 0 and np.std(b) > 0:
                mirror_corr = float(np.corrcoef(a, b)[0, 1])
    detected = bool(c is not None and best_d is not None and best_d < 500.0)
    return {
        "checked": True,
        "cutoff_hz": c,
        "nearest_original_rate_hz": int(best) if best else None,
        "distance_hz": best_d,
        "mirror_correlation": mirror_corr,
        "suspected_upsampled": detected,
        "confidence": "medium" if detected else "low",
        "confidence_reason": "a cutoff near half the file's rate is also what a "
                             "deliberate anti-alias filter produces; the mirror "
                             "correlation is reported so the reading can be checked",
        "method": "cutoff proximity to 22.05/24/44.1/48 kHz within 500 Hz, plus the "
                  "correlation between the spectrum below the cutoff and its "
                  "reflection above it",
    }


def _analog_signatures(src: AudioSource, f: np.ndarray, db: np.ndarray,
                       stereo: dict[str, Any], profile: str,
                       collector: Collector) -> dict[str, Any]:
    P = PARAMS["forensics"]
    nyq = src.sr / 2.0
    smooth = log_smooth(f, db, P["hum"]["local_median_octaves"] * 2)

    def _excess(target: float) -> float | None:
        if target >= nyq or f.size == 0:
            return None
        i = int(np.argmin(np.abs(f - target)))
        return float(db[i] - smooth[i])

    hum = {}
    for mains in P["hum"]["mains_hz"]:
        harmonics = []
        for h in range(1, P["hum"]["harmonics"] + 1):
            harmonics.append({"hz": mains * h, "excess_db": _excess(mains * h)})
        vals = [h["excess_db"] for h in harmonics if h["excess_db"] is not None]
        hum[f"{int(mains)}hz"] = {
            "harmonics": harmonics,
            "max_excess_db": max(vals) if vals else None,
            "mean_excess_db": float(np.mean(vals)) if vals else None,
        }
    hum_conf = "medium"
    collector.low_confidence("forensics.mains_hum", hum_conf,
                             "musical content at 50/60 Hz and its harmonics raises "
                             "the same measure as mains hum")

    # Rumble
    lo, hi = P["rumble_hz"]
    total = band_power(f, 10 ** (db / 10.0), 20.0, min(20000.0, nyq))
    rumble = band_power(f, 10 ** (db / 10.0), lo, hi)
    rumble_res = {
        "band_hz": [lo, hi],
        "level_db_rel_total": db_pow(rumble / total) if total > 0 else None,
        "confidence": "high",
        "method": "energy below 30 Hz relative to 20 Hz-20 kHz, from the LTAS",
    }

    # Elliptical EQ: bass mono-ness with a sharp transition
    ell = {"mono_crossover_hz": None, "side_minus_mid_below_120hz_db": None,
           "detected": None, "confidence": "low",
           "confidence_reason": "a mono bass end is also a routine modern mixing "
                                "choice, not only a vinyl-era elliptical EQ"}
    if stereo.get("available"):
        ell["mono_crossover_hz"] = stereo.get("mono_crossover_hz")
        ell["side_minus_mid_below_120hz_db"] = stereo.get("side_minus_mid_below_120hz_db")
        c = stereo.get("mono_crossover_hz")
        ell["detected"] = bool(c is not None and 80.0 <= c <= 400.0)
    collector.low_confidence("forensics.elliptical_eq", "low", ell["confidence_reason"])

    # Tape bias whine
    bias = {"peaks": [], "confidence": "medium",
            "confidence_reason": "narrowband energy above 15 kHz also comes from "
                                 "switch-mode supplies, CRT whine and synth content"}
    lo_b, hi_b = P["tape_bias_hz"]
    m = (f >= lo_b) & (f <= min(hi_b, nyq * 0.99))
    if m.sum() > 8:
        excess = (db - smooth)[m]
        idx, props = sps.find_peaks(excess, prominence=6.0)
        for i, p in zip(idx, props["prominences"]):
            bias["peaks"].append({"frequency_hz": round(float(f[m][i]), 1),
                                  "excess_db": float(excess[i]),
                                  "prominence_db": float(p)})
        bias["peaks"] = sorted(bias["peaks"], key=lambda r: -r["excess_db"])[:5]
    collector.low_confidence("forensics.tape_bias", "medium", bias["confidence_reason"])

    wow = _wow_flutter(src, profile, collector)
    return {"mains_hum": hum, "hum_confidence": hum_conf, "rumble": rumble_res,
            "elliptical_eq": ell, "tape_bias": bias, "wow_flutter": wow}


def _wow_flutter(src: AudioSource, profile: str, collector: Collector) -> dict[str, Any]:
    if profile == "quick":
        return {"available": False, "reason": "skipped by --profile quick"}
    try:
        import librosa
    except ImportError:
        collector.warn("forensics.wow_flutter", "librosa not installed")
        return {"available": False, "reason": "librosa not installed"}
    P = PARAMS["forensics"]["wow_flutter"]
    y, sr = src.lib_mono, src.lib_sr
    w = int(round(P["frame_s"] * sr))
    if y.size < w * 8:
        return {"available": False, "reason": "file too short for frame-wise tuning"}
    cents = []
    times = []
    # Cap the trace: estimate_tuning is the most expensive call per second of
    # audio in the whole tool, and a few hundred frames already pin the spread.
    n_frames_total = y.size // w
    stride = max(1, n_frames_total // P["max_frames"])
    for i in range(0, n_frames_total, stride):
        seg = y[i * w : (i + 1) * w]
        if float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) < 1e-5:
            continue
        try:
            t = float(librosa.estimate_tuning(y=seg, sr=sr))
        except Exception:
            continue
        cents.append(t * 100.0)
        times.append(i * P["frame_s"])
    frame_stride_s = stride * P["frame_s"]
    if len(cents) < 8:
        collector.low_confidence("forensics.wow_flutter", "low",
                                 "too few usable frames for a tuning trace")
        return {"available": False, "reason": "too few usable frames",
                "frames": len(cents)}
    c = np.array(cents)
    t = np.array(times)
    coef = np.polyfit(t, c, 1) if t.size > 2 else [0.0, 0.0]
    resid = c - np.polyval(coef, t)
    return {
        "available": True,
        "frame_s": P["frame_s"],
        "frame_stride_s": frame_stride_s,
        "method": P["method"],
        "frames": int(c.size),
        "cents_std": float(np.std(c)),
        "cents_detrended_std": float(np.std(resid)),
        "slow_drift_cents_per_min": float(coef[0] * 60.0),
        "cents_range": [float(np.min(c)), float(np.max(c))],
        "times_s": t, "cents": c,
        "confidence": "low",
        "confidence_reason": "librosa.estimate_tuning quantises to a chroma grid and "
                             "is confounded by key changes, vibrato and dense "
                             "material; use the number as an upper bound on stability",
    }


def analyse(src: AudioSource, collector: Collector, stereo: dict[str, Any],
            profile: str = "full") -> dict[str, Any]:
    P = PARAMS["forensics"]
    HC = P["hf_cutoff"]
    # Forensics run at the file's own rate: a band-rate resample would add a
    # rolloff that does not exist in the file.
    sr = src.sr
    f, db = _smoothed_ltas(src.mono, sr, HC["nfft"])
    if f.size == 0:
        collector.warn("forensics", "file too short for a spectrum; forensics are null")
        return {"available": False, "reason": "file too short"}

    cutoff = _find_cutoff(f, db, sr, HC["collapse_depth_db"])
    shelf = None
    if cutoff.get("cutoff_hz") is not None:
        c = cutoff["cutoff_hz"]
        nearest = min(CODEC_SHELVES_HZ, key=lambda s: abs(s - c))
        shelf = {"nearest_shelf_hz": nearest, "distance_hz": round(abs(nearest - c), 1),
                 "sharpness_db_per_oct": cutoff.get("rolloff_slope_db_per_oct"),
                 "note": "a shelf at 15.5/16/19/20/20.5 kHz with a steep slope is the "
                         "MP3/AAC fingerprint; the distance and slope are given so the "
                         "match can be judged rather than trusted"}

    if profile == "quick":
        stability = {"available": False, "reason": "skipped by --profile quick"}
    else:
        stability = _frame_cutoffs(src, HC["collapse_depth_db"], HC["frame_s"],
                                   HC["nfft"], cutoff.get("cutoff_hz"))
    frac_below = stability.get("frames_above_cutoff_below_floor")

    cutoff_out = dict(cutoff)
    cutoff_out.update({
        "params": HC,
        "measured_on": "mono at the file's own sample rate, smoothed 1/12 octave",
        "codec_shelf_match": shelf,
        "fraction_of_frames_above_cutoff_below_floor": frac_below,
    })
    if cutoff.get("full_bandwidth"):
        collector.low_confidence("forensics.hf_cutoff", "high",
                                 "spectrum runs to the Nyquist frequency; no shelf to report")
    elif cutoff.get("cutoff_hz") is not None and cutoff["cutoff_hz"] < 20000.0:
        collector.warn("forensics.hf_cutoff",
                       f"spectrum collapses at {cutoff['cutoff_hz']:.0f} Hz "
                       f"(reference minus {HC['collapse_depth_db']} dB); "
                       "check the codec shelf match before drawing tonal conclusions")

    return {
        "available": True,
        "hf_cutoff": cutoff_out,
        "cutoff_stability": stability,
        "spectral_holes": _spectral_holes(f, db, P["spectral_hole"]["min_depth_db"],
                                          P["spectral_hole"]["neighbour_octaves"]),
        "effective_bit_depth": _effective_bit_depth(src, collector),
        "upsampling": _upsampling(src, cutoff, f, db),
        "noise_floor": _noise_floor(src, collector),
        "silence": _silence(src, collector),
        "analog_signatures": _analog_signatures(src, f, db, stereo, profile, collector),
    }
