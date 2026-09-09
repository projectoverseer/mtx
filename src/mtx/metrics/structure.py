"""4.7 Structure, tempo, key.

Every headline metric also exists per section, because a track average hides
exactly the behaviour that distinguishes one master from another.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage
from scipy import signal as sps

from ..audio import AudioSource
from ..bands import get_band_pack
from ..dsp import (band_power, block_loudness, crest_db, gated_integrated,
                   linear_fit_db_per_octave, spectrum_table, welch_psd)
from ..params import BANDS, PARAMS
from ..util import Collector, db_amp, db_pow, fmt_time

# Krumhansl-Schmuckler key profiles.
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52,
                     5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54,
                     4.75, 3.98, 2.69, 3.34, 3.17])
PITCHES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _feature_stack(src: AudioSource) -> tuple[np.ndarray, np.ndarray, Any]:
    import librosa
    y, sr = src.lib_mono, src.lib_sr
    hop = PARAMS["general"]["librosa_hop_length"]
    n_fft = PARAMS["general"]["librosa_n_fft"]
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S ** 2), n_mfcc=20)
    chroma = src.chroma_cqt()
    rms = librosa.feature.rms(S=S)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    feats = []
    width = min(mfcc.shape[1], chroma.shape[1], rms.shape[1], contrast.shape[1])
    for f in (mfcc, chroma, rms, contrast):
        f = np.asarray(f, dtype=np.float64)[:, :width]
        mu = f.mean(axis=1, keepdims=True)
        sd = f.std(axis=1, keepdims=True)
        feats.append((f - mu) / np.where(sd > 0, sd, 1.0))
    stack = np.vstack(feats)
    times = librosa.frames_to_time(np.arange(stack.shape[1]), sr=sr, hop_length=hop)
    return stack, times, librosa


def _downsample_frames(stack: np.ndarray, factor: int) -> np.ndarray:
    """Average groups of `factor` feature frames.

    The self-similarity matrix is quadratic in the frame count, so running it
    at the raw 23 ms hop costs 16x what a 93 ms hop costs and buys nothing:
    the shortest section reported is 4 s.
    """
    if factor <= 1:
        return stack
    n = (stack.shape[1] // factor) * factor
    if n < factor:
        return stack
    return stack[:, :n].reshape(stack.shape[0], -1, factor).mean(axis=2)


def _novelty(stack: np.ndarray, kernel_frames: int) -> np.ndarray:
    """Foote novelty from a cosine self-similarity matrix."""
    X = stack / np.maximum(np.linalg.norm(stack, axis=0, keepdims=True), 1e-12)
    ssm = X.T @ X
    ssm = ndimage.median_filter(ssm, size=3)
    k = max(4, kernel_frames // 2 * 2)
    half = k // 2
    g = np.outer(sps.windows.gaussian(k, k / 4.0), sps.windows.gaussian(k, k / 4.0))
    sign = np.ones((k, k))
    sign[:half, half:] = -1.0
    sign[half:, :half] = -1.0
    kern = g * sign
    n = ssm.shape[0]
    nov = np.zeros(n)
    for i in range(half, n - half):
        nov[i] = float(np.sum(ssm[i - half : i + half, i - half : i + half] * kern))
    if nov.max() > nov.min():
        nov = (nov - nov.min()) / (nov.max() - nov.min())
    return nov


def _section_metrics(src: AudioSource, t0: float, t1: float,
                     integrated: float | None) -> dict[str, Any]:
    sr = src.sr
    a, b = int(t0 * sr), min(int(t1 * sr), src.n_frames)
    seg = src.x[a:b].astype(np.float64)
    out: dict[str, Any] = {"start_s": t0, "end_s": t1, "duration_s": t1 - t0,
                           "start": fmt_time(t0), "end": fmt_time(t1)}
    if seg.shape[0] < int(0.4 * sr):
        out["note"] = "section shorter than one loudness block"
        return out
    t, bl = block_loudness(seg, sr, 0.4, 0.1)
    lufs, _ = gated_integrated(bl)
    t3, st = block_loudness(seg, sr, 3.0, 0.1)
    mono = seg.mean(axis=1)
    out["lufs_i"] = lufs
    out["shortterm_max_lufs"] = float(np.max(st)) if st.size else None
    out["crest_db"] = crest_db(mono)
    out["delta_vs_track_lufs"] = (lufs - integrated) if (lufs is not None and integrated is not None) else None

    bsr = src.band_sr
    ba, bb = int(t0 * bsr), min(int(t1 * bsr), src.band_x.shape[0])
    bmid = src.band_mid[ba:bb]
    bside = src.band_side[ba:bb]
    f, p = welch_psd(bmid, bsr, 8192)
    if f.size:
        slope, r2 = linear_fit_db_per_octave(f, db_pow(np.maximum(p, 1e-30)),
                                             100.0, min(10000.0, bsr / 2 * 0.98))
        out["tilt_db_per_oct"] = slope
        out["tilt_r2"] = r2
        rows = spectrum_table(f, p, [(0.5 * (lo + hi), lo, hi) for _, lo, hi in BANDS],
                              bsr / 2.0)
        out["band_energy_pct"] = {n: (r["pct"] if r["pct"] is not None else None)
                                  for (n, _, _), r in zip(BANDS, rows)}
    pm = float(np.mean(bmid * bmid)) if bmid.size else 0.0
    ps = float(np.mean(bside * bside)) if bside.size else 0.0
    out["side_minus_mid_db"] = db_pow(ps / pm) if (pm > 0 and ps > 0) else (
        -200.0 if pm > 0 else None)
    return out


def _widest_band_in_section(src: AudioSource, section: dict[str, Any]) -> dict[str, Any] | None:
    """Which of the 8 bands carries the most side energy inside one section.

    Reported for the widest section, so "this record opens up in the chorus" can
    be read as a band rather than as a single number.
    """
    if src.n_ch < 2:
        return None
    sr = src.band_sr
    a = int(section["start_s"] * sr)
    b = min(int(section["end_s"] * sr), src.band_x.shape[0])
    if b - a < sr:
        return None
    L, R = src.band_x[a:b, 0], src.band_x[a:b, 1]
    f, p_ll = welch_psd(L, sr, 8192)
    _, p_rr = welch_psd(R, sr, 8192)
    if f.size == 0:
        return None
    nps = int(min(8192, L.size))
    _, cross = sps.csd(L, R, fs=sr, window="hann", nperseg=nps, noverlap=nps // 2,
                       detrend=False, scaling="density")
    p_lr = np.real(cross)
    best = None
    per_band = []
    for name, lo, hi in BANDS:
        if lo >= sr / 2.0:
            continue
        e_ll = band_power(f, p_ll, lo, min(hi, sr / 2.0))
        e_rr = band_power(f, p_rr, lo, min(hi, sr / 2.0))
        e_lr = band_power(f, p_lr, lo, min(hi, sr / 2.0))
        e_mid = (e_ll + e_rr + 2 * e_lr) / 4.0
        e_side = (e_ll + e_rr - 2 * e_lr) / 4.0
        val = db_pow(e_side / e_mid) if (e_mid > 0 and e_side > 0) else None
        per_band.append({"band": name, "side_minus_mid_db": val})
        if val is not None and (best is None or val > best[1]):
            best = (name, val)
    if best is None:
        return None
    return {"section_index": section.get("index"), "band": best[0],
            "side_minus_mid_db": best[1], "per_band": per_band,
            "method": "band-integrated L/R auto- and cross-spectra within the section"}


def _arrangement_gaps(src: AudioSource) -> list[dict[str, Any]]:
    P = PARAMS["structure"]["arrangement_gap"]
    pack = get_band_pack(src)
    gaps: list[dict[str, Any]] = []
    min_frames = max(1, int(round(P["min_ms"] / 1000.0 / pack.hop_s)))
    for name in pack.names:
        e = pack.envelopes[name]
        if e.size < min_frames * 2:
            continue
        db = db_amp(np.maximum(e, 1e-20))
        ref = db_amp(float(np.sqrt(np.mean(e * e))))
        low = db < (ref - P["drop_db"])
        if not low.any():
            continue
        d = np.diff(low.astype(np.int8))
        starts = np.flatnonzero(d == 1) + 1
        ends = np.flatnonzero(d == -1) + 1
        if low[0]:
            starts = np.concatenate([[0], starts])
        if low[-1]:
            ends = np.concatenate([ends, [low.size]])
        for s, e_ in zip(starts, ends):
            if (e_ - s) >= min_frames:
                gaps.append({
                    "band": name,
                    "start_s": float(s * pack.hop_s),
                    "start": fmt_time(float(s * pack.hop_s)),
                    "duration_ms": float((e_ - s) * pack.hop_s * 1000.0),
                    "band_level_db_rel_track": float(np.mean(db[s:e_]) - ref),
                })
    gaps.sort(key=lambda g: (g["start_s"], g["band"]))
    return gaps[:200]


def _tempo(src: AudioSource, librosa, collector: Collector) -> dict[str, Any]:
    y, sr = src.lib_mono, src.lib_sr
    hop = PARAMS["general"]["librosa_hop_length"]
    if y.size < sr * 4:
        return {"available": False, "reason": "file too short for tempo estimation"}
    env = src.onset_envelope()
    bpm_raw, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr,
                                             hop_length=hop, units="time")
    bpm_raw = float(np.atleast_1d(bpm_raw)[0])
    beats = np.asarray(beats, dtype=float)
    ibi = np.diff(beats)
    stability = None
    if ibi.size > 4:
        med = float(np.median(ibi))
        stability = float(np.mean(np.abs(ibi - med) < 0.05 * med)) if med > 0 else None
    # beat_track places beats on the 512-sample analysis grid, which quantises
    # its own BPM readout by up to a couple of percent.  Regressing beat time on
    # beat index recovers the period from the whole grid instead of one interval.
    bpm = bpm_raw
    bpm_fit_r2 = None
    if beats.size >= 8:
        k = np.arange(beats.size)
        A = np.vstack([k, np.ones_like(k, dtype=float)]).T
        coef, *_ = np.linalg.lstsq(A, beats, rcond=None)
        period = float(coef[0])
        resid = beats - A @ coef
        ss_tot = float(np.sum((beats - beats.mean()) ** 2))
        bpm_fit_r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else None
        if period > 0 and abs(60.0 / period - bpm_raw) < 0.15 * bpm_raw:
            bpm = 60.0 / period
    # Drift: re-estimate on 30 s windows, reusing the onset envelope rather
    # than recomputing it once per window.
    W = PARAMS["structure"]["tempo_drift_window_s"]
    win_frames = int(round(W * sr / hop))
    per_window = []
    for i in range(max(1, env.size // win_frames)):
        seg = env[i * win_frames : (i + 1) * win_frames]
        if seg.size < int(5 * sr / hop):
            continue
        try:
            b2, _ = librosa.beat.beat_track(onset_envelope=seg, sr=sr,
                                            hop_length=hop)
            per_window.append(float(np.atleast_1d(b2)[0]))
        except Exception:
            continue
    drift = float(np.std(per_window)) if len(per_window) > 1 else None
    conf = "high" if (stability or 0) > 0.9 else ("medium" if (stability or 0) > 0.7 else "low")
    if conf != "high":
        collector.low_confidence("structure.tempo", conf,
                                 f"only {100 * (stability or 0):.0f}% of inter-beat "
                                 "intervals lie within 5% of the median")
    return {
        "available": True,
        "bpm": bpm,
        "bpm_beat_track_raw": bpm_raw,
        "bpm_source": "least-squares fit of beat time on beat index"
                      if bpm != bpm_raw else "librosa.beat.beat_track",
        "bpm_grid_fit_r2": bpm_fit_r2,
        "method": "librosa.beat.beat_track on the onset strength envelope "
                  "(22.05 kHz, hop 512); BPM refined by regressing beat time on "
                  "beat index to remove the analysis-hop quantisation",
        "beat_count": int(np.asarray(beats).size),
        "beat_times_s": np.asarray(beats, dtype=float),
        "inter_beat_interval_stability": stability,
        "confidence": conf,
        "drift_window_s": W,
        "bpm_per_window": per_window,
        "bpm_drift_std": drift,
        "drift_note": "near-zero drift indicates a programmed grid; a moving value "
                      "indicates played or tape-sourced material",
    }


def _key(src: AudioSource, librosa, collector: Collector) -> dict[str, Any]:
    y, sr = src.lib_mono, src.lib_sr
    if y.size < sr * 2:
        return {"available": False, "reason": "file too short for key estimation"}
    prof = src.chroma_cqt().mean(axis=1)
    if float(np.std(prof)) <= 0:
        return {"available": False, "reason": "flat chroma; no key content"}
    scores = []
    for mode, template in (("major", KS_MAJOR), ("minor", KS_MINOR)):
        for i in range(12):
            t = np.roll(template, i)
            c = float(np.corrcoef(prof, t)[0, 1])
            scores.append((c, f"{PITCHES[i]} {mode}"))
    scores.sort(reverse=True)
    top, runner = scores[0], scores[1]
    margin = top[0] - runner[0]
    conf = "low" if margin < PARAMS["structure"]["key_low_confidence_margin"] else "medium"
    if conf == "low":
        collector.low_confidence("structure.key", "low",
                                 f"top two candidates ({top[1]}, {runner[1]}) are "
                                 f"within {margin:.3f} correlation")
    # `estimate_tuning` returns fractions of a chroma bin, and a bin is a
    # semitone -- so the deviation is in semitones, not in octaves, and the
    # reference pitch it implies moves by a twelfth of that.
    tuning = float(librosa.estimate_tuning(y=y, sr=sr))
    cents = tuning * 100.0
    if abs(cents) >= PARAMS["structure"]["tuning_report_cents"]:
        collector.low_confidence(
            "structure.tuning", "medium",
            f"the master sits {cents:+.0f} cents from A440 (A4 = "
            f"{440.0 * 2.0 ** (tuning / 12.0):.1f} Hz). Chroma is estimated "
            "against the track's own reference, so key and chords are "
            "unaffected; note names elsewhere are still named from A440")
    return {
        "available": True,
        "key": top[1], "correlation": top[0],
        "runner_up": runner[1], "runner_up_correlation": runner[0],
        "margin": margin,
        "confidence": conf,
        "method": "mean chroma-CQT correlated against Krumhansl-Schmuckler profiles",
        "tuning_cents": cents,
        "implied_a4_hz": 440.0 * (2.0 ** (tuning / 12.0)),
        "tuning_method": "librosa.estimate_tuning",
        "tuning_note": "deviation from A440 within one semitone. The estimate "
                       "wraps at +/-50 cents, so a master a whole semitone "
                       "off -- Baroque A=415, a tape played a semitone fast "
                       "-- reads as 0 cents and shifts the reported key by a "
                       "semitone instead. chroma-CQT estimates the reference "
                       "from the track itself, so a fractional detune does "
                       "not move the key, the chords or the form clustering.",
    }


def analyse(src: AudioSource, collector: Collector, integrated: float | None,
            profile: str = "full") -> dict[str, Any]:
    if profile == "quick":
        return {"available": False, "reason": "skipped by --profile quick",
                "arrangement_gaps": _arrangement_gaps(src)}
    try:
        stack, times, librosa = _feature_stack(src)
    except ImportError:
        collector.warn("structure", "librosa not installed; structure, tempo and "
                                    "key are null")
        return {"available": False, "reason": "librosa not installed"}
    except Exception as exc:
        collector.warn("structure", f"feature stack failed: {exc!r}")
        return {"available": False, "reason": repr(exc)}

    P = PARAMS["structure"]
    ds = int(P["ssm_downsample_frames"])
    stack = _downsample_frames(stack, ds)
    times = times[: stack.shape[1] * ds : ds]
    frame_s = float(times[1] - times[0]) if times.size > 1 else 0.023 * ds
    sections: list[dict[str, Any]] = []
    boundaries: list[float] = []
    if stack.shape[1] > 32:
        nov = _novelty(stack, int(round(P["novelty_kernel_s"] / frame_s)))
        pk = P["peak_pick"]
        f = lambda s: max(1, int(round(s / frame_s)))
        idx = librosa.util.peak_pick(nov, pre_max=f(pk["pre_max_s"]),
                                     post_max=f(pk["post_max_s"]),
                                     pre_avg=f(pk["pre_avg_s"]),
                                     post_avg=f(pk["post_avg_s"]),
                                     delta=pk["delta"], wait=f(pk["wait_s"]))
        boundaries = [0.0] + [float(times[i]) for i in np.atleast_1d(idx)] + [src.duration]
        merged = [boundaries[0]]
        for b in boundaries[1:]:
            if b - merged[-1] >= P["min_section_s"]:
                merged.append(b)
        if merged[-1] < src.duration - 1e-6:
            merged[-1] = src.duration
        boundaries = merged
        for i in range(len(boundaries) - 1):
            m = _section_metrics(src, boundaries[i], boundaries[i + 1], integrated)
            m["index"] = i
            sections.append(m)
    else:
        collector.warn("structure", "too few analysis frames for segmentation")

    # Onset rate per section, from the shared onset envelope.
    try:
        on_times, _ = src.onset_times()
    except Exception:
        on_times = np.zeros(0)
    for s in sections:
        d = s["duration_s"]
        n = int(np.sum((on_times >= s["start_s"]) & (on_times < s["end_s"])))
        s["onset_rate_per_s"] = (n / d) if d > 0 else None
    prev = None
    for s in sections:
        s["delta_vs_previous_lufs"] = (
            (s.get("lufs_i") - prev) if (prev is not None and s.get("lufs_i") is not None) else None)
        if s.get("lufs_i") is not None:
            prev = s["lufs_i"]

    lufs_vals = [(s.get("lufs_i"), s["index"]) for s in sections if s.get("lufs_i") is not None]
    loudest = max(lufs_vals)[1] if lufs_vals else None
    quietest = min(lufs_vals)[1] if lufs_vals else None
    jumps = [(abs(s["delta_vs_previous_lufs"]), s["index"], s["start_s"])
             for s in sections if s.get("delta_vs_previous_lufs") is not None]
    biggest = max(jumps) if jumps else None

    widest = None
    widest_band = None
    if sections:
        w = [(s.get("side_minus_mid_db"), s["index"]) for s in sections
             if s.get("side_minus_mid_db") is not None]
        widest = max(w)[1] if w else None
        if widest is not None:
            widest_band = _widest_band_in_section(src, sections[widest])

    return {
        "available": True,
        "method": P["features"],
        "section_count": len(sections),
        "boundaries_s": boundaries,
        "sections": sections,
        "loudest_section_index": loudest,
        "quietest_section_index": quietest,
        "widest_section_index": widest,
        "widest_band_in_widest_section": widest_band,
        "biggest_jump": ({"db": biggest[0], "at_section": biggest[1],
                          "time_s": biggest[2], "time": fmt_time(biggest[2])}
                         if biggest else None),
        "arrangement_gaps": _arrangement_gaps(src),
        "arrangement_gap_rule": f"a band more than {PARAMS['structure']['arrangement_gap']['drop_db']} dB "
                                f"below its own track RMS for at least "
                                f"{PARAMS['structure']['arrangement_gap']['min_ms']} ms",
        "tempo": _tempo(src, librosa, collector),
        "key": _key(src, librosa, collector),
    }
