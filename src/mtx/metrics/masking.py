"""Inter-stem masking: how each stem sits in the bands the others occupy.

Every other stem measurement in this tool is taken in isolation, or against the
mix.  Nothing is measured against another stem, and that is the whole of mix
engineering: a vocal is not quiet, it is quiet *underneath something*.

Everything here is pure DSP over signals that are already on disk once
separation has run, so it costs one band-energy pass per stem and no new
dependency.  It is still a measurement of separated signals: the caveat that
governs `stems` governs this block too.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from scipy import signal as sps

from ..audio import AudioSource
from ..dsp import (block_loudness, gated_integrated,
                   linear_fit_db_per_octave, log_smooth, third_octave_edges,
                   welch_psd)
from ..params import PARAMS, THIRD_OCTAVE_CENTRES
from ..util import Collector, db_pow, fmt_time

# One band-energy frame every ~93 ms at 44.1 kHz: long enough for a 20 Hz band
# to exist at all, short enough that a section boundary lands inside one frame.
NFFT = 8192


def _band_matrix(x: np.ndarray, sr: float,
                 edges: list[tuple[float, float, float]]
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Band power per STFT frame.  Returns (times_s, matrix[n_frames, n_bands]).

    The spectrogram is reduced to bands and dropped immediately: holding four
    stems' worth of full-resolution spectrograms would cost more memory than
    the decoded audio it came from.
    """
    n = int(np.asarray(x).shape[0])
    if n < NFFT:
        return np.zeros(0), np.zeros((0, len(edges)))
    f, t, sxx = sps.spectrogram(np.asarray(x, dtype=np.float64), fs=sr,
                                window="hann", nperseg=NFFT, noverlap=NFFT // 2,
                                detrend=False, scaling="density", mode="psd")
    nyq = sr / 2.0
    out = np.zeros((sxx.shape[1], len(edges)))
    for j, (_, lo, hi) in enumerate(edges):
        if lo >= nyq:
            continue
        m = (f >= lo) & (f < min(hi, nyq))
        if not np.any(m):
            continue
        out[:, j] = np.trapezoid(sxx[m], f[m], axis=0) if m.sum() > 1 else sxx[m][0]
    del sxx
    return np.asarray(t, dtype=float), out


def _lufs(x: np.ndarray, sr: float) -> float | None:
    """Gated integrated loudness of a raw (frames, channels) slice."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if x.shape[0] < x.shape[1]:
        x = x.T
    if x.shape[0] < int(0.4 * sr):
        return None
    _, bl = block_loudness(x, sr, 0.4, 0.1)
    val, _ = gated_integrated(bl)
    return val


def _norm(v: np.ndarray) -> np.ndarray:
    s = float(np.sum(v))
    return v / s if s > 0 else np.zeros_like(v)


def _masking_index_db(target: np.ndarray, masker: np.ndarray) -> float | None:
    """The masker's level inside the target's own band-energy distribution.

    Weights are the target's normalised band energies, so a masker is judged
    where the target actually lives and not where it happens to be loud.  The
    result is a ratio in dB: positive means the masker carries more energy than
    the target does, band for band, weighted by where the target is.
    """
    w = _norm(np.asarray(target, dtype=np.float64))
    if not np.any(w):
        return None
    num = float(np.sum(w * masker))
    den = float(np.sum(w * target))
    if num <= 0 or den <= 0:
        return None
    return db_pow(num / den)


def _overlap(a: np.ndarray, b: np.ndarray) -> dict[str, float | None]:
    """How much two band distributions occupy the same bands."""
    pa, pb = _norm(a), _norm(b)
    na, nb = float(np.linalg.norm(pa)), float(np.linalg.norm(pb))
    cos = float(np.dot(pa, pb) / (na * nb)) if (na > 0 and nb > 0) else None
    bhat = float(np.sum(np.sqrt(pa * pb))) if (na > 0 and nb > 0) else None
    return {"cosine": cos, "bhattacharyya": bhat}


def _sibilance(src: AudioSource, collector: Collector) -> dict[str, Any]:
    """Sibilance-band behaviour on the vocal stem: de-esser evidence, measured.

    A de-esser is a compressor on one band, and a compressor is visible as a
    slope: if the 5-10 kHz level rises less than 1 dB for every dB the 1-4 kHz
    level rises, something is holding it down.  The slope is the measurement;
    calling it a de-esser is the inference, and it is labelled as one.
    """
    P = PARAMS["masking"]
    sr = src.band_sr
    x = src.band_mono
    frame = max(64, int(round(P["sibilance_frame_ms"] / 1000.0 * sr)))
    n = (x.size // frame) * frame
    if n < frame * 8:
        return {"available": False, "reason": "vocal stem too short for a "
                                              "sibilance timeline"}
    sib_lo, sib_hi = P["sibilance_band_hz"]
    ref_lo, ref_hi = P["sibilance_reference_band_hz"]
    nyq = sr / 2.0
    if sib_lo >= nyq:
        return {"available": False,
                "reason": f"sibilance band starts above Nyquist ({nyq:.0f} Hz)"}

    def band_env(lo: float, hi: float) -> np.ndarray:
        sos = sps.butter(4, [max(lo, 1.0) / nyq, min(hi, nyq * 0.99) / nyq],
                         btype="bandpass", output="sos")
        y = sps.sosfiltfilt(sos, x)[:n].reshape(-1, frame)
        return np.sqrt(np.mean(y * y, axis=1))

    sib = band_env(sib_lo, sib_hi)
    ref = band_env(ref_lo, ref_hi)
    sib_db, ref_db = db_pow(np.maximum(sib ** 2, 1e-24)), db_pow(np.maximum(ref ** 2, 1e-24))
    # Only frames where the voice is actually sounding: the ratio in a silent
    # frame is a ratio of two noise floors.
    live = ref_db > (np.percentile(ref_db, 95) - 30.0)
    if int(np.sum(live)) < 16:
        return {"available": False, "reason": "too few sounding frames"}
    sd, rd = sib_db[live], ref_db[live]
    A = np.vstack([rd, np.ones_like(rd)]).T
    coef, *_ = np.linalg.lstsq(A, sd, rcond=None)
    pred = A @ coef
    ss_tot = float(np.sum((sd - sd.mean()) ** 2))
    r2 = 1.0 - float(np.sum((sd - pred) ** 2)) / ss_tot if ss_tot > 0 else None
    ratio = sd - rd
    slope = float(coef[0])
    conf = "medium" if (r2 or 0) > 0.5 else "low"
    if conf != "medium":
        collector.low_confidence("stems.masking.sibilance", conf,
                                 f"sibilance-on-reference regression R2={r2}")
    return {
        "available": True,
        "band_hz": [sib_lo, sib_hi],
        "reference_band_hz": [ref_lo, ref_hi],
        "frame_ms": P["sibilance_frame_ms"],
        "sounding_frames": int(np.sum(live)),
        "ratio_db": {"median": float(np.median(ratio)),
                     "p90": float(np.percentile(ratio, 90)),
                     "p99": float(np.percentile(ratio, 99)),
                     "max": float(np.max(ratio)),
                     "range_p10_p99": float(np.percentile(ratio, 99)
                                            - np.percentile(ratio, 10))},
        "regression_slope_db_per_db": slope,
        "regression_r2": r2,
        "confidence": conf,
        "inference": {
            "band_compression": bool(slope < 0.8),
            "basis": "a slope below 1 dB/dB means the sibilance band rises more "
                     "slowly than the band under it; a de-esser is one cause and "
                     "a dark vocal chain is another",
        },
    }


def _highpass_corner(src: AudioSource) -> dict[str, Any]:
    """Where the vocal stem's low end stops, and how steeply."""
    P = PARAMS["masking"]
    sr = src.band_sr
    f, p = welch_psd(src.band_mono, sr, 16384)
    if f.size == 0:
        return {"available": False, "reason": "stem too short for an LTAS"}
    db = db_pow(np.maximum(p, 1e-30))
    ds = log_smooth(f, db, 1.0 / 12.0)
    fs = f
    plo, phi = P["highpass_plateau_hz"]
    m = (fs >= plo) & (fs <= phi)
    if not np.any(m):
        return {"available": False, "reason": "no plateau band in range"}
    plateau = float(np.median(ds[m]))
    lo, hi = P["highpass_probe_hz"]
    probe = (fs >= lo) & (fs < plo)
    if not np.any(probe):
        return {"available": False, "reason": "no probe band in range"}
    fp, dp = fs[probe], ds[probe]
    below = np.flatnonzero(dp <= plateau - 3.0)
    corner = float(fp[below[-1]]) if below.size else None
    slope = None
    if corner is not None:
        slope, _ = linear_fit_db_per_octave(fs, ds, max(lo, corner / 2.0),
                                            min(plo, corner * 2.0))
    return {
        "available": True,
        "plateau_band_hz": [plo, phi],
        "plateau_db": plateau,
        "corner_hz": corner,
        "corner_rule": "highest frequency below the plateau band that is 3 dB "
                       "or more under the plateau median, on a 1/12-octave "
                       "smoothed LTAS",
        "slope_below_corner_db_per_oct": slope,
        "slope_sign_note": "positive is the level rising with frequency, which "
                           "is what a high-pass looks like from below its corner",
        "confidence": "medium" if corner is not None else "low",
    }


def _reverb_send(src: AudioSource, collector: Collector) -> dict[str, Any]:
    """Vocal tail behaviour: how much decay follows a note, and after how long.

    Pre-delay is read off the ensemble-averaged envelope after a note offset:
    a dry vocal falls monotonically, a sent one falls and then holds or rises
    again once the send returns.  It is an inference and says so.
    """
    from . import processing as m_processing

    sub = Collector()
    rev = m_processing._reverb(src, sub)
    sr = src.band_sr
    x = np.abs(src.band_mono)
    hop = max(1, int(round(0.005 * sr)))
    n = (x.size // hop) * hop
    pre_delay_ms: float | None = None
    ensemble: list[float] = []
    if n >= hop * 200:
        env = np.sqrt(np.mean((x[:n].reshape(-1, hop)) ** 2, axis=1))
        edb = db_pow(np.maximum(env ** 2, 1e-24))
        span = int(round(0.4 / 0.005))          # 400 ms of tail
        lead = int(round(0.05 / 0.005))         # 50 ms of the note itself
        loud = np.percentile(edb, 90)
        offsets = []
        for i in range(lead, edb.size - span):
            # A note offset: loud just before, and 12 dB down 100 ms later.
            if edb[i] > loud - 12.0 and edb[i + 20] < edb[i] - 12.0:
                offsets.append(i)
        # Keep offsets that do not overlap, newest wins nothing: first come.
        keep: list[int] = []
        for i in offsets:
            if not keep or i - keep[-1] > span:
                keep.append(i)
        if len(keep) >= 4:
            stack = np.vstack([edb[i:i + span] - edb[i] for i in keep])
            ensemble = [float(v) for v in np.mean(stack, axis=0)]
            arr = np.asarray(ensemble)
            # The first frame after the fall where the decay stops falling.
            d = np.diff(arr)
            flat = np.flatnonzero(d[4:] > -0.05)
            if flat.size:
                pre_delay_ms = float((flat[0] + 4) * 5.0)
    conf = "low" if pre_delay_ms is None else "medium"
    collector_note = ("pre-delay is inferred from the shape of the averaged "
                      "post-offset envelope, not from a measured impulse")
    if conf == "low":
        collector.low_confidence("stems.masking.vocal_reverb", "low",
                                 "fewer than four isolated note offsets, or no "
                                 "flattening in the averaged tail")
    return {
        "available": True,
        "source": "separated",
        "per_octave_band": rev.get("per_octave_band") if isinstance(rev, dict) else None,
        "tail_stereo_correlation": rev.get("tail_stereo_correlation") if isinstance(rev, dict) else None,
        "pre_delay_ms": pre_delay_ms,
        "post_offset_envelope_db": ensemble,
        "post_offset_envelope_hop_ms": 5.0,
        "confidence": conf,
        "note": collector_note,
        "warnings": sub.warnings,
    }


def _delay_throws(src: AudioSource, bpm: float | None) -> dict[str, Any]:
    """Autocorrelation of the vocal envelope at musical subdivisions of the beat.

    A tempo-synced throw puts a copy of a word one subdivision later; the
    envelope correlates with itself at exactly that lag.  Reported as the
    correlation at each subdivision, never as a verdict that a delay exists.
    """
    P = PARAMS["masking"]["delay_throw_subdivisions"]
    if not bpm or bpm <= 0:
        return {"available": False, "reason": "no tempo; subdivision lags are "
                                              "undefined"}
    sr = src.band_sr
    hop = max(1, int(round(0.005 * sr)))
    x = np.abs(src.band_mono)
    n = (x.size // hop) * hop
    if n < hop * 400:
        return {"available": False, "reason": "stem too short"}
    env = np.sqrt(np.mean(x[:n].reshape(-1, hop) ** 2, axis=1))
    env = env - env.mean()
    denom = float(np.dot(env, env))
    if denom <= 0:
        return {"available": False, "reason": "silent stem"}
    beat_s = 60.0 / float(bpm)

    def acf(lag: int) -> float:
        return float(np.dot(env[:-lag], env[lag:]) / denom)

    rows = []
    for sub in P:
        lag = int(round(sub * beat_s / 0.005))
        if lag <= 0 or lag >= env.size // 2:
            continue
        r = acf(lag)
        # An envelope correlates with itself more at short lags than at long
        # ones whatever the record does, so the raw value is dominated by the
        # subdivision being small.  What a throw actually looks like is a peak
        # standing above its own neighbourhood, so the baseline is the median
        # of the lags 10-25% away on either side and the excess is reported.
        span = max(2, int(round(lag * 0.25)))
        near = [acf(l) for l in range(max(1, lag - span), lag + span + 1)
                if 0 < l < env.size // 2 and abs(l - lag) >= max(1, int(lag * 0.10))]
        base = float(np.median(near)) if near else None
        rows.append({"subdivision_beats": sub, "lag_ms": lag * 5.0,
                     "autocorrelation": r,
                     "local_baseline": base,
                     "excess_over_baseline": (r - base) if base is not None else None})
    if not rows:
        return {"available": False, "reason": "no usable lag"}
    scored = [r for r in rows if r["excess_over_baseline"] is not None]
    best = (max(scored, key=lambda r: r["excess_over_baseline"])
            if scored else max(rows, key=lambda r: r["autocorrelation"]))
    return {"available": True, "bpm_used": float(bpm), "envelope_hop_ms": 5.0,
            "per_subdivision": rows, "strongest": best,
            "read_this_one": "excess_over_baseline; the raw autocorrelation "
                             "rises as the lag shortens whatever the record does",
            "method": "normalised autocorrelation of the 5 ms amplitude "
                      "envelope at each subdivision of the measured beat, minus "
                      "the median autocorrelation of the lags 10-25% away"}


def analyse(mix: AudioSource, stems: dict[str, AudioSource],
            sections: list[dict[str, Any]], tempo: dict[str, Any],
            collector: Collector) -> dict[str, Any]:
    """The masking block of `stems`.

    `stems` maps stem name to a decoded source; `sections` are the measured
    structure boundaries, so masking can be read where it changes rather than
    only as a track average.
    """
    names = sorted(stems)
    if len(names) < 2:
        return {"available": False, "reason": "fewer than two stems"}
    edges = third_octave_edges(THIRD_OCTAVE_CENTRES)
    centres = [c for c, _, _ in edges]

    mats: dict[str, np.ndarray] = {}
    times: dict[str, np.ndarray] = {}
    for nm in names:
        t, m = _band_matrix(stems[nm].band_mono, stems[nm].band_sr, edges)
        times[nm], mats[nm] = t, m
    whole = {nm: (mats[nm].mean(axis=0) if mats[nm].size else np.zeros(len(edges)))
             for nm in names}

    pairs: list[dict[str, Any]] = []
    for target, masker in itertools.permutations(names, 2):
        et, em = whole[target], whole[masker]
        per_band = []
        for j, c in enumerate(centres):
            if et[j] <= 0 or em[j] <= 0:
                per_band.append({"centre_hz": c, "masker_minus_target_db": None,
                                 "target_energy_share_pct": None})
                continue
            per_band.append({
                "centre_hz": c,
                "masker_minus_target_db": db_pow(em[j] / et[j]),
                "target_energy_share_pct": 100.0 * et[j] / float(np.sum(et)),
            })
        pairs.append({
            "target": target, "masker": masker,
            "masking_index_db": _masking_index_db(et, em),
            "overlap": _overlap(et, em),
            "per_third_octave": per_band,
        })

    # The symmetric half, reported once rather than twice.
    overlaps = [{"stems": [a, b], **_overlap(whole[a], whole[b])}
                for a, b in itertools.combinations(names, 2)]

    # Per-section masking, and the vocal-versus-everything-else balance that
    # only exists once the stems are compared with each other.
    per_section: list[dict[str, Any]] = []
    for s in sections:
        t0, t1 = float(s.get("start_s", 0.0)), float(s.get("end_s", 0.0))
        if t1 - t0 <= 0:
            continue
        vec: dict[str, np.ndarray] = {}
        for nm in names:
            tt, mm = times[nm], mats[nm]
            if tt.size == 0:
                continue
            sel = (tt >= t0) & (tt < t1)
            if np.any(sel):
                vec[nm] = mm[sel].mean(axis=0)
        if len(vec) < 2:
            continue
        row: dict[str, Any] = {"index": s.get("index"), "start_s": t0,
                               "start": fmt_time(t0), "end_s": t1,
                               "masking_index_db": {}}
        for target, masker in itertools.permutations(sorted(vec), 2):
            val = _masking_index_db(vec[target], vec[masker])
            row["masking_index_db"][f"{masker}_into_{target}"] = val
        if "vocals" in stems:
            sr = stems["vocals"].sr
            a, b = int(t0 * sr), int(t1 * sr)
            v = _lufs(stems["vocals"].x[a:b], sr)
            rest = None
            others = [nm for nm in names if nm != "vocals"]
            if others:
                acc = None
                for nm in others:
                    o = stems[nm]
                    seg = o.x[int(t0 * o.sr):int(t1 * o.sr)].astype(np.float64)
                    if acc is None:
                        acc = seg.copy()
                    elif seg.shape[0] == acc.shape[0] and seg.shape[1] == acc.shape[1]:
                        acc += seg
                if acc is not None:
                    rest = _lufs(acc, stems[others[0]].sr)
            row["vocal_lufs"] = v
            row["instrumental_lufs"] = rest
            row["vocal_minus_instrumental_lu"] = (
                (v - rest) if (v is not None and rest is not None) else None)
        per_section.append(row)

    release = None
    if per_section:
        key_pairs = sorted({k for r in per_section for k in r["masking_index_db"]})
        release = {}
        for k in key_pairs:
            vals = [r["masking_index_db"].get(k) for r in per_section]
            good = [v for v in vals if v is not None]
            release[k] = {
                "per_section_db": vals,
                "range_db": (max(good) - min(good)) if len(good) > 1 else None,
                "quietest_section_index": (
                    per_section[int(np.argmin([v if v is not None else np.inf
                                               for v in vals]))].get("index")
                    if good else None),
            }

    out: dict[str, Any] = {
        "available": True,
        "source": "separated",
        "method": PARAMS["masking"]["pair_metric"],
        "band_centres_hz": centres,
        "stems_compared": names,
        "whole_track_band_energy_pct": {
            nm: [float(v) for v in 100.0 * _norm(whole[nm])] for nm in names},
        "pairs": pairs,
        "spectral_overlap": overlaps,
        "per_section": per_section,
        "masking_release": release,
    }
    if "vocals" in stems:
        v = stems["vocals"]
        bpm = (tempo or {}).get("bpm") if isinstance(tempo, dict) else None
        out["vocal"] = {
            "sibilance": _sibilance(v, collector),
            "high_pass": _highpass_corner(v),
            "reverb": _reverb_send(v, collector),
            "delay_throws": _delay_throws(v, bpm),
        }
    return out
