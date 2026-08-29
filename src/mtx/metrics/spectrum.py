"""4.5 Spectrum.

Two Welch resolutions are computed: a broadband pass over the whole file, and a
131072-point pass over an automatically chosen body section.  The second exists
because the first cannot separate a 62 Hz fundamental from a 70 Hz one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal as sps

from ..audio import AudioSource
from ..bands import get_band_pack
from ..dsp import (band_power, linear_fit_db_per_octave, log_smooth,
                   log_smooth_indices, spectrum_table, third_octave_edges,
                   welch_psd)
from ..params import BANDS, BARK_EDGES, PARAMS, THIRD_OCTAVE_CENTRES
from ..spectra import get_spectra
from ..util import Collector, db_pow, fmt_time, note_for_frequency, percentiles


def pick_body_section(src: AudioSource, target_s: float,
                      collector: Collector) -> tuple[int, int, float, float]:
    """Densest sustained ~target_s of material, by 20-2000 Hz RMS on a 1 s grid.

    Returns (start_sample, end_sample, start_s, end_s) at the band rate.
    """
    sr = src.band_sr
    x = src.band_mono
    n = x.size
    if n <= int(target_s * sr):
        return 0, n, 0.0, n / float(sr)
    from ..dsp import band_filter
    body = band_filter(x, sr, 20.0, 2000.0)
    w = int(sr)
    m = (body.size // w) * w
    frame_ms = np.mean(body[:m].reshape(-1, w) ** 2, axis=1)
    k = int(round(target_s))
    if frame_ms.size <= k:
        return 0, n, 0.0, n / float(sr)
    cs = np.concatenate([[0.0], np.cumsum(frame_ms)])
    window_sum = cs[k:] - cs[:-k]
    best = int(np.argmax(window_sum))
    a, b = best * w, min(n, (best + k) * w)
    return a, b, a / float(sr), b / float(sr)


def _peak_q(freqs: np.ndarray, db: np.ndarray, idx: int) -> float | None:
    """Q from the -3 dB width around a spectral peak."""
    top = db[idx]
    target = top - 3.0
    i = idx
    while i > 0 and db[i] > target:
        i -= 1
    j = idx
    while j < db.size - 1 and db[j] > target:
        j += 1
    f_lo, f_hi = freqs[i], freqs[j]
    if f_hi <= f_lo or freqs[idx] <= 0:
        return None
    return float(freqs[idx] / (f_hi - f_lo))


def _bass_fundamentals(freqs: np.ndarray, psd: np.ndarray,
                       collector: Collector) -> dict[str, Any]:
    P = PARAMS["spectrum"]
    lo, hi = P["bass_peak_search_hz"]
    m = (freqs >= lo) & (freqs <= hi)
    if m.sum() < 8:
        collector.warn("spectrum.bass_fundamentals",
                       "high-resolution spectrum has too few bins below 200 Hz")
        return {"peaks": [], "single_note": None, "resolution_hz": None}
    f = freqs[m]
    db = db_pow(np.maximum(psd[m], 1e-30))
    idx, props = sps.find_peaks(db, prominence=P["bass_peak_prominence_db"])
    if idx.size == 0:
        return {"peaks": [], "single_note": None,
                "resolution_hz": float(freqs[1] - freqs[0]) if freqs.size > 1 else None,
                "note": "no peak below 200 Hz cleared the prominence threshold"}
    order = np.argsort(db[idx])[::-1]
    strongest_db = float(db[idx[order[0]]])
    peaks = []
    for rank, o in enumerate(order[:12]):
        i = int(idx[o])
        name, midi, cents = note_for_frequency(float(f[i]))
        peaks.append({
            "rank": rank + 1,
            "frequency_hz": round(float(f[i]), 1),
            "level_db_rel_strongest": float(db[i]) - strongest_db,
            "prominence_db": float(props["prominences"][o]),
            "nearest_note": name,
            "cents_from_note": round(float(cents), 1),
            "q": _peak_q(f, db, i),
        })
    single = None
    if len(peaks) >= 2:
        single = bool(peaks[0]["level_db_rel_strongest"] - peaks[1]["level_db_rel_strongest"] > 6.0)
    elif len(peaks) == 1:
        single = True
    return {
        "peaks": peaks,
        "single_note": single,
        "single_note_rule": "top peak more than 6 dB above the next",
        "resolution_hz": float(f[1] - f[0]) if f.size > 1 else None,
    }


def _resonances(src: AudioSource, freqs: np.ndarray, psd: np.ndarray,
                collector: Collector) -> list[dict[str, Any]]:
    """Narrow persistent peaks, with the fraction of frames each appears in."""
    P = PARAMS["spectrum"]
    if freqs.size < 32:
        return []
    db = db_pow(np.maximum(psd, 1e-30))
    smooth = log_smooth(freqs, db, P["resonance_max_bandwidth_oct"] * 2)
    excess = db - smooth
    idx, props = sps.find_peaks(excess, prominence=P["resonance_prominence_db"])
    if idx.size == 0:
        return []
    order = np.argsort(props["prominences"])[::-1][:20]
    # Per-frame presence, from a coarse STFT of the band-rate mono signal.
    sr = src.band_sr
    nper = 8192
    x = src.band_mono
    presence: dict[int, float] = {}
    if x.size >= nper * 2:
        f_t, _, Z = sps.stft(x, fs=sr, nperseg=nper, noverlap=0, boundary=None,
                             padded=False)
        mag = db_pow(np.maximum(np.abs(Z) ** 2, 1e-30))
        # One set of window bounds for the whole STFT, smoothed in one pass.
        widx = log_smooth_indices(f_t, P["resonance_max_bandwidth_oct"] * 2)
        ex = mag - log_smooth(f_t, mag, P["resonance_max_bandwidth_oct"] * 2, widx)
        for o in order:
            fi = int(np.argmin(np.abs(f_t - freqs[idx[o]])))
            presence[int(idx[o])] = float(np.mean(ex[fi] > P["resonance_prominence_db"] * 0.6))
    rows = []
    for o in order:
        i = int(idx[o])
        rows.append({
            "frequency_hz": round(float(freqs[i]), 1),
            "prominence_db": float(props["prominences"][o]),
            "q": _peak_q(freqs, db, i),
            "frame_presence_fraction": presence.get(i),
            "level_db": float(db[i]),
        })
    rows.sort(key=lambda r: -r["prominence_db"])
    return rows


def _descriptor_timeline(src: AudioSource, profile: str = "full") -> dict[str, Any]:
    """Centroid/spread/skew/kurtosis/flatness/rolloff/ZCR, whole track and per second."""
    sr = src.band_sr
    x = src.band_mono
    hop = int(round(PARAMS["spectrum"]["descriptor_timeline_hop_s"] * sr))
    nper = 4096
    if x.size < nper:
        return {"available": False, "reason": "file shorter than one 4096-sample window"}
    if profile == "quick":
        f, p = welch_psd(x, sr, 16384)
        whole = _descriptors_from_psd(f, p)
        whole["zcr"] = float(np.mean(np.abs(np.diff(np.signbit(x))))) if x.size > 1 else None
        return {"available": True, "whole_track": whole, "timeline": None,
                "timeline_skipped_reason": "skipped by --profile quick"}
    n_frames = max(1, x.size // hop)
    keys = ["centroid_hz", "spread_hz", "skewness", "kurtosis", "flatness",
            "rolloff85_hz", "rolloff95_hz", "rolloff99_hz", "zcr"]
    series: dict[str, list[float]] = {k: [] for k in keys}
    times: list[float] = []
    for i in range(n_frames):
        seg = x[i * hop : (i + 1) * hop]
        if seg.size < nper:
            break
        f, p = welch_psd(seg, sr, nper)
        vals = _descriptors_from_psd(f, p)
        zc = float(np.mean(np.abs(np.diff(np.signbit(seg))))) if seg.size > 1 else 0.0
        vals["zcr"] = zc
        for k in keys:
            series[k].append(vals.get(k, float("nan")))
        times.append(i * hop / float(sr))
    f, p = welch_psd(x, sr, 16384)
    whole = _descriptors_from_psd(f, p)
    whole["zcr"] = float(np.mean(np.abs(np.diff(np.signbit(x))))) if x.size > 1 else None
    return {
        "available": True,
        "hop_s": PARAMS["spectrum"]["descriptor_timeline_hop_s"],
        "window": "Welch nperseg=4096 within each 1 s frame; whole-track values use nperseg=16384",
        "whole_track": whole,
        "times_s": times,
        "timeline": series,
        "percentiles": {k: percentiles(np.array(v, dtype=float), [10, 50, 90])
                        for k, v in series.items()},
    }


def _descriptors_from_psd(f: np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    if f.size < 4 or p.sum() <= 0:
        return {}
    m = f > 0
    f, p = f[m], np.maximum(p[m], 1e-30)
    tot = float(p.sum())
    centroid = float(np.sum(f * p) / tot)
    spread = float(np.sqrt(np.sum(((f - centroid) ** 2) * p) / tot))
    if spread > 0:
        skew = float(np.sum(((f - centroid) ** 3) * p) / (tot * spread ** 3))
        kurt = float(np.sum(((f - centroid) ** 4) * p) / (tot * spread ** 4))
    else:
        skew = kurt = None
    flatness = float(np.exp(np.mean(np.log(p))) / np.mean(p))
    c = np.cumsum(p) / tot
    out = {"centroid_hz": centroid, "spread_hz": spread, "skewness": skew,
           "kurtosis": kurt, "flatness": flatness}
    for q in (85, 95, 99):
        k = int(np.searchsorted(c, q / 100.0))
        out[f"rolloff{q}_hz"] = float(f[min(k, f.size - 1)])
    return out


def _band_timeline(src: AudioSource) -> dict[str, Any]:
    pack = get_band_pack(src)
    hop = PARAMS["spectrum"]["band_timeline_hop_ms"] / 1000.0
    out: dict[str, Any] = {"hop_ms": PARAMS["spectrum"]["band_timeline_hop_ms"],
                           "bands": {}, "times_s": None}
    for name in pack.names:
        t, e = pack.resample_envelope(name, hop)
        if out["times_s"] is None:
            out["times_s"] = t
        out["bands"][name] = db_pow(np.maximum(e ** 2, 1e-30))
    return out


def analyse(src: AudioSource, collector: Collector, profile: str = "full") -> dict[str, Any]:
    P = PARAMS["spectrum"]
    sr = src.band_sr
    nyq = sr / 2.0

    pack = get_spectra(src)
    freqs = pack.freqs
    psds: dict[str, tuple[np.ndarray, np.ndarray]] = {
        name: (freqs, p) for name, p in pack.psd.items()
    }
    ltas: dict[str, Any] = {}
    if freqs.size == 0:
        collector.warn("spectrum", "file too short for a Welch spectrum; "
                                   "all spectral metrics are null")
        return {"available": False, "reason": "file too short for Welch analysis"}

    ltas["broadband"] = {
        "params": pack.params(),
        "sample_rate_hz": sr,
        "resolution_hz": float(freqs[1] - freqs[0]) if freqs.size > 1 else None,
        "frequencies_hz": freqs,
        "psd_db": {k: db_pow(np.maximum(v[1], 1e-30)) for k, v in psds.items()},
    }

    # --- 8-band table -------------------------------------------------------
    band_edges = [(0.5 * (lo + hi), lo, hi) for _, lo, hi in BANDS]
    band_tables: dict[str, Any] = {}
    for name, (f, p) in psds.items():
        rows = spectrum_table(f, p, band_edges, nyq)
        for row, (bname, lo, hi) in zip(rows, BANDS):
            row["band"] = bname
        band_tables[name] = rows

    # --- third-octave and Bark ---------------------------------------------
    to_edges = third_octave_edges(THIRD_OCTAVE_CENTRES)
    third_octave: dict[str, Any] = {}
    for name in ("mid", "side"):
        f, p = psds[name]
        rows = spectrum_table(f, p, to_edges, nyq)
        vals = [r["db"] for r in rows if r["db"] is not None]
        peak = max(vals) if vals else None
        for r_ in rows:
            r_["db_rel_loudest"] = (r_["db"] - peak) if (r_["db"] is not None and peak is not None) else None
        third_octave[name] = rows
    # side/mid per third-octave, the number the stereo section leans on
    sm_rows = []
    for rm, rs in zip(third_octave["mid"], third_octave["side"]):
        val = None
        if rm["power"] and rs["power"] and rm["power"] > 0 and rs["power"] > 0:
            val = db_pow(rs["power"] / rm["power"])
        sm_rows.append({"centre_hz": rm["centre_hz"], "side_minus_mid_db": val})

    bark_edges = [(0.5 * (BARK_EDGES[i] + BARK_EDGES[i + 1]), BARK_EDGES[i], BARK_EDGES[i + 1])
                  for i in range(len(BARK_EDGES) - 1)]
    bark_edges.append((17750.0, 15500.0, 20000.0))
    bark: dict[str, Any] = {}
    for name in ("mid", "side"):
        f, p = psds[name]
        rows = spectrum_table(f, p, bark_edges, nyq)
        for i, r_ in enumerate(rows):
            r_["bark_band"] = i + 1
        bark[name] = rows

    # --- tilt ---------------------------------------------------------------
    f_m, p_m = psds["mid"]
    db_m = db_pow(np.maximum(p_m, 1e-30))
    lo_f, hi_f = P["tilt_fit_range_hz"]
    slope, r2 = linear_fit_db_per_octave(f_m, db_m, lo_f, min(hi_f, nyq * 0.98))
    piecewise = []
    for a, b in P["tilt_piecewise_hz"]:
        s, rr = linear_fit_db_per_octave(f_m, db_m, a, min(b, nyq * 0.98))
        piecewise.append({"low_hz": a, "high_hz": b, "slope_db_per_oct": s, "r2": rr})
    if r2 is not None and r2 < 0.5:
        collector.low_confidence("spectrum.tilt", "low",
                                 f"tilt fit R2 is {r2:.2f}; a single slope describes "
                                 "this spectrum poorly, read the band table instead")

    # --- high-resolution low-end pass --------------------------------------
    lowres: dict[str, Any] = {"computed": False,
                              "reason": "skipped by --profile quick"}
    bass = {"peaks": [], "single_note": None}
    if profile != "quick":
        LP = P["ltas_lowfreq"]
        a, b, t0, t1 = pick_body_section(src, LP["section_target_s"], collector)
        seg = src.band_mid[a:b]
        f_hi, p_hi = welch_psd(seg, sr, LP["nperseg"])
        if f_hi.size == 0:
            collector.warn("spectrum.ltas_lowfreq",
                           "section shorter than the 131072-point window; "
                           "high-resolution low end unavailable")
            lowres = {"computed": False,
                      "reason": "section shorter than nperseg=131072"}
        else:
            lowres = {
                "computed": True,
                "params": LP,
                "section_start_s": t0, "section_end_s": t1,
                "section_start": fmt_time(t0), "section_end": fmt_time(t1),
                "section_duration_s": t1 - t0,
                "resolution_hz": float(f_hi[1] - f_hi[0]),
                "frequencies_hz": f_hi[f_hi <= 500.0],
                "psd_db_mid": db_pow(np.maximum(p_hi[f_hi <= 500.0], 1e-30)),
            }
            bass = _bass_fundamentals(f_hi, p_hi, collector)

    total_mono = band_power(freqs, psds["mono"][1], 20.0, min(20000.0, nyq))
    air = band_power(freqs, psds["mono"][1], 12000.0, min(20000.0, nyq))
    sub = band_power(freqs, psds["mono"][1], 20.0, 60.0)

    return {
        "available": True,
        "ltas": ltas,
        "ltas_lowfreq": lowres,
        "band_energy": {
            "definition": "power integrated between the band edges of the Welch PSD",
            "bands_hz": [[n, lo, hi] for n, lo, hi in BANDS],
            "tables": band_tables,
        },
        "air_band_pct": (100.0 * air / total_mono) if total_mono > 0 else None,
        "sub_band_pct": (100.0 * sub / total_mono) if total_mono > 0 else None,
        "third_octave": {"reference": "dB relative to the loudest band of that signal",
                         "mid": third_octave["mid"], "side": third_octave["side"],
                         "side_minus_mid_db": sm_rows},
        "bark": bark,
        "tilt": {
            "fit_range_hz": [lo_f, hi_f],
            "slope_db_per_oct": slope,
            "r2": r2,
            "measured_on": "mid",
            "piecewise": piecewise,
        },
        "bass_fundamentals": bass,
        "resonances": (_resonances(src, freqs, psds["mid"][1], collector)
                       if profile != "quick" else []),
        "resonances_skipped": profile == "quick",
        "descriptors": _descriptor_timeline(src, profile),
        "band_timeline": _band_timeline(src),
    }
