"""4.8 Processing forensics.

Everything in this module is an inference from a measurement, not a
measurement of the processor.  Each result carries the method that produced it,
a confidence, and the reason whenever that confidence is below high.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scipy import signal as sps

from ..audio import AudioSource
from ..bands import get_band_pack
from ..dsp import band_filter, octave_band_edges
from ..params import BANDS, PARAMS
from ..util import Collector, db_amp, db_pow


def _frame_levels(x: np.ndarray, sr: float, frame_s: float) -> np.ndarray:
    w = max(1, int(round(frame_s * sr)))
    n = (x.size // w) * w
    if n == 0:
        return np.zeros(0)
    return np.sqrt(np.mean(x[:n].reshape(-1, w) ** 2, axis=1))


def _fit_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    """Least-squares slope, closed form.

    The reverb estimator fits thousands of short decays; np.polyfit's setup
    cost dominates at that size, and the closed form is the same answer.
    """
    n = x.size
    if n < 2:
        return None
    mx = x.mean()
    my = y.mean()
    dx = x - mx
    den = float(np.dot(dx, dx))
    if den <= 0:
        return None
    return float(np.dot(dx, y - my) / den)


def _regress(xdb: np.ndarray, ydb: np.ndarray) -> tuple[float | None, float | None, int]:
    m = np.isfinite(xdb) & np.isfinite(ydb)
    if m.sum() < 20:
        return None, None, int(m.sum())
    a = np.vstack([xdb[m], np.ones(int(m.sum()))]).T
    coef, *_ = np.linalg.lstsq(a, ydb[m], rcond=None)
    pred = a @ coef
    ss_res = float(np.sum((ydb[m] - pred) ** 2))
    ss_tot = float(np.sum((ydb[m] - np.mean(ydb[m])) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return float(coef[0]), r2, int(m.sum())


def _saturation(src: AudioSource, sections: list[dict[str, Any]],
                collector: Collector) -> dict[str, Any]:
    P = PARAMS["processing"]["saturation"]
    sr = src.band_sr
    x = src.band_mono
    lo, hi = P["hf_band_hz"]
    if lo >= sr / 2.0:
        return {"available": False, "reason": "HF band above Nyquist"}
    hf = band_filter(x, sr, lo, min(hi, sr / 2.0 * 0.98))
    frame_s = P["frame_ms"] / 1000.0
    bb = _frame_levels(x, sr, frame_s)
    hh = _frame_levels(hf, sr, frame_s)
    n = min(bb.size, hh.size)
    if n < 20:
        collector.warn("processing.saturation", "fewer than 20 frames; slope is null")
        return {"available": False, "reason": "file too short"}
    # Frames near digital silence carry no level-dependence information.
    keep = (bb[:n] > 1e-5) & (hh[:n] > 1e-7)
    xdb = db_amp(np.maximum(bb[:n], 1e-20))
    ydb = db_amp(np.maximum(hh[:n], 1e-20))
    slope, r2, used = _regress(np.where(keep, xdb, np.nan), np.where(keep, ydb, np.nan))
    per_section = []
    for s in sections:
        a = int(s["start_s"] / frame_s)
        b = int(s["end_s"] / frame_s)
        if b - a < 20:
            continue
        sl, rr, u = _regress(np.where(keep[a:b], xdb[a:b], np.nan),
                             np.where(keep[a:b], ydb[a:b], np.nan))
        per_section.append({"index": s.get("index"), "start_s": s["start_s"],
                            "slope_db_per_db": sl, "r2": rr, "frames": u})
    conf = "medium" if (r2 or 0) >= 0.5 else "low"
    collector.low_confidence("processing.saturation", conf,
                             f"regression R2 is {r2:.2f}" if r2 is not None
                             else "regression did not converge")
    return {
        "available": True,
        "params": P,
        "slope_db_per_db": slope,
        "r2": r2,
        "frames_used": used,
        "per_section": per_section,
        "method": "least-squares regression of 5-10 kHz frame level (dB) on "
                  "broadband frame level (dB) over 50 ms frames",
        "reading": "slope above 1 means the material gets brighter as it gets "
                   "louder; below 1 means it gets duller. The slope is reported "
                   "raw; no threshold is applied.",
        "confidence": conf,
        "confidence_reason": "the slope also moves with arrangement (a chorus adds "
                             "cymbals as well as level), so it is an indication of "
                             "level-dependent HF behaviour, not proof of saturation",
    }


def _pumping(src: AudioSource, collector: Collector) -> dict[str, Any]:
    P = PARAMS["processing"]["pumping"]
    sr = src.band_sr
    hop_s = P["envelope_hop_ms"] / 1000.0
    lf = _frame_levels(band_filter(src.band_mono, sr, *P["lf_band_hz"]), sr, hop_s)
    md = _frame_levels(band_filter(src.band_mono, sr,
                                   P["mid_band_hz"][0],
                                   min(P["mid_band_hz"][1], sr / 2 * 0.98)), sr, hop_s)
    n = min(lf.size, md.size)
    if n < 200:
        collector.warn("processing.pumping", "fewer than 200 envelope frames; null")
        return {"available": False, "reason": "file too short"}
    a = db_amp(np.maximum(lf[:n], 1e-20))
    b = db_amp(np.maximum(md[:n], 1e-20))
    a = a - a.mean()
    b = b - b.mean()
    max_lag = int(round(P["lag_range_ms"][1] / 1000.0 / hop_s))
    lags = np.arange(-max_lag, max_lag + 1)
    denom = np.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    if denom <= 0:
        return {"available": False, "reason": "flat envelopes"}
    corr = sps.correlate(b, a, mode="full", method="fft") / denom
    mid = n - 1
    seg = corr[mid - max_lag : mid + max_lag + 1]
    k_neg = int(np.argmin(seg))
    k_pos = int(np.argmax(seg))
    lag_ms = float(lags[k_neg] * hop_s * 1000.0)
    # Depth and recovery: mid-band dip following the loudest bass hits.
    depth_db, release_ms = None, None
    thr = np.percentile(lf[:n], 98)
    hits = np.flatnonzero(lf[:n] >= thr)
    if hits.size >= 3:
        span = int(round(0.4 / hop_s))
        dips, rels = [], []
        for h in hits[:200]:
            w = b[h : h + span]
            if w.size < 8:
                continue
            base = float(np.median(b[max(0, h - span) : h])) if h > 8 else 0.0
            dip = float(np.min(w)) - base
            dips.append(dip)
            j = int(np.argmin(w))
            rec = w[j:]
            tgt = base - abs(dip) * 0.37  # 1/e recovery
            idx = np.flatnonzero(rec >= tgt)
            if idx.size:
                rels.append(float(idx[0] * hop_s * 1000.0))
        depth_db = float(np.median(dips)) if dips else None
        release_ms = float(np.median(rels)) if rels else None
    conf = "medium" if abs(float(seg[k_neg])) > 0.15 else "low"
    collector.low_confidence("processing.pumping", conf,
                             "envelope cross-correlation reflects arrangement as "
                             "well as gain reduction")
    return {
        "available": True,
        "params": P,
        # "most negative" is literal: on material with no ducking the minimum
        # over the lag range can itself be positive, and that is the finding.
        "most_negative_correlation": float(seg[k_neg]),
        "most_negative_lag_ms": lag_ms,
        "most_positive_correlation": float(seg[k_pos]),
        "most_positive_lag_ms": float(lags[k_pos] * hop_s * 1000.0),
        "zero_lag_correlation": float(seg[max_lag]),
        "dip_depth_db": depth_db,
        "estimated_release_ms": release_ms,
        "release_method": "median time for the 500 Hz-6 kHz envelope to recover 63% "
                          "of its dip after the loudest 2% of sub-120 Hz frames",
        "reading": "a mid-band envelope that dips after bass hits (negative "
                   "correlation at a positive lag) is consistent with bus "
                   "compression or a sidechain",
        "confidence": conf,
        "confidence_reason": "arrangement can produce the same envelope relationship "
                             "without any compressor",
    }


def _modulation(src: AudioSource, tempo: dict[str, Any],
                collector: Collector) -> dict[str, Any]:
    P = PARAMS["processing"]["modulation"]
    pack = get_band_pack(src)
    hop_s = pack.hop_s
    fs_env = 1.0 / hop_s
    bpm = tempo.get("bpm") if tempo.get("available") else None
    beat_hz = (bpm / 60.0) if bpm else None
    out: dict[str, Any] = {
        "available": True, "params": P, "envelope_rate_hz": fs_env,
        "beat_rate_hz": beat_hz, "bands": {},
        "method": "FFT of each band's 5 ms RMS envelope (modulation spectrum); depth "
                  "is the magnitude at the rate relative to the envelope mean",
        "confidence": "medium" if beat_hz else "low",
        "confidence_reason": ("no tempo estimate available, so beat-rate bins could "
                              "not be located" if not beat_hz else
                              "modulation at the beat rate is produced by the "
                              "arrangement as well as by a sidechain"),
    }
    if beat_hz is None:
        out["available"] = False
        collector.low_confidence("processing.modulation", "low", out["confidence_reason"])
        return out
    rates = {"beat": beat_hz, "half_beat": beat_hz * 2.0, "quarter_beat": beat_hz * 4.0}
    bt = tempo.get("beat_times_s")
    beat_times = np.asarray(bt if bt is not None else [], dtype=float)
    for name in pack.names:
        e = pack.envelopes[name]
        if e.size < 512:
            out["bands"][name] = {"note": "band envelope too short"}
            continue
        e = e - e.mean()
        win = np.hanning(e.size)
        spec = np.abs(np.fft.rfft(e * win)) / (0.5 * e.size)
        freqs = np.fft.rfftfreq(e.size, hop_s)
        ref = float(np.sqrt(np.mean(pack.envelopes[name] ** 2)))
        row: dict[str, Any] = {}
        for rname, rhz in rates.items():
            if rhz >= fs_env / 2:
                row[f"{rname}_depth_db"] = None
                continue
            i = int(np.argmin(np.abs(freqs - rhz)))
            lo, hi = max(0, i - 2), min(freqs.size, i + 3)
            amp = float(np.max(spec[lo:hi]))
            row[f"{rname}_hz"] = float(rhz)
            row[f"{rname}_depth_db"] = db_amp(amp / ref) if ref > 0 else None
        # Phase of the envelope dip relative to the beat grid.
        if beat_times.size > 4:
            period = 60.0 / bpm
            t = np.arange(pack.envelopes[name].size) * hop_s
            phase = ((t - beat_times[0]) % period) / period
            bins = np.clip((phase * 16).astype(int), 0, 15)
            prof = np.array([float(np.mean(pack.envelopes[name][bins == k]))
                             if np.any(bins == k) else np.nan for k in range(16)])
            if np.all(np.isfinite(prof)):
                row["dip_phase_fraction_of_beat"] = float(np.argmin(prof) / 16.0)
                row["beat_profile_depth_db"] = db_amp(float(np.min(prof)) /
                                                      float(np.max(prof))) if prof.max() > 0 else None
        out["bands"][name] = row
    return out


def _multiband_timeline(src: AudioSource) -> dict[str, Any]:
    hop = PARAMS["processing"]["multiband_timeline_hop_ms"] / 1000.0
    pack = get_band_pack(src)
    out: dict[str, Any] = {"hop_ms": PARAMS["processing"]["multiband_timeline_hop_ms"],
                           "times_s": None, "rms_db": {}, "crest_db": {}}
    for name in pack.names:
        t, e = pack.resample_envelope(name, hop)
        if out["times_s"] is None:
            out["times_s"] = t
        out["rms_db"][name] = db_amp(np.maximum(e, 1e-20))
        # Crest over a 1 s sliding window of the 10 ms envelope.
        k = max(2, int(round(1.0 / hop)))
        if e.size >= k:
            n = (e.size // k) * k
            fr = e[:n].reshape(-1, k)
            pk = fr.max(axis=1)
            rm = np.sqrt(np.mean(fr ** 2, axis=1))
            out["crest_db"][name] = np.where(rm > 0, db_amp(np.maximum(pk, 1e-20)) -
                                             db_amp(np.maximum(rm, 1e-20)), np.nan)
        else:
            out["crest_db"][name] = np.zeros(0)
    # How independently the bands move: correlation of their dB envelopes.
    names = [n for n in pack.names if out["rms_db"][n].size > 8]
    if len(names) > 1:
        L = min(out["rms_db"][n].size for n in names)
        M = np.vstack([out["rms_db"][n][:L] for n in names])
        M = M - M.mean(axis=1, keepdims=True)
        sd = M.std(axis=1)
        ok = sd > 0
        C = np.corrcoef(M[ok]) if ok.sum() > 1 else np.zeros((0, 0))
        iu = np.triu_indices(C.shape[0], 1) if C.size else ([], [])
        kept = [n for n, o in zip(names, ok) if o]
        vals = C[iu] if C.size else np.zeros(0)
        # The off-diagonal summary and the extreme pairs carry nearly all of
        # what the full matrix says, in a form that fits the digest.
        order = np.argsort(vals) if vals.size else np.zeros(0, dtype=int)

        def _pair(idx: int) -> dict[str, Any]:
            i, j = int(iu[0][idx]), int(iu[1][idx])
            return {"bands": [kept[i], kept[j]], "r": float(vals[idx])}

        out["band_envelope_correlation"] = {
            "bands": kept,
            "matrix": C,
            "mean_offdiagonal": float(np.mean(vals)) if vals.size else None,
            "offdiagonal": {
                "min": float(np.min(vals)) if vals.size else None,
                "median": float(np.median(vals)) if vals.size else None,
                "max": float(np.max(vals)) if vals.size else None,
                "mean": float(np.mean(vals)) if vals.size else None,
                "pairs": int(vals.size),
            },
            "least_correlated_pairs": [_pair(int(k)) for k in order[:2]],
            "most_correlated_pair": _pair(int(order[-1])) if vals.size else None,
            "reading": "bands that move together indicate broadband processing; "
                       "bands that move independently indicate multiband",
        }
    return out


def _hpss(src: AudioSource, collector: Collector) -> dict[str, Any]:
    try:
        import librosa
    except ImportError:
        return {"available": False, "reason": "librosa not installed"}
    y, sr = src.lib_mono, src.lib_sr
    if y.size < sr:
        return {"available": False, "reason": "file too short"}
    # A coarser hop than the rest of the librosa work: the HPSS median filters
    # are quadratic in the frame count and the outputs here are per-second
    # energy ratios, which a 46 ms hop resolves perfectly well.
    hop = PARAMS["processing"]["hpss"]["hop_length"]
    n_fft = PARAMS["general"]["librosa_n_fft"]
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    H, Pc = librosa.decompose.hpss(S, kernel_size=PARAMS["processing"]["hpss"]["kernel_size"],
                                   power=PARAMS["processing"]["hpss"]["power"])
    hmag, pmag = np.abs(H) ** 2, np.abs(Pc) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    eh, ep = float(hmag.sum()), float(pmag.sum())
    per_band = {}
    for name, lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        if not m.any():
            per_band[name] = None
            continue
        a, b = float(hmag[m].sum()), float(pmag[m].sum())
        per_band[name] = db_pow(b / a) if a > 0 and b > 0 else None
    # Vocal-band proxy: harmonic energy in 1-4 kHz relative to total, per second.
    lo, hi = PARAMS["processing"]["vocal_band_hz"]
    vm = (freqs >= lo) & (freqs < hi)
    frames_per_s = max(1, int(round(sr / hop)))
    vocal_times, vocal_vals = [], []
    total_per_frame = hmag.sum(axis=0) + pmag.sum(axis=0)
    voc_per_frame = hmag[vm].sum(axis=0)
    n_sec = hmag.shape[1] // frames_per_s
    for i in range(n_sec):
        a = slice(i * frames_per_s, (i + 1) * frames_per_s)
        tot = float(total_per_frame[a].sum())
        vocal_times.append(float(i))
        vocal_vals.append(db_pow(float(voc_per_frame[a].sum()) / tot) if tot > 0 else None)
    return {
        "available": True,
        "method": "librosa.decompose.hpss on a 2048-point STFT at 22.05 kHz",
        "percussive_to_harmonic_db": db_pow(ep / eh) if eh > 0 and ep > 0 else None,
        "percussive_energy_fraction": (ep / (ep + eh)) if (ep + eh) > 0 else None,
        "per_band_percussive_to_harmonic_db": per_band,
        "vocal_band_proxy": {
            "band_hz": [lo, hi],
            "definition": "harmonic-component energy in 1-4 kHz relative to total "
                          "energy, per second, in dB",
            "times_s": vocal_times, "values_db": vocal_vals,
            "confidence": "low",
            "confidence_reason": "the 1-4 kHz harmonic band contains guitars, keys "
                                 "and synths as well as voice; this is a proxy, not "
                                 "a vocal level",
        },
        "confidence": "medium",
        "confidence_reason": "HPSS separation leaks between components on dense material",
    }


def _reverb(src: AudioSource, collector: Collector) -> dict[str, Any]:
    P = PARAMS["processing"]["reverb"]
    sr = src.band_sr
    x = src.band_mono
    hop = 0.002
    results = []
    events_total = 0
    for centre, lo, hi in octave_band_edges(P["octave_bands_hz"]):
        if lo >= sr / 2.0:
            results.append({"centre_hz": centre, "t20_s": None, "t30_s": None,
                            "note": "band above Nyquist"})
            continue
        y = band_filter(x, sr, lo, min(hi, sr / 2.0 * 0.98))
        env = _frame_levels(y, sr, hop)
        if env.size < 200:
            results.append({"centre_hz": centre, "t20_s": None, "t30_s": None,
                            "note": "too few frames"})
            continue
        edb = db_amp(np.maximum(env, 1e-20))
        span = int(round(P["decay_min_ms"] / 1000.0 / hop))
        t20s, t30s, early_late = [], [], []
        # Candidate decays: a local envelope maximum followed by a monotone fall.
        peaks = np.flatnonzero((edb[1:-1] > edb[:-2]) & (edb[1:-1] >= edb[2:])) + 1
        strong = peaks[edb[peaks] > np.percentile(edb, 90)] if peaks.size else peaks
        for p in strong[:400]:
            seg = env[p : p + span * 3]
            if seg.size < span or seg[0] <= 0:
                continue
            # Schroeder reverse integration of the squared envelope.
            e2 = seg ** 2
            sch = np.cumsum(e2[::-1])[::-1]
            sch_db = db_pow(np.maximum(sch / sch[0], 1e-12))
            t = np.arange(sch_db.size) * hop
            for rng, store, factor in ((P["t20_range_db"], t20s, 3.0),
                                       (P["t30_range_db"], t30s, 2.0)):
                m = (sch_db <= rng[0]) & (sch_db >= rng[1])
                if m.sum() < 6:
                    continue
                slope = _fit_slope(t[m], sch_db[m])
                if slope is None or slope >= -1e-6:
                    continue
                store.append(float(factor * (rng[1] - rng[0]) / slope))
            split = int(round(P["early_late_split_ms"] / 1000.0 / hop))
            if e2.size > split * 2:
                early = float(e2[:split].sum())
                late = float(e2[split:].sum())
                if late > 0:
                    early_late.append(db_pow(early / late))
        events_total += len(t20s)
        results.append({
            "centre_hz": centre,
            "events_used": len(t20s),
            "t20_s": float(np.median(t20s)) if t20s else None,
            "t30_s": float(np.median(t30s)) if t30s else None,
            "t20_iqr_s": (float(np.percentile(t20s, 75) - np.percentile(t20s, 25))
                          if len(t20s) > 3 else None),
            "early_to_late_db": float(np.median(early_late)) if early_late else None,
        })
    # Stereo decorrelation of the tails.
    tail_corr = None
    if src.n_ch >= 2:
        L, R = src.band_x[:, 0], src.band_x[:, 1]
        env = _frame_levels(src.band_mono, sr, 0.05)
        if env.size > 20:
            quiet = np.flatnonzero(env < np.percentile(env, 25))
            w = int(round(0.05 * sr))
            cs = []
            for q in quiet[:300]:
                a, b = q * w, (q + 1) * w
                if b > L.size:
                    break
                la, ra = L[a:b] - L[a:b].mean(), R[a:b] - R[a:b].mean()
                d = np.sqrt(float(np.dot(la, la)) * float(np.dot(ra, ra)))
                if d > 0:
                    cs.append(float(np.dot(la, ra) / d))
            tail_corr = float(np.median(cs)) if cs else None
    conf = "low" if events_total < 40 else "medium"
    collector.low_confidence("processing.reverb", conf,
                             f"{events_total} usable decays across all bands; on dense "
                             "material the decay after an onset is masked by the next "
                             "one, so T20/T30 are upper bounds at best")
    return {
        "available": True,
        "params": P,
        "per_octave_band": results,
        "tail_stereo_correlation": tail_corr,
        "tail_correlation_method": "median L/R correlation over the quietest quartile "
                                   "of 50 ms frames",
        "method": "Schroeder reverse integration of the squared envelope after strong "
                  "onsets, per octave band, 2 ms envelope resolution",
        "confidence": conf,
        "confidence_reason": "reverberation time is being estimated from programme "
                             "material, not from an impulse response",
    }


def _transient_density(src: AudioSource) -> dict[str, Any]:
    pack = get_band_pack(src)
    hop = pack.hop_s
    out: dict[str, Any] = {"hop_s": 1.0, "bands": {}, "rate_per_s": {},
                           "method": "per-band envelope rises of at least 6 dB within "
                                     "20 ms, counted per second"}
    step = max(1, int(round(0.020 / hop)))
    for name in pack.names:
        e = pack.envelopes[name]
        if e.size < step * 4:
            out["bands"][name] = []
            out["rate_per_s"][name] = None
            continue
        db = db_amp(np.maximum(e, 1e-20))
        rise = db[step:] - db[:-step]
        hits = np.flatnonzero(rise > 6.0)
        if hits.size:
            keep = np.concatenate([[True], np.diff(hits) > step])
            hits = hits[keep]
        per_s = int(round(1.0 / hop))
        n_sec = max(1, e.size // per_s)
        counts = np.zeros(n_sec)
        for h in hits:
            k = min(n_sec - 1, h // per_s)
            counts[k] += 1
        out["bands"][name] = counts
        out["rate_per_s"][name] = float(hits.size / (e.size * hop)) if e.size else None
    return out


def analyse(src: AudioSource, collector: Collector, structure: dict[str, Any],
            profile: str = "full") -> dict[str, Any]:
    sections = structure.get("sections", []) if structure.get("available") else []
    out: dict[str, Any] = {
        "saturation_proxy": _saturation(src, sections, collector),
        "bus_compression": _pumping(src, collector),
    }
    if profile == "quick":
        for k in ("modulation_spectrum", "multiband_timeline", "hpss", "reverb",
                  "transient_density"):
            out[k] = {"available": False, "reason": "skipped by --profile quick"}
        return out
    out["modulation_spectrum"] = _modulation(src, structure.get("tempo", {}), collector)
    out["multiband_timeline"] = _multiband_timeline(src)
    out["hpss"] = _hpss(src, collector)
    out["reverb"] = _reverb(src, collector)
    out["transient_density"] = _transient_density(src)
    return out
