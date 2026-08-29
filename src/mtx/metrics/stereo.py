"""4.6 Stereo field.

Convention, stated in the output as well as here: mid = (L+R)/2, side = (L-R)/2.
Every side/mid figure is 20*log10(rms(side)/rms(mid)) or its per-band
equivalent, so 0 dB means equal side and mid energy and -inf means mono.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal as sps

from ..audio import AudioSource
from ..dsp import (band_power, block_loudness, gated_integrated, spectrum_table,
                   third_octave_edges)
from ..params import BANDS, PARAMS, THIRD_OCTAVE_CENTRES
from ..spectra import get_spectra
from ..util import Collector, db_amp, db_pow, fmt_time, percentiles

MONO_RESULT = {
    "available": False,
    "reason": "file has a single channel; every stereo metric is undefined",
}


def _side_mid_db(mid: np.ndarray, side: np.ndarray) -> float | None:
    pm = float(np.mean(mid * mid))
    ps = float(np.mean(side * side))
    if pm <= 0:
        return None
    if ps <= 0:
        return -200.0
    return db_pow(ps / pm)


def _band_side_mid(f: np.ndarray, p_ll: np.ndarray, p_rr: np.ndarray,
                   p_lr: np.ndarray, lo: float, hi: float) -> float | None:
    """Side/mid in dB over one band, from the L/R auto- and cross-spectra."""
    e_ll = band_power(f, p_ll, lo, hi)
    e_rr = band_power(f, p_rr, lo, hi)
    e_lr = band_power(f, p_lr, lo, hi)
    e_mid = (e_ll + e_rr + 2 * e_lr) / 4.0
    e_side = (e_ll + e_rr - 2 * e_lr) / 4.0
    if e_mid <= 0:
        return None
    return db_pow(e_side / e_mid) if e_side > 0 else -200.0


def _correlation_timeline(L: np.ndarray, R: np.ndarray, sr: int,
                          win_s: float) -> tuple[np.ndarray, np.ndarray]:
    w = max(2, int(round(win_s * sr)))
    n = (L.size // w) * w
    if n == 0:
        return np.zeros(0), np.zeros(0)
    a = L[:n].reshape(-1, w)
    b = R[:n].reshape(-1, w)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = np.sum(a * b, axis=1)
    den = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
    out = np.full(num.shape, np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return np.arange(a.shape[0]) * w / float(sr), out


def _itd(L: np.ndarray, R: np.ndarray, sr: int, search_ms: float) -> dict[str, Any]:
    """Inter-channel time offset by cross-correlation over +/- search_ms."""
    max_lag = int(round(search_ms / 1000.0 * sr))
    if L.size < 4 * max_lag or max_lag < 1:
        return {"lag_samples": None, "lag_us": None, "correlation_at_lag": None,
                "reason": "file too short for the requested lag search"}
    # Use a central excerpt: enough for a stable estimate, cheap to correlate.
    n = min(L.size, sr * 30)
    s = max(0, (L.size - n) // 2)
    a = L[s : s + n] - np.mean(L[s : s + n])
    b = R[s : s + n] - np.mean(R[s : s + n])
    corr = sps.correlate(a, b, mode="full", method="fft")
    mid = a.size - 1
    seg = corr[mid - max_lag : mid + max_lag + 1]
    norm = np.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    k = int(np.argmax(np.abs(seg)))
    lag = k - max_lag
    return {
        "lag_samples": int(lag),
        "lag_us": float(lag / sr * 1e6),
        "correlation_at_lag": float(seg[k] / norm) if norm > 0 else None,
        "correlation_at_zero": float(seg[max_lag] / norm) if norm > 0 else None,
        "search_ms": search_ms,
        "excerpt_s": [float(s / sr), float((s + n) / sr)],
        "note": "positive lag means L leads R",
    }


def _goniometer(mid: np.ndarray, side: np.ndarray, bin_deg: int) -> dict[str, Any]:
    energy = mid * mid + side * side
    m = energy > 0
    if not np.any(m):
        return {"histogram": None, "fraction_energy_outside_45_deg": None}
    ang = np.degrees(np.arctan2(side[m], mid[m]))
    ang = ((ang + 90.0) % 180.0) - 90.0  # fold to (-90, 90]
    e = energy[m]
    edges = np.arange(-90, 91, bin_deg)
    hist, _ = np.histogram(ang, bins=edges, weights=e)
    total = float(hist.sum())
    return {
        "bin_deg": bin_deg,
        "bin_edges_deg": edges.tolist(),
        "energy_fraction_per_bin": (hist / total).tolist() if total > 0 else None,
        "fraction_energy_outside_45_deg": float(np.sum(e[np.abs(ang) > 45.0]) / total) if total > 0 else None,
        "definition": "instantaneous angle atan2(side, mid), energy-weighted",
    }


def analyse(src: AudioSource, collector: Collector, profile: str = "full") -> dict[str, Any]:
    P = PARAMS["stereo"]
    if src.n_ch < 2:
        return dict(MONO_RESULT)
    sr = src.band_sr
    nyq = sr / 2.0
    mid, side = src.band_mid, src.band_side
    L = src.band_x[:, 0]
    R = src.band_x[:, 1]

    overall = _side_mid_db(mid, side)

    # --- per third-octave side/mid and mono-sum damage ---------------------
    # Mid and side spectra come from the L/R auto- and cross-spectra, which are
    # exact for a linear decomposition and cost one FFT pass instead of one
    # band-pass filter per band:
    #   P_mid  = (S_LL + S_RR + 2 Re S_LR) / 4
    #   P_side = (S_LL + S_RR - 2 Re S_LR) / 4
    pack = get_spectra(src)
    f_ll = f_m = f_s = pack.freqs
    p_m, p_s = pack.psd["mid"], pack.psd["side"]
    p_ll, p_rr, p_lr = pack.psd["ch0"], pack.psd["ch1"], pack.csd_lr
    edges = third_octave_edges(THIRD_OCTAVE_CENTRES)
    rows_m = spectrum_table(f_m, p_m, edges, nyq)
    rows_s = spectrum_table(f_s, p_s, edges, nyq)
    per_third: list[dict[str, Any]] = []
    mono_damage: list[dict[str, Any]] = []
    for rm, rs in zip(rows_m, rows_s):
        pm, ps = rm["power"], rs["power"]
        sm = db_pow(ps / pm) if (pm and ps and pm > 0 and ps > 0) else (
            -200.0 if (pm and pm > 0) else None)
        per_third.append({"centre_hz": rm["centre_hz"], "side_minus_mid_db": sm})
        loss = None
        if pm is not None and ps is not None and (pm + ps) > 0:
            loss = db_pow(pm / (pm + ps))
        mono_damage.append({"centre_hz": rm["centre_hz"], "mono_sum_loss_db": loss})

    # --- mono crossover ----------------------------------------------------
    crossover = None
    thr = P["mono_crossover_threshold_db"]
    for row in per_third:
        v = row["side_minus_mid_db"]
        if v is None:
            continue
        if v < thr:
            crossover = row["centre_hz"]
        else:
            break
    if crossover is None:
        collector.low_confidence("stereo.mono_crossover", "high",
                                 f"no third-octave band below {thr} dB side/mid at the "
                                 "bottom of the spectrum; the low end is not mono")

    # --- per-band side/mid and correlation ---------------------------------
    # Both come from the same auto/cross spectra; the per-band correlation is
    # the band-integrated real cross-power normalised by the band energies.
    per_band = []
    for name, lo, hi in BANDS:
        if lo >= nyq:
            per_band.append({"band": name, "side_minus_mid_db": None,
                             "correlation": None, "note": "band above Nyquist"})
            continue
        hi_c = min(hi, nyq)
        e_ll = band_power(f_ll, p_ll, lo, hi_c)
        e_rr = band_power(f_ll, p_rr, lo, hi_c)
        e_lr = band_power(f_ll, p_lr, lo, hi_c)
        e_mid = (e_ll + e_rr + 2 * e_lr) / 4.0
        e_side = (e_ll + e_rr - 2 * e_lr) / 4.0
        den = np.sqrt(max(e_ll, 0.0) * max(e_rr, 0.0))
        per_band.append({
            "band": name, "low_hz": lo, "high_hz": hi_c,
            "side_minus_mid_db": (db_pow(e_side / e_mid) if e_mid > 0 and e_side > 0
                                  else (-200.0 if e_mid > 0 else None)),
            "correlation": float(e_lr / den) if den > 0 else None,
            "method": "band-integrated L/R auto- and cross-spectra",
        })

    c_t, c_v = _correlation_timeline(L, R, sr, P["correlation_window_s"])
    finite = c_v[np.isfinite(c_v)]
    worst_idx = np.argsort(np.where(np.isfinite(c_v), c_v, np.inf))[:3]
    worst = [{"start_s": float(c_t[i]), "start": fmt_time(float(c_t[i])),
              "correlation": float(c_v[i])}
             for i in worst_idx if np.isfinite(c_v[i])]
    den_all = np.sqrt(float(np.dot(L, L)) * float(np.dot(R, R)))
    corr_all = float(np.dot(L - L.mean(), R - R.mean()) / den_all) if den_all > 0 else None

    # --- channel balance ---------------------------------------------------
    rms_l = float(np.sqrt(np.mean(src.channel(0) ** 2)))
    rms_r = float(np.sqrt(np.mean(src.channel(1) ** 2)))
    lufs = []
    for c in (0, 1):
        t, bl_ = block_loudness(src.x[:, c : c + 1].astype(np.float64), src.sr, 0.4, 0.1)
        v, _ = gated_integrated(bl_)
        lufs.append(v)

    # --- width timeline ----------------------------------------------------
    w = max(2, int(round(sr)))
    n = (mid.size // w) * w
    width_t, width_v = np.zeros(0), np.zeros(0)
    if n > 0:
        fm = mid[:n].reshape(-1, w)
        fs = side[:n].reshape(-1, w)
        pm = np.mean(fm * fm, axis=1)
        ps = np.mean(fs * fs, axis=1)
        width_v = np.where(pm > 0, db_pow(np.maximum(ps, 1e-30) / np.maximum(pm, 1e-30)), np.nan)
        width_t = np.arange(fm.shape[0]) * w / float(sr)

    return {
        "available": True,
        "convention": PARAMS["general"]["midside_convention"],
        "side_minus_mid_db": overall,
        "side_minus_mid_below_120hz_db": _band_side_mid(f_ll, p_ll, p_rr, p_lr,
                                                        20.0, min(120.0, nyq)),
        "side_minus_mid_per_third_octave": per_third,
        "side_minus_mid_per_band": per_band,
        "mono_crossover_hz": crossover,
        "mono_crossover_rule": f"highest third-octave centre below which side/mid "
                               f"stays continuously under {thr} dB",
        "correlation": {
            "overall": corr_all,
            "window_s": P["correlation_window_s"],
            "times_s": c_t,
            "values": c_v,
            "min": float(np.min(finite)) if finite.size else None,
            "p5": percentiles(finite, [5])["p5"] if finite.size else None,
            "median": float(np.median(finite)) if finite.size else None,
            "pct_time_below_0": float(100.0 * np.mean(finite < 0.0)) if finite.size else None,
            "pct_time_below_0_3": float(100.0 * np.mean(finite < 0.3)) if finite.size else None,
            "most_negative_windows": worst,
        },
        "channel_balance": {
            "rms_l_dbfs": db_amp(rms_l) if rms_l > 0 else None,
            "rms_r_dbfs": db_amp(rms_r) if rms_r > 0 else None,
            "rms_l_minus_r_db": (db_amp(rms_l) - db_amp(rms_r)) if (rms_l > 0 and rms_r > 0) else None,
            "lufs_l": lufs[0], "lufs_r": lufs[1],
            "lufs_l_minus_r": (lufs[0] - lufs[1]) if (lufs[0] is not None and lufs[1] is not None) else None,
        },
        "inter_channel_time_offset": _itd(L, R, sr, P["itd_search_ms"]),
        "width_timeline": {"hop_s": 1.0, "times_s": width_t, "side_minus_mid_db": width_v},
        "mono_sum_damage": {
            "definition": "10*log10(P_mid / (P_mid + P_side)) per third-octave: the "
                          "energy lost when the file is summed to mono",
            "per_third_octave": mono_damage,
            "broadband_loss_db": db_pow(
                float(np.mean(mid * mid)) /
                max(float(np.mean(mid * mid)) + float(np.mean(side * side)), 1e-30))
            if float(np.mean(mid * mid)) > 0 else None,
        },
        "goniometer": (_goniometer(mid, side, P["goniometer_bins_deg"])
                       if profile != "quick"
                       else {"available": False,
                             "reason": "skipped by --profile quick"}),
    }
