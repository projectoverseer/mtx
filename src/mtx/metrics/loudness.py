"""4.3 Loudness, true peak, PLR/PSR, DR14.

Two independent implementations of the integrated loudness are reported (this
package's own K-weighting, and ffmpeg's ebur128 filter), plus pyloudnorm as a
third opinion when it is installed.  Nothing is averaged: the deltas are the
output.
"""

from __future__ import annotations

import re
import subprocess
from typing import Any

import numpy as np

from ..audio import AudioSource
from ..dsp import (block_loudness, gated_integrated, loudness_range, rolling_max,
                   true_peak_scan)
from ..params import PARAMS
from ..util import Collector, db_amp, fmt_time, percentiles

# Resolution of the max-envelope kept from each oversampling pass.  1 ms is
# well below the 3 s PSR window and keeps the envelope small.
PSR_ENVELOPE_HOP_S = 0.001

FF_I = re.compile(r"^\s*I:\s*(-?\d+\.?\d*)\s*LUFS", re.M)
FF_LRA = re.compile(r"^\s*LRA:\s*(-?\d+\.?\d*)\s*LU", re.M)
FF_TP = re.compile(r"Peak:\s*(-?\d+\.?\d*|-inf)\s*dBFS")
FF_THRESH = re.compile(r"^\s*Threshold:\s*(-?\d+\.?\d*)\s*LUFS", re.M)


def ffmpeg_ebur128(path: str, collector: Collector) -> dict[str, Any]:
    """Cross-check via `ffmpeg -af ebur128`.  Never fatal."""
    out: dict[str, Any] = {"available": False, "integrated_lufs": None,
                           "lra_lu": None, "true_peak_dbtp": None,
                           "threshold_lufs": None, "reason": None}
    try:
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-i", path,
             "-map", "0:a:0", "-af", "ebur128=peak=true:framelog=quiet",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=1800,
        )
    except FileNotFoundError:
        out["reason"] = "ffmpeg not found on PATH; loudness cross-validation unavailable"
        collector.warn("loudness", out["reason"])
        return out
    except subprocess.TimeoutExpired:
        out["reason"] = "ffmpeg ebur128 timed out"
        collector.warn("loudness", out["reason"])
        return out
    text = proc.stderr or ""
    mi, ml, mt = FF_I.search(text), FF_LRA.search(text), FF_THRESH.search(text)
    peaks = FF_TP.findall(text)
    if not mi:
        out["reason"] = f"could not parse ebur128 summary (ffmpeg exit {proc.returncode})"
        collector.warn("loudness", out["reason"])
        return out
    out["available"] = True
    out["integrated_lufs"] = float(mi.group(1))
    out["lra_lu"] = float(ml.group(1)) if ml else None
    out["threshold_lufs"] = float(mt.group(1)) if mt else None
    if peaks:
        v = peaks[-1]
        out["true_peak_dbtp"] = None if v == "-inf" else float(v)
    return out


def pyloudnorm_check(x: np.ndarray, sr: int, collector: Collector) -> dict[str, Any]:
    try:
        import pyloudnorm as pyln
    except ImportError:
        return {"available": False, "integrated_lufs": None,
                "reason": "pyloudnorm not installed"}
    try:
        meter = pyln.Meter(sr)
        val = float(meter.integrated_loudness(np.asarray(x, dtype=np.float64)))
    except Exception as exc:
        collector.warn("loudness", f"pyloudnorm failed: {exc!r}")
        return {"available": False, "integrated_lufs": None, "reason": repr(exc)}
    return {"available": True, "integrated_lufs": val,
            "version": getattr(pyln, "__version__", "unknown"),
            "note": "pyloudnorm normalises its RLB stage to a 0.995 pass-band "
                    "gain, worth about -0.04 LU against the BS.1770-4 table"}


def dr14(src: AudioSource, collector: Collector) -> dict[str, Any]:
    """TT Dynamic Range Meter, offline variant.  See params.dr14."""
    p = PARAMS["dr14"]
    sr = src.sr
    bs = int(round(p["block_s"] * sr))
    per_ch: list[dict[str, Any]] = []
    if src.n_frames < bs:
        collector.warn("dr14", f"file shorter than one {p['block_s']} s block; DR14 is null")
        return {"per_channel": [], "dr_unrounded": None, "dr": None,
                "blocks_used": 0, "validation": _dr_validation(False)}
    for c in range(src.n_ch):
        x = src.channel(c)
        n_blocks = int(x.size // bs)
        blocks = x[: n_blocks * bs].reshape(n_blocks, bs)
        rms = np.sqrt(2.0 * np.mean(blocks * blocks, axis=1))
        peaks = np.max(np.abs(blocks), axis=1)
        order = np.argsort(rms)[::-1]
        n_top = max(1, int(round(p["loudest_fraction"] * n_blocks)))
        top = rms[order[:n_top]]
        rms_top = float(np.sqrt(np.mean(top * top)))
        peaks_sorted = np.sort(peaks)[::-1]
        peak2_block = float(peaks_sorted[min(1, peaks_sorted.size - 1)])
        # The alternative reading of "second-highest sample peak": the second
        # largest distinct sample magnitude in the channel.
        ax = np.abs(x)
        pk_max = float(ax.max()) if ax.size else 0.0
        below = ax[ax < pk_max]
        peak2_sample = float(below.max()) if below.size else pk_max
        dr_block = (db_amp(peak2_block) - db_amp(rms_top)) if rms_top > 0 else None
        dr_sample = (db_amp(peak2_sample) - db_amp(rms_top)) if rms_top > 0 else None
        per_ch.append({
            "channel": c, "blocks": n_blocks, "blocks_in_top20": n_top,
            "rms_top20_dbfs": db_amp(rms_top) if rms_top > 0 else None,
            "peak2_block_dbfs": db_amp(peak2_block) if peak2_block > 0 else None,
            "peak2_sample_dbfs": db_amp(peak2_sample) if peak2_sample > 0 else None,
            "dr_from_block_peak2": dr_block,
            "dr_from_sample_peak2": dr_sample,
        })
    vals = [c["dr_from_block_peak2"] for c in per_ch if c["dr_from_block_peak2"] is not None]
    alt = [c["dr_from_sample_peak2"] for c in per_ch if c["dr_from_sample_peak2"] is not None]
    dr = float(np.mean(vals)) if vals else None
    n_blocks_min = min((c["blocks"] for c in per_ch), default=0)
    if n_blocks_min < 5:
        collector.low_confidence("dr14", "low",
                                 f"only {n_blocks_min} blocks of {p['block_s']} s; "
                                 "the DR statistic is unstable on short material")
    return {
        "per_channel": per_ch,
        "dr_unrounded": dr,
        "dr": int(round(dr)) if dr is not None else None,
        "dr_alt_sample_peak2_unrounded": float(np.mean(alt)) if alt else None,
        "peak2_definition": "second highest of the per-block sample peaks (TT DR); "
                            "dr_alt_* uses the second largest distinct sample magnitude",
        "blocks_used": n_blocks_min,
        "validation": _dr_validation(True),
    }


def _dr_validation(computed: bool) -> dict[str, Any]:
    return {
        "validated_against_published_reference": False,
        "status": "NOT VALIDATED against a published DR rating",
        "reason": "mtx runs offline and ships no copyrighted reference track, so "
                  "the implementation could not be checked against a published "
                  "DR value. It is checked against analytically known synthetic "
                  "cases in `mtx selftest` (a full-scale sine must give DR 0.0, "
                  "a 20 dB crest square-burst must give DR 20.0). Treat the DR "
                  "number as unverified against the reference implementation "
                  "until you compare one track you already have a rating for.",
        "self_checked_synthetically": computed,
    }


def analyse(src: AudioSource, collector: Collector,
            profile: str = "full") -> dict[str, Any]:
    P = PARAMS["loudness"]
    TP = PARAMS["true_peak"]
    sr = src.sr
    x = src.x.astype(np.float64)

    # ---- momentary / short-term timelines --------------------------------
    m_t, m_l = block_loudness(x, sr, P["block_ms"] / 1000.0,
                              (P["block_ms"] / 1000.0) * (1 - P["block_overlap_pct"] / 100.0))
    s_t, s_l = block_loudness(x, sr, P["shortterm_block_s"], 0.1)
    integrated, gate_mask = gated_integrated(m_l, P["absolute_gate_lufs"],
                                             P["relative_gate_lu"])
    lra = loudness_range(s_l, P["lra_absolute_gate_lufs"], P["lra_relative_gate_lu"],
                         tuple(P["lra_percentiles"]))
    if integrated is None:
        collector.warn("loudness", "no block survived the absolute gate; "
                                   "integrated loudness is null (file may be silent)")
    if m_l.size == 0:
        collector.warn("loudness", f"file shorter than one {P['block_ms']} ms block")
    if s_l.size == 0:
        collector.warn("loudness", f"file shorter than one {P['shortterm_block_s']} s block; "
                                   "LRA and PSR are null")

    # ---- cross-checks -----------------------------------------------------
    ff = ffmpeg_ebur128(src.path, collector)
    pyln = pyloudnorm_check(x, sr, collector)
    delta_ff = (integrated - ff["integrated_lufs"]) if (integrated is not None and ff["integrated_lufs"] is not None) else None
    delta_pyln = (integrated - pyln["integrated_lufs"]) if (integrated is not None and pyln.get("integrated_lufs") is not None) else None
    if delta_ff is not None:
        collector.disagreement("loudness.integrated", "mtx", integrated, "ffmpeg ebur128",
                               ff["integrated_lufs"], P["cross_check_tolerance_lu"], "LU")
    if delta_pyln is not None:
        collector.disagreement("loudness.integrated", "mtx", integrated, "pyloudnorm",
                               pyln["integrated_lufs"], P["cross_check_tolerance_lu"], "LU")
    delta_lra = (lra - ff["lra_lu"]) if (lra is not None and ff.get("lra_lu") is not None) else None

    # ---- peaks ------------------------------------------------------------
    sample_peak_ch = np.max(np.abs(x), axis=0) if src.n_frames else np.zeros(src.n_ch)
    # 4x runs over the whole file because the PSR timeline needs its envelope;
    # 16x needs only a peak and the over counts, so it runs pruned.
    scan4 = true_peak_scan(x, sr, 4, env_hop_s=PSR_ENVELOPE_HOP_S)
    tp4_ch = scan4["peak_per_channel"]
    if profile == "quick":
        collector.warn("true_peak",
                       "--profile quick computes the 4x true peak only; the 16x "
                       "cross-check and the inter-sample over counts are skipped, "
                       "so the two-method agreement is unverified for this run")
        scan16 = None
        tp16_ch = np.full(src.n_ch, np.nan)
        overs = {"oversampling": 16, "skipped": True,
                 "reason": "skipped by --profile quick"}
    else:
        scan16 = true_peak_scan(x, sr, 16, thresholds_dbtp=TP["over_thresholds_dbtp"],
                                env_hop_s=None)
        tp16_ch = scan16["peak_per_channel"]
        overs = {
            "oversampling": 16,
            "skipped": False,
            "thresholds_dbtp": list(TP["over_thresholds_dbtp"]),
            "counts_in_order": scan16["over_counts"],
            "counts": {f"above_{t:+.1f}_dbtp".replace("+", "plus_").replace("-", "minus_"): c
                       for t, c in zip(TP["over_thresholds_dbtp"], scan16["over_counts"])},
            "highest_peak_dbtp": db_amp(scan16["peak"]) if scan16["peak"] > 0 else None,
            "highest_peak_time_s": scan16["peak_time_s"],
            "highest_peak_time": fmt_time(scan16["peak_time_s"]),
            "over_definition": "one contiguous excursion above the threshold in the "
                               "16x reconstructed waveform, not one sample",
            "scanned_fraction_of_file": scan16["scanned_fraction"],
            "pruning": "the 16x scan skips stretches that provably cannot reach the "
                       "sample peak or the lowest threshold, using the interpolation "
                       f"filter's per-phase L1 bound of "
                       f"{scan16['phase_gain_bound_db']:.2f} dB; the result is "
                       "identical to a full scan (asserted in mtx selftest)",
        }
    sp_db = [db_amp(v) if v > 0 else None for v in sample_peak_ch]
    tp4_db = [db_amp(v) if v > 0 else None for v in tp4_ch]
    tp16_db = [db_amp(v) if v > 0 else None for v in tp16_ch]
    sp_all = db_amp(float(sample_peak_ch.max())) if sample_peak_ch.size and sample_peak_ch.max() > 0 else None
    tp4_all = db_amp(float(tp4_ch.max())) if tp4_ch.size and tp4_ch.max() > 0 else None
    tp16_all = db_amp(float(tp16_ch.max())) if tp16_ch.size and tp16_ch.max() > 0 else None
    if tp4_all is not None and tp16_all is not None:
        collector.disagreement("true_peak", "16x", tp16_all, "4x", tp4_all,
                               TP["cross_check_tolerance_db"], "dB")
        if tp16_all < tp4_all - 1e-9:
            collector.warn("true_peak",
                           f"16x estimate ({tp16_all:.3f} dBTP) is below the 4x estimate "
                           f"({tp4_all:.3f} dBTP), which should not happen")
    if ff.get("true_peak_dbtp") is not None and tp16_all is not None:
        collector.disagreement("true_peak", "mtx 16x", tp16_all, "ffmpeg ebur128",
                               ff["true_peak_dbtp"], 0.5, "dB")

    # ---- PLR / PSR --------------------------------------------------------
    # PLR and the streaming preview use the 16x figure where it exists, and say
    # so when they had to fall back to 4x.
    tp_ref = tp16_all if tp16_all is not None else tp4_all
    tp_ref_source = "16x" if tp16_all is not None else ("4x" if tp4_all is not None else None)
    plr = (tp_ref - integrated) if (tp_ref is not None and integrated is not None) else None
    psr = _psr(x, sr, scan4, collector)

    # ---- streaming normalisation preview ---------------------------------
    preview = {}
    for name, target in PARAMS["streaming_targets_lufs"].items():
        if integrated is None:
            preview[name] = {"target_lufs": target, "gain_db": None,
                             "true_peak_after_dbtp": None, "gain_is_positive": None}
            continue
        gain = target - integrated
        preview[name] = {
            "target_lufs": target,
            "gain_db": gain,
            "true_peak_after_dbtp": (tp_ref + gain) if tp_ref is not None else None,
            "gain_is_positive": bool(gain > 0),
            "note": "positive gain means the platform turns the track up; the "
                    "resulting true peak is stated so the headroom can be checked"
                    if gain > 0 else None,
        }

    return {
        "integrated_lufs": integrated,
        "lra_lu": lra,
        "gated_block_fraction": float(np.mean(gate_mask)) if gate_mask.size else None,
        "cross_check": {
            "ffmpeg_ebur128": ff,
            "pyloudnorm": pyln,
            "delta_lufs_mtx_minus_ffmpeg": delta_ff,
            "delta_lufs_mtx_minus_pyloudnorm": delta_pyln,
            "delta_lra_mtx_minus_ffmpeg": delta_lra,
            "tolerance_lu": P["cross_check_tolerance_lu"],
        },
        "momentary": {
            "block_ms": P["block_ms"], "hop_ms": P["block_ms"] * 0.25,
            "times_s": m_t, "lufs": m_l,
            "max_lufs": float(np.max(m_l)) if m_l.size else None,
            "percentiles_lufs": percentiles(m_l, [10, 25, 50, 75, 90, 95]),
        },
        "shortterm": {
            "block_s": P["shortterm_block_s"], "hop_s": 0.1,
            "times_s": s_t, "lufs": s_l,
            "max_lufs": float(np.max(s_l)) if s_l.size else None,
            "percentiles_lufs": percentiles(s_l, [10, 25, 50, 75, 90, 95]),
        },
        "sample_peak": {
            "per_channel_dbfs": sp_db, "overall_dbfs": sp_all,
            "per_channel_linear": [float(v) for v in sample_peak_ch],
        },
        "true_peak": {
            "per_channel_dbtp_4x": tp4_db, "per_channel_dbtp_16x": tp16_db,
            "overall_dbtp_4x": tp4_all, "overall_dbtp_16x": tp16_all,
            "delta_16x_minus_4x_db": (tp16_all - tp4_all) if (tp4_all is not None and tp16_all is not None) else None,
            "delta_truepeak16x_minus_samplepeak_db": (tp16_all - sp_all) if (tp16_all is not None and sp_all is not None) else None,
            "overs": overs,
        },
        "plr_db": plr,
        "plr_definition": f"true peak ({tp_ref_source}, dBTP) minus integrated "
                          "loudness (LUFS)",
        "plr_true_peak_source": tp_ref_source,
        "psr": psr,
        "streaming_preview": preview,
        "dr14": dr14(src, collector),
    }


def _psr(x: np.ndarray, sr: int, scan4: dict[str, Any],
         collector: Collector) -> dict[str, Any]:
    """PSR as a timeline: short-term true peak minus short-term LUFS.

    The short-term true peak is a rolling maximum over the 4x max-envelope that
    the true-peak scan already produced, so no window is oversampled twice.
    """
    P = PARAMS["psr"]
    st_t, st_l = block_loudness(x, sr, P["window_s"], P["hop_s"])
    env = scan4["envelope"]
    hop_s = scan4["envelope_hop_s"]
    win = int(round(P["window_s"] / hop_s))
    step = max(1, int(round(P["hop_s"] / hop_s)))
    tp_lin = rolling_max(env, win, step)
    tp_db = db_amp(np.maximum(tp_lin, 1e-20)) if tp_lin.size else np.zeros(0)
    n = min(st_l.size, tp_db.size)
    if n == 0:
        collector.warn("psr", f"file shorter than one {P['window_s']} s window; PSR is null")
        return {"window_s": P["window_s"], "hop_s": P["hop_s"], "times_s": [],
                "psr_db": [], "min_db": None, "min_time_s": None, "min_time": None,
                "p10_db": None, "median_db": None, "max_db": None}
    psr = tp_db[:n] - st_l[:n]
    times = st_t[:n]
    k = int(np.argmin(psr))
    q = percentiles(psr, [10, 50])
    return {
        "window_s": P["window_s"], "hop_s": P["hop_s"],
        "definition": P["definition"],
        "times_s": times, "psr_db": psr,
        "min_db": float(psr[k]), "min_time_s": float(times[k]),
        "min_time": fmt_time(float(times[k])),
        "p10_db": q["p10"], "median_db": q["p50"],
        "max_db": float(np.max(psr)),
        "shortterm_true_peak_dbtp": tp_db[:n],
        "shortterm_lufs": st_l[:n],
    }
