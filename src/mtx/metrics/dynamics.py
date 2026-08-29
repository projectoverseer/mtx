"""4.4 Dynamics and limiting fingerprints.

The flat-top detector derives its threshold from the file's own maximum, so it
still finds clipping on a master whose ceiling sits below -0.1 dBFS.  That is
the regression case in `mtx selftest`.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from ..audio import AudioSource
from ..bands import get_band_pack
from ..dsp import band_filter, crest_db
from ..params import PARAMS
from ..util import Collector, db_amp, fmt_time, percentiles

FLAT_TOP_FACTOR = 0.99999


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start indices and lengths of each True run in a boolean array."""
    if mask.size == 0 or not mask.any():
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    d = np.diff(mask.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [mask.size]])
    return starts, ends - starts


def _run_histogram(lengths: np.ndarray) -> dict[str, int]:
    bins = {"1": 0, "2": 0, "3-5": 0, "6-10": 0, "11-20": 0, "21+": 0}
    for L in lengths:
        if L == 1:
            bins["1"] += 1
        elif L == 2:
            bins["2"] += 1
        elif L <= 5:
            bins["3-5"] += 1
        elif L <= 10:
            bins["6-10"] += 1
        elif L <= 20:
            bins["11-20"] += 1
        else:
            bins["21+"] += 1
    return bins


def _crest_timeline(x: np.ndarray, sr: int, hop_s: float) -> tuple[np.ndarray, np.ndarray]:
    hop = max(1, int(round(hop_s * sr)))
    n = (x.size // hop) * hop
    if n == 0:
        return np.zeros(0), np.zeros(0)
    fr = x[:n].reshape(-1, hop)
    pk = np.max(np.abs(fr), axis=1)
    rm = np.sqrt(np.mean(fr * fr, axis=1))
    ok = (pk > 0) & (rm > 0)
    out = np.full(pk.shape, np.nan)
    out[ok] = db_amp(pk[ok]) - db_amp(rm[ok])
    return np.arange(fr.shape[0]) * hop / float(sr), out


def _loudest_window(x: np.ndarray, sr: int, win_s: float) -> tuple[float | None, float | None]:
    """(crest_db, start_time_s) of the highest-RMS window of the given length."""
    w = int(round(win_s * sr))
    if x.size < w or w <= 0:
        return None, None
    hop = max(1, int(round(0.5 * sr)))
    cs = np.concatenate([[0.0], np.cumsum(x * x)])
    starts = np.arange(0, x.size - w + 1, hop)
    ms = (cs[starts + w] - cs[starts]) / w
    k = int(np.argmax(ms))
    s = int(starts[k])
    return crest_db(x[s : s + w]), s / float(sr)


def _flat_top(src: AudioSource, collector: Collector) -> dict[str, Any]:
    P = PARAMS["flat_top"]
    sr = src.sr
    lf = band_filter(src.band_mono, src.band_sr, 20.0, 120.0)
    lf_sr = src.band_sr
    lf_win = max(1, int(round(P["lf_context_window_ms"] / 1000.0 * lf_sr)))
    lf_frame_rms = None
    if lf.size >= lf_win:
        n = (lf.size // lf_win) * lf_win
        lf_frame_rms = np.sqrt(np.mean(lf[:n].reshape(-1, lf_win) ** 2, axis=1))
    lf_track_db = db_amp(np.sqrt(np.mean(lf * lf))) if lf.size else None

    per_ch: list[dict[str, Any]] = []
    all_events: list[tuple[int, int, int]] = []  # (length, start, channel)
    for c in range(src.n_ch):
        x = src.channel(c)
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        thr = peak * FLAT_TOP_FACTOR
        mask = np.abs(x) >= thr if peak > 0 else np.zeros(x.size, dtype=bool)
        starts, lengths = _runs(mask)
        order = np.argsort(lengths)[::-1][: P["longest_runs_reported"]]
        longest = int(lengths.max()) if lengths.size else 0
        # Ceiling density: distance of every sample from this channel's ceiling.
        density = {}
        if peak > 0 and x.size:
            ax = np.abs(x)
            for d in P["ceiling_density_db"]:
                density[f"within_{d}_db"] = float(np.mean(ax >= peak * 10 ** (-d / 20.0)))
        per_ch.append({
            "channel": c,
            "channel_peak_dbfs": db_amp(peak) if peak > 0 else None,
            "threshold_dbfs": db_amp(thr) if thr > 0 else None,
            "threshold_rule": P["threshold_rule"],
            "flat_sample_count": int(mask.sum()),
            "flat_sample_fraction": float(mask.mean()) if mask.size else None,
            "event_count": int(starts.size),
            "run_length_histogram": _run_histogram(lengths),
            "longest_run_samples": longest,
            "longest_run_ms": 1000.0 * longest / sr,
            "longest_runs": [
                {"start_s": float(starts[i] / sr), "start": fmt_time(float(starts[i] / sr)),
                 "length_samples": int(lengths[i]), "length_ms": 1000.0 * int(lengths[i]) / sr}
                for i in order
            ],
            "ceiling_density": density,
        })
        for i in range(starts.size):
            all_events.append((int(lengths[i]), int(starts[i]), c))

    # --- clip-then-normalise: a flat value that sits below full scale --------
    file_peak = max((float(np.max(np.abs(src.channel(c)))) for c in range(src.n_ch)),
                    default=0.0)
    flat_value_dbfs = db_amp(file_peak) if file_peak > 0 else None
    n_long = sum(1 for L, _, _ in all_events if L >= 3)
    clip_then_normalise = {
        "flat_value_dbfs": flat_value_dbfs,
        "flat_runs_of_3_or_more": n_long,
        "ceiling_below_full_scale": bool(flat_value_dbfs is not None and flat_value_dbfs < -0.05),
        "detected": bool(n_long > 0 and flat_value_dbfs is not None and flat_value_dbfs < -0.05),
        "method": "flat-topped runs of 3+ samples whose flat value is below 0 dBFS",
        "confidence": "medium",
        "confidence_reason": "a flat run below full scale is also produced by a "
                             "clipper set below 0 dBFS, and by any signal that "
                             "genuinely holds its maximum for several samples",
    }
    if clip_then_normalise["detected"]:
        collector.low_confidence("dynamics.clip_then_normalise", "medium",
                                 clip_then_normalise["confidence_reason"])

    # --- correlation of flat-top events with low-frequency energy -----------
    lf_corr: dict[str, Any] = {"correlation": None, "lf_at_events_db": None,
                               "lf_track_db": lf_track_db, "delta_db": None,
                               "window_ms": P["lf_context_window_ms"]}
    if lf_frame_rms is not None and lf_frame_rms.size > 4 and all_events:
        ev_frames = np.zeros(lf_frame_rms.size)
        scale = lf_sr / float(sr)
        for L, s, _ in all_events:
            f = int((s * scale) // lf_win)
            if 0 <= f < ev_frames.size:
                ev_frames[f] += L
        lf_db = db_amp(np.maximum(lf_frame_rms, 1e-20))
        if np.std(ev_frames) > 0 and np.std(lf_db) > 0:
            lf_corr["correlation"] = float(np.corrcoef(ev_frames, lf_db)[0, 1])
        hit = ev_frames > 0
        if hit.any():
            lf_corr["lf_at_events_db"] = float(np.mean(lf_db[hit]))
            if lf_track_db is not None:
                lf_corr["delta_db"] = float(np.mean(lf_db[hit])) - lf_track_db
        lf_corr["frames_with_events"] = int(hit.sum())
        lf_corr["note"] = ("positive delta means the flat-top events sit where the "
                           "sub-120 Hz level is above the track mean")

    return {
        "per_channel": per_ch,
        "total_flat_samples": int(sum(c["flat_sample_count"] for c in per_ch)),
        "total_events": int(sum(c["event_count"] for c in per_ch)),
        "longest_run_samples": max((c["longest_run_samples"] for c in per_ch), default=0),
        "longest_run_ms": max((c["longest_run_ms"] for c in per_ch), default=0.0),
        "clip_then_normalise": clip_then_normalise,
        "low_frequency_association": lf_corr,
        "limiter_vs_clipper": _entry_slopes(src, all_events),
    }


def _entry_slopes(src: AudioSource, events: list[tuple[int, int, int]]) -> dict[str, Any]:
    """Waveform slope in the 2 ms either side of each ceiling event.

    A clipper enters and leaves the ceiling steeply; a limiter rides in on an
    envelope.  The finding is an inference from the slope distribution.
    """
    P = PARAMS["flat_top"]
    sr = src.sr
    w = max(2, int(round(P["slope_window_ms"] / 1000.0 * sr)))
    if not events:
        return {"events_measured": 0, "entry_slope_db_per_ms": None,
                "exit_slope_db_per_ms": None, "inference": None,
                "confidence": "low",
                "confidence_reason": "no ceiling events to measure"}
    entry, exit_ = [], []
    # One |x| per channel, not one per event: this loop runs thousands of times.
    abs_ch = {c: np.abs(src.x[:, c].astype(np.float64)) for c in range(src.n_ch)}
    # Measure the longest events first; they carry the clearest shape.
    for L, s, c in sorted(events, reverse=True)[:2000]:
        x = abs_ch[c]
        a0, a1 = s - w, s
        b0, b1 = s + L, s + L + w
        if a0 < 0 or b1 > x.size or L <= 0:
            continue
        pre, post = x[a0:a1], x[b0:b1]
        if pre.size < 2 or post.size < 2:
            continue
        pre_db = db_amp(np.maximum(pre, 1e-12))
        post_db = db_amp(np.maximum(post, 1e-12))
        ms = P["slope_window_ms"]
        entry.append(float((pre_db[-1] - pre_db[0]) / ms))
        exit_.append(float((post_db[0] - post_db[-1]) / ms))
    if not entry:
        return {"events_measured": 0, "entry_slope_db_per_ms": None,
                "exit_slope_db_per_ms": None, "inference": None,
                "confidence": "low",
                "confidence_reason": "ceiling events sit at the file edges; "
                                     "no 2 ms context available"}
    ea, xa = np.array(entry), np.array(exit_)
    med_entry, med_exit = float(np.median(ea)), float(np.median(xa))
    steep = float(np.mean((np.abs(ea) > 6.0) | (np.abs(xa) > 6.0)))
    return {
        "events_measured": int(ea.size),
        "window_ms": P["slope_window_ms"],
        "entry_slope_db_per_ms": {"median": med_entry,
                                  **percentiles(ea, [10, 50, 90])},
        "exit_slope_db_per_ms": {"median": med_exit,
                                 **percentiles(xa, [10, 50, 90])},
        "fraction_of_events_with_steep_edges": steep,
        "steep_edge_threshold_db_per_ms": 6.0,
        "inference": ("vertical entry/exit dominates (clipper-like)" if steep > 0.5
                      else "rounded entry/exit dominates (limiter-like)"),
        "inference_is_an_inference": True,
        "confidence": "medium" if ea.size >= 20 else "low",
        "confidence_reason": ("slope statistics over fewer than 20 events are noisy"
                              if ea.size < 20 else
                              "slope shape is also affected by programme content "
                              "at the ceiling, not only by the processor"),
        "method": "mean dB slope of |x| across the 2 ms before entry and after exit "
                  "of each flat-top run, measured on the 2000 longest runs",
    }


def _onsets(src: AudioSource, collector: Collector, profile: str) -> dict[str, Any]:
    if profile == "quick":
        # Skipped not because it is slow in itself but because it is the first
        # librosa call, and importing/JIT-compiling that stack is most of a
        # quick run's wall time.
        return {"available": False, "reason": "skipped by --profile quick"}
    if importlib.util.find_spec("librosa") is None:
        collector.warn("dynamics.onsets", "librosa not installed; onset statistics are null")
        return {"available": False, "reason": "librosa not installed"}
    y = src.lib_mono
    hop = PARAMS["general"]["librosa_hop_length"]
    if y.size < hop * 4:
        collector.warn("dynamics.onsets", "file too short for onset analysis")
        return {"available": False, "reason": "file too short"}
    env = src.onset_envelope()
    times, frames = src.onset_times()
    dur = src.duration
    # Attack slope of the strongest onsets, measured on the native-rate signal.
    strengths = env[frames] if frames.size else np.zeros(0)
    order = np.argsort(strengths)[::-1][:100]
    slopes = []
    xm = np.abs(src.mono)
    for idx in order:
        t = float(times[idx])
        a = int(t * src.sr)
        b = a + int(0.010 * src.sr)
        if a < 1 or b >= xm.size:
            continue
        pre = float(np.max(xm[max(0, a - int(0.010 * src.sr)) : a]))
        post = float(np.max(xm[a:b]))
        if pre > 1e-9 and post > 1e-9:
            slopes.append((db_amp(post) - db_amp(pre)) / 10.0)  # dB per ms
    return {
        "available": True,
        "method": "librosa.onset.onset_strength + onset_detect at 22.05 kHz, hop 512",
        "onset_count": int(frames.size),
        "onset_rate_per_s": float(frames.size / dur) if dur > 0 else None,
        "median_onset_strength": float(np.median(strengths)) if strengths.size else None,
        "onset_strength_percentiles": percentiles(env, [10, 50, 90]),
        "median_attack_slope_db_per_ms": float(np.median(slopes)) if slopes else None,
        "attack_slope_onsets_used": len(slopes),
        "attack_slope_method": "peak |x| in the 10 ms after the onset minus the peak "
                               "in the 10 ms before, divided by 10 ms, over the 100 "
                               "strongest onsets",
        "onset_times_s": times,
    }


def analyse(src: AudioSource, collector: Collector, profile: str = "full") -> dict[str, Any]:
    P = PARAMS["crest"]
    sr = src.sr
    mono = src.mono

    crest_whole = crest_db(mono)
    crest_loud, crest_loud_t = _loudest_window(mono, sr, P["loudest_window_s"])
    ct_t, ct_v = _crest_timeline(mono, sr, P["timeline_hop_s"])

    # Per-band crest comes from the shared single band-split pass.
    band_crest = [dict(b) for b in get_band_pack(src).bands]
    spread = [b["crest_db"] for b in band_crest if b["crest_db"] is not None]

    # DC offset, whole file and worst 1 s window.
    dc = []
    for c in range(src.n_ch):
        x = src.channel(c)
        w = int(sr)
        worst, worst_t = 0.0, None
        if x.size >= w:
            cs = np.concatenate([[0.0], np.cumsum(x)])
            starts = np.arange(0, x.size - w + 1, max(1, w // 2))
            means = (cs[starts + w] - cs[starts]) / w
            k = int(np.argmax(np.abs(means)))
            worst, worst_t = float(means[k]), float(starts[k] / sr)
        dc.append({
            "channel": c,
            "dc_offset": float(np.mean(x)) if x.size else None,
            "dc_offset_dbfs": db_amp(abs(float(np.mean(x)))) if x.size and abs(float(np.mean(x))) > 0 else None,
            "max_1s_window_dc": worst,
            "max_1s_window_time_s": worst_t,
            "max_1s_window_time": fmt_time(worst_t),
        })

    return {
        "crest": {
            "definition": P["definition"],
            "whole_file_db": crest_whole,
            "loudest_window": {
                "window_s": P["loudest_window_s"],
                "crest_db": crest_loud,
                "start_s": crest_loud_t,
                "start": fmt_time(crest_loud_t),
            },
            "timeline_hop_s": P["timeline_hop_s"],
            "timeline_times_s": ct_t,
            "timeline_db": ct_v,
            "timeline_percentiles_db": percentiles(ct_v[np.isfinite(ct_v)], [10, 50, 90]),
        },
        "per_band_crest": {
            "measured_on": "mono (channel mean) at the band analysis rate",
            "bands": band_crest,
            "spread_db": (float(max(spread) - min(spread)) if len(spread) > 1 else None),
        },
        "flat_top": _flat_top(src, collector),
        "onsets": _onsets(src, collector, profile),
        "dc_offset": dc,
    }
