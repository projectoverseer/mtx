"""5. compare -- two files, level-matched before anything else is measured.

Comparing two masters at their own levels is the most reliable way to reach a
wrong conclusion, so the gain is applied first and reported in the output.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from scipy import signal as sps

from . import SCHEMA_VERSION, __version__
from .analyze import analyze_file
from .audio import AudioSource, resample_to
from .digest import hz, kv_rows, n, table
from .dsp import (block_loudness, correlation, gated_integrated,
                  spectrum_table, third_octave_edges, welch_psd)
from .params import BANDS, PARAMS, THIRD_OCTAVE_CENTRES
from .util import Collector, db_amp, db_pow, jsonable

# Metrics whose value moves with the level match, and those that do not.
LEVEL_DEPENDENT = ("lufs_i", "true_peak_dbtp_16x", "sample_peak_dbfs")
HEADLINE_ORDER = [
    "lufs_i", "lra_lu", "true_peak_dbtp_16x", "sample_peak_dbfs", "plr_db",
    "psr_min_db", "psr_median_db", "dr14", "crest_whole_db",
    "crest_loudest_10s_db", "spectral_tilt_db_per_oct", "air_band_pct",
    "sub_band_pct", "side_minus_mid_db", "side_minus_mid_below_120hz_db",
    "mono_crossover_hz", "correlation_mean", "correlation_min",
    "flat_top_sample_count", "flat_top_longest_run_ms", "hf_cutoff_hz",
    "effective_bit_depth", "tempo_bpm", "section_count", "duration_s",
]


def _integrated(x: np.ndarray, sr: int) -> float | None:
    _, bl = block_loudness(x, sr, 0.4, 0.1)
    v, _ = gated_integrated(bl)
    return v


def _third_octave_db(x: np.ndarray, sr: int) -> list[dict[str, Any]]:
    f, p = welch_psd(x, sr, PARAMS["spectrum"]["ltas_broadband"]["nperseg"])
    return spectrum_table(f, p, third_octave_edges(THIRD_OCTAVE_CENTRES), sr / 2.0)


def _align(a: np.ndarray, b: np.ndarray, sr: int,
           search_s: float) -> tuple[int, float]:
    """Best integer sample offset of b relative to a, and the correlation there."""
    # Coarse search on an 8 kHz downmix keeps a 10 s window affordable.
    coarse_sr = 8000
    ca = resample_to(a.astype(np.float32), sr, coarse_sr).astype(np.float64)
    cb = resample_to(b.astype(np.float32), sr, coarse_sr).astype(np.float64)
    nmax = int(search_s * coarse_sr)
    L = min(ca.size, cb.size, coarse_sr * 120)
    ca, cb = ca[:L], cb[:L]
    ca = ca - ca.mean()
    cb = cb - cb.mean()
    corr = sps.correlate(ca, cb, mode="full", method="fft")
    mid = cb.size - 1
    lo, hi = max(0, mid - nmax), min(corr.size, mid + nmax + 1)
    seg = corr[lo:hi]
    k = int(np.argmax(np.abs(seg)))
    lag_coarse = (lo + k) - mid
    lag = int(round(lag_coarse * sr / coarse_sr))
    # Refine at full rate over +/- 2 ms around the coarse estimate.
    win = int(0.002 * sr)
    best, best_c = lag, -2.0
    n_test = min(a.size, b.size, sr * 30)
    for cand in range(lag - win, lag + win + 1, max(1, win // 64)):
        aa, bb = _shift_pair(a, b, cand, n_test)
        if aa is None:
            continue
        c = correlation(aa, bb)
        if c is not None and c > best_c:
            best_c, best = c, cand
    return best, best_c


def _shift_pair(a: np.ndarray, b: np.ndarray, lag: int,
                limit: int | None = None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Align b to a by `lag` samples and trim both to the common region."""
    if lag >= 0:
        aa, bb = a[lag:], b[: a.size - lag]
    else:
        aa, bb = a[: b.size + lag], b[-lag:]
    m = min(aa.size, bb.size)
    if m < 1000:
        return None, None
    if limit:
        m = min(m, limit)
    return aa[:m], bb[:m]


def run_null_test(pa: str, pb: str, collector: Collector) -> dict[str, Any]:
    P = PARAMS["compare"]["null_test"]
    sa = AudioSource(pa, collector)
    sb = AudioSource(pb, collector)
    sr = max(sa.sr, sb.sr)
    xa = resample_to(sa.x, sa.sr, sr).astype(np.float64)
    xb = resample_to(sb.x, sb.sr, sr).astype(np.float64)
    ch = min(xa.shape[1], xb.shape[1])
    xa, xb = xa[:, :ch], xb[:, :ch]
    ma, mb = xa.mean(axis=1), xb.mean(axis=1)
    lag, corr = _align(ma, mb, sr, P["align_search_s"])
    if corr is None or corr < P["min_correlation"]:
        return {
            "performed": False,
            "reason": f"correlation after alignment is {corr if corr is None else round(corr, 3)}, "
                      f"below the {P['min_correlation']} floor: these are not "
                      "plausibly the same performance, so a null test would be "
                      "meaningless",
            "lag_samples": lag, "alignment_correlation": corr,
            "resampled_to_hz": sr,
        }
    aa, bb = _shift_pair(ma, mb, lag)
    la, lb = _integrated(aa[:, None], sr), _integrated(bb[:, None], sr)
    gain_db = (la - lb) if (la is not None and lb is not None) else 0.0
    g = 10 ** (gain_db / 20.0)
    resid = aa - bb * g
    res_db = db_amp(float(np.sqrt(np.mean(resid ** 2))))
    a_db = db_amp(float(np.sqrt(np.mean(aa ** 2))))
    rows_r = _third_octave_db(resid, sr)
    rows_a = _third_octave_db(aa, sr)
    per_third = []
    for rr, ra in zip(rows_r, rows_a):
        per_third.append({
            "centre_hz": rr["centre_hz"],
            "residual_db": rr["db"],
            "residual_rel_a_db": (rr["db"] - ra["db"])
            if (rr["db"] is not None and ra["db"] is not None) else None,
        })
    w = sr
    m = (resid.size // w) * w
    timeline = (db_amp(np.sqrt(np.mean(resid[:m].reshape(-1, w) ** 2, axis=1)))
                if m else np.zeros(0))
    return {
        "performed": True,
        "resampled_to_hz": sr,
        "lag_samples": lag,
        "lag_ms": 1000.0 * lag / sr,
        "alignment_correlation": corr,
        "gain_applied_to_b_db": gain_db,
        "residual_dbfs": res_db,
        "reference_a_dbfs": a_db,
        "residual_rel_reference_db": res_db - a_db if a_db is not None else None,
        "residual_per_third_octave": per_third,
        "residual_timeline_hop_s": 1.0,
        "residual_timeline_dbfs": timeline,
        "note": "with the mix held constant, the residual is the mastering "
                "difference, isolated",
    }


def compare_files(pa: str, pb: str, out_dir: str, *, null_test: bool = False,
                  profile: str = "full", log=None) -> dict[str, str]:
    do_null = null_test
    if log:
        log(f"analysing A: {os.path.basename(pa)}")
    ra = analyze_file(pa, profile=profile)
    if log:
        log(f"analysing B: {os.path.basename(pb)}")
    rb = analyze_file(pb, profile=profile)

    la = ra["headline"]["lufs_i"]
    lb = rb["headline"]["lufs_i"]
    gain_db = (la - lb) if (la is not None and lb is not None) else None
    collector = Collector()
    if gain_db is None:
        collector.warn("compare", "one file has no integrated loudness; the "
                                  "comparison is NOT level-matched")
        gain_db = 0.0

    # Headline side by side.
    rows = []
    for k in HEADLINE_ORDER:
        va, vb = ra["headline"].get(k), rb["headline"].get(k)
        d = None
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            d = float(vb) - float(va)
        matched = None
        if k in LEVEL_DEPENDENT and d is not None:
            matched = d + gain_db
        rows.append({"metric": k, "a": va, "b": vb, "delta_b_minus_a": d,
                     "delta_level_matched": matched,
                     "level_dependent": k in LEVEL_DEPENDENT})

    # Spectral difference on level-matched signals.
    if log:
        log("level-matched spectral difference")
    sa = AudioSource(pa, collector)
    sb = AudioSource(pb, collector)
    sr = min(sa.band_sr, sb.band_sr)
    g = 10 ** (gain_db / 20.0)
    a_mid = resample_to(sa.band_mid.astype(np.float32), sa.band_sr, sr).astype(np.float64)
    b_mid = resample_to(sb.band_mid.astype(np.float32), sb.band_sr, sr).astype(np.float64) * g
    a_side = resample_to(sa.band_side.astype(np.float32), sa.band_sr, sr).astype(np.float64)
    b_side = resample_to(sb.band_side.astype(np.float32), sb.band_sr, sr).astype(np.float64) * g

    diff: dict[str, list[dict[str, Any]]] = {}
    for label, xa, xb in (("mid", a_mid, b_mid), ("side", a_side, b_side)):
        ta, tb = _third_octave_db(xa, sr), _third_octave_db(xb, sr)
        diff[label] = [
            {"centre_hz": x["centre_hz"], "a_db": x["db"], "b_db": y["db"],
             "b_minus_a_db": (y["db"] - x["db"])
             if (x["db"] is not None and y["db"] is not None) else None}
            for x, y in zip(ta, tb)
        ]

    # Per-band side/mid and correlation differences.
    from .dsp import band_filter
    band_diff = []
    for name, lo, hi in BANDS:
        if lo >= sr / 2.0:
            continue
        def sm(mid: np.ndarray, side: np.ndarray) -> float | None:
            bm = band_filter(mid, sr, lo, hi)
            bs = band_filter(side, sr, lo, hi)
            pm, ps = float(np.mean(bm ** 2)), float(np.mean(bs ** 2))
            return db_pow(ps / pm) if pm > 0 and ps > 0 else None
        va, vb = sm(a_mid, a_side), sm(b_mid, b_side)
        band_diff.append({"band": name, "a_side_minus_mid_db": va,
                          "b_side_minus_mid_db": vb,
                          "delta_db": (vb - va) if (va is not None and vb is not None) else None})

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool_version": __version__,
        "profile": profile,
        "file_a": {"path": os.path.abspath(pa), "filename": os.path.basename(pa),
                   "sha256": ra["file"]["sha256"], "lufs_i": la},
        "file_b": {"path": os.path.abspath(pb), "filename": os.path.basename(pb),
                   "sha256": rb["file"]["sha256"], "lufs_i": lb},
        "level_match": {
            "rule": PARAMS["compare"]["level_match"],
            "gain_applied_to_b_db": gain_db,
            "matched_lufs_i": la,
        },
        "headline_comparison": rows,
        "third_octave_difference_db": diff,
        "per_band_side_mid_difference": band_diff,
        "correlation_difference": _delta(ra, rb, ["stereo", "correlation", "overall"]),
        "psr_difference": _delta(ra, rb, ["loudness", "psr", "min_db"]),
        "crest_difference": _delta(ra, rb, ["dynamics", "crest", "whole_file_db"]),
        "warnings": collector.warnings,
    }
    if do_null:
        if log:
            log("null test")
        result["null_test"] = run_null_test(pa, pb, collector)
    else:
        result["null_test"] = {"performed": False,
                               "reason": "not requested (pass --null-test)"}
    result["warnings"] = collector.warnings

    os.makedirs(out_dir, exist_ok=True)
    jp = os.path.join(out_dir, "comparison.json")
    with open(jp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(jsonable(result), f, indent=1, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        f.write("\n")
    mp = os.path.join(out_dir, "comparison.md")
    with open(mp, "w", encoding="utf-8", newline="\n") as f:
        f.write(render_comparison(result))
    return {"comparison.json": jp, "comparison.md": mp}


def _delta(ra: dict, rb: dict, path: list[str]) -> dict[str, Any]:
    def dig(d):
        for k in path:
            if not isinstance(d, dict):
                return None
            d = d.get(k)
        return d
    va, vb = dig(ra), dig(rb)
    return {"a": va, "b": vb,
            "delta_b_minus_a": (vb - va) if isinstance(va, (int, float))
            and isinstance(vb, (int, float)) else None}


# Counts, Hz and bit depths are integers; showing them to two decimals in the
# side-by-side table just makes it harder to read.
INTEGER_METRICS = ("flat_top_sample_count", "effective_bit_depth", "hf_cutoff_hz",
                   "mono_crossover_hz", "section_count", "dr14")


def _decimals(metric: str) -> int:
    return 0 if metric in INTEGER_METRICS else 2


def render_comparison(res: dict[str, Any]) -> str:
    out = ["# mtx compare\n",
           f"A: {res['file_a']['filename']}",
           f"B: {res['file_b']['filename']}",
           f"level match: B gained by {n(res['level_match']['gain_applied_to_b_db'], 3)} dB "
           f"to A's {n(res['file_a']['lufs_i'], 2)} LUFS\n",
           "## HEADLINE (B minus A)\n", "```"]
    rows = []
    for r in res["headline_comparison"]:
        nd = _decimals(r["metric"])
        rows.append([r["metric"], n(r["a"], nd), n(r["b"], nd),
                     n(r["delta_b_minus_a"], nd),
                     n(r["delta_level_matched"], 2) if r["level_dependent"] else "-"])
    out.append(table(["metric", "A", "B", "delta", "delta (matched)"], rows))
    out.append("```\n")

    out.append("## THIRD-OCTAVE DIFFERENCE, B minus A (dB)\n")
    for label in ("mid", "side"):
        cells = [f"{hz(r['centre_hz'])}:{n(r['b_minus_a_db'], 1)}"
                 for r in res["third_octave_difference_db"][label]]
        out.append(f"### {label}\n\n```")
        out.extend("  ".join(cells[i:i + 5]) for i in range(0, len(cells), 5))
        out.append("```\n")

    out.append("## PER-BAND SIDE/MID\n\n```")
    out.append(table(["band", "A dB", "B dB", "delta"],
                     [[b["band"], n(b["a_side_minus_mid_db"], 1),
                       n(b["b_side_minus_mid_db"], 1), n(b["delta_db"], 1)]
                      for b in res["per_band_side_mid_difference"]]))
    out.append("```\n")

    d = res
    out.append("## OTHER DELTAS\n\n```")
    out.append(kv_rows([
        ("correlation", f"A {n(d['correlation_difference']['a'], 2)}  "
                        f"B {n(d['correlation_difference']['b'], 2)}  "
                        f"delta {n(d['correlation_difference']['delta_b_minus_a'], 2)}"),
        ("PSR min", f"A {n(d['psr_difference']['a'], 1)}  "
                    f"B {n(d['psr_difference']['b'], 1)}  "
                    f"delta {n(d['psr_difference']['delta_b_minus_a'], 1)} dB"),
        ("crest", f"A {n(d['crest_difference']['a'], 1)}  "
                  f"B {n(d['crest_difference']['b'], 1)}  "
                  f"delta {n(d['crest_difference']['delta_b_minus_a'], 1)} dB"),
    ]))
    out.append("```\n")

    nt = res["null_test"]
    out.append("## NULL TEST\n\n```")
    if nt.get("performed"):
        out.append(kv_rows([
            ("alignment", f"lag {nt['lag_samples']} samples ({n(nt['lag_ms'], 3)} ms), "
                          f"correlation {n(nt['alignment_correlation'], 3)}"),
            ("resampled to", f"{nt['resampled_to_hz']} Hz"),
            ("gain applied to B", f"{n(nt['gain_applied_to_b_db'], 3)} dB"),
            ("residual", f"{n(nt['residual_dbfs'], 2)} dBFS "
                         f"({n(nt['residual_rel_reference_db'], 2)} dB below A)"),
        ]))
        worst = sorted([r for r in nt["residual_per_third_octave"]
                        if r["residual_rel_a_db"] is not None],
                       key=lambda r: -r["residual_rel_a_db"])[:8]
        out.append("\nloudest residual bands (dB relative to A):")
        out.append("  ".join(f"{hz(r['centre_hz'])}:{n(r['residual_rel_a_db'], 1)}"
                             for r in worst))
    else:
        out.append(f"not performed: {nt.get('reason')}")
    out.append("```\n")
    if res.get("warnings"):
        out.append("## WARNINGS\n")
        out.extend(f"- {w}" for w in res["warnings"])
        out.append("")
    return "\n".join(out) + "\n"
