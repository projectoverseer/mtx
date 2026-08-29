"""digest.md: the compact, paste-able view of analysis.json.

Default budget of 12 KB.  Blocks are assembled in a fixed order and dropped
from the lowest priority upward until the budget is met; whatever was dropped
is named in the output, so the digest never silently omits a section.  The
budget is a default and not a law: `--digest-budget` raises it and `--sections`
selects what is worth spending it on, because a fixed cap with a fixed drop
order otherwise loses the block a given session actually needed.

`--stems` adds a whole section of measurements that exist nowhere else in the
paste-able output, so the budget grows by `STEMS_BUDGET_BONUS` when it renders
rather than pushing the stems out of their own run.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

SIZE_BUDGET_BYTES = 12 * 1024
STEMS_BUDGET_BONUS = 4 * 1024

# Redaction marker used by the prediction sheet.  Wide enough to write in.
BLANK = "____"


# ------------------------------------------------------------------ formatting
def n(v: Any, nd: int = 1, unit: str = "") -> str:
    """Round for display.  A missing value is `n/a`, never a substituted zero."""
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return "n/a"
    return f"{f:.{nd}f}{unit}"


def db(v: Any, nd: int = 1) -> str:
    """A dB value, rendering the -200 dB floor as -inf.

    Only for quantities that use that floor (side/mid ratios); other metrics go
    through `n`, so a legitimate -200 ms lag is never mistaken for silence.
    """
    if v is not None and not isinstance(v, str):
        try:
            if float(v) <= -199.9:
                return "-inf"
        except (TypeError, ValueError):
            pass
    return n(v, nd)


def hz(v: Any) -> str:
    """Hz, integer above 1 kHz as the digest contract requires."""
    if v is None:
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return "n/a"
    return f"{f:.0f}" if abs(f) >= 1000 else f"{f:.1f}"


def table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    rows = [list(map(str, r)) for r in rows]
    if not rows:
        return ""
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r[: len(widths)]):
            widths[i] = max(widths[i], len(c))
    out = ["  ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip(),
           "  ".join("-" * w for w in widths)]
    for r in rows:
        out.append("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())
    return "\n".join(out)


def kv_rows(pairs: Sequence[tuple[str, str]]) -> str:
    w = max((len(k) for k, _ in pairs), default=0)
    return "\n".join(f"{k.ljust(w)}  {v}" for k, v in pairs)


def wrap_series(values: Sequence[Any], per_line: int, nd: int = 1) -> str:
    out, line = [], []
    for i, v in enumerate(values):
        line.append(n(v, nd).rjust(6))
        if len(line) == per_line:
            out.append(" ".join(line))
            line = []
    if line:
        out.append(" ".join(line))
    return "\n".join(out)


def _grid(series: Sequence[float] | None, times: Sequence[float] | None,
          step_s: float) -> list[float]:
    """Resample a timeline onto a fixed grid by nearest sample."""
    if series is None or times is None or len(series) == 0 or len(times) == 0:
        return []
    series = list(series)
    out = []
    t = 0.0
    ti = 0
    last = times[-1]
    times = list(times)
    while t <= last:
        while ti + 1 < len(times) and times[ti + 1] <= t:
            ti += 1
        out.append(series[min(ti, len(series) - 1)])
        t += step_s
    return out


def _grid_mean_db(series: Sequence[float] | None, times: Sequence[float] | None,
                  step_s: float) -> list[float]:
    """Average a dB series within each grid cell, in the power domain.

    On a coarse grid a nearest-sample reading of one instant is misleading: a
    cell that starts in the leading silence would report the silence rather than
    the ten seconds of music behind it.
    """
    if series is None or times is None or len(series) == 0 or len(times) == 0:
        return []
    vals = np.asarray(series, dtype=float)
    t = np.asarray(times, dtype=float)
    ok = np.isfinite(vals)
    if not ok.any():
        return []
    cells = (t / step_s).astype(int)
    out: list[float] = []
    for c in range(int(cells.max()) + 1):
        m = (cells == c) & ok
        if not m.any():
            out.append(float("nan"))
            continue
        out.append(float(10.0 * np.log10(np.mean(np.power(10.0, vals[m] / 10.0)))))
    return out


# The headline fields a prediction can be committed against, with the unit and
# the rounding the digest prints.  One table, so the prediction sheet, the
# `--check` arithmetic and the HEADLINE block can never drift apart; the labels
# are asserted against the rendered HEADLINE in the tests.
PREDICT_FIELDS: tuple[tuple[str, str, str, int], ...] = (
    # (label, headline key, unit, decimals)
    ("LUFS-I", "lufs_i", "LUFS", 2),
    ("LRA", "lra_lu", "LU", 1),
    ("True peak (16x)", "true_peak_dbtp_16x", "dBTP", 2),
    ("Sample peak", "sample_peak_dbfs", "dBFS", 2),
    ("PLR", "plr_db", "dB", 1),
    ("PSR min", "psr_min_db", "dB", 1),
    ("PSR median", "psr_median_db", "dB", 1),
    ("DR14", "dr14", "", 0),
    ("Crest (whole)", "crest_whole_db", "dB", 1),
    ("Crest (loudest 10 s)", "crest_loudest_10s_db", "dB", 1),
    ("Spectral tilt", "spectral_tilt_db_per_oct", "dB/oct", 2),
    ("Air 12-20k", "air_band_pct", "%", 2),
    ("Sub 20-60", "sub_band_pct", "%", 2),
    ("Side/mid overall", "side_minus_mid_db", "dB", 1),
    ("Side/mid <120 Hz", "side_minus_mid_below_120hz_db", "dB", 1),
    ("Mono crossover", "mono_crossover_hz", "Hz", 0),
    ("Correlation mean", "correlation_mean", "", 2),
    ("Correlation min", "correlation_min", "", 2),
    ("HF cutoff", "hf_cutoff_hz", "Hz", 0),
    ("Tempo", "tempo_bpm", "BPM", 2),
)


# ---------------------------------------------------------------- the sections
def _headline(res: dict[str, Any]) -> str:
    h = res["headline"]
    L = res.get("loudness", {})
    tp = L.get("true_peak", {})
    pairs = [
        ("LUFS-I", n(h["lufs_i"], 2) + " LUFS"),
        ("LRA", n(h["lra_lu"], 1) + " LU"),
        ("True peak (16x)", n(h["true_peak_dbtp_16x"], 2) + " dBTP"),
        ("True peak (4x)", n(tp.get("overall_dbtp_4x"), 2) + " dBTP"),
        ("Sample peak", n(h["sample_peak_dbfs"], 2) + " dBFS"),
        ("TP-SP delta", n(tp.get("delta_truepeak16x_minus_samplepeak_db"), 2) + " dB"),
        ("PLR", n(h["plr_db"], 1) + " dB"),
        ("PSR min", f"{n(h['psr_min_db'], 1)} dB @ {h.get('psr_min_time') or 'n/a'}"),
        ("PSR median", n(h["psr_median_db"], 1) + " dB"),
        ("DR14", n(h["dr14"], 0)),
        ("Crest (whole)", n(h["crest_whole_db"], 1) + " dB"),
        ("Crest (loudest 10 s)", n(h["crest_loudest_10s_db"], 1) + " dB"),
        ("Spectral tilt", f"{n(h['spectral_tilt_db_per_oct'], 2)} dB/oct "
                          f"(R2 {n(h['spectral_tilt_r2'], 2)})"),
        ("Air 12-20k", n(h["air_band_pct"], 2) + " % of energy"),
        ("Sub 20-60", n(h["sub_band_pct"], 2) + " % of energy"),
        ("Side/mid overall", db(h["side_minus_mid_db"]) + " dB"),
        ("Side/mid <120 Hz", db(h["side_minus_mid_below_120hz_db"]) + " dB"),
        ("Mono crossover", hz(h["mono_crossover_hz"]) + " Hz"),
        ("Correlation mean", n(h["correlation_mean"], 2)),
        ("Correlation min", n(h["correlation_min"], 2)),
        ("Flat-top samples", f"{h['flat_top_sample_count']}"),
        ("Flat-top longest", n(h["flat_top_longest_run_ms"], 2) + " ms"),
        ("HF cutoff", hz(h["hf_cutoff_hz"]) + " Hz"),
        ("Effective bit depth", n(h["effective_bit_depth"], 0) + " bits"),
        ("Tempo", n(h["tempo_bpm"], 2) + " BPM"),
        ("Key", h["key"] or "n/a"),
        ("Sections", n(h["section_count"], 0)),
        ("Duration", n(h["duration_s"], 3) + " s"),
    ]
    return "## HEADLINE\n\n```\n" + kv_rows(pairs) + "\n```\n"


def _flags(res: dict[str, Any]) -> str:
    lines: list[str] = []
    for w in res.get("warnings", []):
        lines.append(f"- {w if len(w) <= 200 else w[:197] + '...'}")
    for note in res.get("confidence_notes", []):
        lines.append(f"- [{note['confidence']}] {note['metric']}: {note['reason'][:160]}")
    dr = res.get("loudness", {}).get("dr14", {}).get("validation", {})
    rec = (dr or {}).get("record", {})
    if dr and dr.get("validated_against_published_reference"):
        lines.append(f"- [validated against {rec.get('tracks_checked')} track(s)] "
                     f"DR14: worst disagreement with a published rating "
                     f"{n(rec.get('max_abs_delta_dr'), 1)} DR "
                     f"(mtx validate-dr record)")
    elif dr and rec.get("tracks_checked"):
        lines.append(f"- [disputed] DR14: {rec['tracks_checked'] - rec.get('tracks_within_tolerance', 0)} "
                     f"of {rec['tracks_checked']} recorded published rating(s) "
                     f"disagree by more than {n(rec.get('tolerance_dr'), 1)} DR; "
                     f"worst {n(rec.get('max_abs_delta_dr'), 1)} DR")
    elif dr:
        lines.append("- [unverified] DR14: not validated against a published DR "
                     "rating; synthetic checks only (run mtx validate-dr once, "
                     "see METHOD)")
    shown = lines[:22]
    extra = len(lines) - len(shown)
    body = "\n".join(shown) if shown else "- none"
    if extra > 0:
        body += f"\n- (+{extra} more in analysis.json: warnings[], confidence_notes[])"
    return "## FLAGS\n\n" + body + "\n"


def _block_bands(res: dict[str, Any]) -> str:
    S = res.get("spectrum", {})
    if not S.get("available"):
        return ""
    tabs = S["band_energy"]["tables"]
    rows = []
    for i, (name, lo, hi) in enumerate(S["band_energy"]["bands_hz"]):
        m = tabs["mid"][i]
        s = tabs["side"][i]
        rows.append([name, f"{hz(lo)}-{hz(hi)}", n(m.get("pct"), 2), n(m.get("db"), 1),
                     n(s.get("pct"), 2), n(s.get("db"), 1)])
    return ("### Band energy (mid / side)\n\n```\n"
            + table(["band", "Hz", "mid%", "mid dB", "side%", "side dB"], rows)
            + "\n```\n")


def _block_band_crest(res: dict[str, Any]) -> str:
    D = res.get("dynamics", {}).get("per_band_crest", {})
    rows = [[b["band"], n(b.get("crest_db"), 1), n(b.get("rms_dbfs"), 1),
             n(b.get("peak_dbfs"), 1)] for b in D.get("bands", [])]
    if not rows:
        return ""
    spread = D.get("spread_db")
    return ("### Per-band crest (mono)\n\n```\n"
            + table(["band", "crest dB", "rms dBFS", "peak dBFS"], rows)
            + f"\nspread across bands: {n(spread, 1)} dB\n```\n")


def _block_sidemid(res: dict[str, Any]) -> str:
    ST = res.get("stereo", {})
    if not ST.get("available"):
        return ""
    rows = ST["side_minus_mid_per_third_octave"]
    cells = [f"{hz(r['centre_hz'])}:{db(r['side_minus_mid_db'])}" for r in rows]
    lines = ["  ".join(cells[i:i + 5]) for i in range(0, len(cells), 5)]
    dmg = ST["mono_sum_damage"]["per_third_octave"]
    worst = sorted([d for d in dmg if d["mono_sum_loss_db"] is not None],
                   key=lambda d: d["mono_sum_loss_db"])[:6]
    worst_s = ", ".join(f"{hz(d['centre_hz'])} Hz {n(d['mono_sum_loss_db'], 1)} dB"
                        for d in worst)
    return ("### Side/mid per third-octave (Hz:dB)\n\n```\n" + "\n".join(lines)
            + f"\n\nmono-sum loss, broadband: {n(ST['mono_sum_damage']['broadband_loss_db'], 2)} dB"
            + f"\nmono-sum loss, worst bands: {worst_s or 'n/a'}\n```\n")


def _block_loudness_timeline(res: dict[str, Any]) -> str:
    L = res.get("loudness", {})
    st = L.get("shortterm", {})
    grid = _grid(st.get("lufs"), st.get("times_s"), 5.0)
    p = st.get("percentiles_lufs", {}) or {}
    m = L.get("momentary", {}).get("percentiles_lufs", {}) or {}
    head = (f"short-term percentiles LUFS  P10 {n(p.get('p10'), 1)}  P25 {n(p.get('p25'), 1)}  "
            f"P50 {n(p.get('p50'), 1)}  P75 {n(p.get('p75'), 1)}  P90 {n(p.get('p90'), 1)}  "
            f"P95 {n(p.get('p95'), 1)}  max {n(st.get('max_lufs'), 1)}\n"
            f"momentary  percentiles LUFS  P10 {n(m.get('p10'), 1)}  P25 {n(m.get('p25'), 1)}  "
            f"P50 {n(m.get('p50'), 1)}  P75 {n(m.get('p75'), 1)}  P90 {n(m.get('p90'), 1)}  "
            f"P95 {n(m.get('p95'), 1)}  max {n(L.get('momentary', {}).get('max_lufs'), 1)}")
    body = wrap_series(grid, 12, 1) if grid else "n/a"
    return ("### Short-term loudness, 5 s grid (LUFS)\n\n```\n" + head
            + "\n\n" + body + "\n```\n")


def _block_psr(res: dict[str, Any]) -> str:
    P = res.get("loudness", {}).get("psr", {})
    series = P.get("psr_db")
    if series is None or len(series) == 0:
        return ""
    grid = _grid(P.get("psr_db"), P.get("times_s"), 5.0)
    return ("### PSR, 5 s grid (dB)\n\n```\n"
            f"min {n(P.get('min_db'), 1)} @ {P.get('min_time')}   "
            f"P10 {n(P.get('p10_db'), 1)}   median {n(P.get('median_db'), 1)}   "
            f"max {n(P.get('max_db'), 1)}\n\n" + wrap_series(grid, 12, 1) + "\n```\n")


def _block_streaming(res: dict[str, Any]) -> str:
    sp = res.get("loudness", {}).get("streaming_preview", {})
    rows = []
    for name, v in sp.items():
        rows.append([name, n(v["target_lufs"], 0), n(v["gain_db"], 2),
                     n(v["true_peak_after_dbtp"], 2),
                     "yes" if v.get("gain_is_positive") else "no"])
    if not rows:
        return ""
    return ("### Streaming normalisation preview\n\n```\n"
            + table(["platform", "target", "gain dB", "TP after dBTP", "turned up"], rows)
            + "\n```\n")


def _block_flat_top(res: dict[str, Any]) -> str:
    F = res.get("dynamics", {}).get("flat_top", {})
    if not F:
        return ""
    rows = []
    for c in F.get("per_channel", []):
        h = c["run_length_histogram"]
        rows.append([str(c["channel"]), n(c["threshold_dbfs"], 3),
                     str(c["flat_sample_count"]), str(c["event_count"]),
                     f"{h['1']}/{h['2']}/{h['3-5']}/{h['6-10']}/{h['11-20']}/{h['21+']}",
                     n(c["longest_run_ms"], 2)])
    cn = F.get("clip_then_normalise", {})
    lv = F.get("limiter_vs_clipper", {})
    lfa = F.get("low_frequency_association", {})
    tail = (f"\nflat value {n(cn.get('flat_value_dbfs'), 3)} dBFS, "
            f"below full scale: {cn.get('ceiling_below_full_scale')}, "
            f"clip-then-normalise: {cn.get('detected')}"
            f"\nentry/exit slope median {n((lv.get('entry_slope_db_per_ms') or {}).get('median'), 2)}"
            f" / {n((lv.get('exit_slope_db_per_ms') or {}).get('median'), 2)} dB/ms"
            f" over {lv.get('events_measured', 0)} events -> {lv.get('inference') or 'n/a'} (inferred)"
            f"\nsub-120 Hz at events vs track: {n(lfa.get('delta_db'), 1)} dB, "
            f"correlation {n(lfa.get('correlation'), 2)}")
    return ("### Flat-top / ceiling forensics\n\n```\n"
            + table(["ch", "thr dBFS", "samples", "events", "runs 1/2/3-5/6-10/11-20/21+",
                     "longest ms"], rows) + tail + "\n```\n")


def _block_ceiling(res: dict[str, Any]) -> str:
    F = res.get("dynamics", {}).get("flat_top", {})
    rows = []
    for c in F.get("per_channel", []):
        d = c.get("ceiling_density", {})
        if not d:
            continue
        rows.append([str(c["channel"])] + [n(100 * d.get(k, 0), 3) for k in
                                           ("within_0.1_db", "within_0.5_db",
                                            "within_1.0_db", "within_3.0_db",
                                            "within_6.0_db")])
    if not rows:
        return ""
    return ("### Ceiling density (% of samples within N dB of the ceiling)\n\n```\n"
            + table(["ch", "0.1", "0.5", "1", "3", "6"], rows) + "\n```\n")


def _block_forensics(res: dict[str, Any]) -> str:
    F = res.get("forensics", {})
    if not F.get("available"):
        return ""
    c = F["hf_cutoff"]
    shelf = c.get("codec_shelf_match") or {}
    stab = F["cutoff_stability"]
    nf = F["noise_floor"]
    bit = F["effective_bit_depth"]
    sil = F["silence"]
    up = F["upsampling"]
    holes = F["spectral_holes"][:4]
    pairs = [
        ("HF cutoff", f"{hz(c.get('cutoff_hz'))} Hz, slope "
                      f"{n(c.get('rolloff_slope_db_per_oct'), 1)} dB/oct, collapse "
                      f"{n(c.get('collapse_depth_db'), 1)} dB, frames below floor "
                      f"{n(100 * (c.get('fraction_of_frames_above_cutoff_below_floor') or 0), 1)}%"),
        ("Nearest codec shelf", f"{hz(shelf.get('nearest_shelf_hz'))} Hz "
                                f"(distance {hz(shelf.get('distance_hz'))} Hz)"
         if shelf else "n/a"),
        ("Cutoff stability", f"mean {hz(stab.get('mean_hz'))} Hz, std "
                             f"{hz(stab.get('std_hz'))} Hz over {stab.get('frames_measured', 0)} frames"),
        ("Upsampling", f"{up.get('suspected_upsampled')}, nearest original "
                       f"{up.get('nearest_original_rate_hz')} Hz, mirror corr "
                       f"{n(up.get('mirror_correlation'), 2)}" if up.get("checked")
         else up.get("reason", "n/a")),
        ("Effective bit depth", f"{bit.get('effective_bits')} of "
                                f"{bit.get('container_bits')} container bits"),
        ("Noise floor", f"{n(nf.get('level_dbfs'), 1)} dBFS, slope above 10k "
                        f"{n(nf.get('slope_above_10k_db_per_oct'), 1)} dB/oct"),
        ("Silence", f"lead {n(sil.get('leading_black_ms'), 1)} ms ({sil.get('start_kind')}, "
                    f"fade {n(sil.get('fade_in_ms'), 0)} ms), tail "
                    f"{n(sil.get('trailing_black_ms'), 1)} ms ({sil.get('end_kind')}, "
                    f"fade {n(sil.get('fade_out_ms'), 0)} ms)"),
        ("Spectral holes", ", ".join(f"{hz(h['centre_hz'])} Hz -{n(h['depth_db'], 1)} dB"
                                     for h in holes) or "none above threshold"),
    ]
    sig = F["analog_signatures"]
    for mains in ("50hz", "60hz"):
        v = sig["mains_hum"].get(mains, {})
        pairs.append((f"Hum {mains}", f"max excess {n(v.get('max_excess_db'), 1)} dB, "
                                      f"mean {n(v.get('mean_excess_db'), 1)} dB"))
    pairs.append(("Rumble <30 Hz", n(sig["rumble"].get("level_db_rel_total"), 1) + " dB rel total"))
    bias = sig["tape_bias"]["peaks"]
    pairs.append(("Tape bias >15k", ", ".join(f"{hz(b['frequency_hz'])} Hz +{n(b['excess_db'], 1)} dB"
                                              for b in bias[:2]) or "none"))
    wf = sig["wow_flutter"]
    pairs.append(("Wow/flutter", f"std {n(wf.get('cents_std'), 1)} cents, detrended "
                                 f"{n(wf.get('cents_detrended_std'), 1)}, drift "
                                 f"{n(wf.get('slow_drift_cents_per_min'), 2)} cents/min"
                  if wf.get("available") else wf.get("reason", "n/a")))
    return "### Source forensics\n\n```\n" + kv_rows(pairs) + "\n```\n"


def _block_sections(res: dict[str, Any]) -> str:
    S = res.get("structure", {})
    if not S.get("available") or not S.get("sections"):
        return ""
    rows = []
    for s in S["sections"][:16]:
        rows.append([str(s.get("index")), s.get("start", ""), n(s.get("duration_s"), 1),
                     n(s.get("lufs_i"), 1), n(s.get("shortterm_max_lufs"), 1),
                     n(s.get("crest_db"), 1), n(s.get("tilt_db_per_oct"), 2),
                     db(s.get("side_minus_mid_db")),
                     n(s.get("onset_rate_per_s"), 2),
                     n(s.get("delta_vs_track_lufs"), 1)])
    j = S.get("biggest_jump") or {}
    wb = S.get("widest_band_in_widest_section") or {}
    tail = (f"\nloudest section {S.get('loudest_section_index')}, quietest "
            f"{S.get('quietest_section_index')}, widest {S.get('widest_section_index')}"
            + (f" (widest band there: {wb['band']} at {db(wb['side_minus_mid_db'])} dB)"
               if wb.get("band") else "")
            + f"\nbiggest jump {n(j.get('db'), 1)} dB at {j.get('time', 'n/a')}")
    extra = len(S["sections"]) - 16
    if extra > 0:
        tail += f"\n(+{extra} further sections in analysis.json)"
    return ("### Sections\n\n```\n"
            + table(["#", "start", "dur", "LUFS", "ST max", "crest", "tilt",
                     "side/mid", "ons/s", "d-track"], rows) + tail + "\n```\n")


def _block_processing(res: dict[str, Any]) -> str:
    P = res.get("processing", {})
    sat = P.get("saturation_proxy", {})
    pump = P.get("bus_compression", {})
    hp = P.get("hpss", {})
    rv = P.get("reverb", {})
    mod = P.get("modulation_spectrum", {})
    pairs = [
        ("Saturation slope", f"{n(sat.get('slope_db_per_db'), 3)} dB/dB "
                             f"(R2 {n(sat.get('r2'), 2)}, {sat.get('frames_used', 0)} frames) "
                             f"[{sat.get('confidence', 'n/a')}]"
         if sat.get("available") else sat.get("reason", "n/a")),
        ("Pumping", f"most negative corr {n(pump.get('most_negative_correlation'), 2)} at "
                    f"{n(pump.get('most_negative_lag_ms'), 0)} ms, dip "
                    f"{n(pump.get('dip_depth_db'), 1)} dB, release "
                    f"{n(pump.get('estimated_release_ms'), 0)} ms [{pump.get('confidence', 'n/a')}]"
         if pump.get("available") else pump.get("reason", "n/a")),
        ("Perc/harm", f"{n(hp.get('percussive_to_harmonic_db'), 1)} dB, percussive "
                      f"fraction {n(hp.get('percussive_energy_fraction'), 3)}"
         if hp.get("available") else hp.get("reason", "n/a")),
    ]
    if rv.get("available"):
        vals = ", ".join(f"{hz(b['centre_hz'])}:{n(b.get('t20_s'), 2)}"
                         for b in rv["per_octave_band"] if b.get("t20_s") is not None)
        pairs.append(("T20 by octave (Hz:s)", vals or "no usable decays"))
        pairs.append(("Tail L/R corr", f"{n(rv.get('tail_stereo_correlation'), 2)} "
                                       f"[{rv.get('confidence')}]"))
    if mod.get("available"):
        rows = []
        for band, v in mod.get("bands", {}).items():
            if not isinstance(v, dict) or "beat_depth_db" not in v:
                continue
            rows.append(f"{band}:{n(v.get('beat_depth_db'), 1)}/"
                        f"{n(v.get('half_beat_depth_db'), 1)}/"
                        f"{n(v.get('quarter_beat_depth_db'), 1)}")
        pairs.append(("Mod depth beat/half/quarter dB", "  ".join(rows) or "n/a"))
    return "### Processing forensics (all inferred)\n\n```\n" + kv_rows(pairs) + "\n```\n"


def _block_bass(res: dict[str, Any]) -> str:
    S = res.get("spectrum", {})
    bf = S.get("bass_fundamentals", {}) if S.get("available") else {}
    peaks = bf.get("peaks", [])[:6]
    if not peaks:
        return ""
    rows = [[str(p["rank"]), n(p["frequency_hz"], 1), n(p["level_db_rel_strongest"], 1),
             p["nearest_note"], n(p["cents_from_note"], 1), n(p.get("q"), 1)]
            for p in peaks]
    lr = S.get("ltas_lowfreq", {})
    tail = (f"\nresolution {n(bf.get('resolution_hz'), 3)} Hz, section "
            f"{lr.get('section_start', 'n/a')}-{lr.get('section_end', 'n/a')}, "
            f"single-note low end: {bf.get('single_note')}")
    return ("### Bass fundamentals (131072-point Welch)\n\n```\n"
            + table(["#", "Hz", "dB rel", "note", "cents", "Q"], rows) + tail + "\n```\n")


def _block_resonances(res: dict[str, Any]) -> str:
    S = res.get("spectrum", {})
    rs = S.get("resonances", [])[:8] if S.get("available") else []
    if not rs:
        return ""
    rows = [[hz(r["frequency_hz"]), n(r["prominence_db"], 1), n(r.get("q"), 1),
             n(r.get("frame_presence_fraction"), 2)] for r in rs]
    return ("### Resonances\n\n```\n"
            + table(["Hz", "prom dB", "Q", "frame presence"], rows) + "\n```\n")


def _block_descriptors(res: dict[str, Any]) -> str:
    S = res.get("spectrum", {})
    d = S.get("descriptors", {}) if S.get("available") else {}
    if not d.get("available"):
        return ""
    w = d["whole_track"]
    pairs = [
        ("Centroid / spread", f"{hz(w.get('centroid_hz'))} / {hz(w.get('spread_hz'))} Hz"),
        ("Skew / kurtosis", f"{n(w.get('skewness'), 2)} / {n(w.get('kurtosis'), 2)}"),
        ("Flatness", n(w.get("flatness"), 4)),
        ("Rolloff 85/95/99", f"{hz(w.get('rolloff85_hz'))} / {hz(w.get('rolloff95_hz'))} / "
                             f"{hz(w.get('rolloff99_hz'))} Hz"),
        ("Zero-crossing rate", n(w.get("zcr"), 4)),
    ]
    return "### Spectral descriptors (whole track)\n\n```\n" + kv_rows(pairs) + "\n```\n"


def _block_stereo_extra(res: dict[str, Any]) -> str:
    ST = res.get("stereo", {})
    if not ST.get("available"):
        return ""
    c = ST["correlation"]
    cb = ST["channel_balance"]
    itd = ST["inter_channel_time_offset"]
    g = ST["goniometer"]
    worst = ", ".join(f"{w['start']} {n(w['correlation'], 2)}"
                      for w in c.get("most_negative_windows", []))
    pairs = [
        ("Correlation", f"mean {n(c.get('overall'), 2)}, min {n(c.get('min'), 2)}, "
                        f"P5 {n(c.get('p5'), 2)}, median {n(c.get('median'), 2)}, "
                        f"<0 {n(c.get('pct_time_below_0'), 1)}%, "
                        f"<0.3 {n(c.get('pct_time_below_0_3'), 1)}%"),
        ("Most negative windows", worst or "n/a"),
        ("Channel balance", f"L-R rms {n(cb.get('rms_l_minus_r_db'), 2)} dB, "
                            f"L-R LUFS {n(cb.get('lufs_l_minus_r'), 2)}"),
        ("Inter-channel offset", f"{itd.get('lag_samples')} samples "
                                 f"({n(itd.get('lag_us'), 1)} us), corr at lag "
                                 f"{n(itd.get('correlation_at_lag'), 3)}"),
        ("Energy outside +/-45 deg", n(100 * (g.get("fraction_energy_outside_45_deg") or 0), 1) + " %"),
    ]
    return "### Stereo detail\n\n```\n" + kv_rows(pairs) + "\n```\n"


def _block_gaps(res: dict[str, Any]) -> str:
    gaps = res.get("structure", {}).get("arrangement_gaps", [])[:8]
    if not gaps:
        return ""
    rows = [[g["band"], g["start"], n(g["duration_ms"], 0),
             n(g["band_level_db_rel_track"], 1)] for g in gaps]
    return ("### Arrangement gaps\n\n```\n"
            + table(["band", "start", "ms", "dB rel band mean"], rows) + "\n```\n")


def _block_band_timeline(res: dict[str, Any]) -> str:
    """Per-band energy over time, on the finest grid that fits the budget.

    The specification asks for a 1 s grid here; at 1 s a four-minute track needs
    more than the whole 12 KB on its own, so the grid is widened until the block
    fits and the grid actually used is printed. The 100 ms series is in
    `spectrum.band_timeline` in the JSON either way.
    """
    S = res.get("spectrum", {})
    if not S.get("available"):
        return ""
    bt = S.get("band_timeline", {})
    times = bt.get("times_s")
    bands = bt.get("bands") or {}
    if times is None or len(times) == 0 or not bands:
        return ""
    duration = float(times[-1])
    for step in (1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 60.0):
        if duration / step <= 14:
            break
    rows = []
    for name, series in bands.items():
        grid = _grid_mean_db(series, times, step)
        rows.append(f"{name:<9}" + " ".join(n(v, 0).rjust(4) for v in grid[:14]))
    return (f"### Per-band level over time (dB, {n(step, 0)} s grid; "
            "100 ms series is in analysis.json)\n\n```\n"
            + "\n".join(rows) + "\n```\n")


def _block_band_envelope(res: dict[str, Any]) -> str:
    """Broadband vs multiband compression, in three lines.

    The full 8x8 matrix is in analysis.json; the off-diagonal spread and the
    two least-correlated pairs carry nearly all of what it says.  Correlated
    envelopes mean one gain reduction moving everything together; decorrelated
    envelopes mean the bands are being controlled independently.
    """
    B = (res.get("processing", {}).get("multiband_timeline", {})
         .get("band_envelope_correlation", {}))
    od = B.get("offdiagonal") or {}
    if not od or od.get("median") is None:
        return ""
    least = B.get("least_correlated_pairs") or []
    most = B.get("most_correlated_pair") or {}
    pairs = [
        ("Off-diagonal r", f"min {n(od.get('min'), 2)}  median {n(od.get('median'), 2)}  "
                           f"max {n(od.get('max'), 2)}  mean {n(od.get('mean'), 2)}  "
                           f"over {od.get('pairs', 0)} band pairs"),
        ("Least correlated", "; ".join(
            f"{'/'.join(x['bands'])} {n(x['r'], 2)}" for x in least) or "n/a"),
        ("Most correlated", (f"{'/'.join(most['bands'])} {n(most['r'], 2)}"
                             if most.get("bands") else "n/a")),
    ]
    return ("### Band-envelope correlation (10 ms envelopes, dB domain)\n\n```\n"
            + kv_rows(pairs) + "\n```\n")


# --------------------------------------------------------------------- stems
STEM_ORDER = ("drums", "bass", "other", "vocals")


def _stem_row(name: str, e: dict[str, Any]) -> list[str]:
    L = e.get("loudness", {}) or {}
    D = e.get("dynamics", {}) or {}
    S = e.get("spectrum", {}) or {}
    ST = e.get("stereo", {}) or {}
    lv = e.get("level_vs_mix", {}) or {}
    crest = D.get("crest", {}) or {}
    tilt = (S.get("tilt", {}) if S.get("available") else {}) or {}
    corr = (ST.get("correlation", {}) if ST.get("available") else {}) or {}
    return [
        name,
        n(lv.get("rms_db"), 1),
        n(lv.get("lufs_delta"), 1),
        n(L.get("integrated_lufs"), 1),
        n(crest.get("whole_file_db"), 1),
        n((crest.get("loudest_window") or {}).get("crest_db"), 1),
        f"{n(tilt.get('slope_db_per_oct'), 1)}({n(tilt.get('r2'), 2)})",
        db(ST.get("side_minus_mid_db")) if ST.get("available") else "mono",
        n(corr.get("overall"), 2),
        n(S.get("sub_band_pct") if S.get("available") else None, 1),
        n(S.get("air_band_pct") if S.get("available") else None, 1),
    ]


def _stems(res: dict[str, Any]) -> str:
    """The `--stems` section.

    Without it the most expensive flag in the tool writes only to
    analysis.json, which by design never leaves the machine: the mix-versus-
    master attribution these numbers exist to settle would be unreachable from
    the paste-able output.  The model is named in the block because stems
    separated by different models are not comparable.
    """
    S = res.get("stems", {}) or {}
    if not S.get("requested"):
        return ""
    if not S.get("available"):
        return ("## STEMS\n\nrequested but unavailable: "
                f"{S.get('reason', 'see FLAGS')}\n")
    entries = S.get("stems", {}) or {}
    order = ([k for k in STEM_ORDER if k in entries]
             + [k for k in entries if k not in STEM_ORDER])
    rows = [_stem_row(k, entries[k]) for k in order]
    if not rows:
        return ""
    warn = []
    for k in order:
        for w in (entries[k].get("warnings") or [])[:2]:
            warn.append(f"{k}: {w}")
    head = (f"model: {S.get('model') or 'unknown'}   source: {S.get('source')}\n"
            "lvl_vs_mix and lufs_delta are this stem against the whole mix; "
            "every other column is measured on the stem alone.\n")
    flags = "FLAGS  " + (S.get("caveat") or "")
    if warn:
        flags += "\n       " + "\n       ".join(warn[:4])
    return ("## STEMS\n\n```\n" + head + "\n"
            + table(["stem", "lvl_vs_mix", "lufs_delta", "LUFS-I", "crest",
                     "crest10s", "tilt(R2)", "side/mid", "corr", "sub%", "air%"],
                    rows)
            + "\n\n" + flags + "\n```\n")


def _corpus_row(res: dict[str, Any]) -> str:
    t = res.get("tags", {}).get("named", {})
    h = res["headline"]
    ST = res.get("stereo", {})
    S = res.get("spectrum", {})
    tilt = S.get("tilt", {}) if S.get("available") else {}
    pw = tilt.get("piecewise", [])
    piece = "; ".join(
        f"{hz(p['low_hz'])}-{hz(p['high_hz'])} {n(p['slope_db_per_oct'], 2)} dB/oct"
        for p in pw)
    tonal = (f"tilt {n(tilt.get('slope_db_per_oct'), 2)} dB/oct "
             f"(R2 {n(tilt.get('r2'), 2)}); {piece}; "
             f"air {n(h['air_band_pct'], 2)}%, sub {n(h['sub_band_pct'], 2)}%")
    width = (f"side/mid {db(h['side_minus_mid_db'])} dB overall, "
             f"{db(h['side_minus_mid_below_120hz_db'])} dB below 120 Hz; "
             f"mono crossover {hz(h['mono_crossover_hz'])} Hz; "
             f"correlation mean {n(h['correlation_mean'], 2)}, min {n(h['correlation_min'], 2)}"
             if ST.get("available") else "mono file")
    pairs = [
        ("Title", t.get("title") or ""),
        ("Artist", t.get("artist") or ""),
        ("Year", (t.get("date") or "")[:10]),
        ("Genre", t.get("genre") or ""),
        ("Engineers", ""),
        ("LUFS-I", n(h["lufs_i"], 2)),
        ("True peak", n(h["true_peak_dbtp_16x"], 2) + " dBTP"),
        ("LRA", n(h["lra_lu"], 1) + " LU"),
        ("PLR", n(h["plr_db"], 1) + " dB"),
        ("PSR min", n(h["psr_min_db"], 1) + " dB"),
        ("PSR median", n(h["psr_median_db"], 1) + " dB"),
        ("DR14", n(h["dr14"], 0)),
        ("Crest (loudest 10s)", n(h["crest_loudest_10s_db"], 1) + " dB"),
        ("Tonal tilt notes", tonal),
        ("Width/mono notes", width),
        ("mtx run", run_provenance(res)),
    ]
    body = "\n".join(f"{k}: {v}" for k, v in pairs)
    return ("## CORPUS ROW\n\nEmpty fields are left empty on purpose: nothing here is "
            "guessed.\n\n```\n" + body + "\n```\n")


def run_provenance(res: dict[str, Any]) -> str:
    """version / schema / profile / sha256 prefix, as one storable string.

    A method change in a later version makes rows measured under an earlier one
    subtly incomparable, and nothing else in a stored corpus would catch it.
    The header already carries all four values; this is them reaching the row
    that actually gets archived.
    """
    run = res.get("run", {})
    sha = (res.get("file", {}).get("sha256") or "")[:16]
    stems = res.get("stems", {}) or {}
    model = f" / stems {stems.get('model')}" if stems.get("available") else ""
    return (f"mtx {run.get('tool_version')} / schema {run.get('schema_version')} "
            f"/ profile {run.get('profile')} / sha256 {sha or 'n/a'}{model}")


def corpus_row_dict(res: dict[str, Any]) -> dict[str, Any]:
    """The corpus row as JSON, keyed by the corpus property names.

    Numbers stay numbers, so the row imports as typed properties instead of
    text, and `_units` says what each one is in.  Anything mtx cannot measure
    is null rather than guessed, exactly as in the pasted block.
    """
    t = res.get("tags", {}).get("named", {})
    h = res["headline"]
    ST = res.get("stereo", {})
    S = res.get("spectrum", {})
    tilt = S.get("tilt", {}) if S.get("available") else {}
    pw = tilt.get("piecewise", [])
    piece = "; ".join(
        f"{hz(p['low_hz'])}-{hz(p['high_hz'])} {n(p['slope_db_per_oct'], 2)} dB/oct"
        for p in pw)
    tonal = (f"tilt {n(tilt.get('slope_db_per_oct'), 2)} dB/oct "
             f"(R2 {n(tilt.get('r2'), 2)}); {piece}; "
             f"air {n(h['air_band_pct'], 2)}%, sub {n(h['sub_band_pct'], 2)}%")
    width = (f"side/mid {db(h['side_minus_mid_db'])} dB overall, "
             f"{db(h['side_minus_mid_below_120hz_db'])} dB below 120 Hz; "
             f"mono crossover {hz(h['mono_crossover_hz'])} Hz; "
             f"correlation mean {n(h['correlation_mean'], 2)}, "
             f"min {n(h['correlation_min'], 2)}"
             if ST.get("available") else "mono file")

    def num(v: Any, nd: int) -> float | None:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return round(f, nd) if math.isfinite(f) else None

    return {
        "Title": t.get("title") or None,
        "Artist": t.get("artist") or None,
        "Year": (t.get("date") or "")[:10] or None,
        "Genre": t.get("genre") or None,
        "Engineers": None,
        "LUFS-I": num(h["lufs_i"], 2),
        "True peak": num(h["true_peak_dbtp_16x"], 2),
        "LRA": num(h["lra_lu"], 1),
        "PLR": num(h["plr_db"], 1),
        "PSR min": num(h["psr_min_db"], 1),
        "PSR median": num(h["psr_median_db"], 1),
        "DR14": num(h["dr14"], 1),
        "Crest (loudest 10s)": num(h["crest_loudest_10s_db"], 1),
        "Tonal tilt notes": tonal,
        "Width/mono notes": width,
        "mtx run": run_provenance(res),
        "_units": {
            "LUFS-I": "LUFS", "True peak": "dBTP", "LRA": "LU", "PLR": "dB",
            "PSR min": "dB", "PSR median": "dB", "DR14": "DR",
            "Crest (loudest 10s)": "dB",
        },
        "_note": ("Fields mtx cannot measure are null, never guessed. Engineers, "
                  "and every session field (calibration, lessons, applied-to, "
                  "verdict), are filled in by the session and not by the tool."),
        "_source": {"file": res.get("file", {}).get("filename"),
                    "sha256": res.get("file", {}).get("sha256"),
                    "psr_min_time": h.get("psr_min_time")},
    }


METHOD_LINES = [
    "LUFS-I/LRA: ITU-R BS.1770-4 K-weighting, 400 ms blocks 75% overlap, gates -70 LUFS / -10 LU; LRA on 3 s blocks, gates -70 / -20 LU, P95-P10. Cross-checked against ffmpeg ebur128 (tolerance 0.2 LU).",
    "True peak: scipy.signal.resample_poly (Kaiser beta 5.0) at 4x and 16x, both reported; overs counted as contiguous excursions at 16x.",
    "PSR: 3 s windows, 1 s hop; short-term true peak (4x) minus short-term LUFS over the same window.",
    "DR14: TT offline DR. 3 s blocks, block RMS sqrt(2*mean(x^2)), loudest 20%, second-highest per-block sample peak. {dr14_validation}",
    "Flat-top: threshold derived per channel as max(|x|)*0.99999, never a fixed -0.1 dBFS. Run lengths, ms, and the ten longest runs reported.",
    "Ceiling density: fraction of samples within N dB of that channel's own ceiling; threshold-free.",
    "Limiter vs clipper: mean dB slope of |x| over the 2 ms before entry and after exit of each flat-top run. Inferred, not measured.",
    "LTAS: Welch, Hann, 50% overlap, nperseg 16384 broadband; nperseg 131072 over an auto-selected ~90 s body section for the low end.",
    "Tilt: least-squares dB/octave over 100 Hz-10 kHz on mid, with R2; piecewise slopes over 30-120, 120-1k, 1k-6k, 6k-20k.",
    "Mid/side: mid=(L+R)/2, side=(L-R)/2. Side/mid is 10*log10(P_side/P_mid). Mono crossover is the highest third-octave centre below which side/mid stays under -20 dB.",
    "Mono-sum damage: 10*log10(P_mid/(P_mid+P_side)) per third-octave.",
    "HF cutoff: highest frequency of the 1/6-octave-smoothed LTAS still within 25 dB of the 1-5 kHz median; per-5 s frames for stability.",
    "Effective bit depth: 32 minus the trailing zero bits of the left-justified int32 sample, maximum over non-zero samples.",
    "Sections: MFCC+chroma+RMS+spectral contrast, cosine SSM, Foote novelty with an 8 s kernel, peak-picked, segments under 4 s merged.",
    "Tempo/key: librosa.beat.beat_track; mean chroma-CQT against Krumhansl-Schmuckler profiles. Both carry a confidence.",
    "Saturation proxy: least-squares regression of 5-10 kHz frame level on broadband frame level, 50 ms frames, in dB/dB.",
    "Pumping: cross-correlation of the sub-120 Hz and 500 Hz-6 kHz dB envelopes (5 ms hop) over -200..+200 ms.",
    "Modulation: FFT of each band's 5 ms RMS envelope; depth at the beat rate relative to the envelope RMS.",
    "Reverb: Schroeder reverse integration after strong onsets, per octave band. Estimate, usually low or medium confidence.",
]


def _method(res: dict[str, Any]) -> str:
    """METHOD, with the DR14 line reporting the machine's validation record.

    A static "NOT validated" line would keep saying so after the check had been
    done, which is the same failure the record exists to prevent.
    """
    val = res.get("loudness", {}).get("dr14", {}).get("validation", {}) or {}
    rec = val.get("record", {}) or {}
    if val.get("validated_against_published_reference"):
        note = (f"Validated against {rec.get('tracks_checked')} published DR "
                f"rating(s), worst disagreement {n(rec.get('max_abs_delta_dr'), 1)} "
                "DR (loudness.dr14.validation.record).")
    elif rec.get("tracks_checked"):
        note = (f"{rec['tracks_checked'] - rec.get('tracks_within_tolerance', 0)} of "
                f"{rec['tracks_checked']} recorded published rating(s) disagree by "
                f"more than {n(rec.get('tolerance_dr'), 1)} DR "
                "(loudness.dr14.validation.record).")
    else:
        note = ("NOT validated against a published DR rating "
                "(see loudness.dr14.validation).")
    lines = [ln.replace("{dr14_validation}", note) for ln in METHOD_LINES]
    return "## METHOD\n\n" + "\n".join(f"- {ln}" for ln in lines) + "\n"


# ------------------------------------------------------------------- assembly
# (priority, name, group, builder).  Higher priority is dropped first when the
# assembled digest is over budget.  The group is what `--sections` selects on.
DETAIL_BLOCKS = [
    (1, "band energy", "spectrum", _block_bands),
    (1, "per-band crest", "dynamics", _block_band_crest),
    (2, "flat-top forensics", "dynamics", _block_flat_top),
    (2, "source forensics", "forensics", _block_forensics),
    (3, "sections", "structure", _block_sections),
    (3, "short-term loudness", "loudness", _block_loudness_timeline),
    (4, "PSR timeline", "loudness", _block_psr),
    (4, "stereo detail", "stereo", _block_stereo_extra),
    (5, "side/mid per third-octave", "stereo", _block_sidemid),
    # Three lines and ~230 bytes for the clearest broadband-vs-multiband tell
    # in the tool: at priority 5 it died alongside the 700-byte processing
    # block on any track long enough to go over budget.
    (3, "band-envelope correlation", "processing", _block_band_envelope),
    (5, "processing forensics", "processing", _block_processing),
    (6, "per-band level over time", "spectrum", _block_band_timeline),
    (6, "streaming preview", "loudness", _block_streaming),
    (6, "bass fundamentals", "spectrum", _block_bass),
    (7, "ceiling density", "dynamics", _block_ceiling),
    (7, "spectral descriptors", "spectrum", _block_descriptors),
    (8, "resonances", "spectrum", _block_resonances),
    (8, "arrangement gaps", "structure", _block_gaps),
]

SECTION_GROUPS = tuple(sorted({g for _, _, g, _ in DETAIL_BLOCKS}))
SECTION_NAMES = tuple(sorted({nm for _, nm, _, _ in DETAIL_BLOCKS}))


def _norm(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").replace("-", " ").split())


def resolve_sections(names: Sequence[str]) -> set[str]:
    """Turn `--sections` arguments into the set of block names to keep.

    An unrecognised name is an error rather than a silent no-op: a digest that
    quietly dropped the one block the session asked for would be worse than no
    selection at all.
    """
    keep: set[str] = set()
    unknown: list[str] = []
    for raw in names:
        want = _norm(raw)
        hits = {nm for _, nm, g, _ in DETAIL_BLOCKS
                if _norm(g) == want or _norm(nm) == want}
        if not hits:
            unknown.append(raw)
        keep |= hits
    if unknown:
        raise ValueError(
            "unknown --sections name(s): " + ", ".join(unknown)
            + "\ngroups: " + ", ".join(SECTION_GROUPS)
            + "\nblocks: " + ", ".join(SECTION_NAMES))
    return keep


def render_digest(res: dict[str, Any], budget: int | None = None,
                  sections: Sequence[str] | None = None) -> str:
    """Assemble digest.md.

    `budget` overrides the default cap; `sections` restricts DETAIL to the
    named groups or blocks.  A `--stems` run raises the cap by
    `STEMS_BUDGET_BONUS`, because the stem table exists nowhere else in the
    paste-able output and dropping it to protect a byte count would defeat the
    run that produced it.
    """
    tags = res.get("tags", {}).get("named", {})
    stems_block = _stems(res)
    stem_model = (res.get("stems", {}) or {}).get("model")
    header = (f"# mtx digest\n\n"
              f"file: {res.get('file', {}).get('filename')}\n"
              f"title: {tags.get('title') or '(no title tag)'}\n"
              f"artist: {tags.get('artist') or '(no artist tag)'}\n"
              f"sha256: {(res.get('file', {}).get('sha256') or '')[:16]}...\n"
              f"tool: mtx {res['run']['tool_version']} / schema "
              f"{res['run']['schema_version']} / profile {res['run']['profile']}\n"
              + (f"stems: {stem_model} (separated; see STEMS)\n"
                 if stems_block and stem_model else "")
              + f"audio: {res['audio']['sample_rate_hz']} Hz, "
              f"{res['audio']['channels']} ch, "
              f"{res['audio']['subtype']}, {res['audio']['duration_s']} s\n\n")

    if budget is None:
        # The bonus pays for the stem table, not for the one-line note that
        # says separation was unavailable.
        has_table = bool(stems_block) and (res.get("stems", {}) or {}).get("available")
        budget = SIZE_BUDGET_BYTES + (STEMS_BUDGET_BONUS if has_table else 0)

    fixed = header + _headline(res) + "\n" + _flags(res) + "\n"
    after = (("\n" + stems_block) if stems_block else "")
    after += "\n" + _corpus_row(res) + "\n" + _method(res)

    keep = resolve_sections(sections) if sections else None
    built = [(prio, name, fn(res)) for prio, name, group, fn in DETAIL_BLOCKS
             if keep is None or name in keep]
    built = [(p, name, txt) for p, name, txt in built if txt]
    dropped: list[str] = []

    def assemble(blocks: list[tuple[int, str, str]], note: str) -> str:
        detail = "## DETAIL\n\n" + "\n".join(txt for _, _, txt in blocks)
        if note:
            detail += "\n" + note + "\n"
        return fixed + detail + after

    selected = ("_DETAIL restricted by --sections to: "
                + ", ".join(nm for _, nm, _ in built) + "._\n") if sections else ""
    out = assemble(built, selected)
    while len(out.encode("utf-8")) > budget and built:
        worst = max(range(len(built)), key=lambda i: (built[i][0], i))
        dropped.append(built[worst][1])
        built.pop(worst)
        note = selected + ("_Dropped to stay under the "
                           f"{budget / 1024:.0f} KB digest budget (full detail is in "
                           "analysis.json; --sections or --digest-budget keeps a "
                           "block that a session needs): " + ", ".join(dropped) + "._")
        out = assemble(built, note)
    if len(out.encode("utf-8")) > budget:
        # HEADLINE, FLAGS, CORPUS ROW and METHOD are never dropped: a digest
        # missing its own provenance would be smaller and useless.  Say that
        # the budget was not met rather than meeting it by lying.
        out += (f"\n_Budget note: the header, HEADLINE, FLAGS, CORPUS ROW and "
                f"METHOD sections are {len(out.encode('utf-8'))} bytes on their "
                f"own and are never dropped, so the {budget}-byte budget could "
                f"not be met. Every DETAIL block was removed._\n")
    return out
