"""digest.md: the compact, paste-able view of analysis.json.

Hard budget of 12 KB.  Blocks are assembled in a fixed order and dropped from
the lowest priority upward until the budget is met; whatever was dropped is
named in the output, so the digest never silently omits a section.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

SIZE_BUDGET_BYTES = 12 * 1024


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
    if dr and not dr.get("validated_against_published_reference"):
        lines.append("- [unverified] DR14: not validated against a published DR "
                     "rating; synthetic checks only (see METHOD)")
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
        ("Tonal tilt notes", tonal),
        ("Width/mono notes", width),
    ]
    body = "\n".join(f"{k}: {v}" for k, v in pairs)
    return ("## CORPUS ROW\n\nEmpty fields are left empty on purpose: nothing here is "
            "guessed.\n\n```\n" + body + "\n```\n")


METHOD_LINES = [
    "LUFS-I/LRA: ITU-R BS.1770-4 K-weighting, 400 ms blocks 75% overlap, gates -70 LUFS / -10 LU; LRA on 3 s blocks, gates -70 / -20 LU, P95-P10. Cross-checked against ffmpeg ebur128 (tolerance 0.2 LU).",
    "True peak: scipy.signal.resample_poly (Kaiser beta 5.0) at 4x and 16x, both reported; overs counted as contiguous excursions at 16x.",
    "PSR: 3 s windows, 1 s hop; short-term true peak (4x) minus short-term LUFS over the same window.",
    "DR14: TT offline DR. 3 s blocks, block RMS sqrt(2*mean(x^2)), loudest 20%, second-highest per-block sample peak. NOT validated against a published DR rating (see loudness.dr14.validation).",
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
    return "## METHOD\n\n" + "\n".join(f"- {ln}" for ln in METHOD_LINES) + "\n"


# ------------------------------------------------------------------- assembly
# (priority, title, builder).  Higher priority is dropped first when over budget.
DETAIL_BLOCKS = [
    (1, "band energy", _block_bands),
    (1, "per-band crest", _block_band_crest),
    (2, "flat-top forensics", _block_flat_top),
    (2, "source forensics", _block_forensics),
    (3, "sections", _block_sections),
    (3, "short-term loudness", _block_loudness_timeline),
    (4, "PSR timeline", _block_psr),
    (4, "stereo detail", _block_stereo_extra),
    (5, "side/mid per third-octave", _block_sidemid),
    (5, "processing forensics", _block_processing),
    (6, "per-band level over time", _block_band_timeline),
    (6, "streaming preview", _block_streaming),
    (6, "bass fundamentals", _block_bass),
    (7, "ceiling density", _block_ceiling),
    (7, "spectral descriptors", _block_descriptors),
    (8, "resonances", _block_resonances),
    (8, "arrangement gaps", _block_gaps),
]


def render_digest(res: dict[str, Any], budget: int = SIZE_BUDGET_BYTES) -> str:
    tags = res.get("tags", {}).get("named", {})
    header = (f"# mtx digest\n\n"
              f"file: {res.get('file', {}).get('filename')}\n"
              f"title: {tags.get('title') or '(no title tag)'}\n"
              f"artist: {tags.get('artist') or '(no artist tag)'}\n"
              f"sha256: {(res.get('file', {}).get('sha256') or '')[:16]}...\n"
              f"tool: mtx {res['run']['tool_version']} / schema "
              f"{res['run']['schema_version']} / profile {res['run']['profile']}\n"
              f"audio: {res['audio']['sample_rate_hz']} Hz, {res['audio']['channels']} ch, "
              f"{res['audio']['subtype']}, {res['audio']['duration_s']} s\n\n")

    fixed = header + _headline(res) + "\n" + _flags(res) + "\n"
    corpus = "\n" + _corpus_row(res) + "\n" + _method(res)

    built = [(prio, name, fn(res)) for prio, name, fn in DETAIL_BLOCKS]
    built = [(p, name, txt) for p, name, txt in built if txt]
    dropped: list[str] = []

    def assemble(blocks: list[tuple[int, str, str]], note: str) -> str:
        detail = "## DETAIL\n\n" + "\n".join(txt for _, _, txt in blocks)
        if note:
            detail += "\n" + note + "\n"
        return fixed + detail + corpus

    out = assemble(built, "")
    while len(out.encode("utf-8")) > budget and built:
        worst = max(range(len(built)), key=lambda i: (built[i][0], i))
        dropped.append(built[worst][1])
        built.pop(worst)
        note = ("_Dropped to stay under the 12 KB digest budget (full detail is in "
                "analysis.json): " + ", ".join(dropped) + "._") if dropped else ""
        out = assemble(built, note)
    return out
