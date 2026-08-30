"""4.7b Rhythm: downbeats, meter, groove, and per-stem microtiming.

`structure.tempo` reports a tempo.  A tempo is not a groove: without a downbeat
there is no bar, and without a bar half the musical questions cannot even be
phrased.  Everything here is built on the beat grid that module already
measured, so it costs one accent pass over signals that are already in memory.

The per-stem microtiming block is the strongest thing in the module and the
cheapest: "the drums are dragging" is the median onset deviation from the beat
grid, in milliseconds, per stem.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..audio import AudioSource
from ..bands import get_band_pack
from ..params import PARAMS
from ..util import Collector, db_amp, fmt_time

# Longuet-Higgins & Lee metrical weights for a 16-step bar: 0 is the strongest
# position, -4 the weakest.  Syncopation is a note on a weak position followed
# by a rest on a stronger one.
LHL_WEIGHTS_16 = (0, -4, -3, -4, -2, -4, -3, -4, -1, -4, -3, -4, -2, -4, -3, -4)


def fine_onsets(src: AudioSource) -> np.ndarray:
    """Onset times at hop 128 (5.8 ms), cached on the source.

    The shared onset envelope runs at hop 512, and 23 ms of quantisation is
    larger than most of the microtiming this module exists to measure -- it
    would report a sequenced grid and a played one as the same number.
    """
    hop = int(PARAMS["rhythm"]["onset_hop_length"])

    def build() -> np.ndarray:
        import librosa
        env = librosa.onset.onset_strength(y=src.lib_mono, sr=src.lib_sr,
                                           hop_length=hop)
        frames = librosa.onset.onset_detect(onset_envelope=env, sr=src.lib_sr,
                                            hop_length=hop, backtrack=False)
        return np.asarray(librosa.frames_to_time(frames, sr=src.lib_sr,
                                                 hop_length=hop), dtype=float)

    try:
        return src.cache_get(f"fine_onsets_{hop}", build)
    except Exception:
        return np.zeros(0)


def onset_resolution_ms() -> float:
    return 1000.0 * PARAMS["rhythm"]["onset_hop_length"] / 22050.0


def _beat_accents(src: AudioSource, beats: np.ndarray) -> dict[str, np.ndarray]:
    """Three independent accent signals sampled at each beat.

    Kept separate as well as summed, because which one carries the downbeat is
    itself informative: a record whose bars are marked only by harmony reads
    differently from one whose bars are marked only by a kick.
    """
    out: dict[str, np.ndarray] = {}
    hop = PARAMS["general"]["librosa_hop_length"]
    env = src.onset_envelope()
    env_t = np.arange(env.size) * hop / float(src.lib_sr)
    out["onset"] = np.interp(beats, env_t, env, left=0.0, right=0.0)

    pack = get_band_pack(src)
    low = None
    for name in ("sub", "bass"):
        if name in pack.envelopes and pack.envelopes[name].size:
            e = pack.envelopes[name] ** 2
            low = e if low is None else low + e
    if low is not None and low.size:
        lt = np.arange(low.size) * pack.hop_s
        # 60 ms after the beat instant: a kick is an attack, not a level.
        win = max(1, int(round(0.06 / pack.hop_s)))
        idx = np.clip(np.searchsorted(lt, beats), 0, low.size - 1)
        out["low_band"] = np.asarray(
            [float(np.max(low[i:i + win])) if i < low.size else 0.0 for i in idx])
    else:
        out["low_band"] = np.zeros(beats.size)

    try:
        chroma = src.chroma_cqt()
        ct = np.arange(chroma.shape[1]) * hop / float(src.lib_sr)
        cols = np.clip(np.searchsorted(ct, beats), 0, chroma.shape[1] - 1)
        bc = chroma[:, cols]
        bc = bc / np.maximum(np.linalg.norm(bc, axis=0, keepdims=True), 1e-12)
        change = np.zeros(beats.size)
        if bc.shape[1] > 1:
            change[1:] = 1.0 - np.sum(bc[:, 1:] * bc[:, :-1], axis=0)
        out["harmonic_change"] = change
    except Exception:
        out["harmonic_change"] = np.zeros(beats.size)
    return out


def _z(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    sd = float(np.std(v))
    return (v - float(np.mean(v))) / sd if sd > 0 else np.zeros_like(v)


def _downbeats(src: AudioSource, beats: np.ndarray,
               collector: Collector) -> dict[str, Any]:
    P = PARAMS["rhythm"]
    if beats.size < 8:
        return {"available": False, "reason": "fewer than eight beats"}
    acc = _beat_accents(src, beats)
    total = _z(acc["onset"]) + _z(acc["low_band"]) + _z(acc["harmonic_change"])
    per_meter: list[dict[str, Any]] = []
    best: tuple[float, int, int] | None = None
    for m in P["meters"]:
        if beats.size < m * 3:
            continue
        m_best: tuple[float, int] | None = None
        for phase in range(m):
            on = total[phase::m]
            mask = np.ones(total.size, dtype=bool)
            mask[phase::m] = False
            off = total[mask]
            if on.size < 2 or off.size < 2:
                continue
            contrast = float(np.mean(on) - np.mean(off))
            if m_best is None or contrast > m_best[0]:
                m_best = (contrast, phase)
        if m_best is None:
            continue
        per_meter.append({"meter": m, "best_phase": m_best[1],
                          "downbeat_accent_contrast": m_best[0]})
        if best is None or m_best[0] > best[0]:
            best = (m_best[0], m, m_best[1])
    if best is None:
        return {"available": False, "reason": "no meter could be scored"}
    contrast, meter, phase = best
    others = [r["downbeat_accent_contrast"] for r in per_meter if r["meter"] != meter]
    margin = contrast - max(others) if others else None
    conf = "high" if (margin or 0) > 0.35 else ("medium" if (margin or 0) > 0.12 else "low")
    if conf != "high":
        collector.low_confidence(
            "rhythm.downbeats", conf,
            f"{meter} beats per bar wins by {margin:.3f} of accent contrast over "
            "the next meter" if margin is not None
            else "only one meter was scorable")
    downbeats = beats[phase::meter]
    return {
        "available": True,
        "beats_per_bar": meter,
        "time_signature": f"{meter}/4",
        "time_signature_note": "only the number of beats per bar is measured; "
                               "the denominator is a notation convention and is "
                               "assumed, not observed",
        "phase_beat_index": phase,
        "downbeat_times_s": downbeats,
        "bar_count": int(downbeats.size),
        "downbeat_accent_contrast": contrast,
        "meter_margin": margin,
        "per_meter": per_meter,
        "accent_components": {
            "onset_strength": acc["onset"],
            "low_band_energy": acc["low_band"],
            "harmonic_change": acc["harmonic_change"],
        },
        "confidence": conf,
        "method": P["downbeat_accent"],
    }


def _tempo_octave(src: AudioSource, beats: np.ndarray, bpm: float | None,
                  collector: Collector) -> dict[str, Any]:
    """Is the reported tempo the metrical level the record is actually on?

    A beat tracker picks one metrical level, and picking the wrong one is the
    single most consequential error in this module: every bar, every
    chords-per-bar and every swing figure is computed against it.  Measured
    against seven published tempos, librosa reported half once and double once.

    The test is made on the beat grid rather than by autocorrelation, because a
    periodic pulse train correlates with itself just as well at twice its
    period as at its own -- an autocorrelation comparison reports half tempo
    even on a plain click track.  Two ratios avoid that:

    * **midpoint** -- onset strength halfway between the beats, over the onset
      strength on them.  Near 1 means the midpoints are carrying beats too, and
      the real pulse is twice the reported one.
    * **alternation** -- the weaker of the two alternating beat phases over the
      stronger.  Well under 1 means every other beat is empty, and the real
      pulse is half the reported one.

    Reported, never applied: silently doubling the tempo here would leave two
    different tempos in one document.

    **How well this works, measured.**  On the seven reference tracks the two
    ratios raised no false alarm on the four correct tempos -- and identified
    none of the three wrong ones either.  The classes overlap: a correct tempo
    produced midpoint ratios from 0.42 to 0.74, and the one halved tempo
    produced 0.54.  Thresholds that caught it would fire on correct tempos too,
    and on a seven-track sample that is fitting noise, so the thresholds are
    left where they raise nothing.

    Read the ratios, not the verdict.  They are a real measurement of the beat
    grid, and the reason the block exists at all is that a reader of
    `harmony.harmonic_rhythm.changes_per_bar` needs to know that the bar it is
    divided by came from a tempo that may be at the wrong metrical level.
    """
    P = PARAMS["rhythm"]["octave_check"]
    if not bpm or bpm <= 0 or beats.size < 8:
        return {"available": False, "reason": "no beat grid to test"}
    hop = PARAMS["general"]["librosa_hop_length"]
    env = np.asarray(src.onset_envelope(), dtype=float)
    if env.size < 16:
        return {"available": False, "reason": "onset envelope too short"}
    env_t = np.arange(env.size) * hop / float(src.lib_sr)
    on_beats = np.interp(beats, env_t, env, left=0.0, right=0.0)
    mids = (beats[:-1] + beats[1:]) / 2.0
    on_mids = np.interp(mids, env_t, env, left=0.0, right=0.0)
    beat_mean = float(np.mean(on_beats))
    if beat_mean <= 0:
        return {"available": False, "reason": "no onset strength on the beat grid"}
    midpoint_ratio = float(np.mean(on_mids)) / beat_mean
    even, odd = on_beats[0::2], on_beats[1::2]
    a, b = float(np.mean(even)), float(np.mean(odd))
    alternation_ratio = (min(a, b) / max(a, b)) if max(a, b) > 0 else None

    suggested = 1.0
    reason = "the midpoints are quiet and neither beat phase is empty"
    if midpoint_ratio >= P["midpoint_ratio_double"]:
        suggested = 2.0
        reason = (f"the midpoints between beats carry {100 * midpoint_ratio:.0f}% "
                  "of the on-beat onset strength, so they are beats too")
    elif alternation_ratio is not None and alternation_ratio <= P["alternation_ratio_half"]:
        suggested = 0.5
        reason = (f"one of the two beat phases carries only "
                  f"{100 * alternation_ratio:.0f}% of the other, so every other "
                  "beat is empty")
    if suggested != 1.0:
        collector.warn(
            "rhythm.tempo_octave",
            f"the beat grid suggests {bpm * suggested:.1f} BPM rather than the "
            f"reported {bpm:.1f} ({suggested}x): {reason}. Every bar-relative "
            "number in this run is computed against the reported level.")
    return {
        "available": True,
        "reported_bpm": float(bpm),
        "midpoint_ratio": midpoint_ratio,
        "alternation_ratio": alternation_ratio,
        "suggested_factor": suggested,
        "suggested_bpm": float(bpm) * suggested,
        "reported_level_is_best_supported": bool(suggested == 1.0),
        "basis": reason,
        "thresholds": {"midpoint_ratio_double": P["midpoint_ratio_double"],
                       "alternation_ratio_half": P["alternation_ratio_half"]},
        "method": P["method"],
        "note": P["note"],
        "measured_on_reference_set": (
            "no false alarms on four correct tempos, and no detection of three "
            "known octave errors; the two ratios are the measurement, the "
            "suggested factor is only raised when the evidence is unambiguous"),
        "confidence": "low",
        "inherits": "bar_count, syncopation per bar, swing and "
                    "harmony.harmonic_rhythm.changes_per_bar are all computed "
                    "against the reported level",
    }


def _grid(beats: np.ndarray, subdivision: int) -> np.ndarray:
    """Every subdivision instant implied by the measured beat times."""
    if beats.size < 2:
        return beats
    out = []
    for a, b in zip(beats, beats[1:]):
        step = (b - a) / subdivision
        out.extend(a + step * k for k in range(subdivision))
    out.append(float(beats[-1]))
    return np.asarray(out, dtype=float)


def _deviations(onsets: np.ndarray, grid: np.ndarray,
                max_ms: float) -> np.ndarray:
    """Signed onset-to-nearest-grid-instant deviation, in milliseconds."""
    if onsets.size == 0 or grid.size == 0:
        return np.zeros(0)
    idx = np.clip(np.searchsorted(grid, onsets), 1, grid.size - 1)
    lo, hi = grid[idx - 1], grid[idx]
    nearest = np.where(np.abs(onsets - lo) <= np.abs(onsets - hi), lo, hi)
    dev = (onsets - nearest) * 1000.0
    return dev[np.abs(dev) <= max_ms]


def _timing_stats(dev: np.ndarray) -> dict[str, Any]:
    if dev.size < 4:
        return {"onsets_measured": int(dev.size), "median_ms": None,
                "note": "fewer than four onsets landed within the search window"}
    return {
        "onsets_measured": int(dev.size),
        "median_ms": float(np.median(dev)),
        "mean_ms": float(np.mean(dev)),
        "iqr_ms": float(np.percentile(dev, 75) - np.percentile(dev, 25)),
        "std_ms": float(np.std(dev)),
        "share_within_10ms": float(np.mean(np.abs(dev) <= 10.0)),
        "p10_ms": float(np.percentile(dev, 10)),
        "p90_ms": float(np.percentile(dev, 90)),
    }


def _swing(beats: np.ndarray, onsets: np.ndarray) -> dict[str, Any]:
    """Where the off-beat actually falls, as a fraction of the beat."""
    P = PARAMS["rhythm"]["swing"]
    lo, hi = P["search_fraction"]
    fracs: list[float] = []
    for a, b in zip(beats, beats[1:]):
        span = b - a
        if span <= 0:
            continue
        inside = onsets[(onsets > a) & (onsets < b)]
        if inside.size == 0:
            continue
        f = (inside - a) / span
        f = f[(f >= lo) & (f <= hi)]
        if f.size:
            fracs.append(float(f[int(np.argmin(np.abs(f - 0.5)))]))
    if len(fracs) < P["min_events"]:
        return {"available": False,
                "reason": f"only {len(fracs)} off-beat events in the "
                          f"{lo}-{hi} search window"}
    med = float(np.median(fracs))
    ratio = med / (1.0 - med) if med < 1.0 else None
    return {
        "available": True,
        "events": len(fracs),
        "offbeat_position_median": med,
        "offbeat_position_iqr": float(np.percentile(fracs, 75)
                                      - np.percentile(fracs, 25)),
        "swing_ratio": ratio,
        "reference": "0.500 (ratio 1.00) is straight eighths; 0.667 "
                     "(ratio 2.00) is a triplet shuffle",
        "method": "median position of the off-beat onset nearest the midpoint "
                  "of each beat interval, as a fraction of that interval",
    }


def _syncopation(downbeats: np.ndarray, beats_per_bar: int,
                 onsets: np.ndarray) -> dict[str, Any]:
    """Longuet-Higgins & Lee syncopation over a 16-step bar."""
    if downbeats.size < 3 or beats_per_bar not in (4, 2):
        return {"available": False,
                "reason": "the 16-step weight table is defined for a bar of "
                          "four beats"}
    per_bar: list[float] = []
    slots = 16
    for a, b in zip(downbeats, downbeats[1:]):
        span = b - a
        if span <= 0:
            continue
        pos = np.zeros(slots, dtype=bool)
        inside = onsets[(onsets >= a) & (onsets < b)]
        for o in inside:
            k = int(round((o - a) / span * slots)) % slots
            pos[k] = True
        score = 0.0
        for i in range(slots):
            if not pos[i]:
                continue
            j = (i + 1) % slots
            while j != i and not pos[j]:
                if LHL_WEIGHTS_16[j] > LHL_WEIGHTS_16[i]:
                    score += LHL_WEIGHTS_16[j] - LHL_WEIGHTS_16[i]
                    break
                j = (j + 1) % slots
        per_bar.append(score)
    if not per_bar:
        return {"available": False, "reason": "no complete bars"}
    return {"available": True, "bars": len(per_bar),
            "mean_per_bar": float(np.mean(per_bar)),
            "median_per_bar": float(np.median(per_bar)),
            "max_per_bar": float(np.max(per_bar)),
            "per_bar": [float(v) for v in per_bar[:400]],
            "method": PARAMS["rhythm"]["syncopation"]}


def _beat_position_profile(src: AudioSource, beats: np.ndarray,
                           beats_per_bar: int) -> dict[str, Any]:
    """Kick-band and snare-band energy at each position in the bar."""
    pack = get_band_pack(src)

    def band_at(names: tuple[str, ...], instants: np.ndarray) -> np.ndarray:
        acc = None
        for nm in names:
            e = pack.envelopes.get(nm)
            if e is None or e.size == 0:
                continue
            acc = e ** 2 if acc is None else acc + e ** 2
        if acc is None:
            return np.zeros(instants.size)
        t = np.arange(acc.size) * pack.hop_s
        win = max(1, int(round(0.06 / pack.hop_s)))
        idx = np.clip(np.searchsorted(t, instants), 0, acc.size - 1)
        return np.asarray([float(np.max(acc[i:i + win])) for i in idx])

    def band_at_beats(names: tuple[str, ...]) -> np.ndarray:
        return band_at(names, beats)

    def band_at_beats_at(instants: np.ndarray) -> np.ndarray:
        return band_at(("sub", "bass"), instants)

    kick = band_at_beats(("sub", "bass"))
    snare = band_at_beats(("low_bass", "low_mid", "high_mid"))
    # Halfway between the beats: if the kick band is no louder on the beat than
    # between them, there is no kick pattern to describe, and a flat profile
    # across four beat positions means absence and not four on the floor.
    mids = (beats[:-1] + beats[1:]) / 2.0
    kick_off = band_at_beats_at(mids) if mids.size else np.zeros(0)
    out: dict[str, Any] = {"beats_per_bar": beats_per_bar,
                           "kick_band_hz": [20.0, 120.0],
                           "snare_band_hz": [120.0, 6000.0]}
    prof: dict[str, list[float]] = {}
    for name, vals in (("kick", kick), ("snare", snare)):
        rows = []
        for p in range(beats_per_bar):
            sel = vals[p::beats_per_bar]
            rows.append(db_amp(float(np.sqrt(np.mean(sel)))) if sel.size else None)
        prof[name] = rows
    out["per_beat_position_db"] = prof
    kv = [v for v in prof["kick"] if v is not None]
    sv = [v for v in prof["snare"] if v is not None]
    on_beat = db_amp(float(np.sqrt(np.mean(kick)))) if kick.size else None
    off_beat = db_amp(float(np.sqrt(np.mean(kick_off)))) if kick_off.size else None
    contrast = (on_beat - off_beat) if (on_beat is not None and off_beat is not None) else None
    out["kick_on_minus_off_beat_db"] = contrast
    has_kick = contrast is not None and contrast >= 3.0

    four_on_floor = None
    backbeat = None
    if len(kv) == beats_per_bar and beats_per_bar == 4:
        spread = max(kv) - min(kv)
        out["kick_spread_db"] = spread
        four_on_floor = bool(spread < 3.0) if has_kick else None
    if len(sv) == 4:
        odd = (sv[1] + sv[3]) / 2.0
        even = (sv[0] + sv[2]) / 2.0
        out["snare_backbeat_minus_downbeat_db"] = odd - even
        backbeat = bool(odd - even > 2.0)
    out["inference"] = {
        "four_on_the_floor": four_on_floor,
        "backbeat_on_2_and_4": backbeat,
        "basis": "the kick band must first be at least 3 dB louder on the beats "
                 "than between them, or there is no kick pattern to describe; "
                 "then a kick-band level within 3 dB across all four positions "
                 "is four on the floor, and a snare band more than 2 dB louder "
                 "on beats 2 and 4 than on 1 and 3 is a backbeat",
        "kick_pattern_detected": has_kick,
        "confidence": ("medium" if (beats_per_bar == 4 and has_kick) else "low"),
    }
    return out


def _pulse_rate(sections: list[dict[str, Any]], beats: np.ndarray,
                onsets: np.ndarray) -> dict[str, Any]:
    """Onsets per beat, per section: where a record goes half- or double-time."""
    if beats.size < 4 or not sections:
        return {"available": False, "reason": "no beat grid or no sections"}
    rows = []
    for s in sections:
        t0, t1 = s.get("start_s"), s.get("end_s")
        if t0 is None or t1 is None or t1 <= t0:
            continue
        nb = int(np.sum((beats >= t0) & (beats < t1)))
        no = int(np.sum((onsets >= t0) & (onsets < t1)))
        rows.append({"section_index": s.get("index"), "start_s": t0,
                     "start": fmt_time(t0), "beats": nb, "onsets": no,
                     "onsets_per_beat": (no / nb) if nb else None})
    vals = [r["onsets_per_beat"] for r in rows if r["onsets_per_beat"]]
    med = float(np.median(vals)) if vals else None
    switches = []
    for r in rows:
        r["relative_pulse_rate"] = (r["onsets_per_beat"] / med) if (med and r["onsets_per_beat"]) else None
    for a, b in zip(rows, rows[1:]):
        ra, rb = a.get("relative_pulse_rate"), b.get("relative_pulse_rate")
        if ra and rb and (rb / ra >= 1.7 or rb / ra <= 0.6):
            switches.append({"at_section": b["section_index"], "start_s": b["start_s"],
                             "start": b["start"], "factor": rb / ra})
    return {"available": True, "track_median_onsets_per_beat": med,
            "per_section": rows, "switch_count": len(switches),
            "switches": switches,
            "rule": "a section whose onsets-per-beat is at least 1.7x or at most "
                    "0.6x the previous section's is recorded as a switch; "
                    "half-time and double-time are the usual causes"}


def _stem_onsets(src: AudioSource) -> np.ndarray:
    return fine_onsets(src)


def stem_microtiming(stems: dict[str, AudioSource], beats: np.ndarray,
                     subdivision: int) -> dict[str, Any]:
    """Per-stem onset deviation from the beat grid.  Is the bass late?"""
    P = PARAMS["rhythm"]
    grid = _grid(beats, subdivision)
    if grid.size < 4:
        return {"available": False, "reason": "no beat grid"}
    out: dict[str, Any] = {
        "available": True,
        "source": "separated",
        "grid_subdivision": subdivision,
        "search_window_ms": P["microtiming_max_deviation_ms"],
        "onset_resolution_ms": onset_resolution_ms(),
        "sign_convention": "positive is late (behind the grid), negative is early",
        "per_stem": {},
    }
    for name in sorted(stems):
        dev = _deviations(_stem_onsets(stems[name]), grid,
                          P["microtiming_max_deviation_ms"])
        stats = _timing_stats(dev)
        if stats.get("std_ms") is not None:
            stats["inference"] = {
                "programmed_grid": bool(stats["std_ms"] <= P["programmed_tightness_ms"]),
                "basis": f"onset deviation standard deviation at or below "
                         f"{P['programmed_tightness_ms']} ms",
            }
        out["per_stem"][name] = stats

    # The beat tracker and the onset detector each carry their own constant
    # lag, so the absolute median is not a statement about the performance.
    # The differences between stems are: that offset is common to all of them
    # and cancels.  Both are reported, and which one to read is stated here.
    medians = {nm: st.get("median_ms") for nm, st in out["per_stem"].items()
               if st.get("median_ms") is not None}
    if medians:
        common = float(np.median(list(medians.values())))
        out["common_mode_offset_ms"] = common
        for nm, st in out["per_stem"].items():
            st["median_minus_common_mode_ms"] = (
                (st["median_ms"] - common) if st.get("median_ms") is not None else None)
        ordered = sorted(medians.items(), key=lambda kv: kv[1])
        out["earliest_stem"] = ordered[0][0]
        out["latest_stem"] = ordered[-1][0]
        out["spread_ms"] = ordered[-1][1] - ordered[0][1]
    out["read_this_one"] = ("median_minus_common_mode_ms: the absolute median "
                            "carries the constant lag of the beat tracker and "
                            "the onset detector, which is identical for every "
                            "stem and cancels between them")
    return out


def analyse(src: AudioSource, structure: dict[str, Any], collector: Collector,
            profile: str = "full") -> dict[str, Any]:
    """Downbeats, meter, groove and quantisation over the mix."""
    if profile == "quick":
        return {"available": False, "reason": "skipped by --profile quick"}
    tempo = (structure or {}).get("tempo") or {}
    if not tempo.get("available"):
        return {"available": False,
                "reason": "structure.tempo is unavailable; every measurement "
                          "here is defined on its beat grid"}
    beats = np.asarray(tempo.get("beat_times_s"), dtype=float)
    if beats.size < 8:
        return {"available": False, "reason": "fewer than eight beats"}
    P = PARAMS["rhythm"]
    onsets = fine_onsets(src)
    if onsets.size == 0:
        collector.warn("rhythm", "no onsets were detected; swing, syncopation "
                                 "and grid deviation are null")

    db = _downbeats(src, beats, collector)
    octave = _tempo_octave(src, beats, tempo.get("bpm"), collector)
    grid = _grid(beats, int(P["grid_subdivision"]))
    dev = _deviations(onsets, grid, P["microtiming_max_deviation_ms"])
    tight = _timing_stats(dev)
    out: dict[str, Any] = {
        "available": True,
        "bpm_used": tempo.get("bpm"),
        "beat_count": int(beats.size),
        "tempo_octave": octave,
        "downbeats": db,
        "swing": _swing(beats, onsets),
        "grid": {
            "subdivision": int(P["grid_subdivision"]),
            "instants": int(grid.size),
            "onset_resolution_ms": onset_resolution_ms(),
            "deviation": tight,
            "inference": {
                "programmed_grid": (bool(tight["std_ms"] <= P["programmed_tightness_ms"])
                                    if tight.get("std_ms") is not None else None),
                "basis": f"a standard deviation at or below "
                         f"{P['programmed_tightness_ms']} ms against a "
                         f"1/{P['grid_subdivision']}-beat grid is what a "
                         "sequencer produces; a played performance is wider",
            },
            "method": "signed onset-to-nearest-grid-instant distance, where the "
                      "grid is the measured beat times subdivided evenly",
        },
        "pulse_rate": _pulse_rate((structure or {}).get("sections") or [],
                                  beats, onsets),
    }
    if db.get("available"):
        out["syncopation"] = _syncopation(np.asarray(db["downbeat_times_s"]),
                                          db["beats_per_bar"], onsets)
        out["beat_position_profile"] = _beat_position_profile(
            src, beats, db["beats_per_bar"])
        out["bar_count"] = db["bar_count"]
    else:
        out["syncopation"] = {"available": False, "reason": "no downbeats"}
        out["beat_position_profile"] = {"available": False, "reason": "no downbeats"}
        out["bar_count"] = None
    return out
