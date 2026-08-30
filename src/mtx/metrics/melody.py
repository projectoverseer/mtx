"""Melody: F0 on the separated stems, and what the pitch track says.

The expensive half of this measurement -- separation -- has already run and is
cached, and `librosa.pyin` adds no dependency, so the pitch content of the
vocal is the cheapest musical measurement available to the tool.

Two things live here that are not melody.  The bass note track is measured with
the same machinery because the arrangement module needs it, and the
pitch-quantisation signature is forensics: an F0 track that snaps to the
semitone grid with near-zero transition time is a measurable artefact of hard
pitch correction.  It is reported as the transition statistics plus an
inference, never as a verdict about a singer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..audio import AudioSource
from ..params import PARAMS
from ..util import Collector, fmt_time

MAJOR_STEPS = (0, 2, 4, 5, 7, 9, 11)
MINOR_STEPS = (0, 2, 3, 5, 7, 8, 10)
PITCHES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _midi_name(midi: float) -> str:
    m = int(round(midi))
    return f"{PITCHES[m % 12]}{m // 12 - 1}"


def track_f0(src: AudioSource, fmin: float, fmax: float,
             collector: Collector, where: str) -> dict[str, Any] | None:
    """Run pyin over one stem and return the raw pitch track.

    Returns `None` (with a warning already recorded) when librosa is missing or
    the stem is too short; every caller degrades to a null block.
    """
    P = PARAMS["melody"]["f0"]
    try:
        import librosa
    except ImportError:
        collector.warn(where, "librosa not installed; no pitch track")
        return None
    y, sr = src.lib_mono, src.lib_sr
    hop, flen = int(P["hop_length"]), int(P["frame_length"])
    if y.size < flen * 4:
        collector.warn(where, "stem too short for a pitch track")
        return None
    try:
        f0, voiced, prob = librosa.pyin(y, fmin=fmin, fmax=min(fmax, sr / 2.0 * 0.98),
                                        sr=sr, frame_length=flen, hop_length=hop)
    except Exception as exc:
        collector.warn(where, f"pyin failed: {exc!r}")
        return None
    times = librosa.frames_to_time(np.arange(f0.size), sr=sr, hop_length=hop)
    midi = np.full(f0.shape, np.nan)
    good = np.isfinite(f0) & (f0 > 0)
    midi[good] = 69.0 + 12.0 * np.log2(f0[good] / 440.0)
    return {"times": np.asarray(times, dtype=float), "f0": np.asarray(f0, dtype=float),
            "midi": midi, "voiced": np.asarray(voiced, dtype=bool),
            "prob": np.asarray(prob, dtype=float),
            "hop_s": hop / float(sr), "sr": sr}


def segment_notes(track: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a continuous pitch track into notes.

    A note ends where the voicing does, or where the semitone the trace rounds
    to changes and stays changed.  The hysteresis is what keeps a vibrato that
    crosses a semitone boundary from being reported as a run of short notes.
    """
    P = PARAMS["melody"]["note_segmentation"]
    midi = np.array(track["midi"], dtype=float)
    voiced = np.array(track["voiced"], dtype=bool) & np.isfinite(midi)
    hop_s = float(track["hop_s"])
    if not np.any(voiced):
        return []
    k = int(P["median_filter_frames"])
    if k > 1:
        from scipy import ndimage
        filled = np.where(voiced, midi, np.nan)
        idx = np.flatnonzero(voiced)
        smoothed = filled.copy()
        smoothed[idx] = ndimage.median_filter(midi[idx], size=k, mode="nearest")
        midi = smoothed
    gap_frames = max(1, int(round(P["unvoiced_gap_ms"] / 1000.0 / hop_s)))
    min_frames = max(1, int(round(P["min_note_ms"] / 1000.0 / hop_s)))
    split = float(P["split_semitones"])

    notes: list[dict[str, Any]] = []
    i = 0
    n = midi.size
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        start = i
        ref = midi[i]
        j = i + 1
        unvoiced_run = 0
        while j < n:
            if not voiced[j]:
                unvoiced_run += 1
                if unvoiced_run >= gap_frames:
                    break
                j += 1
                continue
            unvoiced_run = 0
            if abs(midi[j] - ref) > split:
                # Confirm the move: one frame off the note is a glitch, two is
                # a new note.
                if j + 1 < n and voiced[j + 1] and abs(midi[j + 1] - ref) > split:
                    break
            else:
                ref = 0.7 * ref + 0.3 * midi[j]
            j += 1
        end = j - unvoiced_run
        seg = midi[start:end]
        seg = seg[np.isfinite(seg)]
        if seg.size >= min_frames:
            notes.append({
                "start_s": float(track["times"][start]),
                "end_s": float(track["times"][min(end, n - 1)]),
                "duration_s": float((end - start) * hop_s),
                "midi": float(np.median(seg)),
                "midi_std": float(np.std(seg)),
                "start_frame": int(start),
                "end_frame": int(end),
            })
        i = max(j, i + 1)
    for nt in notes:
        nt["note"] = _midi_name(nt["midi"])
        nt["cents_off_grid"] = float((nt["midi"] - round(nt["midi"])) * 100.0)
    return notes


def _weighted_percentile(values: np.ndarray, weights: np.ndarray,
                         q: float) -> float:
    """Percentile of `values` weighted by `weights` (here, note duration).

    A note list is not a sample of a melody: a two-second held note and a
    grace note count once each.  Every distribution figure here is weighted by
    duration so that it describes what was sung rather than what was segmented.
    """
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    if cum[-1] <= 0:
        return float(np.percentile(values, q))
    cut = (cum - 0.5 * w) / cum[-1]
    return float(np.interp(q / 100.0, cut, v))


def _range_block(notes: list[dict[str, Any]], collector: Collector,
                 where: str = "stems.melody.range"
                 ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The vocal range, with the tracker's octave errors kept out of it.

    `librosa.pyin` on a separated stem is monophonic and occasionally an octave
    out.  Measured over four reference vocals those errors were 7-12% of note
    time -- but the raw minimum and maximum are set by exactly those notes, so
    a raw range came out 40 to 58 semitones wide for singers whose real range
    in the song is under two octaves.

    So the reported range is taken over the notes inside a band around the
    duration-weighted median, the excluded notes are counted and reported as
    their own measurement, and the raw extremes stay in the document under a
    name that says what they are.
    """
    P = PARAMS["melody"]
    midis = np.asarray([nt["midi"] for nt in notes])
    durs = np.asarray([nt["duration_s"] for nt in notes])
    thr = float(P["octave_outlier_semitones"])
    median = _weighted_percentile(midis, durs, 50.0)
    inside = np.abs(midis - median) <= thr
    kept = [nt for nt, ok in zip(notes, inside) if ok]
    if len(kept) < 4:                       # nothing to be robust about
        kept, inside = notes, np.ones(len(notes), dtype=bool)
    k_midis = np.asarray([nt["midi"] for nt in kept])
    k_durs = np.asarray([nt["duration_s"] for nt in kept])
    hi_i = int(np.argmax(k_midis))
    lo_i = int(np.argmin(k_midis))
    out_time = float(np.sum(durs[~inside]))
    total_time = float(np.sum(durs)) or 1.0
    share = 100.0 * out_time / total_time
    conf = "high" if share < 3.0 else ("medium" if share < 10.0 else "low")
    if conf != "high":
        collector.low_confidence(
            where, conf,
            f"{share:.1f}% of note time sits more than {thr:.0f} semitones from "
            "the duration-weighted median pitch, which is what a monophonic "
            "tracker's octave errors look like")
    block = {
        "semitones": float(k_midis.max() - k_midis.min()),
        "lowest": {"midi": float(k_midis[lo_i]), "note": kept[lo_i]["note"],
                   "time_s": kept[lo_i]["start_s"],
                   "time": fmt_time(kept[lo_i]["start_s"])},
        "highest": {"midi": float(k_midis[hi_i]), "note": kept[hi_i]["note"],
                    "time_s": kept[hi_i]["start_s"],
                    "time": fmt_time(kept[hi_i]["start_s"])},
        "p5_p95_semitones": (_weighted_percentile(k_midis, k_durs, 95.0)
                             - _weighted_percentile(k_midis, k_durs, 5.0)),
        "p5_note": _midi_name(_weighted_percentile(k_midis, k_durs, 5.0)),
        "p95_note": _midi_name(_weighted_percentile(k_midis, k_durs, 95.0)),
        "notes_used": len(kept),
        "rule": P["range_rule"],
        "read_this_one": "p5_p95_semitones. Checked against two published "
                         "vocal ranges, the duration-weighted 5th and 95th "
                         "percentiles landed within a semitone of both, while "
                         "the surviving extremes still overshot the top by up "
                         "to a fourth: one falsetto ad-lib or one stray "
                         "tracker octave sets a maximum, and neither sets a "
                         "percentile.",
        "confidence": conf,
        "absolute": {
            "semitones": float(midis.max() - midis.min()),
            "lowest_note": _midi_name(float(midis.min())),
            "highest_note": _midi_name(float(midis.max())),
            "caveat": "the raw extremes of the pitch track, octave errors "
                      "included; not a statement about the singer's range",
        },
        "octave_outliers": {
            "threshold_semitones": thr,
            "notes": int(np.sum(~inside)),
            "time_s": out_time,
            "time_share_pct": share,
            "definition": "notes whose median pitch is further than the "
                          "threshold from the duration-weighted median pitch",
        },
    }
    return block, kept


def _vibrato(track: dict[str, Any], notes: list[dict[str, Any]]) -> dict[str, Any]:
    P = PARAMS["melody"]["vibrato"]
    hop_s = float(track["hop_s"])
    midi = np.asarray(track["midi"], dtype=float)
    lo, hi = P["rate_range_hz"]
    min_frames = int(round(P["min_note_ms"] / 1000.0 / hop_s))
    rates: list[float] = []
    depths: list[float] = []
    checked = 0
    for nt in notes:
        a, b = nt["start_frame"], nt["end_frame"]
        seg = midi[a:b]
        seg = seg[np.isfinite(seg)]
        if seg.size < max(16, min_frames):
            continue
        checked += 1
        x = np.arange(seg.size, dtype=float)
        coef = np.polyfit(x, seg, 1)
        resid = (seg - np.polyval(coef, x)) * 100.0     # cents
        win = np.hanning(resid.size)
        spec = np.abs(np.fft.rfft(resid * win))
        freqs = np.fft.rfftfreq(resid.size, d=hop_s)
        band = (freqs >= lo) & (freqs <= hi)
        if not np.any(band) or spec[band].size == 0:
            continue
        k = int(np.flatnonzero(band)[int(np.argmax(spec[band]))])
        # Peak amplitude of a Hann-windowed sinusoid is 4/N of the bin height.
        depth = float(4.0 * spec[k] / resid.size)
        if depth >= P["min_depth_cents"]:
            rates.append(float(freqs[k]))
            depths.append(depth)
    if not rates:
        return {"available": False,
                "reason": f"no note of at least {P['min_note_ms']:.0f} ms carried "
                          f"{P['min_depth_cents']:.0f} cents of {lo}-{hi} Hz modulation",
                "notes_examined": checked}
    return {"available": True, "notes_examined": checked,
            "notes_with_vibrato": len(rates),
            "share_of_long_notes": (len(rates) / checked) if checked else None,
            "rate_hz_median": float(np.median(rates)),
            "depth_cents_median": float(np.median(depths)),
            "depth_cents_p90": float(np.percentile(depths, 90)),
            "method": "linear-detrended cents trace of each long note, "
                      "Hann-windowed FFT, peak in the vibrato band"}


def _quantisation(track: dict[str, Any], notes: list[dict[str, Any]],
                  collector: Collector) -> dict[str, Any]:
    """Pitch-quantisation signature: forensics, not a judgement about singing."""
    P = PARAMS["melody"]["quantisation"]
    if len(notes) < 8:
        return {"available": False, "reason": "fewer than eight notes"}
    hop_s = float(track["hop_s"])
    midi = np.asarray(track["midi"], dtype=float)
    dev = np.abs(np.asarray([nt["cents_off_grid"] for nt in notes]))
    tol = float(P["grid_tolerance_cents"])
    on_grid = float(np.mean(dev <= tol))

    trans: list[float] = []
    frac = float(P["transition_target_fraction"])
    for a, b in zip(notes, notes[1:]):
        gap_frames = b["start_frame"] - a["end_frame"]
        if gap_frames > int(round(0.06 / hop_s)):
            continue
        interval = b["midi"] - a["midi"]
        if abs(interval) < 1.0:
            continue
        lo_i = max(a["end_frame"] - 4, 0)
        hi_i = min(b["start_frame"] + 8, midi.size)
        seg = midi[lo_i:hi_i]
        if seg.size < 3 or not np.all(np.isfinite(seg)):
            seg = seg[np.isfinite(seg)]
            if seg.size < 3:
                continue
        start_v = a["midi"]
        lo_t = start_v + 0.1 * interval
        hi_t = start_v + frac * interval
        if interval > 0:
            i1 = np.flatnonzero(seg >= lo_t)
            i2 = np.flatnonzero(seg >= hi_t)
        else:
            i1 = np.flatnonzero(seg <= lo_t)
            i2 = np.flatnonzero(seg <= hi_t)
        if i1.size and i2.size and i2[0] >= i1[0]:
            trans.append(float((i2[0] - i1[0]) * hop_s * 1000.0))
    median_ms = float(np.median(trans)) if trans else None
    fast = (median_ms is not None and median_ms <= P["fast_transition_ms"])
    hard = bool(on_grid > 0.6 and fast)
    conf = "medium" if (len(notes) >= 30 and len(trans) >= 10) else "low"
    if conf == "low":
        collector.low_confidence("stems.melody.pitch_quantisation", "low",
                                 f"{len(notes)} notes and {len(trans)} measurable "
                                 "note-to-note transitions")
    return {
        "available": True,
        "notes": len(notes),
        "cents_off_grid": {"median_abs": float(np.median(dev)),
                           "p90_abs": float(np.percentile(dev, 90)),
                           "share_within_tolerance": on_grid,
                           "tolerance_cents": tol},
        "transitions_measured": len(trans),
        "transition_ms": {"median": median_ms,
                          "p10": float(np.percentile(trans, 10)) if trans else None,
                          "p90": float(np.percentile(trans, 90)) if trans else None},
        "confidence": conf,
        "inference": {
            "grid_snapped": hard,
            "basis": f"{100 * on_grid:.0f}% of note medians sit within "
                     f"{tol:.0f} cents of a semitone and the median note-to-note "
                     f"transition is {median_ms if median_ms is not None else float('nan'):.0f} ms; "
                     "a retune with a short transition time produces both",
        },
        "method": "note-median deviation from the nearest semitone, plus the "
                  "time the trace takes to cross from 10% to "
                  f"{100 * frac:.0f}% of each note-to-note interval",
    }


def _scale_pitch_classes(key: dict[str, Any] | None) -> tuple[int, str] | None:
    if not key or not key.get("available") or not key.get("key"):
        return None
    parts = str(key["key"]).split()
    if len(parts) != 2 or parts[0] not in PITCHES:
        return None
    return PITCHES.index(parts[0]), parts[1]


def _delivery(track: dict[str, Any], notes: list[dict[str, Any]],
              collector: Collector) -> dict[str, Any]:
    """Sung, spoken or rapped, from how long the pitch holds still."""
    P = PARAMS["melody"]
    voiced = np.asarray(track["voiced"], dtype=bool)
    midi = np.asarray(track["midi"], dtype=float)
    voiced_frames = int(np.sum(voiced & np.isfinite(midi)))
    if voiced_frames < 20:
        return {"available": False, "reason": "almost no voiced frames"}
    stable_frames = 0
    for nt in notes:
        seg = midi[nt["start_frame"]:nt["end_frame"]]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            stable_frames += int(np.sum(np.abs(seg - nt["midi"]) * 100.0
                                        <= P["stable_frame_cents"]))
    share = stable_frames / voiced_frames
    durations = [nt["duration_s"] for nt in notes]
    if share >= 0.55:
        verdict, margin = "sung", share - 0.55
    elif share <= 0.30:
        verdict, margin = "spoken_or_rapped", 0.30 - share
    else:
        verdict, margin = "mixed", min(share - 0.30, 0.55 - share)
    conf = "medium" if margin > 0.08 else "low"
    if conf == "low":
        collector.low_confidence("stems.melody.delivery", "low",
                                 f"stable-pitch share {share:.2f} sits close to a "
                                 "category boundary")
    return {
        "available": True,
        "voiced_frames": voiced_frames,
        "stable_pitch_share": share,
        "median_note_duration_s": float(np.median(durations)) if durations else None,
        "inference": {"delivery": verdict, "confidence": conf},
        "basis": "share of voiced frames whose pitch sits within "
                 f"{P['stable_frame_cents']:.0f} cents of its own note median; "
                 "a held pitch is sung, a moving one is spoken or rapped",
    }


def _melody_block(track: dict[str, Any], notes: list[dict[str, Any]],
                  sections: list[dict[str, Any]], key: dict[str, Any] | None,
                  collector: Collector, profile: str) -> dict[str, Any]:
    P = PARAMS["melody"]
    if not notes:
        return {"available": False, "reason": "no notes segmented from the pitch track"}
    midis = np.asarray([nt["midi"] for nt in notes])
    durs = np.asarray([nt["duration_s"] for nt in notes])
    voiced_s = float(np.sum(durs))

    range_block, kept = _range_block(notes, collector,
                                     "stems.melody.vocals.range")
    k_midis = np.asarray([nt["midi"] for nt in kept])
    k_durs = np.asarray([nt["duration_s"] for nt in kept])
    intervals = np.diff(midis)
    rounded = np.round(intervals).astype(int)
    hist: dict[str, int] = {}
    for v in rounded:
        hist[str(int(v))] = hist.get(str(int(v)), 0) + 1
    steps = int(np.sum(np.abs(rounded) <= 2))
    leaps = int(np.sum(np.abs(rounded) > 2))

    # Phrases: voiced runs separated by a gap of at least phrase_gap_ms.
    gap = P["phrase_gap_ms"] / 1000.0
    phrases: list[dict[str, Any]] = []
    cur = [notes[0]]
    for a, b in zip(notes, notes[1:]):
        if b["start_s"] - a["end_s"] >= gap:
            phrases.append({"start_s": cur[0]["start_s"], "end_s": cur[-1]["end_s"],
                            "notes": len(cur)})
            cur = [b]
        else:
            cur.append(b)
    phrases.append({"start_s": cur[0]["start_s"], "end_s": cur[-1]["end_s"],
                    "notes": len(cur)})
    for ph in phrases:
        ph["duration_s"] = ph["end_s"] - ph["start_s"]
    breaths = [round(b["start_s"], 3) for a, b in zip(phrases, phrases[1:])]

    # Self-similarity: how often the same short interval shape comes back.
    n_gram = int(P["self_similarity_ngram"])
    repeat_share = None
    if rounded.size >= n_gram * 2:
        grams = [tuple(rounded[i:i + n_gram]) for i in range(rounded.size - n_gram + 1)]
        counts: dict[tuple, int] = {}
        for g in grams:
            counts[g] = counts.get(g, 0) + 1
        repeat_share = float(sum(c for g, c in counts.items() if c > 1) / len(grams))

    # Chromaticism against the measured key, when there is one.
    chromatic = None
    sc = _scale_pitch_classes(key)
    if sc is not None:
        tonic, mode = sc
        steps_set = set(MAJOR_STEPS if mode == "major" else MINOR_STEPS)
        inside = 0.0
        for nt in notes:
            pc = (int(round(nt["midi"])) - tonic) % 12
            if pc in steps_set:
                inside += nt["duration_s"]
        chromatic = {"key": key.get("key"),
                     "in_scale_time_pct": 100.0 * inside / voiced_s if voiced_s else None,
                     "out_of_scale_time_pct": 100.0 * (1 - inside / voiced_s) if voiced_s else None}

    contour = []
    for s in sections or []:
        t0, t1 = s.get("start_s"), s.get("end_s")
        if t0 is None or t1 is None:
            continue
        inside = [nt["midi"] for nt in notes if t0 <= nt["start_s"] < t1]
        contour.append({
            "section_index": s.get("index"), "start_s": t0,
            "notes": len(inside),
            "median_midi": float(np.median(inside)) if inside else None,
            "median_note": _midi_name(float(np.median(inside))) if inside else None,
            "max_midi": float(np.max(inside)) if inside else None,
        })

    out: dict[str, Any] = {
        "available": True,
        "source": "separated",
        "note_count": len(notes),
        "voiced_time_s": voiced_s,
        "notes_per_second_of_voicing": (len(notes) / voiced_s) if voiced_s > 0 else None,
        "median_note_duration_s": float(np.median(durs)),
        "range": range_block,
        "tessitura": {
            "median_midi": _weighted_percentile(k_midis, k_durs, 50.0),
            "median_note": _midi_name(_weighted_percentile(k_midis, k_durs, 50.0)),
            "iqr_semitones": (_weighted_percentile(k_midis, k_durs, 75.0)
                              - _weighted_percentile(k_midis, k_durs, 25.0)),
            "definition": "interquartile spread of the note pitches, weighted by "
                          "note duration, over the notes the range block kept",
        },
        "intervals": {
            "histogram_semitones": dict(sorted(hist.items(), key=lambda kv: int(kv[0]))),
            "stepwise_count": steps, "leap_count": leaps,
            "stepwise_share": (steps / (steps + leaps)) if (steps + leaps) else None,
            "median_abs_semitones": float(np.median(np.abs(intervals))) if intervals.size else None,
            "step_rule": "an interval of two semitones or fewer is stepwise",
        },
        "phrases": {
            "count": len(phrases),
            "gap_threshold_ms": P["phrase_gap_ms"],
            "duration_s": {"median": float(np.median([p["duration_s"] for p in phrases])),
                           "max": float(np.max([p["duration_s"] for p in phrases]))},
            "notes_per_phrase_median": float(np.median([p["notes"] for p in phrases])),
            "breath_positions_s": breaths[:200],
            "list": phrases[:200],
        },
        "melisma_index": float(np.mean([p["notes"] for p in phrases])),
        "melisma_definition": "mean notes per phrase; a syllable-aware melisma "
                              "needs an aligned lyric (see lyrics.alignment)",
        "self_similarity": {
            "ngram": n_gram,
            "repeated_ngram_share": repeat_share,
            "definition": "share of rounded-interval n-grams that occur more "
                          "than once in the melody",
        },
        "chromaticism": chromatic,
        "contour_per_section": contour,
        "delivery": _delivery(track, notes, collector),
        "notes_list": [{k: nt[k] for k in
                        ("start_s", "duration_s", "midi", "note", "cents_off_grid")}
                       for nt in notes[:600]],
        "notes_list_truncated": len(notes) > 600,
    }
    if profile != "quick":
        out["vibrato"] = _vibrato(track, notes)
    else:
        out["vibrato"] = {"available": False, "reason": "skipped by --profile quick"}
    return out


def analyse(stems: dict[str, AudioSource], sections: list[dict[str, Any]],
            key: dict[str, Any] | None, collector: Collector,
            profile: str = "full") -> dict[str, Any]:
    """Pitch measurements over the stems where pitch means something."""
    P = PARAMS["melody"]["f0"]
    out: dict[str, Any] = {
        "source": "separated",
        "caveat": "pitch is measured on a separated stem; separation artefacts "
                  "and the tracker's own octave errors are part of the number",
        "method": f"librosa.pyin at {P['sr_hz']} Hz, frame {P['frame_length']}, "
                  f"hop {P['hop_length']}",
    }
    if "vocals" not in stems:
        out["vocals"] = {"available": False, "reason": "no vocals stem"}
    else:
        track = track_f0(stems["vocals"], P["vocal_fmin_hz"], P["vocal_fmax_hz"],
                         collector, "stems.melody")
        if track is None:
            out["vocals"] = {"available": False, "reason": "pitch track unavailable"}
        else:
            notes = segment_notes(track)
            out["vocals"] = _melody_block(track, notes, sections, key, collector,
                                          profile)
            out["vocals"]["pitch_quantisation"] = _quantisation(track, notes, collector)
            out["_vocal_notes"] = notes
            out["_vocal_track"] = track

    if "bass" not in stems:
        out["bass"] = {"available": False, "reason": "no bass stem"}
    else:
        track = track_f0(stems["bass"], P["bass_fmin_hz"], P["bass_fmax_hz"],
                         collector, "stems.melody.bass")
        if track is None:
            out["bass"] = {"available": False, "reason": "pitch track unavailable"}
        else:
            notes = segment_notes(track)
            if not notes:
                out["bass"] = {"available": False, "reason": "no notes segmented"}
            else:
                b_range, b_kept = _range_block(notes, collector,
                                               "stems.melody.bass.range")
                med = _weighted_percentile(
                    np.asarray([nt["midi"] for nt in b_kept]),
                    np.asarray([nt["duration_s"] for nt in b_kept]), 50.0)
                out["bass"] = {
                    "available": True,
                    "source": "separated",
                    "note_count": len(notes),
                    "median_midi": med,
                    "median_note": _midi_name(med),
                    "range": b_range,
                    "range_semitones": b_range["semitones"],
                    "lowest_note": b_range["lowest"]["note"],
                    "median_note_duration_s": float(np.median(
                        [nt["duration_s"] for nt in notes])),
                    "notes_list": [{k: nt[k] for k in
                                    ("start_s", "duration_s", "midi", "note")}
                                   for nt in notes[:600]],
                    "notes_list_truncated": len(notes) > 600,
                }
            out["_bass_notes"] = notes
            out["_bass_track"] = track
    return out


def pitch_class_timeline(notes: list[dict[str, Any]]) -> list[tuple[float, float, int]]:
    """(start, end, pitch class) for a note list.  Used by the harmony module
    to test a chord root against the measured bass note."""
    return [(nt["start_s"], nt["start_s"] + nt["duration_s"],
             int(round(nt["midi"])) % 12) for nt in notes]
