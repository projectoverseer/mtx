"""4.7c Harmony: the chord track, and everything that needs one.

Nothing downstream could ask a harmonic question before this module existed:
`structure.key` was one global guess from a mean chroma, and there was no chord
data anywhere in the schema.

The recogniser here is deliberately not a learned model.  It is a template
match against binary chord-tone masks, smoothed by a Viterbi pass with one
self-transition probability, over the chroma the tool already computes -- which
means every number it produces is reproducible from the parameter block and
adds no dependency.  A learned recogniser (Chordino, BTC, madmom) would score
better on a benchmark and worse on the only property that matters here.

`structure.key` stays where it is and gains a second opinion: the key implied
by the chord track is an independent estimate, and the disagreement between the
two is the interesting number.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..audio import AudioSource
from ..params import PARAMS
from ..util import Collector, fmt_time

PITCHES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# (quality, chord tones as semitones above the root, label suffix)
QUALITIES: tuple[tuple[str, tuple[int, ...], str], ...] = (
    ("maj", (0, 4, 7), ""),
    ("min", (0, 3, 7), "m"),
    ("dim", (0, 3, 6), "dim"),
    ("aug", (0, 4, 8), "aug"),
    ("sus2", (0, 2, 7), "sus2"),
    ("sus4", (0, 5, 7), "sus4"),
    ("maj7", (0, 4, 7, 11), "maj7"),
    ("min7", (0, 3, 7, 10), "m7"),
    ("dom7", (0, 4, 7, 10), "7"),
    ("min6", (0, 3, 7, 9), "m6"),
    ("maj6", (0, 4, 7, 9), "6"),
    ("hdim7", (0, 3, 6, 10), "m7b5"),
    ("dim7", (0, 3, 6, 9), "dim7"),
)

KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52,
                     5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54,
                     4.75, 3.98, 2.69, 3.34, 3.17])

MAJOR_STEPS = (0, 2, 4, 5, 7, 9, 11)
MINOR_STEPS = (0, 2, 3, 5, 7, 8, 10)
# Roman numerals for the seven scale degrees, by mode.
MAJOR_NUMERALS = {0: "I", 2: "ii", 4: "iii", 5: "IV", 7: "V", 9: "vi", 11: "vii"}
MINOR_NUMERALS = {0: "i", 2: "ii", 3: "III", 5: "iv", 7: "v", 8: "VI", 10: "VII"}
# Which triad a chord quality reduces to, for the tonic test in the key estimate.
MAJOR_QUALITIES = frozenset({"maj", "maj7", "maj6", "dom7", "aug"})
MINOR_QUALITIES = frozenset({"min", "min7", "min6"})


def _templates() -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """The chord templates, mean-removed and L2-normalised, plus their names.

    Mean removal is the whole trick.  Against raw binary masks the match score
    is a plain cosine, and a four-tone mask that *contains* a triad can never
    lose to it: adding a tone only ever adds to the numerator, so any energy at
    all on the fourth tone makes the seventh win.  Measured against published
    chord charts, that turned nearly every triad in every song into a seventh.

    Once both the mask and the chroma frame have their means removed, the score
    is a Pearson correlation: energy on a tone the chord does *not* contain
    counts against it, and a seventh is preferred only when the seventh is
    really sounding.  The complexity penalty applied by the caller is the
    second half of the same idea.
    """
    rows: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    extra: list[float] = []
    for root in range(12):
        for quality, tones, suffix in QUALITIES:
            v = np.zeros(12)
            for t in tones:
                v[(root + t) % 12] = 1.0
            c = v - v.mean()
            rows.append(c / np.linalg.norm(c))
            extra.append(float(len(tones) - 3))
            meta.append({"root_pc": root, "quality": quality,
                         "label": PITCHES[root] + suffix,
                         "tones": [(root + t) % 12 for t in tones]})
    return np.vstack(rows), np.asarray(extra), meta


def _viterbi(scores: np.ndarray, p_self: float, gain: float) -> np.ndarray:
    """Best state path with a uniform off-diagonal transition.

    Because every off-diagonal transition has the same probability, the usual
    K x K maximisation collapses to a comparison against the running best, so
    the pass is O(frames * states) rather than O(frames * states^2).
    """
    n_frames, k = scores.shape
    log_self = math.log(p_self)
    log_off = math.log((1.0 - p_self) / max(k - 1, 1))
    delta = scores[0] * gain
    back = np.zeros((n_frames, k), dtype=np.int32)
    for t in range(1, n_frames):
        best_j = int(np.argmax(delta))
        best_v = delta[best_j] + log_off
        stay = delta + log_self
        take_stay = stay >= best_v
        back[t] = np.where(take_stay, np.arange(k), best_j)
        delta = np.where(take_stay, stay, best_v) + scores[t] * gain
        delta -= delta.max()          # keep the accumulator bounded
    path = np.zeros(n_frames, dtype=np.int32)
    path[-1] = int(np.argmax(delta))
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _beat_sync(chroma: np.ndarray, frame_times: np.ndarray,
               beats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median-aggregate chroma between consecutive beat instants."""
    edges = np.searchsorted(frame_times, beats)
    cols: list[np.ndarray] = []
    spans: list[tuple[float, float]] = []
    for a, b, t0, t1 in zip(edges, edges[1:], beats, beats[1:]):
        if b <= a:
            b = a + 1
        seg = chroma[:, a:min(b, chroma.shape[1])]
        if seg.size == 0:
            continue
        cols.append(np.median(seg, axis=1))
        spans.append((float(t0), float(t1)))
    if not cols:
        return np.zeros((12, 0)), np.zeros((0, 2))
    return np.vstack(cols).T, np.asarray(spans)


def _bass_chroma(src: AudioSource, collector: Collector) -> tuple[np.ndarray, np.ndarray] | None:
    """Chroma of the bottom two octaves only, for inversions and pedal points."""
    try:
        import librosa
        y, sr = src.lib_mono, src.lib_sr
        hop = PARAMS["general"]["librosa_hop_length"]
        c = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop,
                                       fmin=librosa.note_to_hz("C1"),
                                       n_octaves=2, bins_per_octave=36)
        t = librosa.frames_to_time(np.arange(c.shape[1]), sr=sr, hop_length=hop)
        return np.asarray(c), np.asarray(t)
    except Exception as exc:
        collector.warn("harmony.bass", f"low-register chroma failed: {exc!r}")
        return None


def _key_from_chords(segments: list[dict[str, Any]]) -> dict[str, Any]:
    """A second key estimate, from the chord track rather than the raw chroma.

    Not a Krumhansl-Schmuckler correlation over a chord-tone histogram: that
    was measured against seven published keys and got three of them, which is
    worse than the mean-chroma estimate it is supposed to be a second opinion
    on.  A histogram of chord tones throws away the thing a chord track knows
    that a chroma does not -- which chords were played, and for how long.

    So the question asked here is the direct one: which key's scale explains
    the most chord time, with a bonus for time spent on that key's own tonic
    chord.  Both halves are reported so the margin is visible.
    """
    P = PARAMS["harmony"]["key_from_chords"]
    total = sum(s["duration_s"] for s in segments)
    if not segments or total <= 0:
        return {"available": False, "reason": "no chord content"}
    w = float(P["tonic_weight"])
    w_first = float(P["first_chord_weight"])
    w_final = float(P["final_chord_weight"])
    first, last = segments[0], segments[-1]

    def is_tonic(seg: dict[str, Any], tonic: int, mode: str) -> bool:
        if (seg["root_pc"] - tonic) % 12 != 0:
            return False
        return (seg["quality"] in MAJOR_QUALITIES if mode == "major"
                else seg["quality"] in MINOR_QUALITIES)

    scored: list[tuple[float, dict[str, Any]]] = []
    for tonic in range(12):
        for mode, steps in (("major", MAJOR_STEPS), ("minor", MINOR_STEPS)):
            scale = set(steps)
            diatonic = 0.0
            tonic_time = 0.0
            for seg in segments:
                deg = (seg["root_pc"] - tonic) % 12
                if deg in scale and all(((pc - tonic) % 12) in scale
                                        for pc in seg.get("tones", [])):
                    diatonic += seg["duration_s"]
                if is_tonic(seg, tonic, mode):
                    tonic_time += seg["duration_s"]
            opens = is_tonic(first, tonic, mode)
            closes = is_tonic(last, tonic, mode)
            score = (diatonic + w * tonic_time
                     + (w_first * total if opens else 0.0)
                     + (w_final * total if closes else 0.0))
            scored.append((score, {
                "key": f"{PITCHES[tonic]} {mode}", "score": score,
                "diatonic_time_s": diatonic, "tonic_chord_time_s": tonic_time,
                "opens_on_tonic": opens, "closes_on_tonic": closes}))
    scored.sort(key=lambda kv: -kv[0])
    top, runner = scored[0][1], scored[1][1]
    # The relative key shares every scale tone, so when it is the runner-up the
    # scale fit contributed nothing to the choice and only the tonic evidence
    # did.  That is exactly when this estimate is least trustworthy.
    t_root, t_mode = top["key"].split()
    r_root, r_mode = runner["key"].split()
    rel = (t_mode != r_mode and
           ((PITCHES.index(r_root) - PITCHES.index(t_root)) % 12
            in ((9,) if t_mode == "major" else (3,))))
    conf = "low" if (rel or top["score"] - runner["score"] < 0.05 * total) else "medium"
    return {
        "available": True,
        "key": top["key"],
        "score": top["score"],
        "diatonic_time_s": top["diatonic_time_s"],
        "diatonic_time_pct": 100.0 * top["diatonic_time_s"] / total,
        "tonic_chord_time_s": top["tonic_chord_time_s"],
        "opens_on_tonic": top["opens_on_tonic"],
        "closes_on_tonic": top["closes_on_tonic"],
        "runner_up": runner["key"],
        "runner_up_score": runner["score"],
        "margin": top["score"] - runner["score"],
        "margin_pct_of_track": 100.0 * (top["score"] - runner["score"]) / total,
        "runner_up_is_relative_key": bool(rel),
        "confidence": conf,
        "confidence_reason": (
            "the runner-up is this key's relative major or minor: they share "
            "every scale tone, so only the tonic evidence separated them"
            if rel else
            "the margin over the runner-up is under 5% of the track"
            if conf == "low" else
            "a different scale, chosen by chord time, with tonic evidence agreeing"),
        "accuracy_note": PARAMS["harmony"]["key_from_chords"]["measured_accuracy"],
        "method": P["method"],
        "weights": {"tonic": w, "first_chord": w_first, "final_chord": w_final},
    }


def _degrees(segments: list[dict[str, Any]], key: str | None) -> dict[str, Any]:
    if not key:
        return {"available": False, "reason": "no key to reduce against"}
    parts = str(key).split()
    if len(parts) != 2 or parts[0] not in PITCHES:
        return {"available": False, "reason": f"unparsable key {key!r}"}
    tonic, mode = PITCHES.index(parts[0]), parts[1]
    steps = set(MAJOR_STEPS if mode == "major" else MINOR_STEPS)
    numerals = MAJOR_NUMERALS if mode == "major" else MINOR_NUMERALS
    total = sum(s["duration_s"] for s in segments) or 1.0
    diatonic = 0.0
    counts: dict[str, float] = {}
    for s in segments:
        deg = (s["root_pc"] - tonic) % 12
        s["degree_semitones"] = deg
        s["degree"] = numerals.get(deg, f"b{deg}")
        # A chord is diatonic when its root is a scale degree and every one of
        # its tones is in the scale.
        in_scale = deg in steps and all(((pc - tonic) % 12) in steps
                                        for pc in s.get("tones", []))
        s["diatonic"] = bool(in_scale)
        if in_scale:
            diatonic += s["duration_s"]
        counts[s["degree"]] = counts.get(s["degree"], 0.0) + s["duration_s"]
    return {
        "available": True,
        "key_used": key,
        "diatonic_time_pct": 100.0 * diatonic / total,
        "borrowed_time_pct": 100.0 * (1.0 - diatonic / total),
        "degree_time_s": {k: round(v, 3) for k, v in
                          sorted(counts.items(), key=lambda kv: -kv[1])},
        "rule": "a chord is diatonic when its root and all of its chord tones "
                "lie in the scale of the key used",
    }


def _cadences(segments: list[dict[str, Any]]) -> dict[str, Any]:
    named = PARAMS["harmony"]["cadence_degrees"]
    out: dict[str, Any] = {"counts": {}, "instances": []}
    if not segments or "degree_semitones" not in segments[0]:
        return {"available": False, "reason": "no degree reduction (no key)"}
    for name in named:
        out["counts"][name] = 0
    for a, b in zip(segments, segments[1:]):
        da, dbb = a.get("degree_semitones"), b.get("degree_semitones")
        for name, (x, y) in named.items():
            if da == x and dbb == y:
                out["counts"][name] += 1
                if len(out["instances"]) < 100:
                    out["instances"].append({"type": name, "at_s": b["start_s"],
                                             "at": fmt_time(b["start_s"]),
                                             "from": a["label"], "to": b["label"]})
    out["available"] = True
    out["definition"] = ("counts of degree-to-degree chord transitions; "
                         "authentic V-I, plagal IV-I, deceptive V-vi, half I-V")
    return out


def _loop(segments: list[dict[str, Any]], downbeats: np.ndarray | None
          ) -> dict[str, Any]:
    """Is the harmony a repeating 2, 4, 8 or 16-bar loop?"""
    P = PARAMS["harmony"]
    if downbeats is None or len(downbeats) < 4:
        return {"available": False,
                "reason": "no downbeats; a loop length is measured in bars"}
    bars: list[str] = []
    for a, b in zip(downbeats, downbeats[1:]):
        inside = [s["label"] for s in segments
                  if s["start_s"] < b and s["end_s"] > a]
        bars.append("|".join(dict.fromkeys(inside)) if inside else "-")
    rows = []
    best = None
    for p in P["loop_candidate_bars"]:
        if len(bars) < p * 2:
            continue
        pairs = [(bars[i], bars[i + p]) for i in range(len(bars) - p)]
        frac = sum(1 for x, y in pairs if x == y) / len(pairs)
        rows.append({"bars": p, "match_fraction": frac})
        if frac >= P["loop_match_threshold"] and (best is None or p < best["bars"]):
            best = {"bars": p, "match_fraction": frac}
    return {"available": True, "bar_labels": bars[:200],
            "candidates": rows, "loop": best,
            "threshold": P["loop_match_threshold"],
            "method": "share of bars whose chord set equals the chord set of the "
                      "bar one candidate period later; the shortest period over "
                      "the threshold wins"}


def _modulation(chroma: np.ndarray, times: np.ndarray,
                collector: Collector) -> dict[str, Any]:
    P = PARAMS["harmony"]["modulation"]
    if times.size < 4:
        return {"available": False, "reason": "too few frames"}
    win, hop = P["window_s"], P["hop_s"]
    duration = float(times[-1])
    if duration < win * 2:
        return {"available": False, "reason": "track shorter than two windows"}
    est: list[dict[str, Any]] = []
    t = 0.0
    while t + win <= duration:
        m = (times >= t) & (times < t + win)
        if np.any(m):
            prof = chroma[:, m].mean(axis=1)
            if float(np.std(prof)) > 0:
                scores = []
                for mode, template in (("major", KS_MAJOR), ("minor", KS_MINOR)):
                    for i in range(12):
                        c = float(np.corrcoef(prof, np.roll(template, i))[0, 1])
                        scores.append((c, f"{PITCHES[i]} {mode}"))
                scores.sort(reverse=True)
                est.append({"start_s": t, "key": scores[0][1],
                            "margin": scores[0][0] - scores[1][0]})
        t += hop
    if not est:
        return {"available": False, "reason": "no window produced a key"}
    changes: list[dict[str, Any]] = []
    run_key, run_start = est[0]["key"], est[0]["start_s"]
    for e in est[1:]:
        if e["key"] != run_key:
            held = e["start_s"] - run_start
            if held >= P["min_hold_s"] and e["margin"] >= P["min_margin"]:
                changes.append({"at_s": e["start_s"], "at": fmt_time(e["start_s"]),
                                "from": run_key, "to": e["key"],
                                "previous_held_s": held,
                                "margin": e["margin"]})
            run_key, run_start = e["key"], e["start_s"]
    if changes:
        collector.low_confidence(
            "harmony.modulation", "low",
            f"{len(changes)} windowed key change(s); a windowed KS estimate "
            "moves on a borrowed chord as readily as on a real modulation")
    return {"available": True, "window_s": win, "hop_s": hop,
            "per_window": est, "changes": changes,
            "change_count": len(changes), "confidence": "low",
            "method": "Krumhansl-Schmuckler over a sliding chroma window; a "
                      "change counts only when the previous key held for at "
                      f"least {P['min_hold_s']:.0f} s"}


def _pedal_points(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    P = PARAMS["harmony"]["pedal_min_chords"]
    out: list[dict[str, Any]] = []
    run: list[dict[str, Any]] = []
    for s in segments:
        bpc = s.get("bass_pc")
        if bpc is None:
            run = []
            continue
        if run and run[-1].get("bass_pc") == bpc:
            run.append(s)
        else:
            run = [s]
        if len(run) >= P and len({r["root_pc"] for r in run}) > 1:
            if out and out[-1]["end_s"] == run[-2]["end_s"]:
                out[-1]["end_s"] = run[-1]["end_s"]
                out[-1]["chords"] = [r["label"] for r in run]
            else:
                out.append({"bass_note": PITCHES[bpc],
                            "start_s": run[0]["start_s"],
                            "start": fmt_time(run[0]["start_s"]),
                            "end_s": run[-1]["end_s"],
                            "chords": [r["label"] for r in run]})
    return out[:50]


def analyse(src: AudioSource, structure: dict[str, Any],
            rhythm: dict[str, Any] | None, collector: Collector,
            profile: str = "full") -> dict[str, Any]:
    if profile == "quick":
        return {"available": False, "reason": "skipped by --profile quick"}
    try:
        import librosa
    except ImportError:
        collector.warn("harmony", "librosa not installed; no chord track")
        return {"available": False, "reason": "librosa not installed"}
    P = PARAMS["harmony"]
    hop = PARAMS["general"]["librosa_hop_length"]
    chroma = np.asarray(src.chroma_cqt(), dtype=np.float64)
    if chroma.shape[1] < 8:
        return {"available": False, "reason": "too few chroma frames"}
    frame_times = librosa.frames_to_time(np.arange(chroma.shape[1]),
                                         sr=src.lib_sr, hop_length=hop)

    tempo = (structure or {}).get("tempo") or {}
    raw_beats = tempo.get("beat_times_s")
    beats = np.asarray(raw_beats if raw_beats is not None else [], dtype=float)
    if beats.size >= 4:
        # Extend the grid to the end of the file so the last chord has a span.
        if beats[-1] < src.duration:
            beats = np.append(beats, src.duration)
        obs, spans = _beat_sync(chroma, frame_times, beats)
        unit = "beat"
    else:
        obs, unit = chroma, "frame"
        spans = np.stack([frame_times, np.append(frame_times[1:], src.duration)],
                         axis=1)
    if obs.shape[1] < 4:
        return {"available": False, "reason": "too few observation frames"}

    centred = obs - obs.mean(axis=0, keepdims=True)
    norms = np.maximum(np.linalg.norm(centred, axis=0, keepdims=True), 1e-12)
    obs_n = (centred / norms).T                   # (frames, 12), mean-removed
    templates, extra_tones, meta = _templates()
    scores = obs_n @ templates.T                  # Pearson r per state
    # Two priors, both subtracted from the score before the Viterbi pass: one
    # per chord tone beyond the third, and one per quality.  A suspended or
    # diminished chord is rarer than a triad, and without a prior saying so the
    # recogniser reaches for one whenever a passing tone lands on the second or
    # the fourth.
    prior = np.asarray([float(PARAMS["harmony"].get("quality_prior", {})
                              .get(m["quality"], 0.0)) for m in meta])
    scores = scores - float(P["complexity_penalty"]) * extra_tones[None, :]
    scores = scores - prior[None, :]
    scores = np.hstack([scores, np.full((scores.shape[0], 1), P["no_chord_score"])])
    path = _viterbi(scores, P["viterbi_self_transition"],
                    float(P.get("emission_gain", 30.0)))

    # Merge equal consecutive states into segments.
    segments: list[dict[str, Any]] = []
    i = 0
    n_states = len(meta)
    while i < path.size:
        j = i
        while j + 1 < path.size and path[j + 1] == path[i]:
            j += 1
        state = int(path[i])
        t0, t1 = float(spans[i][0]), float(spans[j][1])
        conf = float(np.mean(scores[i:j + 1, state]))
        if state == n_states:
            segments.append({"label": "N", "root_pc": None, "quality": "no_chord",
                             "tones": [], "start_s": t0, "end_s": t1,
                             "duration_s": t1 - t0, "units": j - i + 1,
                             "match": conf})
        else:
            m = meta[state]
            segments.append({"label": m["label"], "root_pc": m["root_pc"],
                             "quality": m["quality"], "tones": list(m["tones"]),
                             "start_s": t0, "end_s": t1, "duration_s": t1 - t0,
                             "units": j - i + 1, "match": conf})
        i = j + 1
    for s in segments:
        s["start"] = fmt_time(s["start_s"])

    # Bass note under each chord: an inversion is a chord whose lowest note is
    # not its root, and the tool already knows how to find a low fundamental.
    bass = _bass_chroma(src, collector)
    if bass is not None:
        bc, bt = bass
        for s in segments:
            m = (bt >= s["start_s"]) & (bt < s["end_s"])
            if not np.any(m):
                continue
            pc = int(np.argmax(bc[:, m].mean(axis=1)))
            s["bass_pc"] = pc
            s["bass_note"] = PITCHES[pc]
            if s["root_pc"] is not None and pc != s["root_pc"]:
                s["slash_label"] = f"{s['label']}/{PITCHES[pc]}"
                s["inversion"] = ("first" if (pc - s["root_pc"]) % 12 in (3, 4)
                                  else "second" if (pc - s["root_pc"]) % 12 == 7
                                  else "other")
            else:
                s["inversion"] = "root"

    sounded = [s for s in segments if s["root_pc"] is not None]
    total_s = sum(s["duration_s"] for s in segments) or 1.0
    labels = [s["label"] for s in sounded]
    counts: dict[str, float] = {}
    for s in sounded:
        counts[s["label"]] = counts.get(s["label"], 0.0) + s["duration_s"]
    entropy = None
    if counts:
        p = np.asarray(list(counts.values())) / sum(counts.values())
        entropy = float(-np.sum(p * np.log2(np.maximum(p, 1e-12))))

    downbeats = None
    if rhythm and (rhythm.get("downbeats") or {}).get("available"):
        downbeats = np.asarray(rhythm["downbeats"]["downbeat_times_s"], dtype=float)
    changes = max(len(sounded) - 1, 0)
    bars = (len(downbeats) - 1) if downbeats is not None and len(downbeats) > 1 else None

    key_block = (structure or {}).get("key") or {}
    key_used = key_block.get("key") if key_block.get("available") else None
    degrees = _degrees(sounded, key_used)
    chord_key = _key_from_chords(sounded)
    agreement = None
    if chord_key.get("available") and key_used:
        agreement = {
            "chroma_key": key_used,
            "chord_track_key": chord_key["key"],
            "agree": bool(chord_key["key"] == key_used),
            "same_tonic_different_mode": bool(
                chord_key["key"].split()[0] == str(key_used).split()[0]
                and chord_key["key"] != key_used),
        }
        agreement["chord_track_confidence"] = chord_key.get("confidence")
        agreement["read_this_one"] = (
            "structure.key, unless the chord track is the more musically "
            "specific answer for what you are asking: measured against seven "
            "published keys the mean-chroma estimate got five and the chord "
            "track got four")
        if not agreement["agree"]:
            collector.warn("harmony.key_cross_check",
                           f"mean-chroma key {key_used} disagrees with the "
                           f"chord-track key {chord_key['key']} "
                           f"[{chord_key.get('confidence')}]"
                           + (" (they are relative keys)"
                              if chord_key.get("runner_up_is_relative_key") else ""))

    mean_match = float(np.mean([s["match"] for s in sounded])) if sounded else 0.0
    conf = "medium" if mean_match > 0.72 else "low"
    if conf != "medium":
        collector.low_confidence("harmony.chords", conf,
                                 f"mean template match {mean_match:.3f} over "
                                 f"{len(sounded)} sounded chord segments")

    return {
        "available": True,
        "observation_unit": unit,
        "method": "binary chord-tone templates matched by cosine against "
                  f"{'beat-synchronous ' if unit == 'beat' else ''}chroma-CQT, "
                  "smoothed by a Viterbi pass with a single self-transition "
                  "probability",
        "qualities": [q for q, _, _ in QUALITIES],
        "chord_count": len(sounded),
        "no_chord_time_pct": 100.0 * sum(s["duration_s"] for s in segments
                                         if s["root_pc"] is None) / total_s,
        "mean_template_match": mean_match,
        "confidence": conf,
        "vocabulary": {
            "distinct_chords": len(counts),
            "entropy_bits": entropy,
            "time_s_by_chord": {k: round(v, 3) for k, v in
                                sorted(counts.items(), key=lambda kv: -kv[1])[:40]},
        },
        "harmonic_rhythm": {
            "changes": changes,
            "changes_per_second": changes / total_s if total_s > 0 else None,
            "changes_per_bar": (changes / bars) if bars else None,
            "bars_used": bars,
            "per_bar_caveat": "a bar is four beats of the tempo structure.tempo "
                              "reported; if that tempo sits at the wrong "
                              "metrical level this figure is out by the same "
                              "factor. See rhythm.tempo_octave.",
            "median_duration_s": float(np.median([s["duration_s"] for s in sounded]))
            if sounded else None,
            "median_duration_beats": (
                float(np.median([s["units"] for s in sounded])) if (sounded and unit == "beat")
                else None),
        },
        "progression": labels[:400],
        "progression_truncated": len(labels) > 400,
        "chords": [{k: s[k] for k in
                    ("start_s", "start", "duration_s", "label", "quality",
                     "match", "bass_note", "inversion", "slash_label", "degree")
                    if k in s} for s in segments[:600]],
        "chords_truncated": len(segments) > 600,
        "degrees": degrees,
        "cadences": _cadences(sounded),
        "loop": _loop(sounded, downbeats),
        "pedal_points": _pedal_points(sounded),
        "modulation": _modulation(chroma, frame_times, collector),
        "key_from_chords": chord_key,
        "key_cross_check": agreement,
    }
