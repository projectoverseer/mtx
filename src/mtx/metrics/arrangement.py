"""4.9b Arrangement: what is playing, when it comes in, and how much of it.

"Arrangement" barely existed in the dump as a working concept:
`structure.arrangement_gaps` is defined on frequency bands, not on instruments,
and four stems collapse guitars, keys, synths, strings, horns and pads into
`other`.

Two things widen that.  Running demucs with a six-source model (`--stems-model
htdemucs_6s`) splits guitar and piano out of `other` at no new dependency, only
a model choice.  And once there are stems at all, entry and exit, concurrent
source count, drum-machine evidence, 808 behaviour, vocal stacking and
lead-versus-backing balance are all ordinary DSP over signals already on disk.

Everything a stem cannot answer -- naming a horn section, telling a Rhodes from
a Wurlitzer -- needs a trained tagger, and the hook for one is here, reporting
`available: false` with what to install rather than guessing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal as sps

from ..audio import AudioSource
from ..dsp import band_power, welch_psd
from ..params import PARAMS
from ..util import Collector, db_amp, db_pow, fmt_time


def _envelope(src: AudioSource, hop_s: float) -> tuple[np.ndarray, np.ndarray]:
    sr = src.band_sr
    hop = max(1, int(round(hop_s * sr)))
    x = src.band_mono
    n = (x.size // hop) * hop
    if n == 0:
        return np.zeros(0), np.zeros(0)
    env = np.sqrt(np.mean(x[:n].reshape(-1, hop) ** 2, axis=1))
    return np.arange(env.size) * hop / sr, env


def _presence(stems: dict[str, AudioSource], hop_s: float, margin_db: float
              ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    """A per-stem present/absent timeline on a common grid."""
    envs: dict[str, np.ndarray] = {}
    thr: dict[str, float] = {}
    times = np.zeros(0)
    for name in sorted(stems):
        t, e = _envelope(stems[name], hop_s)
        if e.size == 0:
            continue
        edb = db_amp(np.maximum(e, 1e-20))
        thr[name] = float(np.percentile(edb, 95)) - margin_db
        envs[name] = edb
        if t.size > times.size:
            times = t
    n = min((v.size for v in envs.values()), default=0)
    for k in list(envs):
        envs[k] = envs[k][:n]
    return times[:n], envs, thr


def _drum_character(src: AudioSource, collector: Collector) -> dict[str, Any]:
    """How alike are the drum hits, and how alike are their levels?

    A sampled or programmed kit fires the same waveform at the same velocity;
    a played one does not.  Both halves are measured, and the label attached to
    them is an inference with the two numbers next to it.
    """
    from .rhythm import fine_onsets

    P = PARAMS["arrangement"]
    onsets = fine_onsets(src)
    if onsets.size < 12:
        return {"available": False, "reason": "fewer than twelve drum onsets"}
    sr = src.band_sr
    win = max(64, int(round(P["drum_hit_window_ms"] / 1000.0 * sr)))
    x = src.band_mono
    specs: list[np.ndarray] = []
    peaks: list[float] = []
    for t in onsets[:600]:
        a = int(t * sr)
        seg = x[a:a + win]
        if seg.size < win:
            continue
        peaks.append(float(np.max(np.abs(seg))))
        mag = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
        specs.append(db_pow(np.maximum(mag ** 2, 1e-24)))
    if len(specs) < 8:
        return {"available": False, "reason": "too few complete hit windows"}
    M = np.vstack(specs)
    M = M - M.mean(axis=1, keepdims=True)
    norms = np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    U = M / norms
    sim = U @ U.T
    iu = np.triu_indices(sim.shape[0], k=1)
    consistency = float(np.median(sim[iu]))
    pk_db = db_amp(np.asarray(peaks))
    level_sd = float(np.std(pk_db))
    programmed = bool(consistency > 0.6 and level_sd < 4.0)
    conf = "medium" if len(specs) >= 40 else "low"
    if conf == "low":
        collector.low_confidence("stems.arrangement.drums", "low",
                                 f"only {len(specs)} usable hit windows")
    return {
        "available": True,
        "source": "separated",
        "hits_examined": len(specs),
        "hit_window_ms": P["drum_hit_window_ms"],
        "spectral_consistency_median": consistency,
        "hit_level_std_db": level_sd,
        "confidence": conf,
        "inference": {
            "sampled_or_programmed": programmed,
            "basis": "median pairwise cosine between hit spectra above 0.6 and "
                     "a hit-to-hit peak-level spread under 4 dB; a sampler "
                     "repeats one waveform at one velocity, a played kit does not",
        },
        "method": "cosine similarity between the mean-removed log spectra of "
                  f"{P['drum_hit_window_ms']:.0f} ms windows at each detected onset",
    }


def _bass_character(src: AudioSource, bass_notes: list[dict[str, Any]] | None,
                    bass_track: dict[str, Any] | None) -> dict[str, Any]:
    """Sub-bass balance, note decay and glide: the 808 question, measured."""
    P = PARAMS["arrangement"]
    sr = src.band_sr
    f, p = welch_psd(src.band_mono, sr, 16384)
    split = P["sub_bass_split_hz"]
    sub = band_power(f, p, 20.0, split) if f.size else 0.0
    upper = band_power(f, p, split, 250.0) if f.size else 0.0
    total = sub + upper
    sub_pct = 100.0 * sub / total if total > 0 else None

    # Harmonic richness at the median bass note: a sine-like sub has almost
    # nothing above its fundamental, a plucked or distorted bass has plenty.
    richness = None
    fundamental = None
    if bass_notes:
        med_midi = float(np.median([n["midi"] for n in bass_notes]))
        fundamental = 440.0 * (2.0 ** ((med_midi - 69.0) / 12.0))
        if f.size and fundamental > 20:
            e1 = band_power(f, p, fundamental * 0.94, fundamental * 1.06)
            harm = sum(band_power(f, p, fundamental * k * 0.97, fundamental * k * 1.03)
                       for k in (2, 3, 4, 5))
            richness = db_pow(harm / e1) if e1 > 0 else None

    decays: list[float] = []
    hop = max(1, int(round(0.005 * sr)))
    x = np.abs(src.band_mono)
    n = (x.size // hop) * hop
    env = (np.sqrt(np.mean(x[:n].reshape(-1, hop) ** 2, axis=1))
           if n >= hop else np.zeros(0))
    edb = db_amp(np.maximum(env, 1e-20)) if env.size else env
    for nt in (bass_notes or [])[:400]:
        a = int(nt["start_s"] / 0.005)
        b = min(a + int(1.5 / 0.005), edb.size)
        seg = edb[a:b]
        if seg.size < 20:
            continue
        pk = int(np.argmax(seg))
        below = np.flatnonzero(seg[pk:] <= seg[pk] - 20.0)
        if below.size:
            decays.append(float(below[0] * 5.0))
    median_decay = float(np.median(decays)) if decays else None

    glides = 0
    pairs = 0
    if bass_notes and bass_track is not None:
        voiced = np.asarray(bass_track["voiced"], dtype=bool)
        hop_s = float(bass_track["hop_s"])
        max_gap = P["glide"]["max_gap_ms"] / 1000.0
        for a, b in zip(bass_notes, bass_notes[1:]):
            gap = b["start_s"] - (a["start_s"] + a["duration_s"])
            if gap > max_gap or abs(b["midi"] - a["midi"]) < P["glide"]["min_semitones"]:
                continue
            pairs += 1
            i0 = int((a["start_s"] + a["duration_s"]) / hop_s)
            i1 = int(b["start_s"] / hop_s)
            if i1 <= i0 or bool(np.all(voiced[i0:max(i1, i0 + 1)])):
                glides += 1

    eight = None
    if median_decay is not None and sub_pct is not None:
        eight = bool(median_decay >= P["eight_o_eight"]["min_decay_ms"]
                     and sub_pct >= P["eight_o_eight"]["sub_share_min_pct"])
    return {
        "available": True,
        "source": "separated",
        "sub_share_pct": sub_pct,
        "sub_split_hz": split,
        "median_fundamental_hz": fundamental,
        "harmonics_2_to_5_minus_fundamental_db": richness,
        "note_decay_ms": {"median": median_decay,
                          "notes_measured": len(decays),
                          "definition": "time from the note's envelope peak to "
                                        "20 dB below it"},
        "glide": {"transitions_examined": pairs, "glides": glides,
                  "share": (glides / pairs) if pairs else None,
                  "rule": f"consecutive bass notes at least "
                          f"{P['glide']['min_semitones']:.0f} semitones apart with "
                          f"under {P['glide']['max_gap_ms']:.0f} ms between them "
                          "and unbroken voicing across the join"},
        "inference": {
            "long_sub_note": eight,
            "basis": f"a median note decay of at least "
                     f"{P['eight_o_eight']['min_decay_ms']:.0f} ms with at least "
                     f"{P['eight_o_eight']['sub_share_min_pct']:.0f}% of the bass "
                     "stem's energy below the sub split; an 808 is the common "
                     "cause and a long synth sub is another",
            "sub_character": (None if richness is None else
                              ("sine-like" if richness < -12.0 else "harmonically rich")),
        },
    }


def _vocal_layers(src: AudioSource, collector: Collector) -> dict[str, Any]:
    """How many pitches sound at once in the vocal stem."""
    P = PARAMS["arrangement"]
    try:
        import librosa
    except ImportError:
        return {"available": False, "reason": "librosa not installed"}
    try:
        C = np.abs(librosa.cqt(src.lib_mono, sr=src.lib_sr, hop_length=512,
                               fmin=librosa.note_to_hz("C2"), n_bins=60,
                               bins_per_octave=12))
    except Exception as exc:
        collector.warn("stems.arrangement.vocal_layers", f"CQT failed: {exc!r}")
        return {"available": False, "reason": repr(exc)}
    freqs = librosa.cqt_frequencies(n_bins=60, fmin=librosa.note_to_hz("C2"),
                                    bins_per_octave=12)
    try:
        sal = librosa.salience(C, freqs=freqs, harmonics=[1, 2, 3, 4],
                               weights=[1.0, 0.5, 0.33, 0.25], fill_value=0.0)
    except Exception:
        sal = C
    sal_db = db_pow(np.maximum(sal ** 2, 1e-24))
    level = sal_db.max(axis=0)
    live = level > (np.percentile(level, 95) - 25.0)
    if int(np.sum(live)) < 20:
        return {"available": False, "reason": "too few sounding frames"}
    counts: list[int] = []
    floor = P["layer_salience_floor_db"]
    for j in np.flatnonzero(live):
        col = sal_db[:, j]
        pk, _ = sps.find_peaks(col, height=col.max() + floor, distance=2)
        counts.append(int(pk.size))
    arr = np.asarray(counts)
    return {
        "available": True,
        "source": "separated",
        "frames_examined": int(arr.size),
        "simultaneous_pitch_count": {
            "median": float(np.median(arr)), "p90": float(np.percentile(arr, 90)),
            "max": int(arr.max()), "mean": float(arr.mean())},
        "confidence": "low",
        "confidence_reason": "a harmonic-salience peak count is not a voice "
                             "count: one voice with strong harmonics and two "
                             "voices a fifth apart can produce the same number",
        "method": f"peaks in a harmonic-summed CQT salience within "
                  f"{abs(floor):.0f} dB of the frame maximum, over frames within "
                  "25 dB of the stem's 95th-percentile level",
    }


def _lead_vs_backing(src: AudioSource, sections: list[dict[str, Any]]
                     ) -> dict[str, Any]:
    """Centre versus sides on the vocal stem, whole track and per section."""
    if src.n_ch < 2:
        return {"available": False, "reason": "vocal stem is mono"}
    sr = src.band_sr
    mid, side = src.band_mid, src.band_side

    def share(a: np.ndarray, b: np.ndarray) -> float | None:
        pm, ps = float(np.mean(a * a)), float(np.mean(b * b))
        return (100.0 * pm / (pm + ps)) if (pm + ps) > 0 else None

    rows = []
    for s in sections or []:
        a, b = int(float(s["start_s"]) * sr), int(float(s["end_s"]) * sr)
        b = min(b, mid.size)
        if b - a < sr:
            continue
        rows.append({"section_index": s.get("index"), "start_s": s.get("start_s"),
                     "centre_energy_pct": share(mid[a:b], side[a:b])})
    return {
        "available": True, "source": "separated",
        "centre_energy_pct": share(mid, side),
        "per_section": rows,
        "definition": "mid power as a percentage of mid plus side power in the "
                      "vocal stem; a centred lead raises it, a spread stack of "
                      "backing vocals lowers it",
    }


def _adlibs_and_response(src: AudioSource, collector: Collector) -> dict[str, Any]:
    """Off-centre vocal events, and whether they answer the centred ones."""
    from .rhythm import fine_onsets

    if src.n_ch < 2:
        return {"available": False, "reason": "vocal stem is mono"}
    P = PARAMS["arrangement"]
    sr = src.band_sr
    mid, side = src.band_mid, src.band_side
    onsets = fine_onsets(src)
    win = int(round(0.2 * sr))
    off = 0
    total = 0
    for t in onsets:
        a = int(t * sr)
        m, s = mid[a:a + win], side[a:a + win]
        if m.size < win // 2:
            continue
        total += 1
        if float(np.mean(s * s)) > float(np.mean(m * m)):
            off += 1
    hop = max(1, int(round(0.02 * sr)))
    n = (mid.size // hop) * hop
    lag_lo, lag_hi = P["call_response_lag_s"]
    peak = None
    if n >= hop * 200:
        em = np.sqrt(np.mean(mid[:n].reshape(-1, hop) ** 2, axis=1))
        es = np.sqrt(np.mean(side[:n].reshape(-1, hop) ** 2, axis=1))
        em, es = em - em.mean(), es - es.mean()
        den = float(np.sqrt(np.dot(em, em) * np.dot(es, es)))
        if den > 0:
            lo = int(round(lag_lo / 0.02))
            hi = min(int(round(lag_hi / 0.02)), em.size - 1)
            best = None
            for lag in range(lo, hi + 1):
                r = float(np.dot(em[:-lag], es[lag:]) / den)
                if best is None or r > best[1]:
                    best = (lag, r)
            if best:
                peak = {"lag_s": best[0] * 0.02, "correlation": best[1]}
    return {
        "available": True, "source": "separated",
        "vocal_onsets": total,
        "side_dominant_onsets": off,
        "side_dominant_share": (off / total) if total else None,
        "side_dominant_definition": "a vocal onset whose next 200 ms carries more "
                                    "side power than mid power; ad-libs and "
                                    "hard-panned doubles are the usual cause",
        "call_and_response": peak,
        "call_and_response_definition": "the lag between "
                                        f"{lag_lo} and {lag_hi} s at which the "
                                        "side envelope best follows the mid "
                                        "envelope, and how strongly",
    }


def _tagger(collector: Collector) -> dict[str, Any]:
    """A trained instrument tagger, if one is installed."""
    for module, name in (("panns_inference", "PANNs"),
                         ("essentia.standard", "Essentia MTG")):
        try:
            __import__(module)
        except Exception:
            continue
        return {"available": False, "backend_found": name,
                "reason": f"{name} is importable but mtx does not ship a model "
                          "checkpoint or a tag vocabulary for it; instrument "
                          "tagging stays unavailable rather than guessing",
                "note": "the stem names below are the tool's own answer to "
                        "'what is playing', at the resolution the separation "
                        "model provides"}
    return {
        "available": False,
        "reason": "no trained tagger is installed",
        "install": ["pip install panns-inference", "pip install essentia"],
        "note": "for more instruments without a tagger, run with "
                "--stems-model htdemucs_6s, which splits guitar and piano out "
                "of `other` at no new dependency",
    }


def analyse(stems: dict[str, AudioSource], sections: list[dict[str, Any]],
            rhythm: dict[str, Any] | None, melody: dict[str, Any] | None,
            collector: Collector) -> dict[str, Any]:
    P = PARAMS["arrangement"]
    hop_s = float(P["density_hop_s"])
    times, envs, thr = _presence(stems, hop_s, float(P["presence_db_below_stem_p95"]))
    if not envs:
        return {"available": False, "reason": "no stem produced an envelope"}

    present = {k: (envs[k] >= thr[k]) for k in envs}
    density = np.sum(np.vstack([present[k] for k in sorted(present)]), axis=0)

    entries: list[dict[str, Any]] = []
    downbeats = None
    if rhythm and (rhythm.get("downbeats") or {}).get("available"):
        downbeats = np.asarray(rhythm["downbeats"]["downbeat_times_s"], dtype=float)
    for name in sorted(present):
        on = np.flatnonzero(present[name])
        if on.size == 0:
            entries.append({"stem": name, "first_present_s": None,
                            "last_present_s": None, "present_time_pct": 0.0})
            continue
        first = float(times[on[0]]) if on[0] < times.size else None
        last = float(times[on[-1]]) if on[-1] < times.size else None
        row: dict[str, Any] = {
            "stem": name, "first_present_s": first,
            "first_present": fmt_time(first) if first is not None else None,
            "last_present_s": last,
            "present_time_pct": 100.0 * on.size / present[name].size,
        }
        if downbeats is not None and first is not None and downbeats.size:
            row["first_present_bar"] = int(np.searchsorted(downbeats, first))
        entries.append(row)

    per_section: list[dict[str, Any]] = []
    for s in sections or []:
        t0, t1 = float(s["start_s"]), float(s["end_s"])
        m = (times >= t0) & (times < t1)
        if not np.any(m):
            continue
        row = {"section_index": s.get("index"), "start_s": t0,
               "start": fmt_time(t0),
               "mean_concurrent_sources": float(np.mean(density[m])),
               "stems_present": [k for k in sorted(present)
                                 if float(np.mean(present[k][m])) > 0.5]}
        per_section.append(row)

    out: dict[str, Any] = {
        "available": True,
        "source": "separated",
        "stems": sorted(stems),
        "presence_rule": f"a stem counts as present in a "
                         f"{hop_s * 1000:.0f} ms frame when its level is within "
                         f"{P['presence_db_below_stem_p95']} dB of its own "
                         "95th-percentile level",
        "density": {
            "hop_s": hop_s,
            "times_s": times,
            "concurrent_sources": density,
            "mean": float(np.mean(density)) if density.size else None,
            "max": int(density.max()) if density.size else None,
        },
        "entry_exit": entries,
        "per_section": per_section,
        "instrument_tagging": _tagger(collector),
    }
    if "drums" in stems:
        out["drums"] = _drum_character(stems["drums"], collector)
    if "bass" in stems:
        out["bass"] = _bass_character(stems["bass"],
                                      (melody or {}).get("_bass_notes"),
                                      (melody or {}).get("_bass_track"))
    if "vocals" in stems:
        out["vocals"] = {
            "layers": _vocal_layers(stems["vocals"], collector),
            "lead_vs_backing": _lead_vs_backing(stems["vocals"], sections),
            "adlibs": _adlibs_and_response(stems["vocals"], collector),
        }
    return out
