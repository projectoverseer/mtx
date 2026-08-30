"""4.7d Song form: letters over the measured boundaries, then function names.

`structure.sections` already produces excellent raw material -- novelty-derived
boundaries with a full measured vector each -- and it produces it
reproducibly.  What it cannot do is say which of them is the chorus, and
without that, none of the questions people actually ask a record can be
computed: time to the first chorus, chorus share, whether the second chorus is
arranged up from the first.

This module is a labelling pass, never a replacement.  It works in two clearly
separated stages:

* **Measured.**  Sections are clustered by similarity into letters -- A, B, C
  in order of first appearance.  That is a distance calculation over vectors
  the tool already computed, it is parameterised, and it is reproducible.
* **Inferred.**  Function names (verse, chorus, bridge) are assigned from
  measured evidence by a rule set that is written down in `params.form` and
  echoed with every label.  Each label carries the evidence that produced it
  and a confidence.

If an `allin1` model is installed its segmentation is reported alongside, as a
second opinion with its own provenance -- never merged into the measured
boundaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..audio import AudioSource
from ..params import PARAMS
from ..util import Collector, db_amp, fmt_time

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _section_features(src: AudioSource, sections: list[dict[str, Any]],
                      collector: Collector) -> np.ndarray | None:
    """One z-scored vector per section: timbre, harmony and the measured scalars."""
    try:
        import librosa
    except ImportError:
        collector.warn("form", "librosa not installed; sections cannot be clustered")
        return None
    hop = PARAMS["general"]["librosa_hop_length"]
    n_fft = PARAMS["general"]["librosa_n_fft"]
    y, sr = src.lib_mono, src.lib_sr
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S ** 2), n_mfcc=20)
    chroma = np.asarray(src.chroma_cqt())
    times = librosa.frames_to_time(np.arange(mfcc.shape[1]), sr=sr, hop_length=hop)
    ct = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)

    rows: list[np.ndarray] = []
    for s in sections:
        t0, t1 = float(s["start_s"]), float(s["end_s"])
        m = (times >= t0) & (times < t1)
        cm = (ct >= t0) & (ct < t1)
        v_mfcc = mfcc[:, m].mean(axis=1) if np.any(m) else np.zeros(mfcc.shape[0])
        v_chroma = chroma[:, cm].mean(axis=1) if np.any(cm) else np.zeros(12)
        bands = s.get("band_energy_pct") or {}
        scalars = [
            s.get("lufs_i"), s.get("crest_db"), s.get("tilt_db_per_oct"),
            s.get("side_minus_mid_db"), s.get("onset_rate_per_s"),
        ] + [bands.get(k) for k in
             ("sub", "bass", "low_bass", "low_mid", "mid", "high_mid",
              "presence", "air")]
        scalars = [0.0 if v is None else float(v) for v in scalars]
        rows.append(np.concatenate([v_mfcc, v_chroma, np.asarray(scalars)]))
    X = np.vstack(rows)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def _cluster(X: np.ndarray, threshold: float) -> list[int]:
    """Average-linkage agglomerative clustering under a cosine cut-off."""
    n = X.shape[0]
    norms = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    U = X / norms
    D = 1.0 - U @ U.T
    np.fill_diagonal(D, np.inf)
    groups: list[list[int]] = [[i] for i in range(n)]
    while len(groups) > 1:
        best = None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                d = float(np.mean(D[np.ix_(groups[a], groups[b])]))
                if best is None or d < best[0]:
                    best = (d, a, b)
        if best is None or best[0] > threshold:
            break
        _, a, b = best
        groups[a] = groups[a] + groups[b]
        groups.pop(b)
    label = [0] * n
    order = sorted(range(len(groups)), key=lambda g: min(groups[g]))
    for rank, g in enumerate(order):
        for i in groups[g]:
            label[i] = rank
    return label


def _vocal_presence(stems: dict[str, AudioSource] | None,
                    sections: list[dict[str, Any]]) -> dict[str, Any]:
    """Is the voice sounding in this section?  Measured, when a stem exists."""
    if not stems or "vocals" not in stems:
        return {"available": False,
                "reason": "no vocals stem; run with --stems for vocal presence"}
    v = stems["vocals"]
    sr = v.sr
    levels: list[float | None] = []
    for s in sections:
        a, b = int(float(s["start_s"]) * sr), int(float(s["end_s"]) * sr)
        seg = v.mono[a:min(b, v.n_frames)]
        levels.append(db_amp(float(np.sqrt(np.mean(seg * seg)))) if seg.size else None)
    good = [x for x in levels if x is not None]
    if not good:
        return {"available": False, "reason": "vocals stem is silent"}
    ref = float(np.percentile(good, 95))
    margin = PARAMS["form"]["vocal_presence_db_below_p95"]
    return {
        "available": True, "source": "separated",
        "reference_db": ref, "margin_db": margin,
        "per_section_db": levels,
        "present": [(None if x is None else bool(x >= ref - margin)) for x in levels],
        "rule": f"a section counts as having a vocal when the vocal stem's RMS "
                f"in it is within {margin} dB of the 95th percentile of the "
                f"per-section vocal RMS of this track",
    }


def _vocal_entry(stems: dict[str, AudioSource] | None) -> dict[str, Any]:
    if not stems or "vocals" not in stems:
        return {"available": False, "reason": "no vocals stem"}
    v = stems["vocals"]
    sr = v.band_sr
    hop = max(1, int(round(0.05 * sr)))
    x = v.band_mono
    n = (x.size // hop) * hop
    if n < hop * 4:
        return {"available": False, "reason": "vocals stem too short"}
    env = np.sqrt(np.mean(x[:n].reshape(-1, hop) ** 2, axis=1))
    edb = db_amp(np.maximum(env, 1e-20))
    thr = float(np.percentile(edb, 95)) - 20.0
    hits = edb >= thr
    need = 4                      # 200 ms of continuous voice
    run = 0
    for i, h in enumerate(hits):
        run = run + 1 if h else 0
        if run >= need:
            t = (i - need + 1) * 0.05
            return {"available": True, "source": "separated", "time_s": t,
                    "time": fmt_time(t), "threshold_db": thr,
                    "rule": "first 200 ms of continuous vocal-stem level within "
                            "20 dB of its own 95th percentile"}
    return {"available": False, "reason": "no sustained vocal level found"}


def _parts(sections: list[dict[str, Any]], letters: list[int],
           vocal: list[bool | None]) -> list[dict[str, Any]]:
    """Merge runs of consecutive identical letters into form parts.

    A novelty boundary fires wherever the texture turns, so one chorus can
    arrive as three sections.  Function labels belong to the part, not to the
    segment: without this merge a chorus that happens to be segmented finely
    would be counted three times and its share of the track would be right for
    the wrong reason.
    """
    parts: list[dict[str, Any]] = []
    for i, L in enumerate(letters):
        if parts and parts[-1]["letter_id"] == L and parts[-1]["last_index"] == i - 1:
            parts[-1]["last_index"] = i
            parts[-1]["members"].append(i)
        else:
            parts.append({"letter_id": L, "first_index": i, "last_index": i,
                          "members": [i]})
    for p in parts:
        members = p["members"]
        p["letter"] = LETTERS[p["letter_id"] % 26]
        p["start_s"] = float(sections[members[0]]["start_s"])
        p["end_s"] = float(sections[members[-1]]["end_s"])
        p["duration_s"] = p["end_s"] - p["start_s"]
        lufs = [sections[i].get("lufs_i") for i in members
                if sections[i].get("lufs_i") is not None]
        p["lufs_i"] = float(np.mean(lufs)) if lufs else None
        seen = [vocal[i] for i in members if vocal[i] is not None]
        p["vocal_present"] = (bool(np.mean(seen) >= 0.5) if seen else None)
    return parts


def _label(sections: list[dict[str, Any]], parts: list[dict[str, Any]],
           collector: Collector) -> None:
    """Assign a function name to each part from measured evidence."""
    n = len(parts)
    lufs = [p.get("lufs_i") for p in parts]
    known = [x for x in lufs if x is not None]
    median_lufs = float(np.median(known)) if known else None
    durs = [float(p["duration_s"]) for p in parts]
    median_dur = float(np.median(durs)) if durs else 0.0
    vocal = [p.get("vocal_present") for p in parts]

    by_letter: dict[int, list[int]] = {}
    for i, p in enumerate(parts):
        by_letter.setdefault(p["letter_id"], []).append(i)

    def mean_lufs(idxs: list[int]) -> float:
        vals = [lufs[i] for i in idxs if lufs[i] is not None]
        return float(np.mean(vals)) if vals else -np.inf

    def vocal_share(idxs: list[int]) -> float:
        vals = [vocal[i] for i in idxs if vocal[i] is not None]
        return float(np.mean(vals)) if vals else 0.5

    repeated = {L: idxs for L, idxs in by_letter.items() if len(idxs) >= 2}
    chorus_letter = None
    if repeated:
        # The chorus is the letter that repeats, carries a vocal and is loud.
        # Loudness alone would pick a drop; repetition alone would pick a verse.
        chorus_letter = max(
            repeated,
            key=lambda L: (mean_lufs(repeated[L]) + 6.0 * vocal_share(repeated[L]),
                           len(repeated[L])))
    verse_letter = None
    if chorus_letter is not None:
        chorus_idx = set(by_letter[chorus_letter])
        best = None
        for L, idxs in repeated.items():
            if L == chorus_letter:
                continue
            leads = sum(1 for i in idxs if (i + 1) in chorus_idx)
            score = (leads, vocal_share(idxs), -mean_lufs(idxs))
            if best is None or score > best[0]:
                best = (score, L)
        if best is not None:
            verse_letter = best[1]

    chorus_idx = set(by_letter.get(chorus_letter, [])) if chorus_letter is not None else set()
    verse_idx = set(by_letter.get(verse_letter, [])) if verse_letter is not None else set()

    for i, p in enumerate(parts):
        ev: list[str] = [f"letter {p['letter']}",
                         f"repeats {len(by_letter[p['letter_id']])}"]
        if vocal[i] is not None:
            ev.append("vocal present" if vocal[i] else "no vocal")
        label = "unlabelled"
        conf = "low"
        if i == 0 and (median_lufs is None or
                       (lufs[i] is not None and lufs[i] < median_lufs)):
            label, conf = "intro", "medium"
            ev.append("first part, quieter than the track median")
        elif i == n - 1 and (vocal[i] is not True or
                             (lufs[i] is not None and median_lufs is not None
                              and lufs[i] < median_lufs)):
            label, conf = "outro", "medium"
            ev.append("last part, no louder than the track median")
        elif i in chorus_idx:
            label, conf = "chorus", "medium"
            ev.append("loudest repeated letter carrying a vocal")
        elif i in verse_idx:
            label, conf = "verse", "medium"
            ev.append("repeated letter that leads into the chorus")
        elif (i + 1) in chorus_idx and durs[i] < median_dur:
            label, conf = "pre-chorus", "low"
            ev.append("short part immediately before a chorus")
        elif (i - 1) in chorus_idx and len(by_letter[p["letter_id"]]) >= 2:
            label, conf = "post-chorus", "low"
            ev.append("repeated part immediately after a chorus")
        elif vocal[i] is False:
            hot = (lufs[i] is not None and median_lufs is not None
                   and lufs[i] > median_lufs)
            label, conf = ("drop" if hot else "instrumental"), "low"
            ev.append("no vocal, above the track median loudness" if hot
                      else "no vocal")
        elif len(by_letter[p["letter_id"]]) == 1 and i > n // 2:
            label, conf = "bridge", "low"
            ev.append("unrepeated part in the second half")
        p["label"] = label
        p["label_confidence"] = conf
        p["label_evidence"] = ev
        for j in p["members"]:
            sections[j]["letter"] = p["letter"]
            sections[j]["label"] = label
            sections[j]["label_confidence"] = conf
            sections[j]["label_evidence"] = ev
            sections[j]["part_index"] = i

    if chorus_letter is None:
        collector.low_confidence("form.labels", "low",
                                 "no part letter repeats, so no chorus could be "
                                 "identified and every measurement downstream "
                                 "of it is unavailable")


def _loopability(src: AudioSource, X: np.ndarray | None,
                 sections: list[dict[str, Any]]) -> dict[str, Any]:
    """How much the last seconds resemble the first."""
    w = PARAMS["form"]["loopability_window_s"]
    if src.duration < w * 3:
        return {"available": False, "reason": "track shorter than three windows"}
    sr = src.band_sr
    a = src.band_mono[:int(w * sr)]
    b = src.band_mono[-int(w * sr):]
    from ..dsp import welch_psd
    fa, pa = welch_psd(a, sr, 8192)
    fb, pb = welch_psd(b, sr, 8192)
    cos = None
    if fa.size and fb.size and fa.size == fb.size:
        na, nb = np.linalg.norm(pa), np.linalg.norm(pb)
        if na > 0 and nb > 0:
            cos = float(np.dot(pa, pb) / (na * nb))
    ra = db_amp(float(np.sqrt(np.mean(a * a)))) if a.size else None
    rb = db_amp(float(np.sqrt(np.mean(b * b)))) if b.size else None
    return {
        "available": True, "window_s": w,
        "spectral_cosine": cos,
        "level_delta_db": (rb - ra) if (ra is not None and rb is not None) else None,
        "definition": "cosine between the Welch spectra of the first and last "
                      f"{w:.0f} s, and the level difference between them; a loop "
                      "that closes has a high cosine and a small delta",
    }


def _ending(forensics: dict[str, Any], sections: list[dict[str, Any]]
            ) -> dict[str, Any]:
    sil = (forensics or {}).get("silence") or {}
    kind = None
    basis = []
    end_kind = sil.get("end_kind") if isinstance(sil, dict) else None
    if end_kind:
        kind = "fade" if str(end_kind).lower() == "fade" else (
            "cold stop" if str(end_kind).lower() == "hard cut" else str(end_kind))
        basis.append(f"forensics.silence.end_kind = {end_kind}")
        if sil.get("fade_out_ms") is not None:
            basis.append(f"fade_out_ms = {sil['fade_out_ms']:.0f}")
    last = sections[-1] if sections else {}
    if last.get("delta_vs_previous_lufs") is not None:
        basis.append(f"last section is {last['delta_vs_previous_lufs']:+.1f} LU "
                     "against the one before it")
    return {"type": kind, "basis": basis,
            "note": "the fade/hard-cut distinction is measured in "
                    "forensics.silence; this field only names it"}


def _allin1(src: AudioSource, collector: Collector) -> dict[str, Any]:
    """A learned segmentation, if one is installed.  Never merged, only reported."""
    try:
        import allin1  # type: ignore
    except Exception:
        return {"available": False,
                "reason": "the optional `allin1` model is not installed; the "
                          "measured letters and the rule-based labels above are "
                          "the tool's own answer",
                "install": "pip install allin1"}
    try:
        result = allin1.analyze(src.path, keep_byproducts=False)
    except Exception as exc:
        collector.warn("form.allin1", f"allin1 failed: {exc!r}")
        return {"available": False, "reason": repr(exc)}
    segs = [{"start_s": float(s.start), "end_s": float(s.end), "label": str(s.label)}
            for s in getattr(result, "segments", [])]
    return {
        "available": True,
        "source": "model",
        "model": "allin1",
        "version": getattr(allin1, "__version__", "unknown"),
        "segments": segs,
        "downbeats_s": [float(x) for x in getattr(result, "downbeats", [])],
        "caveat": "a learned model's opinion, reported next to the measured "
                  "boundaries and never substituted for them",
    }


def analyse(src: AudioSource, structure: dict[str, Any],
            rhythm: dict[str, Any] | None, forensics: dict[str, Any] | None,
            stems: dict[str, AudioSource] | None, collector: Collector,
            profile: str = "full") -> dict[str, Any]:
    if profile == "quick":
        return {"available": False, "reason": "skipped by --profile quick"}
    sections = [dict(s) for s in ((structure or {}).get("sections") or [])]
    if len(sections) < 3:
        return {"available": False,
                "reason": "fewer than three measured sections to label"}
    P = PARAMS["form"]
    X = _section_features(src, sections, collector)
    if X is None:
        return {"available": False, "reason": "section features unavailable"}
    letters = _cluster(X, float(P["cluster"]["merge_threshold"]))
    vocal_block = _vocal_presence(stems, sections)
    vocal = (list(vocal_block.get("present")) if vocal_block.get("available")
             else [None] * len(sections))
    for i, s in enumerate(sections):
        s["vocal_present"] = vocal[i]
    parts = _parts(sections, letters, vocal)
    _label(sections, parts, collector)

    letter_string = "".join(p["letter"] for p in parts)
    labels = [p["label"] for p in parts]
    total = float(src.duration) or 1.0

    chorus = [p for p in parts if p["label"] == "chorus"]
    chorus_time = sum(float(p["duration_s"]) for p in chorus)
    first_chorus = chorus[0] if chorus else None
    intro = [p for p in parts if p["label"] == "intro"]

    second_vs_first = None
    if len(chorus) >= 2:
        # Compare the first section of each chorus part: the measured section
        # vector is what structure.sections already produced, and taking the
        # opening of each keeps the two comparable in length.
        a = sections[chorus[0]["members"][0]]
        b = sections[chorus[1]["members"][0]]
        keys = ("lufs_i", "crest_db", "tilt_db_per_oct", "side_minus_mid_db",
                "onset_rate_per_s", "shortterm_max_lufs")
        second_vs_first = {
            k: ((b.get(k) - a.get(k)) if (a.get(k) is not None and b.get(k) is not None)
                else None) for k in keys}
        ba, bb = a.get("band_energy_pct") or {}, b.get("band_energy_pct") or {}
        second_vs_first["band_energy_pct_delta"] = {
            k: ((bb.get(k) - ba.get(k)) if (ba.get(k) is not None and bb.get(k) is not None)
                else None) for k in ba}
        second_vs_first["note"] = ("differences between the second chorus and "
                                   "the first, from the section vectors that "
                                   "structure.sections already measured")

    entry = _vocal_entry(stems)
    switches = ((rhythm or {}).get("pulse_rate") or {}).get("switch_count")

    return {
        "available": True,
        "section_count": len(sections),
        "part_count": len(parts),
        "letters": letter_string,
        "letters_per_section": "".join(s["letter"] for s in sections),
        "letter_method": P["cluster"],
        "part_rule": "consecutive sections with the same letter are one part; "
                     "function labels belong to parts, because a novelty "
                     "boundary fires wherever the texture turns and one chorus "
                     "can arrive as three sections",
        "labels": labels,
        "label_method": P["rule_note"],
        "parts": [{k: p.get(k) for k in
                   ("letter", "label", "label_confidence", "label_evidence",
                    "start_s", "end_s", "duration_s", "lufs_i",
                    "vocal_present", "members")} for p in parts],
        "sections": [{k: s.get(k) for k in
                      ("index", "start_s", "start", "end_s", "duration_s",
                       "letter", "label", "label_confidence", "label_evidence",
                       "vocal_present", "part_index", "lufs_i")} for s in sections],
        "vocal_presence": vocal_block,
        "vocal_entry": entry,
        "chorus_count": len(chorus),
        "chorus_share_pct": 100.0 * chorus_time / total,
        "time_to_first_chorus_s": float(first_chorus["start_s"]) if first_chorus else None,
        "time_to_first_chorus": fmt_time(first_chorus["start_s"]) if first_chorus else None,
        "time_to_first_chorus_fraction": (float(first_chorus["start_s"]) / total
                                          if first_chorus else None),
        "intro_length_s": float(intro[0]["duration_s"]) if intro else None,
        "time_to_vocal_entry_s": entry.get("time_s") if entry.get("available") else None,
        "time_to_title_s": None,
        "time_to_title_note": "needs a time-aligned lyric; see lyrics.alignment",
        "second_chorus_vs_first": second_vs_first,
        "beat_switch_count": switches,
        "ending": _ending(forensics or {}, sections),
        "loopability": _loopability(src, X, sections),
        "learned_segmentation": _allin1(src, collector) if P.get("use_allin1") else {
            "available": False,
            "reason": "not requested; set params.form.use_allin1 to compare "
                      "against the optional allin1 model"},
    }
