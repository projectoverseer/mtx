"""4.9 Optional stem separation (--stems).

demucs is heavy (it pulls in torch) and slow on CPU, so it is entirely
optional: without it the tool loses only this section.  Every number derived
from a stem carries source="separated", because separation artefacts are real
and a stem measurement is not a mix measurement.

Separation is also the gate on the musical half of the tool.  Once it has run
and been cached, pitch (`melody`), inter-stem masking (`masking`) and
arrangement (`arrangement`) are ordinary DSP over signals that are already on
disk, so all three are computed here from one load of the stems rather than
each re-decoding them.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

from ..audio import AudioSource
from ..params import PARAMS
from ..util import Collector, db_amp

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "mtx", "stems")

# What each demucs model separates into.  The six-source model is the cheapest
# way to stop calling a guitar `other`: same dependency, same runtime order,
# two more stems.
STEM_SETS: dict[str, tuple[str, ...]] = {
    "htdemucs": ("drums", "bass", "other", "vocals"),
    "htdemucs_ft": ("drums", "bass", "other", "vocals"),
    "hdemucs_mmi": ("drums", "bass", "other", "vocals"),
    "mdx_extra": ("drums", "bass", "other", "vocals"),
    "htdemucs_6s": ("drums", "bass", "other", "vocals", "guitar", "piano"),
}
STEMS = STEM_SETS["htdemucs"]


def stem_names(model: str) -> tuple[str, ...]:
    return STEM_SETS.get(model, STEM_SETS["htdemucs"])


def _cache_key(path: str) -> str:
    st = os.stat(path)
    h = hashlib.sha256()
    h.update(os.path.abspath(path).encode("utf-8"))
    h.update(str(st.st_size).encode("ascii"))
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
    return h.hexdigest()[:24]


def separate(path: str, collector: Collector,
             model: str | None = None) -> dict[str, str] | None:
    """Run demucs and return {stem_name: wav_path}.  Cached across runs."""
    key = _cache_key(path)
    out_root = os.path.join(CACHE_DIR, key)
    model = model or PARAMS["stems"]["model"]
    stem_dir = os.path.join(out_root, model,
                            os.path.splitext(os.path.basename(path))[0])
    have = {s: os.path.join(stem_dir, f"{s}.wav") for s in stem_names(model)}
    if all(os.path.isfile(p) for p in have.values()):
        return have
    if shutil.which("demucs") is None and importlib.util.find_spec("demucs") is None:
        collector.warn("stems", "demucs is not installed; install the 'stems' "
                                "extra (pip install 'mtx[stems]') to use --stems")
        return None
    os.makedirs(out_root, exist_ok=True)
    cmd = [sys.executable, "-m", "demucs", "-n", model, "-o", out_root, path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=7200)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        collector.warn("stems", f"demucs failed to run: {exc!r}")
        return None
    if proc.returncode != 0:
        collector.warn("stems", f"demucs exited {proc.returncode}: "
                                f"{(proc.stderr or '')[-400:]}")
        return None
    if not all(os.path.isfile(p) for p in have.values()):
        collector.warn("stems", f"demucs produced no stems under {stem_dir}")
        return None
    return have


def load(src: AudioSource, collector: Collector,
         model: str | None = None) -> dict[str, AudioSource] | None:
    """Separate if needed, then decode every stem once for the whole run."""
    paths = separate(src.path, collector, model)
    if paths is None:
        return None
    out: dict[str, AudioSource] = {}
    for name, p in paths.items():
        try:
            out[name] = AudioSource(p, Collector())
        except Exception as exc:
            collector.warn("stems", f"could not decode the {name} stem: {exc!r}")
    return out or None


def analyse(src: AudioSource, collector: Collector, profile: str = "full",
            sources: dict[str, AudioSource] | None = None,
            structure: dict[str, Any] | None = None,
            rhythm: dict[str, Any] | None = None,
            model: str | None = None) -> dict[str, Any]:
    from . import (arrangement as m_arrangement, dynamics as m_dynamics,
                   loudness as m_loudness, masking as m_masking,
                   melody as m_melody, rhythm as m_rhythm,
                   spectrum as m_spectrum, stereo as m_stereo)

    if sources is None:
        sources = load(src, collector, model)
    if not sources:
        return {"requested": True, "available": False,
                "reason": "stem separation unavailable; see warnings"}
    model = model or PARAMS["stems"]["model"]
    mix_rms = float(np.sqrt(np.mean(src.mono ** 2)))
    from ..dsp import block_loudness, gated_integrated
    _, bl = block_loudness(src.x.astype(np.float64), src.sr, 0.4, 0.1)
    mix_lufs, _ = gated_integrated(bl)

    out: dict[str, Any] = {
        "requested": True, "available": True,
        "model": model,
        "stem_names": list(sorted(sources)),
        "cache_dir": CACHE_DIR,
        "source": "separated",
        "caveat": "every number below is measured on a separated stem, not on the "
                  "mix; separation artefacts are part of the measurement",
        "stems": {},
    }
    for name in sorted(sources):
        s = sources[name]
        sub = Collector()
        loud = m_loudness.analyse(s, sub)
        entry = {
            "source": "separated",
            "path": s.path,
            "loudness": loud,
            "dynamics": m_dynamics.analyse(s, sub, profile),
            "spectrum": m_spectrum.analyse(s, sub, profile),
            "stereo": m_stereo.analyse(s, sub, profile),
            "level_vs_mix": {
                "rms_db": db_amp(float(np.sqrt(np.mean(s.mono ** 2)))) - db_amp(mix_rms)
                if mix_rms > 0 else None,
                "lufs_delta": (loud.get("integrated_lufs") - mix_lufs)
                if (loud.get("integrated_lufs") is not None and mix_lufs is not None)
                else None,
            },
            "warnings": sub.warnings,
        }
        out["stems"][name] = entry

    sections = ((structure or {}).get("sections") or [])
    tempo = (structure or {}).get("tempo") or {}
    key = (structure or {}).get("key") or {}

    # The measurements that only exist once there is more than one signal.
    out["masking"] = m_masking.analyse(src, sources, sections, tempo, collector)

    melody = m_melody.analyse(sources, sections, key, collector, profile)
    out["arrangement"] = m_arrangement.analyse(sources, sections, rhythm, melody,
                                               collector)
    # The raw pitch tracks are working state for the arrangement pass, not
    # output: they are one float per 11.6 ms and say nothing a note list does not.
    for k in ("_vocal_notes", "_vocal_track", "_bass_notes", "_bass_track"):
        melody.pop(k, None)
    out["melody"] = melody

    beats = tempo.get("beat_times_s")
    if beats is not None and np.asarray(beats).size >= 8:
        out["microtiming"] = m_rhythm.stem_microtiming(
            sources, np.asarray(beats, dtype=float),
            int(PARAMS["rhythm"]["grid_subdivision"]))
    else:
        out["microtiming"] = {"available": False,
                              "reason": "no beat grid from structure.tempo"}
    return out
