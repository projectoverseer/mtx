"""4.9 Optional stem separation (--stems).

demucs is heavy (it pulls in torch) and slow on CPU, so it is entirely
optional: without it the tool loses only this section.  Every number derived
from a stem carries source="separated", because separation artefacts are real
and a stem measurement is not a mix measurement.
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
STEMS = ("drums", "bass", "other", "vocals")


def _cache_key(path: str) -> str:
    st = os.stat(path)
    h = hashlib.sha256()
    h.update(os.path.abspath(path).encode("utf-8"))
    h.update(str(st.st_size).encode("ascii"))
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
    return h.hexdigest()[:24]


def separate(path: str, collector: Collector) -> dict[str, str] | None:
    """Run htdemucs and return {stem_name: wav_path}.  Cached across runs."""
    key = _cache_key(path)
    out_root = os.path.join(CACHE_DIR, key)
    model = PARAMS["stems"]["model"]
    stem_dir = os.path.join(out_root, model,
                            os.path.splitext(os.path.basename(path))[0])
    have = {s: os.path.join(stem_dir, f"{s}.wav") for s in STEMS}
    if all(os.path.isfile(p) for p in have.values()):
        return have
    if shutil.which("demucs") is None and importlib.util.find_spec("demucs") is None:
        collector.warn("stems", "demucs is not installed; install the 'stems' "
                                "extra (pip install 'mtx[stems]') to use --stems")
        return None
    os.makedirs(out_root, exist_ok=True)
    cmd = [sys.executable, "-m", "demucs", "-n", model, "-o", out_root, path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
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


def analyse(src: AudioSource, collector: Collector, profile: str = "full") -> dict[str, Any]:
    from . import dynamics as m_dynamics, loudness as m_loudness, \
        spectrum as m_spectrum, stereo as m_stereo

    paths = separate(src.path, collector)
    if paths is None:
        return {"requested": True, "available": False,
                "reason": "stem separation unavailable; see warnings"}
    mix_rms = float(np.sqrt(np.mean(src.mono ** 2)))
    mix_lufs = None
    from ..dsp import block_loudness, gated_integrated
    _, bl = block_loudness(src.x.astype(np.float64), src.sr, 0.4, 0.1)
    mix_lufs, _ = gated_integrated(bl)

    out: dict[str, Any] = {
        "requested": True, "available": True,
        "model": PARAMS["stems"]["model"],
        "cache_dir": CACHE_DIR,
        "source": "separated",
        "caveat": "every number below is measured on a separated stem, not on the "
                  "mix; separation artefacts are part of the measurement",
        "stems": {},
    }
    for name, p in paths.items():
        sub = Collector()
        s = AudioSource(p, sub)
        loud = m_loudness.analyse(s, sub)
        entry = {
            "source": "separated",
            "path": p,
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
    return out
