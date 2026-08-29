"""Orchestration: run every metric group over one file and assemble the JSON."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

from . import SCHEMA_VERSION, __version__
from .audio import AudioSource
from .metrics import (dynamics as m_dynamics, fileinfo as m_fileinfo,
                      forensics as m_forensics, loudness as m_loudness,
                      processing as m_processing, spectrum as m_spectrum,
                      stereo as m_stereo, structure as m_structure)
from .params import PARAMS, profile_params
from .util import Collector, jsonable

SEED = 0


def _versions() -> dict[str, Any]:
    out: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "mtx": __version__,
    }
    for mod in ("numpy", "scipy", "soundfile", "mutagen", "pyloudnorm",
                "librosa", "numba", "matplotlib", "sklearn"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            out[mod] = None
    try:
        import soundfile as sf
        out["libsndfile"] = sf.__libsndfile_version__
    except Exception:
        out["libsndfile"] = None
    for tool in ("ffmpeg", "ffprobe"):
        try:
            p = subprocess.run([tool, "-version"], capture_output=True, text=True,
                               timeout=30)
            out[tool] = p.stdout.splitlines()[0] if p.stdout else None
        except Exception:
            out[tool] = None
    return out


def _headline(res: dict[str, Any]) -> dict[str, Any]:
    """The fixed set of numbers the digest table is built from."""
    L = res.get("loudness", {})
    D = res.get("dynamics", {})
    S = res.get("spectrum", {})
    ST = res.get("stereo", {})
    F = res.get("forensics", {})
    STR = res.get("structure", {})
    tp = L.get("true_peak", {})
    psr = L.get("psr", {})
    ft = D.get("flat_top", {})
    corr = ST.get("correlation", {}) if ST.get("available") else {}
    tilt = S.get("tilt", {}) if S.get("available") else {}
    bit = F.get("effective_bit_depth", {}) if F.get("available") else {}
    cut = F.get("hf_cutoff", {}) if F.get("available") else {}
    tempo = STR.get("tempo", {}) if STR.get("available") else {}
    key = STR.get("key", {}) if STR.get("available") else {}
    crest = D.get("crest", {})
    return {
        "lufs_i": L.get("integrated_lufs"),
        "lra_lu": L.get("lra_lu"),
        "true_peak_dbtp_16x": tp.get("overall_dbtp_16x"),
        "sample_peak_dbfs": L.get("sample_peak", {}).get("overall_dbfs"),
        "plr_db": L.get("plr_db"),
        "psr_min_db": psr.get("min_db"),
        "psr_min_time": psr.get("min_time"),
        "psr_median_db": psr.get("median_db"),
        "dr14": L.get("dr14", {}).get("dr"),
        "crest_whole_db": crest.get("whole_file_db"),
        "crest_loudest_10s_db": crest.get("loudest_window", {}).get("crest_db"),
        "spectral_tilt_db_per_oct": tilt.get("slope_db_per_oct"),
        "spectral_tilt_r2": tilt.get("r2"),
        "air_band_pct": S.get("air_band_pct") if S.get("available") else None,
        "sub_band_pct": S.get("sub_band_pct") if S.get("available") else None,
        "side_minus_mid_db": ST.get("side_minus_mid_db") if ST.get("available") else None,
        "side_minus_mid_below_120hz_db": ST.get("side_minus_mid_below_120hz_db") if ST.get("available") else None,
        "mono_crossover_hz": ST.get("mono_crossover_hz") if ST.get("available") else None,
        "correlation_mean": corr.get("overall"),
        "correlation_min": corr.get("min"),
        "flat_top_sample_count": ft.get("total_flat_samples"),
        "flat_top_longest_run_ms": ft.get("longest_run_ms"),
        "hf_cutoff_hz": cut.get("cutoff_hz"),
        "effective_bit_depth": bit.get("effective_bits"),
        "tempo_bpm": tempo.get("bpm"),
        "key": key.get("key"),
        "section_count": STR.get("section_count") if STR.get("available") else None,
        "duration_s": res.get("audio", {}).get("duration_s"),
    }


def analyze_file(path: str, profile: str = "full", want_stems: bool = False,
                 log=None) -> dict[str, Any]:
    """Run the full metric set over one file and return the result dictionary."""
    random.seed(SEED)
    np.random.seed(SEED)
    t_start = time.time()
    collector = Collector()

    state = {"t": time.time(), "name": None}

    def step(name: str | None) -> None:
        """Log the stage that just finished, then announce the next one."""
        if not log:
            return
        now = time.time()
        if state["name"]:
            log(f"  {state['name']}: {now - state['t']:.1f} s")
        state["t"] = now
        state["name"] = name
        if name:
            log(f"{name} ...")

    step("decoding")
    src = AudioSource(path, collector)
    if src.duration < 10.0:
        collector.warn("audio", f"file is {src.duration:.3f} s long; metrics that need "
                                "3 s or 10 s windows degrade or return null")

    res: dict[str, Any] = {}
    step("file and container")
    res.update(m_fileinfo.analyse(src, collector))
    res["audio"] = src.summary()

    step("loudness, true peak, DR")
    res["loudness"] = m_loudness.analyse(src, collector, profile)
    integrated = res["loudness"].get("integrated_lufs")

    step("stereo field")
    res["stereo"] = m_stereo.analyse(src, collector, profile)

    step("source forensics")
    res["forensics"] = m_forensics.analyse(src, collector, res["stereo"], profile)

    step("spectrum")
    res["spectrum"] = m_spectrum.analyse(src, collector, profile)

    step("dynamics")
    res["dynamics"] = m_dynamics.analyse(src, collector, profile)

    step("structure, tempo, key")
    res["structure"] = m_structure.analyse(src, collector, integrated, profile)

    step("processing forensics")
    res["processing"] = m_processing.analyse(src, collector, res["structure"], profile)

    if want_stems:
        step("stem separation")
        from .metrics import stems as m_stems
        res["stems"] = m_stems.analyse(src, collector, profile)
    else:
        res["stems"] = {"requested": False,
                        "note": "run with --stems to separate and measure stems"}

    step(None)
    res["headline"] = _headline(res)
    res["warnings"] = collector.warnings
    res["confidence_notes"] = collector.notes
    res["run"] = {
        "tool_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - t_start, 3),
        "profile": profile,
        "stems_requested": bool(want_stems),
        "random_seed": SEED,
        "versions": _versions(),
        "reproducibility": (
            "Two runs over the same file on the same machine and library set "
            "produce byte-identical JSON apart from run.generated_utc, "
            "run.elapsed_seconds and file.path_absolute."
        ),
    }
    res["params"] = dict(PARAMS)
    res["params"]["profile"] = profile_params(profile)
    return res


def write_outputs(res: dict[str, Any], out_dir: str, *, json_only: bool = False,
                  plots: bool = False, src_path: str | None = None,
                  log=None) -> dict[str, str]:
    """Write analysis.json, digest.md and optionally plots/.  Returns paths."""
    from .digest import render_digest

    os.makedirs(out_dir, exist_ok=True)
    written: dict[str, str] = {}
    t0 = time.time()
    json_path = os.path.join(out_dir, "analysis.json")
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(jsonable(res), f, indent=1, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        f.write("\n")
    written["analysis.json"] = json_path
    if log:
        log(f"  writing outputs: {time.time() - t0:.1f} s")

    if not json_only:
        digest = render_digest(res)
        d_path = os.path.join(out_dir, "digest.md")
        with open(d_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(digest)
        written["digest.md"] = d_path

    if plots and src_path:
        try:
            from .plots import render_plots
            written["plots"] = render_plots(res, src_path, os.path.join(out_dir, "plots"),
                                            log=log)
        except ImportError as exc:
            if log:
                log(f"plots skipped: {exc}")
    return written
