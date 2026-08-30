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
from . import declared as declared_mod
from .audio import AudioSource
from .coverage import build as build_coverage
from .parallel import resolve_threads
from .metrics import (delivery as m_delivery, dynamics as m_dynamics,
                      embedding as m_embedding, fileinfo as m_fileinfo,
                      forensics as m_forensics, form as m_form,
                      harmony as m_harmony, loudness as m_loudness,
                      lyrics as m_lyrics, processing as m_processing,
                      rhythm as m_rhythm, spectrum as m_spectrum,
                      stereo as m_stereo, structure as m_structure)
from .params import PARAMS, profile_params
from .split import DEFAULT_PART_BYTES
from .util import Collector, jsonable

SEED = 0


_VERSIONS_CACHE: dict[str, Any] | None = None


def _versions() -> dict[str, Any]:
    """The library and tool versions this run was produced by.

    Cached for the life of the process.  Nothing here can change while a run is
    in progress, and the two subprocess spawns it costs are charged once per
    file otherwise -- which a scan pays several hundred times for one answer.
    """
    global _VERSIONS_CACHE
    if _VERSIONS_CACHE is None:
        _VERSIONS_CACHE = _probe_versions()
    return dict(_VERSIONS_CACHE)


def _probe_versions() -> dict[str, Any]:
    out: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "mtx": __version__,
    }
    for mod in ("numpy", "scipy", "soundfile", "mutagen", "pyloudnorm",
                "librosa", "numba", "matplotlib", "sklearn", "demucs",
                "torch", "pyarrow", "vaderSentiment", "allin1"):
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
                               encoding="utf-8", errors="replace", timeout=30)
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
    out = {
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
    out.update(_headline_musical(res))
    return out


def _headline_musical(res: dict[str, Any]) -> dict[str, Any]:
    """The musical half of the headline: harmony, rhythm, form, melody, lyric.

    Every one of these is null on a run that could not compute it -- a quick
    profile, a file with no beat grid, a run without --stems -- and never a
    stand-in value.
    """
    H = res.get("harmony", {}) or {}
    R = res.get("rhythm", {}) or {}
    F = res.get("form", {}) or {}
    LY = res.get("lyrics", {}) or {}
    STM = res.get("stems", {}) or {}
    down = R.get("downbeats", {}) if R.get("available") else {}
    grid = R.get("grid", {}) if R.get("available") else {}
    swing = R.get("swing", {}) if R.get("available") else {}
    sync = R.get("syncopation", {}) if R.get("available") else {}
    hr = H.get("harmonic_rhythm", {}) if H.get("available") else {}
    deg = H.get("degrees", {}) if H.get("available") else {}
    voc = ((STM.get("melody") or {}).get("vocals") or {}) if STM.get("available") else {}
    arr = (STM.get("arrangement") or {}) if STM.get("available") else {}
    stats = LY.get("statistics", {}) if LY.get("available") else {}
    return {
        "beats_per_bar": down.get("beats_per_bar"),
        "bar_count": R.get("bar_count") if R.get("available") else None,
        "swing_ratio": swing.get("swing_ratio"),
        "grid_deviation_std_ms": (grid.get("deviation") or {}).get("std_ms"),
        "syncopation_per_bar": sync.get("mean_per_bar"),
        "chord_count": H.get("chord_count") if H.get("available") else None,
        "distinct_chords": (H.get("vocabulary") or {}).get("distinct_chords"),
        "chord_changes_per_bar": hr.get("changes_per_bar"),
        "diatonic_time_pct": deg.get("diatonic_time_pct"),
        "key_from_chords": (H.get("key_from_chords") or {}).get("key"),
        "chorus_count": F.get("chorus_count") if F.get("available") else None,
        "chorus_share_pct": F.get("chorus_share_pct") if F.get("available") else None,
        "time_to_first_chorus_s": F.get("time_to_first_chorus_s") if F.get("available") else None,
        "time_to_vocal_entry_s": F.get("time_to_vocal_entry_s") if F.get("available") else None,
        "form_letters": F.get("letters") if F.get("available") else None,
        "form_part_count": F.get("part_count") if F.get("available") else None,
        "form_unnamed_parts": (F.get("unnamed_part_count")
                               if F.get("available") else None),
        "vocal_range_p5_p95_semitones": (voc.get("range") or {}).get("p5_p95_semitones"),
        "vocal_p5_note": (voc.get("range") or {}).get("p5_note"),
        "vocal_p95_note": (voc.get("range") or {}).get("p95_note"),
        "vocal_median_note": (voc.get("tessitura") or {}).get("median_note"),
        "vocal_notes_per_second": voc.get("notes_per_second_of_voicing"),
        "concurrent_sources_mean": (arr.get("density") or {}).get("mean"),
        "lyric_word_count": stats.get("words"),
        "lyric_source": LY.get("source"),
    }


def analyze_file(path: str, profile: str = "full", want_stems: bool = False,
                 log=None, threads: int | None = None, *,
                 stems_model: str | None = None, declared_path: str | None = None,
                 want_transcript: bool = False, want_embedding: bool = False,
                 ) -> dict[str, Any]:
    """Run the full metric set over one file and return the result dictionary.

    `threads` is how many threads the metrics inside this one file may use.
    `None` means "decide from the machine", which is what a single-file run
    wants; `mtx scan` passes 1 because it is already running one process per
    file and the two layers must not multiply.

    `stems_model` picks the demucs model, and with it how many stems there are:
    `htdemucs_6s` splits guitar and piano out of `other`.  `declared_path`
    points at a `declared.json` sidecar, whose contents are passed through
    labelled as declared and never merged into anything measured.
    `want_transcript` and `want_embedding` each enable one optional, heavy,
    model-backed block that is off by default.
    """
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
    src = AudioSource(path, collector, threads=resolve_threads(threads))
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

    step("rhythm, downbeats, groove")
    res["rhythm"] = m_rhythm.analyse(src, res["structure"], collector, profile)

    step("harmony, chords")
    res["harmony"] = m_harmony.analyse(src, res["structure"], res["rhythm"],
                                       collector, profile)

    stem_sources = None
    if want_stems:
        step("stem separation")
        from .metrics import stems as m_stems
        stem_sources = m_stems.load(src, collector, stems_model)
        step("stems, masking, melody, arrangement")
        res["stems"] = m_stems.analyse(src, collector, profile, stem_sources,
                                       res["structure"], res["rhythm"],
                                       stems_model)
    else:
        res["stems"] = {"requested": False,
                        "note": "run with --stems to separate and measure stems; "
                                "pitch, inter-stem masking and arrangement all "
                                "depend on it"}

    step("song form")
    res["form"] = m_form.analyse(src, res["structure"], res["rhythm"],
                                 res["forensics"], stem_sources, collector,
                                 profile)

    step("delivery conditions")
    res["delivery"] = m_delivery.analyse(src, collector, res["structure"],
                                         res["form"], profile)

    step("declared metadata and version identity")
    res["declared"] = declared_mod.load(path, collector, explicit=declared_path)
    res["version"] = declared_mod.version_identity(res.get("tags") or {})

    step("lyrics")
    res["lyrics"] = m_lyrics.analyse(res.get("tags") or {}, res["declared"],
                                     res["stems"], res["structure"], collector,
                                     want_transcript=want_transcript)

    step("embedding")
    res["embedding"] = m_embedding.analyse(
        src, (res["structure"].get("sections") or []) if res["structure"].get("available") else [],
        collector, enabled=want_embedding)

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
        "stems_model": stems_model or PARAMS["stems"]["model"],
        "transcript_requested": bool(want_transcript),
        "embedding_requested": bool(want_embedding),
        "declared_sidecar": res.get("declared", {}).get("path"),
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
    # Last, so it sees every block: the uniform present/trusted mask over the
    # whole document, which saves every consumer from walking it themselves.
    res["coverage"] = build_coverage(res)
    return res


def write_outputs(res: dict[str, Any], out_dir: str, *, json_only: bool = False,
                  plots: bool = False, src_path: str | None = None,
                  digest_budget: int | None = None,
                  sections: list[str] | None = None,
                  max_part_bytes: int | None = DEFAULT_PART_BYTES,
                  blind: bool = False, log=None) -> dict[str, str]:
    """Write analysis.json, digest.md, corpus_row.json and optionally plots/.

    `analysis.json` is written whole when it fits under `max_part_bytes`, and
    as an index plus `analysis.partNN.json` files when it does not: the
    exhaustive dump of a four-minute track runs past the 5 MB per-file cap most
    places put on an upload, and a file that cannot be uploaded stays on one
    machine.  `max_part_bytes=None` always writes the single file.

    With `blind`, a prediction sheet is written as well: the digest is still
    produced, but the caller is expected to hand over only `predict.md` until
    the prediction has been committed.  Returns the paths written.
    """
    from .digest import corpus_row_dict, render_digest
    from .split import write_analysis

    os.makedirs(out_dir, exist_ok=True)
    written: dict[str, str] = {}
    t0 = time.time()
    written.update(write_analysis(jsonable(res), out_dir, "analysis",
                                  max_bytes=max_part_bytes, log=log))
    if log:
        log(f"  writing outputs: {time.time() - t0:.1f} s")

    if not json_only:
        digest = render_digest(res, budget=digest_budget, sections=sections)
        d_path = os.path.join(out_dir, "digest.md")
        with open(d_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(digest)
        written["digest.md"] = d_path

        # The corpus row, already typed: the last transcription step between a
        # measurement and the archive it is stored in.
        c_path = os.path.join(out_dir, "corpus_row.json")
        with open(c_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(jsonable(corpus_row_dict(res)), f, indent=1, sort_keys=True,
                      ensure_ascii=False, allow_nan=False)
            f.write("\n")
        written["corpus_row.json"] = c_path

        if blind:
            from .predict import render_predict_sheet
            p_path = os.path.join(out_dir, "predict.md")
            with open(p_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(render_predict_sheet(res))
            written["predict.md"] = p_path

    if plots and src_path:
        try:
            from .plots import render_plots
            written["plots"] = render_plots(res, src_path, os.path.join(out_dir, "plots"),
                                            log=log)
        except ImportError as exc:
            if log:
                log(f"plots skipped: {exc}")
    return written
