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

ENV_CACHE = "MTX_STEMS_CACHE"

# Separation output is uncompressed wav, four of them per track, so a library
# scan puts tens of gigabytes somewhere -- and the home directory is usually on
# the smallest disk in the machine.  The variable names a better one.  It is
# read once, at import, so a scan and every worker it spawns agree on where the
# cache lives without the path having to be threaded through them.
CACHE_DIR = (os.environ.get(ENV_CACHE) or
             os.path.join(os.path.expanduser("~"), ".cache", "mtx", "stems"))

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


# ------------------------------------------------------------------ the device

# demucs is a torch model, and on a GPU it runs roughly an order of magnitude
# faster than on the CPU cores a scan can spare it.  What limits a consumer
# card is its memory rather than its speed, and the knob for that is the
# segment: demucs holds one segment of audio on the device at a time, so a
# shorter one fits a smaller card at a little cost in throughput.  These are
# tried in order, longest first, and only after an out-of-memory failure --
# nothing is given up until the card says it has to be.  demucs takes whole
# seconds here, and 7 is the ceiling: htdemucs was trained at 7.8 s and refuses
# to be asked for more than it has seen.
GPU_SEGMENTS: tuple[int, ...] = (7, 5, 3, 2)

ENV_DEVICE = "MTX_STEMS_DEVICE"
ENV_SEGMENT = "MTX_STEMS_SEGMENT"

_CUDA: bool | None = None


def cuda_available() -> bool:
    """Whether torch can see a usable GPU.  Answered once per process.

    Importing torch costs seconds, so this is only ever called when stems were
    actually asked for, and the answer is kept.
    """
    global _CUDA
    if _CUDA is None:
        try:
            import torch
            _CUDA = bool(torch.cuda.is_available())
        except Exception:
            _CUDA = False
    return _CUDA


def resolve_device(requested: str | None = None) -> str:
    """Where separation should run: what was asked for, or what is there.

    The environment carries the answer because a scan's workers are separate
    processes: they inherit it, where a parsed flag would have to be threaded
    through every layer between the command line and this call.
    """
    choice = (requested or os.environ.get(ENV_DEVICE) or
              PARAMS["stems"].get("device") or "auto")
    if choice != "auto":
        return choice
    return "cuda" if cuda_available() else "cpu"


def resolve_segment(requested: int | None = None) -> int | None:
    """Seconds of audio held on the device at once; whole seconds only."""
    if requested is not None:
        return int(requested)
    env = os.environ.get(ENV_SEGMENT) or PARAMS["stems"].get("segment")
    try:
        return int(float(env)) if env else None
    except (TypeError, ValueError):
        return None


# A card is small but it is not busy.  Measured on a GTX 1650 (4 GiB) over 45
# tracks at the default segment: one separation at a time holds 875 MiB and
# leaves the SM 68 % busy, because a large share of every track is spent
# decoding the input and writing four uncompressed wavs with the device idle.
# Overlapping separations fills those gaps -- 1.36x at two streams, 1.51x at
# three, where utilisation reaches 94 % and there is nothing left to fill.
#
# What stops it is memory, and that scales exactly linearly: 875 MiB a stream,
# so three fit a 4 GiB card and four do not.  The reserve below is what keeps
# the fourth from being attempted on this card -- an out-of-memory failure
# costs a step down the segment ladder and the separation that hit it, which
# is more than the last few percent of utilisation is worth.
STREAM_VRAM_MIB = 900
VRAM_RESERVE_MIB = 1100
MAX_STREAMS = 4

ENV_STREAMS = "MTX_STEMS_JOBS"


def device_vram_mib() -> int | None:
    """Total memory on the CUDA device, or None when there is no card.

    Only ever called after `cuda_available()` has already paid for the torch
    import, so this costs nothing a stems run has not already spent.
    """
    if not cuda_available():
        return None
    try:
        import torch
        total = torch.cuda.get_device_properties(0).total_memory
        return int(total // (1024 * 1024))
    except Exception:
        return None


def separation_streams(requested: int | None = None) -> int:
    """How many separations may share the card at once.

    An explicit request wins, so a card this arithmetic reads wrongly can
    still be driven by hand.  Otherwise it is what the device's memory holds
    with a reserve left over, capped: past about four streams the SM is
    already saturated and the extra memory buys nothing.
    """
    if requested is None:
        env = os.environ.get(ENV_STREAMS)
        requested = int(env) if env and env.isdigit() else None
    if requested is not None:
        return max(1, int(requested))
    vram = device_vram_mib()
    if not vram:
        return 1
    fits = (vram - VRAM_RESERVE_MIB) // STREAM_VRAM_MIB
    return max(1, min(MAX_STREAMS, int(fits)))


def _out_of_memory(stderr: str) -> bool:
    low = (stderr or "").lower()
    return "out of memory" in low or "cuda error" in low or "cublas" in low


def _child_env(device: str) -> dict[str, str]:
    """Environment for the demucs child process.

    A scan pins each of its workers to one thread and demucs inherits that,
    which is right while several separations share the CPU and wrong on a GPU,
    where they run one at a time and the CPU-side work -- decode, STFT, write
    -- is all that is left to spread over the cores.
    """
    env = dict(os.environ)
    if device.startswith("cuda"):
        for key in ("MTX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS"):
            env.pop(key, None)
    return env


# What demucs produced for one file is named after that file, both in the
# cache key and in the folder underneath it.  Neither may be true: separation
# depends on the audio bytes and the model and on nothing else, so a second
# copy of one master -- the single next to the album track, the same rip under
# a tidier name -- has to land on the work already done rather than spend
# another few minutes of CPU reproducing it.
STEM_DIR = "stems"


def _content_key(path: str) -> str:
    """Cache identity of a file's contents, independent of where it lives."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:24]


def _legacy_key(path: str) -> str:
    """The key caches written before separation became content-addressed used.

    Kept only to read them: an existing cache is hours of CPU, and there is no
    reason to make a user re-separate a library to pick up the fix.
    """
    st = os.stat(path)
    h = hashlib.sha256()
    h.update(os.path.abspath(path).encode("utf-8"))
    h.update(str(st.st_size).encode("ascii"))
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
    return h.hexdigest()[:24]


def _stem_paths(root: str, model: str, name: str) -> dict[str, str]:
    return {s: os.path.join(root, model, name, f"{s}.wav")
            for s in stem_names(model)}


def _complete(paths: dict[str, str]) -> bool:
    return bool(paths) and all(os.path.isfile(p) for p in paths.values())


# What this card turned out to hold, once a file has found out.  A scan puts
# every separation through this one process, so the second file need not
# rediscover the first one's out-of-memory failures: each of those costs a
# model load before it fails, and paying that per track is most of what a
# small card would otherwise cost.
_LEARNED: dict[tuple[str, str], int | None] = {}


def _attempts(device: str, segment: int | None,
              model: str) -> list[tuple[str, int | None]]:
    """The (device, segment) pairs to try, in order.

    On a GPU the shorter segments are held in reserve for an out-of-memory
    failure, and the CPU is the last resort: slow beats absent.  An explicitly
    requested segment is taken as an instruction and not second-guessed.
    """
    if not device.startswith("cuda"):
        return [(device, segment)]
    if segment is not None:
        return [(device, segment), ("cpu", None)]
    key = (device, model)
    if key not in _LEARNED:
        ladder = list(GPU_SEGMENTS)                  # nothing known yet
    elif _LEARNED[key] is None:
        ladder = []                                  # the card never held it
    else:
        ladder = [s for s in GPU_SEGMENTS if s <= _LEARNED[key]]
    return [(device, s) for s in ladder] + [("cpu", None)]


def separate(path: str, collector: Collector, model: str | None = None, *,
             device: str | None = None,
             segment: int | None = None) -> dict[str, str] | None:
    """Run demucs and return {stem_name: wav_path}.  Cached across runs.

    The cache is keyed on the file's contents, so the same master separates
    once however many copies of it the library holds.
    """
    model = model or PARAMS["stems"]["model"]
    out_root = os.path.join(CACHE_DIR, _content_key(path))
    stem_dir = os.path.join(out_root, model, STEM_DIR)
    have = _stem_paths(out_root, model, STEM_DIR)
    if _complete(have):
        return have
    legacy = _stem_paths(os.path.join(CACHE_DIR, _legacy_key(path)), model,
                         os.path.splitext(os.path.basename(path))[0])
    if _complete(legacy):
        return legacy
    if shutil.which("demucs") is None and importlib.util.find_spec("demucs") is None:
        collector.warn("stems", "demucs is not installed; install the 'stems' "
                                "extra (pip install 'mtx[stems]') to use --stems")
        return None
    os.makedirs(out_root, exist_ok=True)

    asked = resolve_device(device)
    attempts = _attempts(asked, resolve_segment(segment), model)
    for i, (dev, seg) in enumerate(attempts):
        cmd = [sys.executable, "-m", "demucs", "-n", model, "-d", dev,
               "-o", out_root]
        if seg is not None:
            cmd += ["--segment", str(int(seg))]
        cmd.append(path)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace",
                                  timeout=7200, env=_child_env(dev))
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            collector.warn("stems", f"demucs failed to run: {exc!r}")
            return None
        if proc.returncode == 0:
            if asked.startswith("cuda"):
                _LEARNED[(asked, model)] = seg if dev == asked else None
            break
        if i == len(attempts) - 1 or not _out_of_memory(proc.stderr):
            collector.warn("stems", f"demucs exited {proc.returncode} on {dev}: "
                                    f"{(proc.stderr or '')[-400:]}")
            return None
        nxt = attempts[i + 1]
        collector.warn("stems",
                       f"{dev} ran out of memory at segment {seg}; retrying on "
                       f"{nxt[0]}" + (f" at segment {nxt[1]}" if nxt[1] else ""))
    # demucs names its output folder after the input file; give it the fixed
    # name the content key expects, so the next copy of this master finds it.
    produced = os.path.join(out_root, model,
                            os.path.splitext(os.path.basename(path))[0])
    if produced != stem_dir and os.path.isdir(produced):
        shutil.rmtree(stem_dir, ignore_errors=True)
        try:
            os.replace(produced, stem_dir)
        except OSError as exc:
            collector.warn("stems", f"could not name the stem cache entry: {exc!r}")
            have = _stem_paths(out_root, model,
                               os.path.splitext(os.path.basename(path))[0])
    if not _complete(have):
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
