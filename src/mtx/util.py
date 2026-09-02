"""Small shared helpers: dB maths, note names, path and JSON sanitising, warnings."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

# EPS is the amplitude clamp applied before every log, and DB_FLOOR is the
# value a dB conversion saturates at.  Both are finite on purpose: a -inf would
# propagate through any mean it touched.  DB_FLOOR also doubles as the sentinel
# for "this ratio is exactly zero" (a side signal with no energy at all).
EPS = 1e-20
DB_FLOOR = -200.0

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# ------------------------------------------------------------- path components

# Characters no Windows path component may hold (POSIX only bars "/"), and the
# device names it still reserves in every directory.
BAD_FS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_TRAILING = re.compile(r"[.\s]+$")
_RESERVED = frozenset(["CON", "PRN", "AUX", "NUL"]
                      + [f"COM{i}" for i in range(1, 10)]
                      + [f"LPT{i}" for i in range(1, 10)])


def safe_component(name: str, fallback: str = "_") -> str:
    """`name` made usable as one path component, on every platform.

    Windows does not refuse a trailing dot or space -- it drops it, silently
    and only sometimes: `makedirs("03. Sometimes...")` creates `03. Sometimes`
    and reports success, and the write into the folder that was asked for then
    fails with ENOENT halfway through the track.  A title that trails off is an
    ordinary thing for a file to be called, so everything this tool names after
    a filename or a tag goes through here.

    The rule is applied on POSIX too, where those names are legal: a mirror of
    a library has to hold the same folder names whichever machine wrote it, or
    the same track measured from two sides of a share is measured twice.
    """
    out = _TRAILING.sub("", BAD_FS_CHARS.sub("_", name))
    if out.upper().split(".")[0] in _RESERVED:
        out += "_"
    return out or fallback



def db_amp(x: Any, floor: float = DB_FLOOR) -> Any:
    """Amplitude ratio -> dB (20 log10).  Returns `floor` for zero/negative."""
    a = np.abs(np.asarray(x, dtype=np.float64))
    out = 20.0 * np.log10(np.maximum(a, EPS))
    out = np.maximum(out, floor)
    return float(out) if np.isscalar(x) or out.ndim == 0 else out


def db_pow(x: Any, floor: float = DB_FLOOR) -> Any:
    """Power ratio -> dB (10 log10)."""
    a = np.abs(np.asarray(x, dtype=np.float64))
    out = 10.0 * np.log10(np.maximum(a, EPS))
    out = np.maximum(out, floor)
    return float(out) if np.isscalar(x) or out.ndim == 0 else out


def percentiles(x: np.ndarray, ps: Sequence[float]) -> dict[str, float | None]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{p:g}": None for p in ps}
    q = np.percentile(x, ps)
    return {f"p{p:g}": float(v) for p, v in zip(ps, np.atleast_1d(q))}


def note_for_frequency(freq: float, a4: float = 440.0) -> tuple[str, int, float]:
    """Nearest equal-tempered note.  Returns (name_with_octave, midi, cents)."""
    if freq is None or freq <= 0:
        return ("n/a", -1, 0.0)
    midi_f = 69.0 + 12.0 * math.log2(freq / a4)
    midi = int(round(midi_f))
    cents = (midi_f - midi) * 100.0
    name = f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"
    return (name, midi, cents)


def fmt_time(seconds: float | None) -> str:
    """Seconds -> M:SS.mmm, the form used in every timestamp field."""
    if seconds is None or not math.isfinite(seconds):
        return "n/a"
    neg = seconds < 0
    s = abs(float(seconds))
    m = int(s // 60)
    rem = s - m * 60
    out = f"{m}:{rem:06.3f}"
    return ("-" + out) if neg else out


def jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to plain JSON types.

    Non-finite floats become None: a JSON document that cannot be re-read is
    worse than a missing number.
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        # Whole-array conversion: the timelines in one analysis run hold
        # hundreds of thousands of samples, and per-element recursion over them
        # costs more than every other part of writing the file.
        if obj.ndim == 1 and obj.dtype.kind in "fiub":
            if obj.dtype.kind == "f":
                bad = ~np.isfinite(obj)
                out = obj.astype(float).tolist()
                if bad.any():
                    for i in np.flatnonzero(bad):
                        out[int(i)] = None
                return out
            return obj.tolist()
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    return str(obj)


@dataclass
class Collector:
    """Warnings and low-confidence notes, gathered in emission order."""

    warnings: list[str] = field(default_factory=list)
    notes: list[dict[str, Any]] = field(default_factory=list)

    def warn(self, where: str, message: str) -> None:
        line = f"{where}: {message}"
        if line not in self.warnings:
            self.warnings.append(line)

    def low_confidence(self, metric: str, confidence: str, reason: str) -> None:
        self.notes.append({"metric": metric, "confidence": confidence, "reason": reason})

    def disagreement(self, metric: str, a_name: str, a: float, b_name: str, b: float,
                     tolerance: float, unit: str) -> bool:
        """Record a cross-method disagreement.  Returns True if beyond tolerance."""
        if a is None or b is None:
            return False
        delta = float(a) - float(b)
        if abs(delta) > tolerance:
            self.warn(
                metric,
                f"{a_name}={a:.3f} vs {b_name}={b:.3f} {unit} "
                f"(delta {delta:+.3f} {unit}, tolerance {tolerance} {unit})",
            )
            return True
        return False
