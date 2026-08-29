"""One band-split pass, shared by the dynamics, spectrum and forensics modules.

Filtering the whole file eight times is the single most expensive part of a
`full` run, so it happens once: each band is filtered, reduced to the summary
statistics and the 5 ms envelope that every consumer needs, and then dropped.
Peak memory stays at one filtered band, not eight.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .audio import AudioSource
from .dsp import band_filter, crest_db
from .params import BANDS
from .util import db_amp

ENVELOPE_HOP_S = 0.005


def _envelope(x: np.ndarray, sr: float, hop_s: float) -> np.ndarray:
    """Frame RMS at a fixed hop, frame length == hop (no overlap, no leakage)."""
    hop = max(1, int(round(hop_s * sr)))
    n = (x.size // hop) * hop
    if n == 0:
        return np.zeros(0)
    return np.sqrt(np.mean(x[:n].reshape(-1, hop) ** 2, axis=1))


class BandPack:
    """Per-band statistics and envelopes at `ENVELOPE_HOP_S`."""

    def __init__(self, src: AudioSource):
        self.sr = src.band_sr
        self.hop_s = ENVELOPE_HOP_S
        self.bands: list[dict[str, Any]] = []
        self.envelopes: dict[str, np.ndarray] = {}
        nyq = self.sr / 2.0
        x = src.band_mono
        for name, lo, hi in BANDS:
            if lo >= nyq:
                self.bands.append({"band": name, "low_hz": lo, "high_hz": hi,
                                   "valid": False, "note": "band above Nyquist",
                                   "crest_db": None, "rms_dbfs": None,
                                   "peak_dbfs": None})
                self.envelopes[name] = np.zeros(0)
                continue
            y = band_filter(x, self.sr, lo, hi)
            pk = float(np.max(np.abs(y))) if y.size else 0.0
            rm = float(np.sqrt(np.mean(y * y))) if y.size else 0.0
            self.bands.append({
                "band": name, "low_hz": lo, "high_hz": min(hi, nyq), "valid": True,
                "crest_db": crest_db(y),
                "rms_dbfs": db_amp(rm) if rm > 0 else None,
                "peak_dbfs": db_amp(pk) if pk > 0 else None,
                "clamped_to_nyquist": bool(hi > nyq),
            })
            self.envelopes[name] = _envelope(y, self.sr, self.hop_s)
            del y

    @property
    def names(self) -> list[str]:
        return [b["band"] for b in self.bands]

    def envelope_times(self, name: str) -> np.ndarray:
        return np.arange(self.envelopes[name].size) * self.hop_s

    def envelope_db(self, name: str) -> np.ndarray:
        e = self.envelopes[name]
        return db_amp(np.maximum(e, 1e-20)) if e.size else e

    def resample_envelope(self, name: str, hop_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate the 5 ms envelope to a coarser hop by RMS within each frame."""
        e = self.envelopes[name]
        if e.size == 0:
            return np.zeros(0), np.zeros(0)
        k = max(1, int(round(hop_s / self.hop_s)))
        n = (e.size // k) * k
        if n == 0:
            return np.zeros(0), np.zeros(0)
        agg = np.sqrt(np.mean(e[:n].reshape(-1, k) ** 2, axis=1))
        return np.arange(agg.size) * (k * self.hop_s), agg


def get_band_pack(src: AudioSource) -> BandPack:
    return src.cache_get("band_pack", lambda: BandPack(src))
