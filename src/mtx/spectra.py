"""One long-term spectrum pass, shared by the spectrum and stereo modules.

Both need the same Welch estimates, and a 16384-point Welch over a whole track
is not cheap.  They are computed once here, from the channel auto-spectra and
the L/R cross-spectrum:

    P_mid  = (S_LL + S_RR + 2 Re S_LR) / 4
    P_side = (S_LL + S_RR - 2 Re S_LR) / 4

Those identities are exact, not approximations: Welch is an average of
per-segment periodograms, and the periodogram of (L+R)/2 is exactly the
expression above for every segment, with the same window and scaling.
`tests/test_spectra.py` asserts it against a direct Welch of the mid signal.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal as sps

from .audio import AudioSource
from .dsp import welch_psd
from .params import PARAMS


class SpectraPack:
    """Welch PSDs for each channel, mid, side and mono, on one frequency grid."""

    def __init__(self, src: AudioSource, nperseg: int | None = None):
        self.sr = src.band_sr
        self.nperseg = int(nperseg or PARAMS["spectrum"]["ltas_broadband"]["nperseg"])
        bx = src.band_x
        self.psd: dict[str, np.ndarray] = {}
        self.freqs = np.zeros(0)
        for c in range(src.n_ch):
            f, p = welch_psd(bx[:, c], self.sr, self.nperseg)
            if self.freqs.size == 0:
                self.freqs = f
            self.psd[f"ch{c}"] = p
        if self.freqs.size == 0:
            self.csd_lr = np.zeros(0)
            self.psd["mid"] = self.psd["side"] = self.psd["mono"] = np.zeros(0)
            return
        if src.n_ch >= 2:
            nps = int(min(self.nperseg, bx.shape[0]))
            _, cross = sps.csd(bx[:, 0], bx[:, 1], fs=self.sr, window="hann",
                               nperseg=nps, noverlap=nps // 2, detrend=False,
                               scaling="density")
            self.csd_lr = np.real(cross)
            ll, rr = self.psd["ch0"], self.psd["ch1"]
            self.psd["mid"] = (ll + rr + 2.0 * self.csd_lr) / 4.0
            self.psd["side"] = (ll + rr - 2.0 * self.csd_lr) / 4.0
        else:
            self.csd_lr = np.zeros_like(self.freqs)
            self.psd["mid"] = self.psd["ch0"]
            self.psd["side"] = np.zeros_like(self.freqs)
        if src.n_ch == 2:
            # The channel mean is the mid signal.
            self.psd["mono"] = self.psd["mid"]
        else:
            _, p = welch_psd(src.band_mono, self.sr, self.nperseg)
            self.psd["mono"] = p

    def signals(self) -> list[str]:
        return list(self.psd.keys())

    def params(self) -> dict[str, Any]:
        return {
            "method": "Welch", "window": "hann", "nperseg": self.nperseg,
            "overlap_pct": 50, "sample_rate_hz": self.sr,
            "mid_side_note": "mid and side PSDs are derived from the channel auto- "
                             "and cross-spectra, which is exact for the "
                             "(L+R)/2, (L-R)/2 decomposition",
        }


def get_spectra(src: AudioSource) -> SpectraPack:
    return src.cache_get("spectra_pack", lambda: SpectraPack(src))
