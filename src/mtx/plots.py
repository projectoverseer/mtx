"""Optional PNG plots (--plots).

These are for the operator's own eyes.  Nothing in digest.md ever refers to
them, so a run without matplotlib installed loses nothing measurable.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np


def render_plots(res: dict[str, Any], src_path: str, out_dir: str, log=None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import signal as sps

    from .audio import AudioSource
    from .bands import get_band_pack
    from .util import Collector, db_amp

    os.makedirs(out_dir, exist_ok=True)
    src = AudioSource(src_path, Collector())

    def save(fig, name: str) -> None:
        p = os.path.join(out_dir, name)
        fig.tight_layout()
        fig.savefig(p, dpi=110)
        plt.close(fig)
        if log:
            log(f"  plot: {name}")

    # 1. waveform
    fig, ax = plt.subplots(figsize=(12, 3))
    step = max(1, src.n_frames // 200000)
    t = np.arange(0, src.n_frames, step) / src.sr
    for c in range(min(src.n_ch, 2)):
        ax.plot(t, src.x[::step, c], lw=0.3, alpha=0.8, label=f"ch{c}")
    ax.set_xlabel("s")
    ax.set_ylabel("amplitude")
    ax.set_title("waveform")
    ax.legend(loc="upper right", fontsize=7)
    save(fig, "waveform.png")

    # 2. spectrogram
    fig, ax = plt.subplots(figsize=(12, 4))
    f, tt, Z = sps.spectrogram(src.band_mono, fs=src.band_sr, nperseg=4096,
                               noverlap=2048, scaling="density")
    ax.pcolormesh(tt, f, 10 * np.log10(np.maximum(Z, 1e-16)), shading="auto")
    ax.set_yscale("symlog", linthresh=100)
    ax.set_ylabel("Hz")
    ax.set_xlabel("s")
    ax.set_title("spectrogram")
    save(fig, "spectrogram.png")

    # 3. short-term loudness
    st = res.get("loudness", {}).get("shortterm", {})
    if st.get("times_s") is not None and len(st["times_s"]):
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(st["times_s"], st["lufs"], lw=0.8)
        i = res["loudness"].get("integrated_lufs")
        if i is not None:
            ax.axhline(i, color="r", lw=0.8, label=f"LUFS-I {i:.2f}")
            ax.legend(fontsize=7)
        ax.set_xlabel("s")
        ax.set_ylabel("LUFS")
        ax.set_title("short-term loudness (3 s)")
        save(fig, "shortterm_loudness.png")

    # 4. LTAS
    S = res.get("spectrum", {})
    if S.get("available"):
        lt = S["ltas"]["broadband"]
        fig, ax = plt.subplots(figsize=(10, 4))
        freqs = np.asarray(lt["frequencies_hz"], dtype=float)
        for key in ("mid", "side"):
            ax.semilogx(freqs[1:], np.asarray(lt["psd_db"][key], dtype=float)[1:],
                        lw=0.7, label=key)
        ax.set_xlim(20, src.band_sr / 2)
        ax.grid(True, which="both", lw=0.3, alpha=0.5)
        ax.set_xlabel("Hz")
        ax.set_ylabel("dB")
        ax.set_title("LTAS (Welch 16384)")
        ax.legend(fontsize=7)
        save(fig, "ltas.png")

        # 5. per-band side/mid
        rows = res["stereo"].get("side_minus_mid_per_third_octave") \
            if res["stereo"].get("available") else None
        if rows:
            fig, ax = plt.subplots(figsize=(10, 3.5))
            xs = [r["centre_hz"] for r in rows]
            ys = [r["side_minus_mid_db"] if r["side_minus_mid_db"] is not None else np.nan
                  for r in rows]
            ax.semilogx(xs, ys, marker="o", ms=3, lw=0.8)
            ax.axhline(-20, color="k", lw=0.5, ls=":")
            ax.set_xlabel("Hz")
            ax.set_ylabel("side/mid dB")
            ax.set_title("side/mid per third-octave")
            ax.grid(True, which="both", lw=0.3, alpha=0.5)
            save(fig, "side_mid_per_band.png")

    # 6. per-band envelopes
    pack = get_band_pack(src)
    fig, ax = plt.subplots(figsize=(12, 4))
    for name in pack.names:
        t_, e = pack.resample_envelope(name, 0.1)
        if e.size:
            ax.plot(t_, db_amp(np.maximum(e, 1e-20)), lw=0.6, label=name)
    ax.set_xlabel("s")
    ax.set_ylabel("dBFS")
    ax.set_title("per-band envelopes (100 ms)")
    ax.legend(fontsize=6, ncol=4)
    save(fig, "band_envelopes.png")

    # 7. modulation spectrum
    fig, ax = plt.subplots(figsize=(10, 4))
    drew = False
    for name in pack.names:
        e = pack.envelopes[name]
        if e.size < 512:
            continue
        ee = e - e.mean()
        spec = np.abs(np.fft.rfft(ee * np.hanning(ee.size))) / (0.5 * ee.size)
        fr = np.fft.rfftfreq(ee.size, pack.hop_s)
        m = (fr > 0.1) & (fr < 30)
        ax.semilogx(fr[m], 20 * np.log10(np.maximum(spec[m], 1e-12)), lw=0.6, label=name)
        drew = True
    if drew:
        ax.set_xlabel("modulation rate Hz")
        ax.set_ylabel("dB")
        ax.set_title("modulation spectrum of the band envelopes")
        ax.legend(fontsize=6, ncol=4)
        save(fig, "modulation_spectrum.png")
    else:
        import matplotlib.pyplot as plt2
        plt2.close(fig)
    return out_dir
