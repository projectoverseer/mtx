"""Band splitting: edges, energy conservation, and the Nyquist guard."""

from __future__ import annotations

import numpy as np
import pytest

from mtx.dsp import band_filter, band_power, spectrum_table, third_octave_edges, welch_psd
from mtx.params import BANDS, THIRD_OCTAVE_CENTRES

SR = 44100


def _tone(freq: float, seconds: float = 2.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return amp * np.sin(2 * np.pi * freq * t)


def test_band_edges_cover_the_audible_range_without_gaps():
    for (_, _, hi), (_, lo_next, _) in zip(BANDS, BANDS[1:]):
        assert hi == lo_next, "band edges must meet exactly, with no gap or overlap"
    assert BANDS[0][1] == 20.0
    assert BANDS[-1][2] == 20000.0


@pytest.mark.parametrize("band_index,freq", [(0, 40.0), (1, 90.0), (2, 180.0),
                                             (3, 350.0), (4, 1000.0), (5, 3500.0),
                                             (6, 9000.0), (7, 15000.0)])
def test_tone_lands_in_its_own_band(band_index, freq):
    """A tone in a band's centre must dominate that band and no other."""
    x = _tone(freq)
    energies = []
    for _, lo, hi in BANDS:
        y = band_filter(x, SR, lo, hi)
        energies.append(float(np.mean(y * y)))
    winner = int(np.argmax(energies))
    assert winner == band_index, f"{freq} Hz landed in band {winner}, expected {band_index}"
    others = [e for i, e in enumerate(energies) if i != band_index]
    assert energies[band_index] > 100 * max(others), "band separation is too weak"


def test_band_filter_is_zero_phase():
    """sosfiltfilt must not shift the signal in time."""
    x = np.zeros(SR)
    x[SR // 2] = 1.0
    y = band_filter(x, SR, 500.0, 2000.0)
    assert abs(int(np.argmax(np.abs(y))) - SR // 2) <= 1


def test_band_above_nyquist_returns_zero_not_garbage():
    x = _tone(1000.0, seconds=0.5)
    y = band_filter(x, 8000, 12000.0, 20000.0)
    assert np.allclose(y, 0.0), "a band entirely above Nyquist must return silence"


def test_band_power_matches_direct_integration():
    x = _tone(1000.0) + _tone(5000.0, amp=0.25)
    f, p = welch_psd(x, SR, 16384)
    total = float(np.trapezoid(p, f))
    parts = sum(band_power(f, p, lo, hi) for _, lo, hi in BANDS)
    # The 8 bands cover 20 Hz-20 kHz; both tones are inside, so almost all of
    # the energy must be accounted for.
    assert parts / total > 0.98


def test_third_octave_edges_are_geometric():
    edges = third_octave_edges(THIRD_OCTAVE_CENTRES)
    for centre, lo, hi in edges:
        assert lo < centre < hi
        assert hi / lo == pytest.approx(2 ** (1 / 3), rel=1e-9)
        assert centre == pytest.approx(np.sqrt(lo * hi), rel=1e-9)


def test_spectrum_table_marks_bands_above_nyquist():
    x = _tone(1000.0, seconds=0.5)
    f, p = welch_psd(x, 16000, 4096)
    rows = spectrum_table(f, p, third_octave_edges(THIRD_OCTAVE_CENTRES), 8000.0)
    above = [r for r in rows if r["centre_hz"] >= 12500]
    assert above, "test needs bands above the Nyquist frequency"
    for r in above:
        assert r["db"] is None and r["note"] == "band above Nyquist"


def test_band_pack_matches_direct_filtering(tmp_path):
    """The shared band pass must agree with filtering each band on its own."""
    import soundfile as sf
    from mtx.audio import AudioSource
    from mtx.bands import get_band_pack
    from mtx.util import Collector

    x = _tone(80.0, seconds=3.0) + _tone(3000.0, seconds=3.0, amp=0.2)
    path = tmp_path / "bands.wav"
    sf.write(path, np.stack([x, x], axis=1), SR, subtype="PCM_24")
    src = AudioSource(str(path), Collector())
    pack = get_band_pack(src)
    for row, (name, lo, hi) in zip(pack.bands, BANDS):
        assert row["band"] == name
        y = band_filter(src.band_mono, src.band_sr, lo, hi)
        expected = 20 * np.log10(max(np.sqrt(np.mean(y * y)), 1e-20))
        assert row["rms_dbfs"] == pytest.approx(expected, abs=1e-9)
