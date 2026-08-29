"""Mid/side maths: the convention, its inverse, and the ratios built on it."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from mtx.audio import AudioSource
from mtx.dsp import welch_psd
from mtx.metrics.stereo import _band_side_mid, _side_mid_db
from mtx.spectra import get_spectra
from mtx.util import Collector

SR = 44100


def _write(tmp_path, name, x, subtype="PCM_24"):
    p = tmp_path / name
    sf.write(p, x, SR, subtype=subtype)
    return AudioSource(str(p), Collector())


def _noise(n=SR * 3, seed=0):
    return np.random.default_rng(seed).standard_normal(n) * 0.2


def test_convention_is_half_sum_and_half_difference(tmp_path):
    L, R = _noise(seed=1), _noise(seed=2)
    src = _write(tmp_path, "ms.wav", np.stack([L, R], axis=1))
    assert np.allclose(src.mid, (src.x[:, 0] + src.x[:, 1]) / 2.0, atol=1e-7)
    assert np.allclose(src.side, (src.x[:, 0] - src.x[:, 1]) / 2.0, atol=1e-7)


def test_mid_side_inverts_back_to_left_and_right(tmp_path):
    L, R = _noise(seed=3), _noise(seed=4)
    src = _write(tmp_path, "inv.wav", np.stack([L, R], axis=1))
    assert np.allclose(src.mid + src.side, src.x[:, 0], atol=1e-7)
    assert np.allclose(src.mid - src.side, src.x[:, 1], atol=1e-7)


def test_identical_channels_have_no_side_energy(tmp_path):
    x = _noise(seed=5)
    src = _write(tmp_path, "mono_in_stereo.wav", np.stack([x, x], axis=1))
    assert float(np.sum(src.side ** 2)) == 0.0
    assert _side_mid_db(src.mid, src.side) == -200.0


def test_opposite_channels_leave_only_quantisation_in_mid(tmp_path):
    """L = -R cancels to the container's own LSB, not to exact zero.

    Two's complement is asymmetric, so a 24-bit file holding x and -x sums to
    one LSB rather than to nothing.  The mid signal must sit at that floor and
    the side/mid ratio must be enormous.
    """
    x = _noise(seed=6)
    src = _write(tmp_path, "antiphase.wav", np.stack([x, -x], axis=1))
    lsb = 2.0 ** -24
    assert float(np.max(np.abs(src.mid))) <= lsb * 2
    assert _side_mid_db(src.mid, src.side) > 100.0


def test_opposite_float_channels_cancel_exactly(tmp_path):
    x = _noise(seed=6)
    src = _write(tmp_path, "antiphase_f32.wav", np.stack([x, -x], axis=1),
                 subtype="FLOAT")
    assert float(np.sum(src.mid ** 2)) == 0.0
    assert _side_mid_db(src.mid, src.side) is None


def test_independent_channels_give_equal_mid_and_side(tmp_path):
    L, R = _noise(seed=7), _noise(seed=8)
    src = _write(tmp_path, "wide.wav", np.stack([L, R], axis=1))
    assert _side_mid_db(src.mid, src.side) == pytest.approx(0.0, abs=0.2)


def test_mono_file_reports_stereo_metrics_as_unavailable(tmp_path):
    from mtx.metrics import stereo

    x = _noise()
    src = _write(tmp_path, "mono.wav", x[:, None])
    res = stereo.analyse(src, Collector())
    assert res["available"] is False
    assert "single channel" in res["reason"]


def test_derived_mid_side_spectra_equal_a_direct_welch(tmp_path):
    """P_mid and P_side from the auto/cross spectra must be exact."""
    L, R = _noise(seed=9), _noise(seed=10) * 0.5 + _noise(seed=9) * 0.5
    src = _write(tmp_path, "spectra.wav", np.stack([L, R], axis=1))
    pack = get_spectra(src)
    for name, sig in (("mid", src.band_mid), ("side", src.band_side)):
        _, direct = welch_psd(sig, src.band_sr, pack.nperseg)
        rel = np.abs(pack.psd[name] - direct) / np.maximum(direct, 1e-30)
        assert float(rel.max()) < 1e-8, f"{name} PSD identity broke"


def test_band_side_mid_matches_time_domain_filtering(tmp_path):
    from mtx.dsp import band_filter

    L, R = _noise(seed=11), _noise(seed=12) * 0.3 + _noise(seed=11) * 0.7
    src = _write(tmp_path, "band_ms.wav", np.stack([L, R], axis=1))
    pack = get_spectra(src)
    lo, hi = 500.0, 2000.0
    from_spectra = _band_side_mid(pack.freqs, pack.psd["ch0"], pack.psd["ch1"],
                                  pack.csd_lr, lo, hi)
    bm = band_filter(src.band_mid, src.band_sr, lo, hi)
    bs = band_filter(src.band_side, src.band_sr, lo, hi)
    from_time = _side_mid_db(bm, bs)
    assert from_spectra == pytest.approx(from_time, abs=0.5)
