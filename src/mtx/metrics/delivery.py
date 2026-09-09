"""4.11 Delivery conditions: what this master does once it leaves the room.

`loudness.streaming_preview` already applies the right instinct once, to the
-14 and -16 LUFS targets.  A record meets several other conditions on the way
to a listener, and each of them is a local operation on a local file:

* the lossy encode a distributor makes of it,
* the 400 Hz - 8 kHz window most listening hardware actually reproduces,
* a fold to mono,
* and the fifteen or thirty seconds somebody hears on their own.

The forensics module already knows how to detect codec damage; here the same
apparatus is pointed at the file's own future rather than at its past.  Nothing
in this module needs the network, and nothing needs the record to have been
released -- which makes it, with the masking block, one of the two measurements
an unfinished mix can actually use.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np

from ..audio import AudioSource
from ..dsp import (band_filter, band_power, block_loudness, crest_db,
                   gated_integrated, linear_fit_db_per_octave, true_peak,
                   welch_psd)
from ..params import PARAMS
from ..util import Collector, db_amp, db_pow, fmt_time


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _measure(x: np.ndarray, sr: int, *, with_16x: bool = False) -> dict[str, Any]:
    """The small fixed set of numbers every rendering below is compared on."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if x.shape[0] < x.shape[1]:
        x = x.T
    out: dict[str, Any] = {"sample_rate_hz": int(sr),
                           "duration_s": round(x.shape[0] / float(sr), 3)}
    if x.shape[0] < int(0.4 * sr):
        out["note"] = "shorter than one loudness block"
        return out
    _, bl = block_loudness(x, sr, 0.4, 0.1)
    lufs, _ = gated_integrated(bl)
    _, st = block_loudness(x, sr, 3.0, 0.1)
    out["integrated_lufs"] = lufs
    out["shortterm_max_lufs"] = float(np.max(st)) if st.size else None
    out["sample_peak_dbfs"] = db_amp(float(np.max(np.abs(x))))
    tp4 = true_peak(x, 4)
    out["true_peak_dbtp_4x"] = db_amp(float(np.max(tp4)))
    if with_16x:
        tp16 = true_peak(x, 16)
        out["true_peak_dbtp_16x"] = db_amp(float(np.max(tp16)))
    for thr in PARAMS["true_peak"]["over_thresholds_dbtp"]:
        out.setdefault("overs", {})[f"above_{thr}_dbtp"] = bool(
            out["true_peak_dbtp_4x"] > thr)
    mono = x.mean(axis=1)
    out["crest_db"] = crest_db(mono)
    out["plr_db"] = ((out["true_peak_dbtp_4x"] - lufs)
                     if lufs is not None else None)
    f, p = welch_psd(mono, sr, 16384)
    if f.size:
        slope, r2 = linear_fit_db_per_octave(f, db_pow(np.maximum(p, 1e-30)),
                                             100.0, min(10000.0, sr / 2 * 0.98))
        out["tilt_db_per_oct"] = slope
        out["tilt_r2"] = r2
    return out


def _hf_damage(ref: tuple[np.ndarray, np.ndarray],
               got: tuple[np.ndarray, np.ndarray],
               lo: float, hi: float) -> dict[str, Any]:
    """How much of the top octave and a half the encode gave back."""
    fr, pr = ref
    fg, pg = got
    if fr.size == 0 or fg.size == 0:
        return {"available": False, "reason": "no spectrum"}
    nyq = min(fr[-1], fg[-1])
    hi = min(hi, nyq)
    if hi <= lo:
        return {"available": False,
                "reason": f"the {lo:.0f}-{hi:.0f} Hz band does not exist at "
                          "one of the two sample rates"}
    er = band_power(fr, pr, lo, hi)
    eg = band_power(fg, pg, lo, hi)
    tr = band_power(fr, pr, 20.0, nyq)
    tg = band_power(fg, pg, 20.0, nyq)
    return {
        "available": True,
        "band_hz": [lo, hi],
        "source_share_pct": 100.0 * er / tr if tr > 0 else None,
        "encoded_share_pct": 100.0 * eg / tg if tg > 0 else None,
        "band_level_delta_db": db_pow(eg / er) if (er > 0 and eg > 0) else None,
        "definition": "encoded minus source energy in the band, and each one's "
                      "share of its own broadband energy",
    }


def _encode_pass(src: AudioSource, collector: Collector) -> dict[str, Any]:
    """Encode, decode, and measure what came back."""
    P = PARAMS["delivery"]
    if not _have_ffmpeg():
        return {"available": False,
                "reason": "ffmpeg was not found on PATH; the encode pass needs it",
                "encodes": [e["name"] for e in P["encodes"]]}
    import soundfile as sf

    ref = _measure(src.x, src.sr, with_16x=False)
    f_ref, p_ref = welch_psd(src.mono, src.sr, 16384)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="mtx_delivery_") as tmp:
        for spec in P["encodes"]:
            enc = os.path.join(tmp, "enc" + spec["suffix"])
            dec = os.path.join(tmp, spec["name"] + ".wav")
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                   "-y", "-i", src.path, "-vn", "-c:a", spec["codec"],
                   "-b:a", spec["bitrate"], enc]
            try:
                p1 = subprocess.run(cmd, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=1800)
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                collector.warn("delivery", f"{spec['name']} encode failed: {exc!r}")
                rows.append({"name": spec["name"], "available": False,
                             "reason": repr(exc)})
                continue
            if p1.returncode != 0:
                collector.warn("delivery",
                               f"{spec['name']}: ffmpeg exited {p1.returncode}: "
                               f"{(p1.stderr or '')[-200:]}")
                rows.append({"name": spec["name"], "available": False,
                             "reason": f"ffmpeg exit {p1.returncode}"})
                continue
            p2 = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                 "-i", enc, "-c:a", "pcm_f32le", dec],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=1800)
            if p2.returncode != 0 or not os.path.isfile(dec):
                rows.append({"name": spec["name"], "available": False,
                             "reason": "decode of the encoded file failed"})
                continue
            y, sr2 = sf.read(dec, dtype="float64", always_2d=True)
            got = _measure(y, sr2)
            f_got, p_got = welch_psd(y.mean(axis=1), sr2, 16384)
            row = {
                "name": spec["name"], "available": True,
                "codec": spec["codec"], "bitrate": spec["bitrate"],
                "encoded_bytes": os.path.getsize(enc),
                "measured": got,
                "delta_vs_source": {
                    "integrated_lufs": (got.get("integrated_lufs") - ref["integrated_lufs"])
                    if (got.get("integrated_lufs") is not None
                        and ref.get("integrated_lufs") is not None) else None,
                    "true_peak_dbtp_4x": (got.get("true_peak_dbtp_4x")
                                          - ref["true_peak_dbtp_4x"])
                    if (got.get("true_peak_dbtp_4x") is not None
                        and ref.get("true_peak_dbtp_4x") is not None) else None,
                    "sample_peak_dbfs": (got.get("sample_peak_dbfs")
                                         - ref["sample_peak_dbfs"])
                    if (got.get("sample_peak_dbfs") is not None
                        and ref.get("sample_peak_dbfs") is not None) else None,
                },
                "new_overs": {
                    thr: bool(got.get("true_peak_dbtp_4x") is not None
                              and ref.get("true_peak_dbtp_4x") is not None
                              and got["true_peak_dbtp_4x"] > thr
                              >= ref["true_peak_dbtp_4x"])
                    for thr in PARAMS["true_peak"]["over_thresholds_dbtp"]},
                "hf_damage": _hf_damage((f_ref, p_ref), (f_got, p_got),
                                        *P["hf_damage_band_hz"]),
                "resampled": bool(sr2 != src.sr),
            }
            rows.append(row)
    return {
        "available": True,
        "source": ref,
        "renderings": rows,
        "method": "ffmpeg encode at the stated bitrate, decoded back to PCM and "
                  "re-measured; the true peak of the decode is what a listener's "
                  "converter has to reproduce, and it is not the peak of the "
                  "file you delivered",
        "reproducibility_note": "the encoder is ffmpeg's; the exact bytes depend "
                                "on the ffmpeg build recorded in run.versions",
    }


def _small_speaker(src: AudioSource) -> dict[str, Any]:
    """What survives the window most listening hardware reproduces."""
    lo, hi = PARAMS["delivery"]["small_speaker_band_hz"]
    sr = src.band_sr
    nyq = sr / 2.0
    if lo >= nyq:
        return {"available": False, "reason": "band above Nyquist"}
    x = src.band_x
    y = np.column_stack([band_filter(x[:, c], sr, lo, min(hi, nyq * 0.98))
                         for c in range(x.shape[1])])
    full = _measure(x, sr)
    band = _measure(y, sr)
    f, p = welch_psd(x.mean(axis=1), sr, 16384)
    inside = band_power(f, p, lo, min(hi, nyq)) if f.size else 0.0
    total = band_power(f, p, 20.0, nyq) if f.size else 0.0
    return {
        "available": True,
        "band_hz": [lo, hi],
        "energy_share_pct": 100.0 * inside / total if total > 0 else None,
        "loudness_delta_lu": ((band.get("integrated_lufs") - full.get("integrated_lufs"))
                              if (band.get("integrated_lufs") is not None
                                  and full.get("integrated_lufs") is not None) else None),
        "band_passed": band,
        "definition": "the track band-passed to the stated window, measured "
                      "against itself unfiltered; the loudness delta is how much "
                      "of the record a small speaker never gets",
    }


def _mono_fold(src: AudioSource) -> dict[str, Any]:
    """The mono fold, measured in loudness and per octave rather than broadband."""
    if src.n_ch < 2:
        return {"available": False, "reason": "mono file"}
    sr = src.band_sr
    stereo = src.band_x[:, :2]
    mono = src.band_mid
    a = _measure(stereo, sr)
    b = _measure(mono[:, None], sr)
    f, ps = welch_psd(stereo.mean(axis=1), sr, 16384)
    _, pm = welch_psd(mono, sr, 16384)
    per_octave = []
    for centre in PARAMS["processing"]["reverb"]["octave_bands_hz"]:
        lo, hi = centre / (2 ** 0.5), centre * (2 ** 0.5)
        if lo >= sr / 2.0:
            continue
        es = band_power(f, ps, lo, min(hi, sr / 2.0))
        em = band_power(f, pm, lo, min(hi, sr / 2.0))
        per_octave.append({"centre_hz": centre,
                           "mono_minus_stereo_db": db_pow(em / es)
                           if (es > 0 and em > 0) else None})
    return {
        "available": True,
        "loudness_delta_lu": ((b.get("integrated_lufs") - a.get("integrated_lufs"))
                              if (a.get("integrated_lufs") is not None
                                  and b.get("integrated_lufs") is not None) else None),
        "true_peak_delta_db": ((b.get("true_peak_dbtp_4x") - a.get("true_peak_dbtp_4x"))
                               if (a.get("true_peak_dbtp_4x") is not None
                                   and b.get("true_peak_dbtp_4x") is not None) else None),
        "per_octave": per_octave,
        "definition": "(L+R)/2 measured against the stereo file: the broadband "
                      "energy view of the same fold is in "
                      "stereo.mono_sum_damage, this one is loudness-weighted "
                      "and per octave",
    }


def _excerpts(src: AudioSource, form: dict[str, Any] | None,
              structure: dict[str, Any] | None) -> dict[str, Any]:
    """What the first fifteen seconds, and the chorus, contain on their own."""
    P = PARAMS["delivery"]
    rows: list[dict[str, Any]] = []
    for span in P["excerpt_s"]:
        n = int(min(span, src.duration) * src.sr)
        if n < int(0.4 * src.sr):
            continue
        rows.append({"name": f"first_{int(span)}s", "start_s": 0.0,
                     "duration_s": n / float(src.sr),
                     "measured": _measure(src.x[:n], src.sr)})
    chorus_start = None
    basis = None
    if form and form.get("available") and form.get("time_to_first_chorus_s") is not None:
        chorus_start = float(form["time_to_first_chorus_s"])
        basis = "form.time_to_first_chorus_s"
    elif structure and structure.get("available"):
        idx = structure.get("loudest_section_index")
        secs = structure.get("sections") or []
        if idx is not None and 0 <= idx < len(secs):
            chorus_start = float(secs[idx]["start_s"])
            basis = "structure.loudest_section_index (no form label available)"
    if chorus_start is not None:
        span = min(15.0, max(0.0, src.duration - chorus_start))
        a = int(chorus_start * src.sr)
        b = a + int(span * src.sr)
        if b - a >= int(0.4 * src.sr):
            rows.append({"name": "chorus_15s", "start_s": chorus_start,
                         "start": fmt_time(chorus_start),
                         "duration_s": span, "basis": basis,
                         "measured": _measure(src.x[a:b], src.sr)})
    return {"available": bool(rows), "excerpts": rows,
            "note": "each excerpt is measured as if it were the whole file; a "
                    "clip that is used on its own is heard on its own"}


def analyse(src: AudioSource, collector: Collector,
            structure: dict[str, Any] | None = None,
            form: dict[str, Any] | None = None,
            profile: str = "full") -> dict[str, Any]:
    if profile == "quick":
        # Every rendering here re-measures a whole second copy of the file --
        # four true-peak scans before the encode pass even starts.  That is the
        # right cost in a full run and the wrong one in a quick look.
        return {"available": False, "reason": "skipped by --profile quick"}
    return {
        "available": True,
        "small_speaker": _small_speaker(src),
        "mono_fold": _mono_fold(src),
        "excerpts": _excerpts(src, form, structure),
        "encode": _encode_pass(src, collector),
    }
