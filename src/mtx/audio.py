"""Audio access layer.

One object owns the decoded signal and every derived rate, so an expensive
resample or STFT is computed once per run and reused by every metric module.
Decoding is chunked, so peak memory is the size of the decoded float32 signal
and not a multiple of it.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Iterator

import numpy as np
import soundfile as sf
from scipy import signal as sps

from .util import Collector

READ_BLOCK_FRAMES = 1 << 19  # 524288 frames per chunked read
MEMORY_WARN_BYTES = 1_000_000_000
BAND_SR_CAP = 48000
LIBROSA_SR = 22050


def resample_to(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resample along axis 0.  Exact rational ratio, no dithering."""
    if sr_in == sr_out:
        return x
    frac = Fraction(int(sr_out), int(sr_in)).limit_denominator(10000)
    return sps.resample_poly(x, frac.numerator, frac.denominator, axis=0).astype(
        np.float32, copy=False
    )


class AudioSource:
    """Decoded audio plus the derived representations the metrics need."""

    def __init__(self, path: str, collector: Collector, threads: int = 1):
        self.path = path
        self.collector = collector
        # How many threads a metric may use inside this one file.  Carried on
        # the source rather than threaded through every `analyse()` signature:
        # every metric module already has the source in hand, and only the
        # true-peak scan currently has a GIL-releasing hot loop to spend it on.
        self.threads = max(1, int(threads))
        self.info = sf.info(path)
        self.sr: int = int(self.info.samplerate)
        self.n_ch: int = int(self.info.channels)
        self.subtype: str = str(self.info.subtype)
        self.format: str = str(self.info.format)
        self.endian: str = str(self.info.endian)

        self.x: np.ndarray = self._decode_float32()
        self.n_frames: int = int(self.x.shape[0])
        self.duration: float = self.n_frames / float(self.sr)
        self._cache: dict[str, Any] = {}

        if self.n_frames == 0:
            collector.warn("audio", "file decodes to zero frames")
        if self.n_ch == 1:
            collector.warn("audio", "mono file: stereo metrics are reported as null")
        elif self.n_ch > 2:
            collector.warn(
                "audio",
                f"{self.n_ch} channels: stereo metrics use channels 1 and 2; "
                "loudness and peak metrics cover all channels",
            )

    # ------------------------------------------------------------ decoding
    def _decode_float32(self) -> np.ndarray:
        n = int(self.info.frames)
        nbytes = n * self.n_ch * 4
        if nbytes > MEMORY_WARN_BYTES:
            self.collector.warn(
                "audio",
                f"decoded signal is {nbytes / 1e9:.2f} GB in float32; "
                "analysis proceeds but memory use is proportional to duration",
            )
        out = np.empty((n, self.n_ch), dtype=np.float32)
        pos = 0
        with sf.SoundFile(self.path) as f:
            while pos < n:
                block = f.read(
                    frames=min(READ_BLOCK_FRAMES, n - pos),
                    dtype="float32",
                    always_2d=True,
                )
                if block.shape[0] == 0:
                    break
                out[pos : pos + block.shape[0]] = block
                pos += block.shape[0]
        return out[:pos]

    def int_samples(self) -> np.ndarray | None:
        """Integer sample representation, left-justified into int32.

        libsndfile scales PCM_16 by 2**16 and PCM_24 by 2**8 when reading as
        int32, so trailing-zero counting on this array is container-agnostic.
        Returns None for float subtypes, where the question does not apply.
        """
        if self.subtype not in ("PCM_S8", "PCM_U8", "PCM_16", "PCM_24", "PCM_32"):
            return None
        key = "int32"
        if key not in self._cache:
            n = self.n_frames
            out = np.empty((n, self.n_ch), dtype=np.int32)
            pos = 0
            with sf.SoundFile(self.path) as f:
                while pos < n:
                    block = f.read(
                        frames=min(READ_BLOCK_FRAMES, n - pos),
                        dtype="int32",
                        always_2d=True,
                    )
                    if block.shape[0] == 0:
                        break
                    out[pos : pos + block.shape[0]] = block
                    pos += block.shape[0]
            self._cache[key] = out[:pos]
        return self._cache[key]

    # ------------------------------------------------------- basic channels
    @property
    def channels(self) -> list[np.ndarray]:
        return [self.x[:, c].astype(np.float64) for c in range(self.n_ch)]

    def channel(self, c: int) -> np.ndarray:
        return self.x[:, c].astype(np.float64)

    @property
    def is_stereo(self) -> bool:
        return self.n_ch >= 2

    def _cached(self, key: str, fn) -> Any:
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    @property
    def mono(self) -> np.ndarray:
        """Channel mean.  Used where a single broadband signal is wanted."""
        return self._cached(
            "mono", lambda: self.x.mean(axis=1).astype(np.float64)
        )

    @property
    def mid(self) -> np.ndarray:
        """mid = (L+R)/2.  For mono files this is the signal itself."""
        def build():
            if self.n_ch == 1:
                return self.x[:, 0].astype(np.float64)
            return ((self.x[:, 0].astype(np.float64) + self.x[:, 1]) * 0.5)
        return self._cached("mid", build)

    @property
    def side(self) -> np.ndarray:
        """side = (L-R)/2.  None-equivalent (all zeros) for mono files."""
        def build():
            if self.n_ch == 1:
                return np.zeros(self.n_frames, dtype=np.float64)
            return ((self.x[:, 0].astype(np.float64) - self.x[:, 1]) * 0.5)
        return self._cached("side", build)

    # ---------------------------------------------------------- rate views
    @property
    def band_sr(self) -> int:
        """Rate used for band-split work: capped at 48 kHz.

        Every analysis band tops out at 20 kHz, so nothing is lost, and the
        filtering cost of a 192 kHz file drops by 4x.
        """
        return int(min(self.sr, BAND_SR_CAP))

    @property
    def band_x(self) -> np.ndarray:
        """Full multichannel signal at band_sr, shape (n, ch), float64."""
        return self._cached(
            "band_x",
            lambda: np.atleast_2d(
                resample_to(self.x, self.sr, self.band_sr)
            ).astype(np.float64).reshape(-1, self.n_ch),
        )

    @property
    def band_mid(self) -> np.ndarray:
        """mid at band_sr.

        At or below the cap there is no resampling, so `band_x` is `x` in
        float64 and this is arithmetic `mid` has already done -- bit for bit,
        because float32 to float64 is exact and both sides then work in
        float64.  Returning it saves a second full-length float64 array per
        source, and a stems run holds five sources, so on a 44.1 kHz track it
        is about a gigabyte of a second copy of the same answer.  Nothing
        writes to either view in place; if something ever does, this has to
        become a copy again.
        """
        def build():
            if self.sr <= BAND_SR_CAP:
                return self.mid
            bx = self.band_x
            if self.n_ch == 1:
                return bx[:, 0]
            return (bx[:, 0] + bx[:, 1]) * 0.5
        return self._cached("band_mid", build)

    @property
    def band_side(self) -> np.ndarray:
        """side at band_sr.  Aliases `side` below the cap; see `band_mid`."""
        def build():
            if self.sr <= BAND_SR_CAP:
                return self.side
            bx = self.band_x
            if self.n_ch == 1:
                return np.zeros(bx.shape[0])
            return (bx[:, 0] - bx[:, 1]) * 0.5
        return self._cached("band_side", build)

    @property
    def band_mono(self) -> np.ndarray:
        return self._cached("band_mono", lambda: self.band_x.mean(axis=1))

    @property
    def lib_mono(self) -> np.ndarray:
        """Mono at 22.05 kHz, float32: the input to every librosa call."""
        return self._cached(
            "lib_mono",
            lambda: np.ascontiguousarray(
                resample_to(
                    self.x.mean(axis=1).astype(np.float32), self.sr, LIBROSA_SR
                ),
                dtype=np.float32,
            ),
        )

    @property
    def lib_sr(self) -> int:
        return LIBROSA_SR

    # --------------------------------------------------------------- blocks
    def blocks(self, block_frames: int, hop_frames: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (start_frame, block) over the native-rate signal."""
        hop = hop_frames or block_frames
        n = self.n_frames
        i = 0
        while i < n:
            yield i, self.x[i : i + block_frames]
            i += hop

    def cache_get(self, key: str, fn):
        """Memoise an expensive derived array for the rest of the run."""
        return self._cached(key, fn)

    # ------------------------------------------------- shared librosa features
    def onset_envelope(self):
        """librosa onset strength at 22.05 kHz / hop 512, computed once."""
        def build():
            import librosa
            return librosa.onset.onset_strength(
                y=self.lib_mono, sr=LIBROSA_SR, hop_length=512)
        return self._cached("onset_env", build)

    def onset_times(self):
        def build():
            import librosa
            frames = librosa.onset.onset_detect(
                onset_envelope=self.onset_envelope(), sr=LIBROSA_SR, hop_length=512)
            return librosa.frames_to_time(frames, sr=LIBROSA_SR, hop_length=512), frames
        return self._cached("onset_times", build)

    def chroma_cqt(self):
        """Constant-Q chroma; the single most expensive librosa call here."""
        def build():
            import librosa
            return librosa.feature.chroma_cqt(
                y=self.lib_mono, sr=LIBROSA_SR, hop_length=512)
        return self._cached("chroma_cqt", build)

    def summary(self) -> dict[str, Any]:
        return {
            "sample_rate_hz": self.sr,
            "channels": self.n_ch,
            "frames": self.n_frames,
            "duration_s": round(self.duration, 3),
            "subtype": self.subtype,
            "format": self.format,
            "endian": self.endian,
            "band_analysis_sr_hz": self.band_sr,
            "librosa_sr_hz": LIBROSA_SR,
        }
