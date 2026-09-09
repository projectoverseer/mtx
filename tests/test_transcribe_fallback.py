"""The transcription backend's fallback, which used to cover half a failure.

A GPU does not fail transcription at load time.  It loads the model, reports
itself healthy, and then runs out of memory part way through decoding a long
track -- because `WhisperModel.transcribe` returns a lazy generator and the
work happens as the caller iterates it, well after any constructor has
returned.

The original fallback wrapped only the constructor, so a card that loaded the
model and then died had no second attempt.  It cost 78 of 1,321 tracks over a
full corpus run -- 6%, every one of them long, which is to say album tracks
rather than singles: a slant in the lyric data and not merely a hole in it.

The second half of the fix is which rung catches them.  An out-of-memory is a
memory problem, not a broken card, so the first retry is a cheaper compute
type on the same GPU; the CPU stays as the last resort it should be.

These tests use a stub backend rather than a real model: the behaviour under
test is which device gets tried after which failure, and that needs no GPU,
no weights, and no audio.
"""

from __future__ import annotations

import sys
import types

import pytest

from mtx.metrics import lyrics as m_lyrics
from mtx.util import Collector


class _Segment:
    def __init__(self, text, words):
        self.text = text
        self.words = words


class _Word:
    def __init__(self, word, start, end):
        self.word, self.start, self.end = word, start, end


class _Info:
    language = "en"


class _StubModel:
    """A whisper model that fails where it is told to."""

    made: list[str] = []

    def __init__(self, name, device, compute_type):
        self.device, self.compute = device, compute_type
        type(self).made.append(f"{device}/{compute_type}")
        if device in self.fail_on_load or compute_type in self.fail_on_load:
            raise RuntimeError(f"{device} would not load")

    def transcribe(self, path, **kw):
        if (self.device in self.fail_on_decode
                or self.compute in self.fail_on_decode):
            # Lazily, exactly like the real one: the generator is what raises.
            def boom():
                yield _Segment("first line", [_Word("first", 0.0, 0.4)])
                raise RuntimeError("CUDA failed with error out of memory")
            return boom(), _Info()
        return iter([
            _Segment(" one two ", [_Word("one", 0.0, 0.4), _Word("two", 0.4, 0.8)]),
            _Segment(" three ", [_Word("three", 0.9, 1.2)]),
        ]), _Info()


@pytest.fixture
def stub(monkeypatch):
    """Install a fake `faster_whisper`, and no `whisper_timestamped`."""
    _StubModel.made = []
    _StubModel.fail_on_load = frozenset()
    _StubModel.fail_on_decode = frozenset()
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = _StubModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setitem(sys.modules, "whisper_timestamped", None)
    monkeypatch.setattr(m_lyrics, "_whisper_devices",
                        lambda _p: [("cuda", "float16"), ("cpu", "int8")])
    return _StubModel


def test_a_decode_failure_falls_back_to_the_next_device(stub):
    """The bug: the GPU loaded, then died mid-track, and that was the end."""
    stub.fail_on_decode = frozenset({"cuda"})
    got = m_lyrics.transcribe("nowhere.wav", Collector())

    assert got["available"] is True, "a working CPU should have caught this"
    assert got["device"] == "cpu"
    assert stub.made == ["cuda/float16", "cpu/int8"], \
        "both devices should have been tried"
    assert got["text"] == "one two\nthree"


def test_a_load_failure_still_falls_back(stub):
    """The case the old code did handle, which must keep working."""
    stub.fail_on_load = frozenset({"cuda"})
    got = m_lyrics.transcribe("nowhere.wav", Collector())

    assert got["available"] is True
    assert got["device"] == "cpu"
    assert stub.made == ["cuda/float16", "cpu/int8"]


def test_the_first_device_is_used_when_it_works(stub):
    """No fallback when there is nothing to fall back from."""
    got = m_lyrics.transcribe("nowhere.wav", Collector())

    assert got["device"] == "cuda"
    assert stub.made == ["cuda/float16"], "the CPU should not have been touched"


def test_failing_everywhere_reports_every_reason(stub):
    """A transcript that cannot be made says why, per device, not just 'no'."""
    stub.fail_on_decode = frozenset({"cuda", "cpu"})
    collector = Collector()
    got = m_lyrics.transcribe("nowhere.wav", collector)

    assert got["available"] is False
    assert got["attempted"] == ["cuda/float16", "cpu/int8"]
    assert "cuda:" in got["reason"] and "cpu:" in got["reason"]
    assert "out of memory" in got["reason"]
    # Every attempt is a warning on the record, not only the last one.
    assert len([w for w in collector.warnings
                if "faster_whisper" in str(w)]) == 2


def test_segments_stay_one_line_each(stub):
    """Line-based lyric measurement depends on this and reads as fine without it."""
    got = m_lyrics.transcribe("nowhere.wav", Collector())

    assert got["text"].splitlines() == ["one two", "three"]
    assert got["lines"] == 2


def test_the_real_plan_retries_on_the_gpu_before_giving_up_on_it(monkeypatch):
    """An OOM is a memory problem, not a broken card.

    `float16` weights plus a long track's activations do not fit in 4 GB, but
    the same weights as `int8_float16` do -- so the retry that matters is a
    cheaper compute type on the same GPU, seconds of work, not a CPU pass that
    costs minutes a track.  Falling straight to CPU turned 6% of the corpus
    into an overnight job of its own.
    """
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    import torch  # noqa: PLC0415
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    plan = m_lyrics._whisper_devices({"device": "auto"})

    assert plan[0] == ("cuda", "float16"), "the fast path stays first"
    assert plan[1] == ("cuda", "int8_float16"), "the cheap GPU rung is second"
    assert plan[-1][0] == "cpu", "the CPU remains the last resort"


def test_an_out_of_memory_lands_on_the_second_gpu_rung(stub, monkeypatch):
    """The 78 tracks this was built for: they must not reach the CPU."""
    monkeypatch.setattr(m_lyrics, "_whisper_devices",
                        lambda _p: [("cuda", "float16"),
                                    ("cuda", "int8_float16"),
                                    ("cpu", "int8")])
    stub.fail_on_decode = frozenset({"float16"})     # not int8_float16
    got = m_lyrics.transcribe("nowhere.wav", Collector())

    assert got["available"] is True
    assert got["device"] == "cuda"
    assert got["compute_type"] == "int8_float16"
    assert "cpu/int8" not in stub.made, "the CPU pass was not needed"


def test_a_padded_model_path_is_still_found(stub, monkeypatch):
    """A stray space in an env var must not fail every track on every device.

    `set VAR=path && cmd` in cmd.exe puts the space before the `&&` into the
    value, and a hand-written `.env` line does the same.  The path then looks
    correct in every log line it appears in -- the space sits inside the
    closing quote -- and ctranslate2 reports `Unable to open file
    'model.bin'`, which reads as a corrupt download rather than a typo.  It
    cost 43 tracks on a run that was otherwise failing none.
    """
    monkeypatch.setenv("MTX_WHISPER_MODEL", "  /models/faster-whisper-small 	")
    seen = {}

    class Recording(_StubModel):
        def __init__(self, name, device, compute_type):
            seen["name"] = name
            super().__init__(name, device, compute_type)

    sys.modules["faster_whisper"].WhisperModel = Recording
    m_lyrics.transcribe("nowhere.wav", Collector())

    assert seen["name"] == "/models/faster-whisper-small"
