"""Helper processes: their output is UTF-8, and a bad decode is never fatal.

Found on a purchased file with Korean tags: `subprocess.run(..., text=True)`
decodes with the machine's locale codec, which on a Windows cp1252 console
cannot decode UTF-8 tag bytes at all. The reader thread died, `proc.stdout`
came back `None`, and `json.loads(None)` raised a TypeError nothing was
catching -- so one accented character in a tag took down the whole run.
"""

from __future__ import annotations

import subprocess

from mtx.metrics.fileinfo import run_ffprobe
from mtx.util import Collector


class _Proc:
    def __init__(self, returncode=0, stdout=None, stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_ffprobe_is_decoded_as_utf8(monkeypatch, tmp_path):
    """Non-Latin-1 tags must survive; the locale codec must not be consulted."""
    seen: dict = {}

    def fake_run(cmd, **kw):
        seen.update(kw)
        return _Proc(stdout='{"format": {"tags": {"artist": "ROSÉ"}}}')

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = run_ffprobe(str(tmp_path / "x.flac"), Collector())
    assert out["format"]["tags"]["artist"] == "ROSÉ"
    assert seen.get("encoding") == "utf-8"
    assert seen.get("errors") == "replace"


def test_a_helper_that_produced_nothing_warns_instead_of_raising(monkeypatch,
                                                                tmp_path):
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: _Proc(stdout=None))
    collector = Collector()
    assert run_ffprobe(str(tmp_path / "x.flac"), collector) is None
    assert any("no output" in w for w in collector.warnings)


def test_every_helper_call_states_its_encoding():
    """One missed call site is one file format away from the same crash."""
    import inspect
    from mtx import analyze
    from mtx.metrics import fileinfo, loudness, stems

    for module in (analyze, fileinfo, loudness, stems):
        src = inspect.getsource(module)
        for chunk in src.split("subprocess.run(")[1:]:
            call = chunk[:chunk.index(")\n") if ")\n" in chunk else 400]
            assert "encoding=" in call, (
                f"{module.__name__} calls subprocess.run without an explicit "
                "encoding; it will decode with the machine's locale codec")
