"""Scope resolution, the skip ledger, duplicates and the parallelism budget.

These are the four things `mtx scan` promises: that the same track lands in
the same place whichever level the command is run from, that work already done
is not done again, that one master is measured once however many copies of it
the library holds, and that the process and thread layers never multiply into
more workers than the machine has.  None of them need audio to test, so none
of these tests decode any.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor

import pytest

from mtx import SCHEMA_VERSION, __version__
from mtx import declared as declared_mod
from mtx.parallel import ordered_window, resolve_threads, single_threaded_env
from mtx.scan import (Duplicate, NoRootRegistered, find_audio, ledger_path,
                      materialize_duplicate, out_dir_for,
                      partition_duplicates, plan, register, resolve_scope,
                      run_scan, split_budget, why_stale, write_ledger,
                      write_summary)


@pytest.fixture()
def library(tmp_path):
    """A three-level library: root / artist / album / tracks."""
    album = tmp_path / "lib" / "Ed Sheeran" / "÷"
    album.mkdir(parents=True)
    for name in ("01. Eraser.flac", "04. Shape of You.flac"):
        (album / name).write_bytes(b"not really audio")
    other = tmp_path / "lib" / "Coldplay" / "Parachutes"
    other.mkdir(parents=True)
    (other / "01. Don't Panic.flac").write_bytes(b"not really audio")
    return tmp_path / "lib"


@pytest.fixture()
def registry(tmp_path):
    return str(tmp_path / "roots.json")


# ------------------------------------------------------------ scope resolution

def test_out_is_required_before_anything_is_registered(library, registry):
    """Guessing a root would strand the results of the first scan."""
    with pytest.raises(NoRootRegistered):
        resolve_scope(str(library / "Ed Sheeran" / "÷"), registry=registry)


def test_same_track_same_folder_from_every_level(library, registry, tmp_path):
    """The point of the registry: scope varies, destination does not."""
    out = str(tmp_path / "out")
    scope, should = resolve_scope(str(library), out=out, registry=registry)
    assert should
    register(scope, registry)

    track = str(library / "Ed Sheeran" / "÷" / "04. Shape of You.flac")
    destinations = set()
    for level in (library, library / "Ed Sheeran", library / "Ed Sheeran" / "÷"):
        sc, again = resolve_scope(str(level), registry=registry)
        assert not again, "an established root should not re-register"
        assert sc.library_root == str(library)
        destinations.add(out_dir_for(sc, track))
    assert len(destinations) == 1
    assert destinations.pop() == os.path.join(
        out, "Ed Sheeran", "÷", "04. Shape of You")


def test_scanning_a_single_file_resolves_to_its_album(library, registry, tmp_path):
    out = str(tmp_path / "out")
    register(resolve_scope(str(library), out=out, registry=registry)[0], registry)
    track = str(library / "Ed Sheeran" / "÷" / "01. Eraser.flac")
    scope, _ = resolve_scope(track, registry=registry)
    assert scope.scan_path == os.path.dirname(track)
    assert out_dir_for(scope, track).startswith(out)


def test_a_file_target_measures_only_that_file(library, registry, tmp_path, capsys):
    """`mtx scan one.flac` must not quietly measure the other eleven tracks."""
    from mtx.scan import run_scan

    register(resolve_scope(str(library), out=str(tmp_path / "out"),
                           registry=registry)[0], registry)
    track = str(library / "Ed Sheeran" / "÷" / "01. Eraser.flac")
    stats = run_scan(track, registry=registry, dry_run=True,
                     log=lambda s: None)
    assert stats["found"] == 1
    album = run_scan(str(library / "Ed Sheeran" / "÷"), registry=registry,
                     dry_run=True, log=lambda s: None)
    assert album["found"] == 2


def test_nested_root_is_absorbed_by_the_wider_one(library, registry, tmp_path):
    """Registering the library over an album root leaves one answer, not two."""
    album = str(library / "Ed Sheeran" / "÷")
    register(resolve_scope(album, out=str(tmp_path / "album_out"),
                           registry=registry)[0], registry)
    register(resolve_scope(str(library), out=str(tmp_path / "lib_out"),
                           registry=registry)[0], registry)
    roots = json.load(open(registry, encoding="utf-8"))["roots"]
    assert [r["root"] for r in roots] == [str(library)]
    scope, _ = resolve_scope(album, registry=registry)
    assert scope.out_dir == str(tmp_path / "lib_out")


def test_library_root_must_contain_the_scanned_path(library, registry, tmp_path):
    with pytest.raises(ValueError):
        resolve_scope(str(library), out=str(tmp_path / "out"),
                      library_root=str(library / "Coldplay"), registry=registry)


def test_mirror_base_tracks_the_scanned_subtree(library, registry, tmp_path):
    out = str(tmp_path / "out")
    register(resolve_scope(str(library), out=out, registry=registry)[0], registry)
    scope, _ = resolve_scope(str(library / "Ed Sheeran"), registry=registry)
    assert scope.mirror_base == os.path.join(out, "Ed Sheeran")
    root_scope, _ = resolve_scope(str(library), registry=registry)
    assert root_scope.mirror_base == out


# ----------------------------------------------------------------- discovery

def test_find_audio_is_recursive_sorted_and_skips_output_trees(library):
    (library / "mtx_out").mkdir()
    (library / "mtx_out" / "stray.flac").write_bytes(b"x")
    (library / "Ed Sheeran" / "cover.jpg").write_bytes(b"x")
    found = find_audio(str(library))
    assert [os.path.basename(f) for f in found] == [
        "01. Don't Panic.flac", "01. Eraser.flac", "04. Shape of You.flac"]
    assert found == sorted(found)


def test_stem_collisions_keep_the_extension(library, registry, tmp_path):
    """Two files whose names differ only by extension must not share a folder."""
    album = library / "Ed Sheeran" / "÷"
    (album / "01. Eraser.wav").write_bytes(b"x")
    out = str(tmp_path / "out")
    scope, _ = resolve_scope(str(library), out=out, registry=registry)
    pairs = plan(scope, find_audio(str(library)))
    dirs = [d for _, d in pairs]
    assert len(set(dirs)) == len(dirs)
    clashing = sorted(os.path.basename(d) for s, d in pairs
                      if os.path.basename(s).startswith("01. Eraser"))
    assert clashing == ["01. Eraser.flac", "01. Eraser.wav"]


# ------------------------------------------------------------- the skip ledger

def _finish(out_dir, source, profile="full", stems=False, schema=SCHEMA_VERSION):
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "analysis.json"), "w").write("{}")
    write_ledger(out_dir, source, profile, stems,
                 {"file": {"sha256": "deadbeef"}},
                 {"analysis.json": os.path.join(out_dir, "analysis.json")}, 1.0)
    if schema != SCHEMA_VERSION:
        p = os.path.join(out_dir, "mtx_source.json")
        data = json.load(open(p, encoding="utf-8"))
        data["run"]["schema_version"] = schema
        json.dump(data, open(p, "w", encoding="utf-8"), indent=1)


def test_a_finished_track_is_skipped(library, tmp_path):
    src = str(library / "Ed Sheeran" / "÷" / "01. Eraser.flac")
    out = str(tmp_path / "out" / "track")
    assert why_stale(src, out, "full", False) == "not analysed yet"
    _finish(out, src)
    assert why_stale(src, out, "full", False) is None


@pytest.mark.parametrize("mutate, expected", [
    (lambda p: os.utime(p, (1, 1)), "source modified"),
    (lambda p: open(p, "ab").write(b"more"), "source size changed"),
])
def test_a_changed_source_is_re_measured(library, tmp_path, mutate, expected):
    src = str(library / "Ed Sheeran" / "÷" / "01. Eraser.flac")
    out = str(tmp_path / "out" / "track")
    _finish(out, src)
    mutate(src)
    assert why_stale(src, out, "full", False) == expected


def test_settings_and_schema_changes_invalidate(library, tmp_path):
    src = str(library / "Ed Sheeran" / "÷" / "01. Eraser.flac")
    out = str(tmp_path / "out" / "track")
    _finish(out, src)
    assert why_stale(src, out, "quick", False).startswith("profile ")
    assert why_stale(src, out, "full", True) == "stems setting changed"

    _finish(out, src, schema="0.0.1")
    assert why_stale(src, out, "full", False) == "schema 0.0.1 -> " + SCHEMA_VERSION


def test_a_deleted_analysis_invalidates_the_receipt(library, tmp_path):
    """The receipt must not outlive what it is a receipt for."""
    src = str(library / "Ed Sheeran" / "÷" / "01. Eraser.flac")
    out = str(tmp_path / "out" / "track")
    _finish(out, src)
    os.remove(os.path.join(out, "analysis.json"))
    assert why_stale(src, out, "full", False) == "analysis.json missing"


def test_recheck_notices_a_touched_but_unchanged_file(library, tmp_path):
    """Copying a library moves every mtime; --recheck must not be fooled."""
    src = str(library / "Ed Sheeran" / "÷" / "01. Eraser.flac")
    out = str(tmp_path / "out" / "track")
    _finish(out, src)
    os.utime(src, (1, 1))
    assert why_stale(src, out, "full", False) == "source modified"
    # The recorded sha256 is a fabricated one, so a real hash disagrees; what
    # matters is that mtime alone no longer decides.
    assert why_stale(src, out, "full", False, recheck=True) == \
        "source contents changed"


def test_ledger_records_what_a_later_run_needs(library, tmp_path):
    src = str(library / "Ed Sheeran" / "÷" / "01. Eraser.flac")
    out = str(tmp_path / "out" / "track")
    _finish(out, src)
    led = json.load(open(os.path.join(out, "mtx_source.json"), encoding="utf-8"))
    assert led["source"]["path"] == src
    assert led["source"]["sha256"] == "deadbeef"
    assert led["run"]["schema_version"] == SCHEMA_VERSION
    assert led["run"]["tool_version"] == __version__


# --------------------------------------------------------------------- summary

def test_summary_covers_earlier_runs_too(tmp_path):
    """A scan of one album must not drop the rest of the library from the CSV."""
    for name, title in (("a", "First"), ("b", "Second")):
        d = tmp_path / "out" / name
        d.mkdir(parents=True)
        (d / "corpus_row.json").write_text(
            json.dumps({"Title": title, "LUFS-I": -8.5,
                        "_source": {"file": name + ".flac"}}), encoding="utf-8")
    path, n = write_summary(str(tmp_path / "out"))
    assert n == 2
    text = open(path, encoding="utf-8").read()
    assert "First" in text and "Second" in text and "LUFS-I" in text


# ------------------------------------------------------------ the two layers

@pytest.mark.parametrize("jobs, todo, expected", [
    (8, 100, (8, 1)),    # a library: every worker a process, one thread each
    (8, 1, (1, 8)),      # one file left: spend the budget on threads instead
    (8, 3, (3, 2)),      # a short tail: split it
    (1, 50, (1, 1)),
    (12, 12, (12, 1)),
])
def test_budget_is_never_exceeded(jobs, todo, expected):
    procs, threads = split_budget(jobs, todo)
    assert (procs, threads) == expected
    assert procs * threads <= jobs
    assert procs <= todo


def test_scan_workers_are_pinned_to_one_thread(monkeypatch):
    env = single_threaded_env()
    assert env["OMP_NUM_THREADS"] == "1"
    monkeypatch.setenv("MTX_THREADS", "1")
    assert resolve_threads(None) == 1
    assert resolve_threads(4) == 4, "an explicit --jobs still wins"


def test_ordered_window_preserves_order_and_bounds_flight():
    """Out-of-order folding would double-count an over across a chunk edge."""
    live = []
    high_water = []

    def slow(i):
        live.append(i)
        high_water.append(len(live))
        live.pop()
        return i * 2

    got = list(ordered_window(slow, list(range(50)), workers=4, lookahead=6))
    assert got == [i * 2 for i in range(50)]
    assert max(high_water) <= 6
    assert list(ordered_window(slow, [], workers=4)) == []


# ------------------------------------------------------- one master, one scan

def _finished(out_root, rel, data, *, profile="full", stems=False,
              schema=None, stems_model=None):
    """A folder that looks like a track measured on an earlier run."""
    folder = os.path.join(out_root, rel)
    os.makedirs(folder, exist_ok=True)
    source = os.path.join(folder, "pretend-source.flac")
    with open(source, "wb") as f:
        f.write(data)
    res = {"file": {"path_absolute": source, "filename": "pretend-source.flac",
                    "sha256": hashlib.sha256(data).hexdigest()},
           "audio": {"duration_s": 1.0},
           "run": {"profile": profile, "stems_model": stems_model},
           "declared": declared_mod.load(source)}
    with open(os.path.join(folder, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(res, f)
    write_ledger(folder, source, profile, stems, res,
                 {"analysis.json": "analysis.json"}, 1.0)
    if schema is not None:
        with open(ledger_path(folder), encoding="utf-8") as f:
            led = json.load(f)
        led["run"]["schema_version"] = schema
        with open(ledger_path(folder), "w", encoding="utf-8") as f:
            json.dump(led, f)
    return folder, source


@pytest.fixture()
def dedup_scope(tmp_path, registry):
    """An empty two-folder library, registered, with nothing measured yet."""
    lib = tmp_path / "lib"
    (lib / "Singles").mkdir(parents=True)
    (lib / "Album").mkdir(parents=True)
    scope, _ = resolve_scope(str(lib), out=str(tmp_path / "out"), registry=registry)
    register(scope, registry)
    return scope, lib


def _todo(scope, lib):
    return [(src, out_dir_for(scope, src), "not analysed yet")
            for src in find_audio(str(lib))]


def test_the_same_master_under_two_names_is_measured_once(dedup_scope):
    """The single next to the album track it was lifted from."""
    scope, lib = dedup_scope
    audio = b"RIFF" + os.urandom(4096)
    (lib / "Singles" / "Watermelon Sugar.flac").write_bytes(audio)
    (lib / "Album" / "02. Watermelon Sugar.flac").write_bytes(audio)

    todo = _todo(scope, lib)
    keep, dupes = partition_duplicates(scope, todo, "full", False, None)

    assert len(keep) == 1 and len(dupes) == 1
    assert dupes[0].within_run, "its twin is measured by this same run"
    assert dupes[0].twin_dir == keep[0][1]
    assert {keep[0][0], dupes[0].source} == {p for p, _, _ in todo}


def test_two_files_of_one_size_but_different_bytes_are_both_measured(dedup_scope):
    """Size is the cheap filter, never the answer."""
    scope, lib = dedup_scope
    (lib / "Singles" / "a.flac").write_bytes(b"A" * 4096)
    (lib / "Album" / "b.flac").write_bytes(b"B" * 4096)

    keep, dupes = partition_duplicates(scope, _todo(scope, lib), "full", False, None)
    assert len(keep) == 2 and dupes == []


def test_a_file_whose_size_is_unique_is_never_opened(dedup_scope, monkeypatch):
    """Hashing a library to learn it holds no duplicates would cost more."""
    scope, lib = dedup_scope
    (lib / "Singles" / "a.flac").write_bytes(b"A" * 4096)
    (lib / "Album" / "b.flac").write_bytes(b"B" * 8192)

    def refuse(path, chunk=1 << 20):
        raise AssertionError(f"hashed {path}, which cannot be a duplicate")

    monkeypatch.setattr("mtx.scan._sha256", refuse)
    keep, dupes = partition_duplicates(scope, _todo(scope, lib), "full", False, None)
    assert len(keep) == 2 and dupes == []


def test_a_copy_measured_on_an_earlier_run_is_found(dedup_scope):
    """The twin need not be in this scan, or anywhere near this album."""
    scope, lib = dedup_scope
    audio = b"RIFF" + os.urandom(4096)
    twin_dir, _ = _finished(scope.out_dir, os.path.join("Elsewhere", "Track"), audio)
    (lib / "Album" / "02. Watermelon Sugar.flac").write_bytes(audio)

    keep, dupes = partition_duplicates(scope, _todo(scope, lib), "full", False, None)
    assert keep == []
    assert len(dupes) == 1 and not dupes[0].within_run
    assert dupes[0].twin_dir == twin_dir


@pytest.mark.parametrize("settings", [
    {"profile": "quick"},
    {"stems": True},
    {"schema": "0.0.0-not-this-one"},
])
def test_a_twin_measured_under_other_settings_is_not_adopted(dedup_scope, settings):
    """A copy is only worth adopting if it answers the question being asked."""
    scope, lib = dedup_scope
    audio = b"RIFF" + os.urandom(4096)
    _finished(scope.out_dir, os.path.join("Elsewhere", "Track"), audio, **settings)
    (lib / "Album" / "02. Watermelon Sugar.flac").write_bytes(audio)

    keep, dupes = partition_duplicates(scope, _todo(scope, lib), "full", False, None)
    assert dupes == [] and len(keep) == 1


def test_a_copy_lands_a_folder_the_next_scan_skips(dedup_scope):
    """A duplicate's folder is an ordinary result: receipt, provenance and all."""
    scope, lib = dedup_scope
    audio = b"RIFF" + os.urandom(4096)
    twin_dir, twin_src = _finished(scope.out_dir, os.path.join("Elsewhere", "T"), audio)
    copy = lib / "Album" / "02. Watermelon Sugar.flac"
    copy.write_bytes(audio)
    out_dir = out_dir_for(scope, str(copy))

    r = materialize_duplicate(Duplicate(str(copy), out_dir, twin_dir, False),
                              profile="full", stems=False, json_only=True,
                              plots=False, max_part_bytes=None)

    assert r["ok"], r.get("error")
    assert r["duplicate_of"] == twin_src
    assert why_stale(str(copy), out_dir, "full", False) is None, \
        "the copy has to be skipped next time, not copied again"

    with open(os.path.join(out_dir, "analysis.json"), encoding="utf-8") as f:
        written = json.load(f)
    assert written["file"]["path_absolute"] == os.path.abspath(str(copy))
    assert written["file"]["filename"] == "02. Watermelon Sugar.flac"
    assert written["file"]["sha256"] == hashlib.sha256(audio).hexdigest()
    assert written["run"]["duplicate_of"] == twin_src, "provenance is not optional"


def test_a_declared_sidecar_that_disagrees_forces_a_real_measurement(dedup_scope):
    """The one input that does not travel with the bytes."""
    scope, lib = dedup_scope
    audio = b"RIFF" + os.urandom(4096)
    (lib / "Singles" / "Watermelon Sugar.flac").write_bytes(audio)
    (lib / "Album" / "02. Watermelon Sugar.flac").write_bytes(audio)
    (lib / "Album" / "02. Watermelon Sugar.declared.json").write_text(
        json.dumps({"title": "Watermelon Sugar", "year": 2019}), encoding="utf-8")

    keep, dupes = partition_duplicates(scope, _todo(scope, lib), "full", False, None)
    assert dupes == [] and len(keep) == 2


def test_dedup_can_be_turned_off(dedup_scope, registry):
    """--no-dedup measures every copy separately."""
    scope, lib = dedup_scope
    audio = b"RIFF" + os.urandom(4096)
    (lib / "Singles" / "a.flac").write_bytes(audio)
    (lib / "Album" / "b.flac").write_bytes(audio)
    common = dict(out=scope.out_dir, library_root=scope.library_root,
                  registry=registry, profile="full", dry_run=True,
                  log=lambda *a: None)

    off = run_scan(str(lib), dedup=False, **common)
    assert (off["todo"], off["duplicates"]) == (2, 0)
    on = run_scan(str(lib), dedup=True, **common)
    assert (on["todo"], on["duplicates"]) == (1, 1)


# --------------------------------------------------------- the stem cache key

def test_stem_cache_is_keyed_on_contents_not_on_the_path(tmp_path):
    """Separation is the expensive half; two copies must not each pay for it."""
    from mtx.metrics.stems import _content_key

    audio = b"RIFF" + os.urandom(4096)
    one = tmp_path / "Watermelon Sugar.flac"
    two = tmp_path / "album" / "02. Watermelon Sugar.flac"
    two.parent.mkdir()
    one.write_bytes(audio)
    two.write_bytes(audio)
    other = tmp_path / "different.flac"
    other.write_bytes(b"RIFF" + os.urandom(4096))

    assert _content_key(str(one)) == _content_key(str(two))
    assert _content_key(str(one)) != _content_key(str(other))


# ------------------------------------------------------- where stems separate

def test_the_device_is_the_gpu_when_there_is_one(monkeypatch):
    from mtx.metrics import stems as m

    monkeypatch.delenv(m.ENV_DEVICE, raising=False)
    monkeypatch.setattr(m, "cuda_available", lambda: True)
    assert m.resolve_device() == "cuda"
    monkeypatch.setattr(m, "cuda_available", lambda: False)
    assert m.resolve_device() == "cpu"


def test_an_asked_for_device_is_not_second_guessed(monkeypatch):
    """--stems-device cpu has to win over a card that is sitting right there."""
    from mtx.metrics import stems as m

    monkeypatch.setattr(m, "cuda_available", lambda: True)
    monkeypatch.setenv(m.ENV_DEVICE, "cpu")
    assert m.resolve_device() == "cpu"
    assert m.resolve_device("cuda") == "cuda", "an argument beats the environment"


def test_the_gpu_steps_down_through_segments_then_falls_back(monkeypatch):
    """Slow beats absent: the CPU is the last rung, never the first."""
    from mtx.metrics import stems as m

    monkeypatch.setattr(m, "_LEARNED", {})
    ladder = m._attempts("cuda", None, "htdemucs")
    assert [d for d, _ in ladder] == ["cuda"] * len(m.GPU_SEGMENTS) + ["cpu"]
    assert [s for _, s in ladder[:-1]] == list(m.GPU_SEGMENTS)
    assert ladder[0][1] > ladder[1][1], "longest segment first"

    assert m._attempts("cpu", None, "htdemucs") == [("cpu", None)]
    assert m._attempts("cuda", 4, "htdemucs") == [("cuda", 4), ("cpu", None)]


def test_what_the_card_held_is_learned_once_not_per_track(monkeypatch):
    """Every failed attempt costs a model load; a scan must pay that once."""
    from mtx.metrics import stems as m

    monkeypatch.setattr(m, "_LEARNED", {("cuda", "htdemucs"): 3})
    ladder = m._attempts("cuda", None, "htdemucs")
    assert [s for _, s in ladder] == [s for s in m.GPU_SEGMENTS if s <= 3] + [None]
    assert all(s <= 3 for _, s in ladder if s is not None)

    monkeypatch.setattr(m, "_LEARNED", {("cuda", "htdemucs"): None})
    assert m._attempts("cuda", None, "htdemucs") == [("cpu", None)], \
        "a card that never held it is not tried again"


def test_a_gpu_child_is_not_pinned_to_one_thread(monkeypatch):
    """The scan pins its workers; on a card that pin is left behind."""
    from mtx.metrics import stems as m

    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MTX_THREADS", "1")
    assert m._child_env("cpu")["OMP_NUM_THREADS"] == "1"
    assert "OMP_NUM_THREADS" not in m._child_env("cuda")
    assert "MTX_THREADS" not in m._child_env("cuda")


@pytest.mark.parametrize("stderr, expected", [
    ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB", True),
    ("RuntimeError: CUDA error: an illegal memory access", True),
    ("FileNotFoundError: no such model", False),
    ("", False),
])
def test_only_a_memory_failure_steps_down(stderr, expected):
    from mtx.metrics import stems as m

    assert m._out_of_memory(stderr) is expected


# ------------------------------------------------- how many share the card

@pytest.mark.parametrize("vram, expected", [
    (4095, 3),      # GTX 1650: three streams measured at 1.51x, four will not fit
    (2048, 1),      # a 2 GiB card has room for one and no reserve to spare
    (6144, 4),      # past four the SM is saturated; the cap, not the memory, binds
    (24564, 4),
])
def test_streams_are_what_the_card_holds(vram, expected, monkeypatch):
    from mtx.metrics import stems as m

    monkeypatch.delenv(m.ENV_STREAMS, raising=False)
    monkeypatch.setattr(m, "device_vram_mib", lambda: vram)
    assert m.separation_streams() == expected


def test_no_card_means_one_at_a_time(monkeypatch):
    from mtx.metrics import stems as m

    monkeypatch.delenv(m.ENV_STREAMS, raising=False)
    monkeypatch.setattr(m, "device_vram_mib", lambda: None)
    assert m.separation_streams() == 1


def test_an_explicit_request_beats_the_arithmetic(monkeypatch):
    """A card this reads wrongly can still be driven by hand."""
    from mtx.metrics import stems as m

    monkeypatch.setattr(m, "device_vram_mib", lambda: 4095)
    assert m.separation_streams(1) == 1
    assert m.separation_streams(8) == 8
    monkeypatch.setenv(m.ENV_STREAMS, "2")
    assert m.separation_streams() == 2


def _jobs(n, prefix="/library"):
    return [{"source": f"{prefix}/{i:02d}.flac", "out_dir": f"/out/{i:02d}"}
            for i in range(n)]


class _ThreadPool:
    """A real pool, for the tests that need submit() to return before the work."""

    def __init__(self, fn, workers):
        self.fn = fn
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def submit(self, job):
        return self.pool.submit(self.fn, job)


class _InlinePool:
    """A pool that measures on the calling thread, so tests stay deterministic."""

    def __init__(self, fn):
        self.fn = fn

    def submit(self, job):
        fut = Future()
        try:
            fut.set_result(self.fn(job))
        except Exception as exc:            # pragma: no cover - test helper
            fut.set_exception(exc)
        return fut


def test_separations_overlap_but_every_track_is_reported_once(monkeypatch):
    """Three streams, eight tracks, one line each and one lock around them."""
    import threading

    from mtx import scan as scan_mod
    from mtx.metrics import stems as m

    live, peak = 0, 0
    guard = threading.Lock()

    def fake_separate(path, collector, model=None, **kw):
        nonlocal live, peak
        with guard:
            live += 1
            peak = max(peak, live)
        time.sleep(0.05)
        with guard:
            live -= 1
        return {"vocals": path}

    monkeypatch.setattr(m, "separate", fake_separate)
    monkeypatch.setattr(m, "entry_for", lambda paths: None)
    lines = []
    jobs = _jobs(8)
    separate, tally = scan_mod.separation_stage(len(jobs), None, lines.append)
    scan_mod.drive(jobs, _InlinePool(lambda j: {"source": j["source"],
                                                "out_dir": j["out_dir"],
                                                "ok": True}).submit,
                   lambda r: None, separate=separate, streams=3,
                   lookahead=8, log=lines.append)

    assert tally["done"] == 8
    assert peak == 3, "the card was given exactly the streams it was promised"
    reported = [ln for ln in lines if ln.startswith("[stems ")]
    assert len(reported) == 8
    assert sorted(int(ln.split("/")[0].rsplit(" ", 1)[1])
                  for ln in reported) == list(range(1, 9))


def test_one_stream_separates_in_order(monkeypatch):
    """`streams=1` hands the card one track at a time, in the order given."""
    from mtx import scan as scan_mod
    from mtx.metrics import stems as m

    order = []
    monkeypatch.setattr(m, "separate", lambda p, c, model=None, **kw:
                        (order.append(p), {"vocals": p})[1])
    monkeypatch.setattr(m, "entry_for", lambda paths: None)
    jobs = _jobs(2)
    separate, _ = scan_mod.separation_stage(len(jobs), None, lambda _: None)
    scan_mod.drive(jobs, _InlinePool(lambda j: {"source": j["source"],
                                                "out_dir": j["out_dir"],
                                                "ok": True}).submit,
                   lambda r: None, separate=separate, streams=1, lookahead=2)
    assert order == ["/library/00.flac", "/library/01.flac"]


def test_measuring_starts_before_every_track_is_separated(monkeypatch):
    """The point of the pipeline: track 1 is measured while 2 is separating.

    The old pass separated the whole todo list first, so the first measurement
    could not begin until the last separation ended.  Here the first job must
    reach the pool while separations are still running, or the card and the
    cores are still taking turns.
    """
    import threading

    from mtx import scan as scan_mod
    from mtx.metrics import stems as m

    separating = 0
    guard = threading.Lock()
    overlapped = []

    def fake_separate(path, collector, model=None, **kw):
        nonlocal separating
        with guard:
            separating += 1
        time.sleep(0.05)
        with guard:
            separating -= 1
        return {"vocals": path}

    def measure(job):
        with guard:
            overlapped.append(separating > 0)
        return {"source": job["source"], "out_dir": job["out_dir"], "ok": True}

    monkeypatch.setattr(m, "separate", fake_separate)
    monkeypatch.setattr(m, "entry_for", lambda paths: None)
    jobs = _jobs(8)
    separate, _ = scan_mod.separation_stage(len(jobs), None, lambda _: None)
    scan_mod.drive(jobs, _InlinePool(measure).submit, lambda r: None,
                   separate=separate, streams=3, lookahead=8)

    assert len(overlapped) == 8
    assert any(overlapped), "no track was measured while the card was busy"


def test_separation_never_runs_further_ahead_than_the_permits(monkeypatch):
    """The disk bound: unmeasured separations may not exceed the lookahead.

    Each one is four uncompressed wavs, so this is the whole reason a library
    scan can run in one pass -- without it the card, six times faster than the
    measuring, separates the library into a full disk.
    """
    import threading

    from mtx import scan as scan_mod
    from mtx.metrics import stems as m

    guard = threading.Lock()
    outstanding = 0
    high_water = 0

    def fake_separate(path, collector, model=None, **kw):
        nonlocal outstanding, high_water
        time.sleep(0.02)
        with guard:
            outstanding += 1
            high_water = max(high_water, outstanding)
        return {"vocals": path}

    def report(r):
        nonlocal outstanding
        with guard:
            outstanding -= 1

    monkeypatch.setattr(m, "separate", fake_separate)
    monkeypatch.setattr(m, "entry_for", lambda paths: None)
    jobs = _jobs(30)
    separate, _ = scan_mod.separation_stage(len(jobs), None, lambda _: None)
    scan_mod.drive(jobs, _InlinePool(
        lambda j: {"source": j["source"], "out_dir": j["out_dir"],
                   "ok": True}).submit,
        report, separate=separate, streams=4, lookahead=5)

    assert high_water <= 5, f"{high_water} tracks were on disk, 5 was the bound"


def test_a_lost_worker_is_reported_against_its_own_track():
    """A worker that dies outright never returned a result to report."""
    from mtx import scan as scan_mod

    def measure(job):
        if job["source"].endswith("02.flac"):
            raise MemoryError("out of memory")
        return {"source": job["source"], "out_dir": job["out_dir"], "ok": True}

    seen = []
    scan_mod.drive(_jobs(4), _InlinePool(measure).submit, seen.append)

    assert len(seen) == 4
    bad = [r for r in seen if not r["ok"]]
    assert len(bad) == 1
    assert bad[0]["source"].endswith("02.flac")
    assert "MemoryError" in bad[0]["error"]


def test_a_separation_that_raises_still_gets_its_track_measured(monkeypatch):
    """One unseparable track must not strand the rest of a library scan."""
    from mtx import scan as scan_mod
    from mtx.metrics import stems as m

    def fake_separate(path, collector, model=None, **kw):
        if path.endswith("01.flac"):
            raise RuntimeError("the card fell over")
        return {"vocals": path}

    monkeypatch.setattr(m, "separate", fake_separate)
    monkeypatch.setattr(m, "entry_for", lambda paths: None)
    jobs = _jobs(4)
    separate, _ = scan_mod.separation_stage(len(jobs), None, lambda _: None)
    seen = []
    scan_mod.drive(jobs, _InlinePool(
        lambda j: {"source": j["source"], "out_dir": j["out_dir"],
                   "ok": True}).submit,
        seen.append, separate=separate, streams=2, lookahead=4)

    assert len(seen) == 4 and all(r["ok"] for r in seen)


def test_big_tracks_narrow_the_pool_instead_of_swapping():
    """The night this was written for: six lanes of 192 kHz against 32 GB.

    The bound is on bytes in flight, not on jobs, so a run of large tracks
    admits fewer of them at once and a run of small ones admits more.
    """
    import threading

    from mtx import scan as scan_mod

    guard = threading.Lock()
    live = 0
    high_water = 0

    def measure(job):
        nonlocal live, high_water
        with guard:
            live += job["bytes"]
            high_water = max(high_water, live)
        time.sleep(0.01)
        return {"source": job["source"], "out_dir": job["out_dir"], "ok": True}

    def report(r):
        nonlocal live
        with guard:
            live -= by_source[r["source"]]["bytes"]

    jobs = _jobs(20)
    for j in jobs:
        j["bytes"] = 5_000_000_000          # 5 GB a track
    by_source = {j["source"]: j for j in jobs}

    scan_mod.drive(jobs, _ThreadPool(measure, 6).submit, report,
                   budget=21_000_000_000)

    assert high_water <= 21_000_000_000, (
        f"{high_water / 1e9:.0f} GB was in flight against a 21 GB budget")


def test_a_track_bigger_than_the_budget_still_gets_measured():
    """Refusing it forever would hang the run; it goes when the pool is empty."""
    from mtx import scan as scan_mod

    jobs = _jobs(3)
    for j in jobs:
        j["bytes"] = 40_000_000_000         # larger than the whole budget
    seen = []
    scan_mod.drive(jobs, _ThreadPool(
        lambda j: {"source": j["source"], "out_dir": j["out_dir"],
                   "ok": True}, 4).submit,
        seen.append, budget=21_000_000_000)
    assert len(seen) == 3 and all(r["ok"] for r in seen)


def test_no_budget_leaves_the_pool_to_decide():
    """A machine whose RAM cannot be read keeps the behaviour it had."""
    from mtx import scan as scan_mod

    jobs = _jobs(12)
    for j in jobs:
        j["bytes"] = 5_000_000_000
    seen = []
    scan_mod.drive(jobs, _ThreadPool(
        lambda j: {"source": j["source"], "out_dir": j["out_dir"],
                   "ok": True}, 6).submit,
        seen.append, budget=0)
    assert len(seen) == 12


def test_the_band_cap_is_what_stops_a_192k_master_costing_four_times_more():
    """Four times the samples, but the band-rate views do not grow at all."""
    from mtx.scan import decoded_bytes

    at_44 = decoded_bytes(13_230_000, 2, 44100, 300, stems=False)
    at_192 = decoded_bytes(57_600_000, 2, 192000, 300, stems=False)
    assert at_192 > at_44
    assert at_192 < 4 * at_44, "the 48 kHz cap bought nothing"


def test_stems_are_sized_by_duration_not_by_the_masters_rate():
    """demucs writes at its own rate, so two masters of one song agree."""
    from mtx.scan import decoded_bytes

    a = decoded_bytes(13_230_000, 2, 44100, 300, stems=True) - \
        decoded_bytes(13_230_000, 2, 44100, 300, stems=False)
    b = decoded_bytes(57_600_000, 2, 192000, 300, stems=True) - \
        decoded_bytes(57_600_000, 2, 192000, 300, stems=False)
    assert a == b


def test_an_unreadable_header_opts_out_of_memory_scheduling(tmp_path):
    from mtx.scan import job_bytes
    bad = tmp_path / "not-audio.flac"
    bad.write_bytes(b"nope")
    assert job_bytes(str(bad), stems=True) == 0


@pytest.mark.parametrize("procs,streams,asked,want", [
    (6, 3, None, 10),        # every worker fed, every stream working, one spare
    (6, 3, 4, 10),           # an ask under the floor cannot starve the pool
    (6, 3, 40, 40),          # an ask over it is honoured
    (1, 1, None, 3),
])
def test_the_lookahead_floor_keeps_every_worker_fed(procs, streams, asked, want):
    from mtx import scan as scan_mod
    assert scan_mod.lookahead_for(procs, streams, asked) == want


class _BreakablePool:
    """A pool whose worker dies partway, the way an out-of-memory kill does.

    A killed worker does not fail its own track politely: it breaks the
    executor, so the running future and every later submission raise
    `BrokenProcessPool` until something rebuilds it.
    """

    def __init__(self, fn, die_on=2, deaths=1):
        self.fn = fn
        self.die_on = die_on
        self.deaths = deaths
        self.n = 0
        self.broken = False
        self.rebuilds = 0

    def submit(self, job):
        from concurrent.futures.process import BrokenProcessPool
        if self.broken:
            raise BrokenProcessPool("pool is broken")
        self.n += 1
        fut = Future()
        if self.n == self.die_on and self.deaths > 0:
            self.deaths -= 1
            self.broken = True
            fut.set_exception(BrokenProcessPool("worker terminated abruptly"))
        else:
            fut.set_result(self.fn(job))
        return fut

    def restart(self):
        self.broken = False
        self.rebuilds += 1


def _ok(j):
    return {"source": j["source"], "out_dir": j["out_dir"], "ok": True}


def test_a_dead_worker_does_not_end_the_run():
    """One killed worker costs its own track a retry, not the whole library."""
    from mtx import scan as scan_mod

    jobs = _jobs(20)
    pool = _BreakablePool(_ok, die_on=2)
    seen = []
    scan_mod.drive(jobs, pool.submit, seen.append,
                   restart=pool.restart, log=lambda m: None)

    assert len(seen) == 20, "every track was accounted for"
    assert all(r["ok"] for r in seen), "and the rebuilt pool measured them"
    assert pool.rebuilds == 1, "rebuilt once, not once per track it took down"


class _BatchKillPool:
    """A worker dies with several tracks in flight, taking all of them down.

    This is what an out-of-memory kill actually looks like: not one future
    failing, but every future the dead worker held, plus every submission
    afterwards, until the executor is replaced.
    """

    def __init__(self, fn, kill_at=3):
        self.fn = fn
        self.kill_at = kill_at
        self.pending = []
        self.n = 0
        self.broken = False
        self.rebuilds = 0

    def submit(self, job):
        from concurrent.futures.process import BrokenProcessPool
        if self.broken:
            raise BrokenProcessPool("pool is broken")
        self.n += 1
        fut = Future()
        if self.n < self.kill_at:
            self.pending.append(fut)        # still measuring
            return fut
        if self.n == self.kill_at:
            self.broken = True
            for f in self.pending + [fut]:
                f.set_exception(BrokenProcessPool("worker terminated abruptly"))
            self.pending = []
            return fut
        fut.set_result(self.fn(job))
        return fut

    def restart(self):
        self.broken = False
        self.kill_at = -1                   # rebuilt narrower; it holds now
        for f in self.pending:
            f.set_result(self.fn({"source": "x", "out_dir": "y"}))
        self.pending = []
        self.rebuilds += 1


def test_the_pool_is_rebuilt_once_per_break_not_once_per_victim():
    """A dying worker takes several tracks with it; they share one rebuild."""
    from mtx import scan as scan_mod

    jobs = _jobs(12)
    pool = _BatchKillPool(_ok, kill_at=3)
    seen = []
    scan_mod.drive(jobs, pool.submit, seen.append,
                   restart=pool.restart, log=lambda m: None)

    assert len(seen) == 12, "every track was accounted for"
    assert all(r["ok"] for r in seen), "and all of them measured in the end"
    assert pool.rebuilds == 1, "one break is one rebuild, not one per victim"


def test_without_a_restart_a_broken_pool_still_terminates():
    """No `restart` is the old behaviour: report the failures, but finish."""
    from mtx import scan as scan_mod

    jobs = _jobs(12)
    pool = _BreakablePool(_ok, die_on=2)
    seen = []
    scan_mod.drive(jobs, pool.submit, seen.append, log=lambda m: None)

    assert len(seen) == 12
    assert sum(1 for r in seen if r.get("ok")) == 1
    assert all("BrokenProcessPool" in r["error"]
               for r in seen if not r.get("ok"))


def test_a_track_that_breaks_the_pool_twice_is_reported_not_retried_forever():
    """The retry is once per track, so a genuinely fatal file cannot loop."""
    from mtx import scan as scan_mod

    jobs = _jobs(3)
    # Dies on every submission, so the retry breaks it again.
    pool = _BreakablePool(_ok, die_on=1, deaths=99)
    pool.die_on = 1

    def submit(job):
        from concurrent.futures.process import BrokenProcessPool
        fut = Future()
        fut.set_exception(BrokenProcessPool("worker terminated abruptly"))
        return fut

    seen = []
    scan_mod.drive(jobs, submit, seen.append,
                   restart=lambda: None, log=lambda m: None)

    assert len(seen) == 3, "it terminated instead of retrying forever"
    assert all("worker lost" in r["error"] for r in seen)


def test_a_break_tightens_the_budget():
    """A pool only breaks because it was too wide, so the next one is narrower."""
    from mtx import scan as scan_mod

    jobs = _jobs(6)
    for j in jobs:
        j["bytes"] = 1_000_000_000
    pool = _BreakablePool(_ok, die_on=2)
    lines = []
    scan_mod.drive(jobs, pool.submit, lambda r: None, budget=8_000_000_000,
                   restart=pool.restart, log=lines.append)

    narrowed = [ln for ln in lines if "rebuilding the pool" in ln]
    assert narrowed, "the rebuild was reported"
    assert "6 GB" in narrowed[0], f"8 GB tightened to 6, got {narrowed[0]!r}"
