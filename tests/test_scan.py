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
