"""Scope resolution, the skip ledger and the parallelism budget.

These are the three things `mtx scan` promises: that the same track lands in
the same place whichever level the command is run from, that work already done
is not done again, and that the process and thread layers never multiply into
more workers than the machine has.  None of them need audio to test, so none
of these tests decode any.
"""

from __future__ import annotations

import json
import os

import pytest

from mtx import SCHEMA_VERSION, __version__
from mtx.parallel import ordered_window, resolve_threads, single_threaded_env
from mtx.scan import (NoRootRegistered, find_audio, out_dir_for, plan,
                      register, resolve_scope, split_budget, why_stale,
                      write_ledger, write_summary)


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
