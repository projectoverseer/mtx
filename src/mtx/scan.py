"""`mtx scan`: measure a folder of masters, once, in parallel, and resume.

Three things make a library scan different from `mtx batch`, and this module
exists for all three.

**Scope follows the working directory.**  The same command run on an album, on
an artist, or on the whole library measures exactly what sits underneath it.
Nothing about the output changes: a track analysed as part of a library scan
and the same track analysed alone land in the same folder, so the second run
finds the first one's work.

**Already-measured tracks are not measured again.**  Each output folder carries
a small `mtx_source.json` naming the file it came from -- size, modification
time, sha256, profile, schema version.  A scan reads those, not the audio, and
a track whose record still matches is skipped in microseconds instead of a
minute.  That is what makes an interrupted scan resumable: every finished
track wrote its own receipt, so there is no separate progress file to lose.

That extends past the file itself.  Every number mtx reports is a function of
the audio bytes, so a second copy of one master -- the single sitting next to
the album it was lifted from, the same rip under a tidier name -- has one
measurement between it and the first, and the second copy adopts it rather
than spending the minutes again.  Copies are found by sha256, but only among
files that share a size with something, so no library is hashed to discover
that it holds no duplicates.  `--no-dedup` measures every copy separately.

**Files are measured in parallel processes.**  Measured on this codebase, a
full-profile run is a serial chain of numpy and scipy calls that keeps roughly
one core busy; the GIL-bound parts of it (the band split, the per-frame Welch
loops) cannot be threaded at all.  Between files there is no shared state, so
processes scale where threads cannot.  See `parallel.py` for how the process
count and the per-process thread count are kept from multiplying.

Output lives in a mirror of the library tree, under an `--out` directory
recorded once per library root:

    E:\\Music\\Ed Sheeran\\<div>\\04. Shape of You.flac
    E:\\mtx_out\\Ed Sheeran\\<div>\\04. Shape of You\\analysis.json

The library root is what makes that mapping stable no matter which level the
command is run from, so it is stored (in the user config directory) the first
time a root is scanned rather than guessed on each run.
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import Counter
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

from . import SCHEMA_VERSION, __version__
from .parallel import (apply_single_threaded_env, cpu_count, default_workers,
                       single_threaded_env)
from .split import DEFAULT_PART_BYTES

AUDIO_EXTENSIONS = (".flac", ".wav", ".aif", ".aiff", ".w64", ".caf", ".ogg",
                    ".opus", ".mp3", ".m4a", ".aac", ".wv", ".ape")

LEDGER_NAME = "mtx_source.json"
ROOT_MARKER = ".mtx-root.json"
SUMMARY_NAME = "summary.csv"

# Directories that never hold masters worth measuring.
# A mirror tree holds no audio, but a stem cache is nothing but audio, and both
# are commonly parked beside the library they describe.  Walking into either is
# how a scan ends up measuring its own output: a separated vocal is not a
# master, and analysing one would quietly poison the corpus.
SKIP_DIRS = {".mtx", ".mtx_cache", "mtx_out", "_mtx_out", "_mtx_stems",
             ".git", ".svn", "__pycache__",
             "$RECYCLE.BIN", "System Volume Information"}

ENV_REGISTRY = "MTX_SCAN_ROOTS"

# Columns of the scan summary, in the corpus database's own property names so
# the file imports as a populated table rather than a mapping exercise.
SUMMARY_FIELDS = [
    "Path", "Artist", "Title", "Year", "Genre", "LUFS-I", "True peak", "LRA",
    "PLR", "PSR min", "PSR median", "DR14", "Crest (loudest 10s)",
    "Tonal tilt notes", "Width/mono notes", "mtx run",
]


# --------------------------------------------------------------- the registry

def registry_path() -> str:
    """Where the library-root registry lives.  `MTX_SCAN_ROOTS` overrides it."""
    override = os.environ.get(ENV_REGISTRY)
    if override:
        return override
    base = os.environ.get("APPDATA") or os.path.join(
        os.path.expanduser("~"), ".config")
    return os.path.join(base, "mtx", "scan_roots.json")


def load_registry(path: str | None = None) -> list[dict[str, str]]:
    p = path or registry_path()
    if not os.path.isfile(p):
        return []
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    roots = data.get("roots") if isinstance(data, dict) else None
    return [r for r in roots or [] if isinstance(r, dict) and r.get("root") and r.get("out")]


def save_registry(roots: list[dict[str, str]], path: str | None = None) -> str:
    p = path or registry_path()
    os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"roots": roots}, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    return p


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _is_within(child: str, parent: str) -> bool:
    """True when `child` is `parent` or sits underneath it."""
    c, p = _norm(child), _norm(parent)
    return c == p or c.startswith(p.rstrip("\\/") + os.sep)


class NoRootRegistered(Exception):
    """No library root covers this path and none was given.

    Raised rather than guessed: defaulting to `./mtx_out` would anchor the
    mirror tree at whatever folder the command happened to be run from, and a
    later scan of the whole library would then not find that work and would
    measure every track a second time.  One setup run is cheaper than that.
    """

    def __init__(self, path: str):
        self.path = path
        suggest = os.path.dirname(path) or path
        lines = [
            f"no library root is registered for {path}.",
            "       Say once where results go, naming the root of the library "
            "so that later",
            "       scans from any level agree:",
            "",
            f'           mtx scan "{suggest}" --out "<results dir>" --dry-run',
            "",
            "       That records the root and shows what would be measured "
            "without measuring",
            "       it.  After that, `mtx scan` works from any folder "
            "underneath with no flags.",
        ]
        super().__init__("\n".join(lines))


@dataclass(frozen=True)
class Scope:
    """What to walk, where results go, and what the paths are relative to."""

    scan_path: str      # the directory (or file) this invocation measures
    library_root: str   # what the mirror tree's paths are relative to
    out_dir: str        # the root of the mirror tree
    source: str         # how the root was decided, for the log line

    @property
    def mirror_base(self) -> str:
        """The out-tree directory corresponding to `scan_path` itself."""
        rel = os.path.relpath(self.scan_path, self.library_root)
        if rel == os.curdir:
            return self.out_dir
        return os.path.join(self.out_dir, rel)


def resolve_scope(scan_path: str, out: str | None = None,
                  library_root: str | None = None,
                  registry: str | None = None) -> tuple[Scope, bool]:
    """Decide (root, out) for this invocation.  Returns (scope, should_register).

    An explicit `--out` always wins and re-anchors the root.  Otherwise the
    registry is consulted for the most specific registered root containing the
    path, which is what lets `mtx scan` inside one album write into the same
    tree a library-wide scan would have used.
    """
    scan_path = os.path.abspath(scan_path)
    base = scan_path if os.path.isdir(scan_path) else os.path.dirname(scan_path)

    if out:
        root = os.path.abspath(library_root) if library_root else base
        if not _is_within(base, root):
            raise ValueError(
                f"--library-root {root} does not contain {base}")
        return Scope(base, root, os.path.abspath(out), "given on the command line"), True

    matches = [r for r in load_registry(registry) if _is_within(base, r["root"])]
    if matches:
        best = max(matches, key=lambda r: len(_norm(r["root"])))
        return Scope(base, os.path.abspath(best["root"]),
                     os.path.abspath(best["out"]), "registered library root"), False

    raise NoRootRegistered(base)


def superseded(scope: Scope, registry: str | None = None) -> list[dict[str, str]]:
    """Registered roots that sit *inside* the one being registered.

    Registering a library root over a root that was set up at album level is
    the moment the earlier tree becomes unreachable, so the caller is told
    rather than left to discover it as a re-measurement of the whole library.
    """
    return [r for r in load_registry(registry)
            if _norm(r["root"]) != _norm(scope.library_root)
            and _is_within(r["root"], scope.library_root)
            and _norm(r["out"]) != _norm(scope.out_dir)]


def register(scope: Scope, registry: str | None = None) -> str:
    """Record this root so a later scan from any level under it agrees.

    Roots nested inside the new one are dropped: the outer root now answers
    for them, and two overlapping answers would make the skip check depend on
    which directory the command was run from.
    """
    roots = [r for r in load_registry(registry)
             if _norm(r["root"]) != _norm(scope.library_root)
             and not _is_within(r["root"], scope.library_root)]
    roots.append({"root": scope.library_root, "out": scope.out_dir,
                  "registered_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    roots.sort(key=lambda r: r["root"])
    path = save_registry(roots, registry)
    os.makedirs(scope.out_dir, exist_ok=True)
    with open(os.path.join(scope.out_dir, ROOT_MARKER), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump({"library_root": scope.library_root,
                   "out_dir": scope.out_dir,
                   "tool_version": __version__}, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return path


# ----------------------------------------------------------------- discovery

def find_audio(path: str) -> list[str]:
    """Every audio file at or under `path`, sorted, output trees excluded."""
    if os.path.isfile(path):
        return [os.path.abspath(path)] if path.lower().endswith(AUDIO_EXTENSIONS) else []
    found: list[str] = []
    for root, dirs, names in os.walk(path):
        dirs[:] = sorted(d for d in dirs
                         if d not in SKIP_DIRS and not d.startswith("."))
        for nm in sorted(names):
            if nm.lower().endswith(AUDIO_EXTENSIONS):
                found.append(os.path.abspath(os.path.join(root, nm)))
    found.sort()
    return found


def out_dir_for(scope: Scope, source: str, collide: bool = False) -> str:
    """Where one track's results go in the mirror tree.

    The folder is named after the audio file, not after its tags: the name has
    to be known before the file is analysed for the skip check to be cheap, and
    a mirror of the library is easier to navigate than a flat list anyway.
    """
    rel_dir = os.path.relpath(os.path.dirname(source), scope.library_root)
    stem = os.path.basename(source) if collide else os.path.splitext(os.path.basename(source))[0]
    parts = [scope.out_dir]
    if rel_dir != os.curdir:
        parts.append(rel_dir)
    parts.append(stem)
    return os.path.join(*parts)


def plan(scope: Scope, sources: list[str]) -> list[tuple[str, str]]:
    """Pair each source with its output folder, disambiguating stem clashes."""
    seen: dict[str, str] = {}
    collisions: set[str] = set()
    for src in sources:
        key = _norm(out_dir_for(scope, src))
        if key in seen:
            collisions.add(seen[key])
            collisions.add(src)
        else:
            seen[key] = src
    return [(src, out_dir_for(scope, src, collide=src in collisions))
            for src in sources]


# -------------------------------------------------------------- the receipts

def ledger_path(out_dir: str) -> str:
    return os.path.join(out_dir, LEDGER_NAME)


def read_ledger(out_dir: str) -> dict[str, Any] | None:
    p = ledger_path(out_dir)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def source_stat(path: str) -> dict[str, Any]:
    st = os.stat(path)
    return {"path": path, "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def why_stale(source: str, out_dir: str, profile: str, stems: bool,
              recheck: bool = False,
              stems_model: str | None = None) -> str | None:
    """None when the existing result still stands; otherwise the reason it does not."""
    led = read_ledger(out_dir)
    if led is None:
        return "not analysed yet"
    if not os.path.isfile(os.path.join(out_dir, "analysis.json")):
        return "analysis.json missing"
    run = led.get("run") or {}
    if run.get("schema_version") != SCHEMA_VERSION:
        return f"schema {run.get('schema_version')} -> {SCHEMA_VERSION}"
    if run.get("profile") != profile:
        return f"profile {run.get('profile')} -> {profile}"
    if bool(run.get("stems")) != bool(stems):
        return "stems setting changed"
    if stems and run.get("stems_model") and stems_model and \
            run["stems_model"] != stems_model:
        return f"stems model changed ({run['stems_model']} -> {stems_model})"
    old = led.get("source") or {}
    try:
        now = source_stat(source)
    except OSError as exc:
        return f"source unreadable: {exc}"
    if int(old.get("size", -1)) != now["size"]:
        return "source size changed"
    if recheck and old.get("sha256"):
        if old["sha256"] != _sha256(source):
            return "source contents changed"
    elif int(old.get("mtime_ns", -1)) != now["mtime_ns"]:
        # Without a recorded hash there is nothing for --recheck to compare
        # against, so the modification time is still the only evidence there is.
        return "source modified"
    return None


def write_ledger(out_dir: str, source: str, profile: str, stems: bool,
                 res: dict[str, Any], written: dict[str, str],
                 elapsed: float) -> None:
    """The receipt a later scan reads instead of the audio."""
    stat = source_stat(source)
    stat["sha256"] = (res.get("file") or {}).get("sha256")
    payload = {
        "source": stat,
        "run": {
            "profile": profile,
            "stems": bool(stems),
            "stems_model": (res.get("run") or {}).get("stems_model")
            if isinstance(res, dict) else None,
            "tool_version": __version__,
            "schema_version": SCHEMA_VERSION,
            "elapsed_seconds": round(elapsed, 3),
            "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "outputs": sorted(os.path.basename(v) for v in written.values()
                          if isinstance(v, str)),
        "note": "Written by `mtx scan`. Delete this file (or pass --force) to "
                "have the track measured again.",
    }
    with open(ledger_path(out_dir), "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")


# ------------------------------------------------------------- the duplicates

@dataclass(frozen=True)
class Duplicate:
    """A file whose bytes have already been measured somewhere else."""
    source: str
    out_dir: str
    twin_dir: str
    within_run: bool


def _declared_body(declared: dict[str, Any] | None) -> Any:
    """A declared sidecar's content, without where it was looked for.

    `path` and `searched` record which folder was searched, so they differ
    between two copies of one master by construction.  What was declared is
    the rest, and that is what has to agree for one measurement to stand in
    for the other.
    """
    if not isinstance(declared, dict):
        return None
    return {k: v for k, v in declared.items() if k not in ("path", "searched")}


def _reusable(led: dict[str, Any], profile: str, stems: bool,
              stems_model: str | None) -> bool:
    """Whether a receipt describes a run whose result this one could adopt."""
    run = led.get("run") or {}
    if run.get("schema_version") != SCHEMA_VERSION:
        return False
    if run.get("profile") != profile or bool(run.get("stems")) != bool(stems):
        return False
    return not (stems and run.get("stems_model") and stems_model
                and run["stems_model"] != stems_model)


def content_index(out_root: str, sizes: set[int], profile: str, stems: bool,
                  stems_model: str | None,
                  exclude: set[str]) -> tuple[dict[str, str], set[int]]:
    """sha256 -> the folder that already holds a result for those bytes.

    Built from the receipts in the mirror tree rather than from this scan's
    scope, so a copy measured last month, under another album, by a scan of a
    different corner of the library, still counts as measured.  Only receipts
    whose size one of the candidates shares are read past; the sizes that were
    found come back too, so the caller knows which candidates are worth
    hashing at all.
    """
    index: dict[str, str] = {}
    seen_sizes: set[int] = set()
    if not sizes or not os.path.isdir(out_root):
        return index, seen_sizes
    for root, dirs, names in os.walk(out_root):
        dirs[:] = sorted(dirs)
        if LEDGER_NAME not in names or _norm(root) in exclude:
            continue
        led = read_ledger(root)
        src = (led or {}).get("source") or {}
        digest = src.get("sha256")
        if not led or not digest or int(src.get("size", -1)) not in sizes:
            continue
        if not _reusable(led, profile, stems, stems_model):
            continue
        if not os.path.isfile(os.path.join(root, "analysis.json")):
            continue
        seen_sizes.add(int(src["size"]))
        index.setdefault(digest, root)
    return index, seen_sizes


def partition_duplicates(scope: Scope, todo: list[tuple[str, str, str]],
                         profile: str, stems: bool, stems_model: str | None,
                         ) -> tuple[list[tuple[str, str, str]], list[Duplicate]]:
    """Split the work into files to measure and copies of files being measured.

    Everything mtx reports is a function of the audio bytes, so two files with
    the same sha256 have one measurement between them: the second is a copy,
    not a scan.  A single sitting next to the album it was lifted from is the
    ordinary case, and paying twice for it is minutes of demucs per track.

    Hashing a whole library to discover that would cost more than it saves, so
    size goes first.  A file whose size matches nothing -- no other candidate,
    no finished result -- cannot be a duplicate of anything and is never read.
    """
    from . import declared as declared_mod

    sized: list[tuple[int, str, str, str]] = []
    for src, odir, reason in todo:
        try:
            sized.append((os.path.getsize(src), src, odir, reason))
        except OSError:
            sized.append((-1, src, odir, reason))
    counts = Counter(size for size, _, _, _ in sized)
    index, known_sizes = content_index(
        scope.out_dir, {s for s, _, _, _ in sized if s >= 0}, profile, stems,
        stems_model, exclude={_norm(o) for _, _, o, _ in sized})

    keep: list[tuple[str, str, str]] = []
    dupes: list[Duplicate] = []
    primaries: dict[str, tuple[str, str]] = {}
    for size, src, odir, reason in sized:
        if size < 0 or (counts[size] == 1 and size not in known_sizes):
            keep.append((src, odir, reason))
            continue
        try:
            digest = _sha256(src)
        except OSError:
            keep.append((src, odir, reason))
            continue
        if digest in primaries:
            twin_src, twin_dir = primaries[digest]
            # Both files are here, so the one input that does not travel with
            # the bytes can be compared directly rather than guessed at.
            if _declared_body(declared_mod.load(src)) == \
                    _declared_body(declared_mod.load(twin_src)):
                dupes.append(Duplicate(src, odir, twin_dir, True))
                continue
        elif digest in index and _norm(index[digest]) != _norm(odir):
            dupes.append(Duplicate(src, odir, index[digest], False))
            continue
        primaries.setdefault(digest, (src, odir))
        keep.append((src, odir, reason))
    return keep, dupes


def materialize_duplicate(dup: Duplicate, *, profile: str, stems: bool,
                          json_only: bool, plots: bool,
                          max_part_bytes: int | None) -> dict[str, Any]:
    """Write a duplicate's folder from its twin's, decoding no audio.

    The result is adopted whole and then corrected everywhere it names the
    file it came from, so the mirror tree stays uniform: a duplicate's folder
    holds the same analysis, digest and corpus row every other track's does,
    and every tool downstream reads it without knowing the difference.  What
    it also holds is the provenance -- `run.duplicate_of` names the file the
    numbers were measured on.

    A `declared.json` sidecar is the one input that does not travel with the
    bytes, since it sits next to the audio; a disagreement means this file has
    to be measured itself, and says so.
    """
    from . import declared as declared_mod
    from .analyze import write_outputs
    from .split import load_analysis

    t0 = time.time()
    try:
        res = load_analysis(os.path.join(dup.twin_dir, "analysis.json"))
        twin_led = read_ledger(dup.twin_dir) or {}
        twin_source = (twin_led.get("source") or {}).get("path") \
            or (res.get("file") or {}).get("path_absolute")
        mine = declared_mod.load(dup.source)
        if _declared_body(mine) != _declared_body(res.get("declared")):
            return {"source": dup.source, "out_dir": dup.out_dir, "ok": False,
                    "requeue": True, "elapsed": time.time() - t0,
                    "error": "declared sidecar differs from the copy already "
                             "measured"}
        res.setdefault("file", {})
        res["file"]["path_absolute"] = os.path.abspath(dup.source)
        res["file"]["filename"] = os.path.basename(dup.source)
        res["declared"] = mine
        run = res.setdefault("run", {})
        run["declared_sidecar"] = mine.get("path")
        run["duplicate_of"] = twin_source
        run["duplicate_copied_utc"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds")
        run["duplicate_note"] = (
            "every measurement here was made on a byte-identical copy of this "
            "file -- same sha256, so the same audio -- and is reproduced "
            "unchanged. Only the file's own path and its declared sidecar "
            "were read again. run.elapsed_seconds is that measurement's, not "
            "the cost of this copy.")
        # Same reasoning as a fresh measurement: once the folder starts being
        # overwritten its receipt has stopped being true.
        try:
            os.remove(ledger_path(dup.out_dir))
        except OSError:
            pass
        written = write_outputs(res, dup.out_dir, json_only=json_only,
                                plots=plots, src_path=dup.source,
                                max_part_bytes=max_part_bytes)
        elapsed = time.time() - t0
        write_ledger(dup.out_dir, dup.source, profile, stems, res, written,
                     elapsed)
        return {"source": dup.source, "out_dir": dup.out_dir, "ok": True,
                "duplicate_of": twin_source, "elapsed": elapsed,
                "duration_s": (res.get("audio") or {}).get("duration_s")}
    except Exception as exc:
        return {"source": dup.source, "out_dir": dup.out_dir, "ok": False,
                "requeue": True, "elapsed": time.time() - t0,
                "error": f"{type(exc).__name__}: {exc}"}


# -------------------------------------------------------- separation up front

def separate_first(sources: list[str], stems_model: str | None,
                   log=print, streams: int | None = None) -> int:
    """Separate every track on the GPU before measuring starts.

    A card is fast but small, so the separations come out of the pool and are
    done here rather than inside workers that would each want their own copy
    of the model.  They are not done one at a time.  A single separation
    leaves a GTX 1650 68 % busy -- the rest of each track goes on decoding the
    input and writing four uncompressed wavs, with the device idle -- and
    overlapping a few streams fills those gaps for 1.51x at three.  How many
    the card's memory holds is `separation_streams()`, which leaves a reserve
    rather than discovering the ceiling through an out-of-memory failure.

    Nothing else changes: each result lands in the same content-addressed cache
    the workers read, so the pool that follows finds every one of them already
    there and spends its processes on the DSP, which is what processes are good
    for here.  Stopping partway through costs only the separations in flight --
    the rest stay cached, and the next run starts from them.
    """
    import threading

    from .metrics import stems as m_stems
    from .util import Collector

    if streams is None:
        streams = m_stems.separation_streams()
    streams = max(1, min(int(streams), len(sources)))

    t0 = time.time()
    lock = threading.Lock()
    done = 0
    seen = 0

    def one(src: str) -> None:
        nonlocal done, seen
        collector = Collector()
        s0 = time.time()
        paths = m_stems.separate(src, collector, stems_model)
        spent = time.time() - s0
        name = os.path.basename(src)
        # One lock for the counters and the line they print, so a track's
        # warnings stay attached to it while several separations report.
        with lock:
            seen += 1
            n = seen
            for w in collector.warnings:
                log(f"  {name}: {w}")
            if paths is None:
                log(f"[stems {n}/{len(sources)}] {name}: not separated; the "
                    "file will be measured without stems")
                return
            done += 1
            # Wall clock over completions already carries the concurrency:
            # `streams` tracks finishing together shorten the mean directly.
            rate = (time.time() - t0) / n
            tail = "" if n == len(sources) else \
                f"  eta {_fmt_hms(rate * (len(sources) - n))}"
            cached = " (cached)" if spent < 1.0 else ""
            log(f"[stems {n}/{len(sources)}] {name}: {spent:.0f} s{cached}{tail}")

    if streams == 1:
        for src in sources:
            one(src)
    else:
        with ThreadPoolExecutor(max_workers=streams) as pool:
            list(pool.map(one, sources))
    return done


# ------------------------------------------------------------------- workers

def analyse_one(job: dict[str, Any]) -> dict[str, Any]:
    """Measure one file and write its outputs.  Runs in a worker process.

    Returns only a summary: the full result is hundreds of megabytes of numpy
    once decoded, and shipping it back through a pickle would cost more than
    the analysis did.
    """
    source = job["source"]
    out_dir = job["out_dir"]
    t0 = time.time()
    try:
        from .analyze import analyze_file, write_outputs

        res = analyze_file(source, profile=job["profile"],
                           want_stems=job["stems"], threads=job["threads"],
                           stems_model=job.get("stems_model"),
                           want_transcript=bool(job.get("transcribe")),
                           want_embedding=bool(job.get("embed")))
        # From here the folder's existing contents start being overwritten, so
        # the receipt for them stops being true.  Drop it first: a run that
        # dies midway through writing should leave the track looking stale,
        # not looking finished.
        try:
            os.remove(ledger_path(out_dir))
        except OSError:
            pass
        written = write_outputs(res, out_dir, json_only=job["json_only"],
                                plots=job["plots"], src_path=source,
                                max_part_bytes=job["max_part_bytes"])
        elapsed = time.time() - t0
        write_ledger(out_dir, source, job["profile"], job["stems"], res,
                     written, elapsed)
        return {
            "source": source, "out_dir": out_dir, "ok": True,
            "elapsed": elapsed,
            "warnings": len(res.get("warnings") or []),
            "duration_s": (res.get("audio") or {}).get("duration_s"),
        }
    except Exception as exc:  # one bad file must not end a library scan
        import traceback
        tb = traceback.extract_tb(exc.__traceback__)
        where = f" at {tb[-1].filename.rsplit(os.sep, 1)[-1]}:{tb[-1].lineno}" if tb else ""
        return {"source": source, "out_dir": out_dir, "ok": False,
                "elapsed": time.time() - t0,
                "error": f"{type(exc).__name__}: {exc}{where}"}


def _worker_init() -> None:
    apply_single_threaded_env()


@contextmanager
def _child_env(enabled: bool) -> Iterator[None]:
    """Pin the numeric stack to one thread per worker.

    The variables have to be in the environment before numpy is imported in the
    child, and on a spawn platform the child inherits this process's copy at
    exec time -- so they are set here, around pool creation, and restored after.
    The parent does no numeric work while the pool runs.
    """
    if not enabled:
        yield
        return
    saved = {k: os.environ.get(k) for k in single_threaded_env()}
    os.environ.update(single_threaded_env())
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def split_budget(jobs: int, n_todo: int) -> tuple[int, int]:
    """Divide a parallelism budget into (processes, threads per process).

    Threads only help inside the true-peak scan; processes help everywhere.  So
    processes are taken first, and threads only pick up the slack when there
    are fewer files left than there is room to run.
    """
    procs = max(1, min(jobs, n_todo))
    threads = max(1, jobs // procs)
    return procs, threads


# -------------------------------------------------------------------- summary

def _summary_rows(base: str) -> list[dict[str, Any]]:
    """Every corpus row under one mirror directory, including earlier runs."""
    rows: list[dict[str, Any]] = []
    for root, dirs, names in os.walk(base):
        dirs[:] = sorted(dirs)
        if "corpus_row.json" not in names:
            continue
        try:
            with open(os.path.join(root, "corpus_row.json"), encoding="utf-8") as f:
                row = json.load(f)
        except (OSError, ValueError):
            continue
        led = read_ledger(root)
        row = dict(row)
        row["Path"] = (led or {}).get("source", {}).get("path") or \
            (row.get("_source") or {}).get("file")
        rows.append(row)
    rows.sort(key=lambda r: str(r.get("Path") or ""))
    return rows


def write_summary(base: str) -> tuple[str, int]:
    """Refresh `summary.csv` over the whole scanned subtree."""
    rows = _summary_rows(base)
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, SUMMARY_NAME)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path, len(rows)


# ----------------------------------------------------------------- the driver

def _fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def run_scan(scan_path: str, *, out: str | None = None,
             library_root: str | None = None, profile: str = "full",
             jobs: int | None = None, stems_jobs: int | None = None,
             force: bool = False,
             recheck: bool = False, stems: bool = False, plots: bool = False,
             json_only: bool = False, stems_model: str | None = None,
             transcribe: bool = False, embed: bool = False,
             max_part_bytes: int | None = DEFAULT_PART_BYTES,
             dry_run: bool = False, no_summary: bool = False,
             dedup: bool = True,
             registry: str | None = None, log=print) -> dict[str, Any]:
    """Measure everything under `scan_path` that has not been measured already.

    "Already" covers two things: this exact file, measured on an earlier run
    and still matching its receipt, and any *other* file with the same bytes,
    whose measurement this one can adopt instead of repeating.  `dedup=False`
    turns the second off and measures every copy on its own.
    """
    scope, should_register = resolve_scope(scan_path, out, library_root, registry)
    log(f"scope:   {scope.scan_path}")
    log(f"library: {scope.library_root}  ({scope.source})")
    log(f"out:     {scope.out_dir}")
    if should_register:
        # The anchor is registered even on a dry run: it is configuration, not
        # output, and `--out ... --dry-run` is the natural way to set one up.
        for old in superseded(scope, registry):
            log(f"note: {old['root']} was registered separately "
                f"(-> {old['out']}); it is now covered by this root, and any "
                "results already under the old tree will not be found")
        log(f"registered: {scope.library_root} -> {scope.out_dir} "
            f"({register(scope, registry)})")

    # A file target measures that file; the scope stays its folder, so the
    # mirror path and the summary still cover the album it belongs to.
    target = os.path.abspath(scan_path)
    if not os.path.isfile(target):
        target = scope.scan_path
    sources = find_audio(target)
    if not sources:
        log(f"no audio files under {target}")
        return {"found": 0, "done": 0, "skipped": 0, "failed": 0,
                "out_dir": scope.out_dir}

    pairs = plan(scope, sources)
    todo: list[tuple[str, str, str]] = []
    skipped = 0
    for src, odir in pairs:
        reason = ("forced" if force
                  else why_stale(src, odir, profile, stems, recheck, stems_model))
        if reason is None:
            skipped += 1
        else:
            todo.append((src, odir, reason))

    dupes: list[Duplicate] = []
    if todo and dedup:
        todo, dupes = partition_duplicates(scope, todo, profile, stems,
                                           stems_model)

    log(f"{len(pairs)} file(s) found: {skipped} already measured, "
        f"{len(todo)} to do"
        + (f", {len(dupes)} identical to another file" if dupes else ""))
    if not todo and not dupes:
        base = scope.mirror_base
        if not no_summary and os.path.isdir(base):
            path, n = write_summary(base)
            log(f"summary: {path} ({n} row(s))")
        log("nothing to do")
        return {"found": len(pairs), "done": 0, "skipped": skipped, "failed": 0,
                "duplicates": 0, "out_dir": scope.out_dir}

    if dry_run:
        for src, odir, reason in todo[:200]:
            log(f"  would measure {os.path.basename(src)}  ({reason})")
        if len(todo) > 200:
            log(f"  ... and {len(todo) - 200} more")
        for dup in dupes[:200]:
            log(f"  would copy {os.path.basename(dup.source)}  "
                f"(same bytes as {os.path.basename(dup.twin_dir)})")
        if len(dupes) > 200:
            log(f"  ... and {len(dupes) - 200} more")
        return {"found": len(pairs), "done": 0, "skipped": skipped, "failed": 0,
                "todo": len(todo), "duplicates": len(dupes),
                "out_dir": scope.out_dir}

    t_run = time.time()
    copied = 0
    results: list[dict[str, Any]] = []

    def copy_from_twin(dup: Duplicate) -> bool:
        """Write one duplicate, or hand it back to be measured properly."""
        nonlocal copied
        r = materialize_duplicate(dup, profile=profile, stems=stems,
                                  json_only=json_only, plots=plots,
                                  max_part_bytes=max_part_bytes)
        results.append(r)
        name = os.path.basename(dup.source)
        if r["ok"]:
            copied += 1
            log(f"copied {name}: same bytes as "
                f"{os.path.basename(r['duplicate_of'] or dup.twin_dir)}")
            return True
        log(f"copying {name} did not stand: {r.get('error')}")
        return False

    # Copies of work finished on an earlier run are resolved before the pool
    # starts: one that does not stand still has to be measured, and this is
    # the last moment it can join the run that would measure it.
    for dup in [d for d in dupes if not d.within_run]:
        os.makedirs(dup.out_dir, exist_ok=True)
        if not copy_from_twin(dup):
            todo.append((dup.source, dup.out_dir, "copy did not stand"))

    budget = jobs if (jobs and jobs > 0) else default_workers()
    procs, threads = split_budget(budget, len(todo))
    if todo:
        log(f"jobs: {procs} process(es) x {threads} thread(s) "
            f"of {cpu_count()} core(s)")

    # A GPU changes where the expensive half of a stems run belongs.  Asking
    # for it here rather than in each worker is also what keeps torch out of
    # the workers entirely: they find the cache warm and never import it.
    if stems and todo:
        from .metrics import stems as m_stems
        device = m_stems.resolve_device()
        if device.startswith("cuda"):
            streams = m_stems.separation_streams(stems_jobs)
            streams = max(1, min(streams, len(todo)))
            vram = m_stems.device_vram_mib()
            room = f", {vram} MiB" if vram else ""
            at_once = ("one at a time" if streams == 1
                       else f"{streams} at a time")
            log(f"stems: separating {len(todo)} file(s) on {device}{room} "
                f"before measuring, {at_once}")
            try:
                separate_first([src for src, _, _ in todo], stems_model, log,
                               streams=streams)
            except KeyboardInterrupt:
                log("interrupted; separations that finished stay cached and "
                    "the next run picks up from there")
                return {"found": len(pairs), "done": 0, "skipped": skipped,
                        "failed": 0, "duplicates": copied,
                        "elapsed": time.time() - t_run,
                        "out_dir": scope.out_dir, "results": results}
        else:
            log(f"stems: separating on {device}, inside the worker processes")

    jobs_list = [{"source": src, "out_dir": odir, "profile": profile,
                  "stems": stems, "plots": plots, "json_only": json_only,
                  "stems_model": stems_model, "transcribe": transcribe,
                  "embed": embed,
                  "max_part_bytes": max_part_bytes, "threads": threads}
                 for src, odir, _ in todo]
    for j in jobs_list:
        os.makedirs(j["out_dir"], exist_ok=True)

    t0 = time.time()
    done = failed = 0
    audio_seconds = 0.0
    served = 0.0          # per-file service time, summed over finished tracks
    served_n = 0
    capacity = 0.0        # worker-seconds the pool has burned, busy lanes only
    last_event = t0

    def eta_seconds(n: int) -> float | None:
        """Wall clock left after `n` of the tracks have been reported.

        Estimated from mean *service* time rather than wall clock per file.
        Wall clock per file is what a pool makes meaningless: the first
        completion of a `procs`-wide run has `procs` tracks' worth of wall
        clock behind it, and charging all of it to the one track that finished
        overstates the early estimates by exactly the worker count.

        Two corrections keep the end of a run honest.  The tail drains
        narrower than the pool once fewer tracks are left than there are
        workers (`lanes`), and those last tracks are already part-separated
        when they become "remaining" -- the pool has burned `capacity`
        worker-seconds, of which only `served` is accounted for by finished
        tracks, so the difference is work the remaining tracks need not redo.
        """
        remaining = len(jobs_list) - n
        if not remaining:
            return None
        if not served_n:
            return (time.time() - t0) / max(n, 1) * remaining
        lanes = min(procs, remaining)
        in_flight = max(0.0, capacity - served)
        return max(0.0, served / served_n * remaining - in_flight) / lanes

    def report(r: dict[str, Any]) -> None:
        nonlocal done, failed, audio_seconds, served, served_n
        nonlocal capacity, last_event
        n = done + failed + 1
        now = time.time()
        # Lanes that were actually turning over the interval that just ended:
        # once fewer tracks are left than there are workers, the idle ones
        # must not be billed as capacity.
        capacity += min(procs, len(jobs_list) - (n - 1)) * (now - last_event)
        last_event = now
        name = os.path.basename(r["source"])
        if r["ok"]:
            done += 1
            audio_seconds += float(r.get("duration_s") or 0.0)
            spent = float(r.get("elapsed") or 0.0)
            if spent > 0:
                served += spent
                served_n += 1
            warn = f", {r['warnings']} warning(s)" if r.get("warnings") else ""
            eta = eta_seconds(n)
            tail = "" if eta is None else f"  eta {_fmt_hms(eta)}"
            log(f"[{n}/{len(jobs_list)}] {name}: {r['elapsed']:.0f} s{warn}{tail}")
        else:
            failed += 1
            log(f"[{n}/{len(jobs_list)}] {name}: FAILED {r.get('error')}")
        results.append(r)

    interrupted = False
    try:
        if procs == 1:
            for job in jobs_list:
                report(analyse_one(job))
        else:
            with _child_env(True):
                pool = ProcessPoolExecutor(max_workers=procs,
                                           initializer=_worker_init)
            futures = {pool.submit(analyse_one, j): j for j in jobs_list}
            try:
                for fut in as_completed(futures):
                    try:
                        report(fut.result())
                    except Exception as exc:
                        # A worker that died outright (killed, out of memory)
                        # never returned a result of its own to report.
                        job = futures[fut]
                        report({"source": job["source"], "out_dir": job["out_dir"],
                                "ok": False, "elapsed": 0.0,
                                "error": f"worker lost: {type(exc).__name__}: {exc}"})
            except KeyboardInterrupt:
                interrupted = True
                # Drop what has not started; do not wait out the tracks that
                # have, since each of them is a minute of work already spent.
                pool.shutdown(wait=False, cancel_futures=True)
            else:
                pool.shutdown(wait=True)
    except KeyboardInterrupt:
        interrupted = True
    if interrupted:
        log("interrupted; finished tracks keep their results and will be "
            "skipped on the next run")

    # The rest of the copies: their twin was measured by the run that just
    # ended, so they could not be written until now.  One whose twin failed or
    # never started is simply left stale, and the next run measures or copies
    # it -- there is nothing to lose by waiting.
    pending = [d for d in dupes if d.within_run]
    if pending and not interrupted:
        measured = {_norm(r["out_dir"]) for r in results if r.get("ok")}
        for dup in pending:
            if _norm(dup.twin_dir) not in measured:
                log(f"{os.path.basename(dup.source)}: left for the next run "
                    "(the file it copies was not measured)")
                continue
            os.makedirs(dup.out_dir, exist_ok=True)
            copy_from_twin(dup)

    elapsed = time.time() - t_run
    log(f"measured {done} file(s) in {_fmt_hms(elapsed)}"
        + (f", copied {copied} from an identical file" if copied else "")
        + (f", {failed} failure(s)" if failed else ""))
    if done and audio_seconds:
        log(f"throughput: {audio_seconds / max(elapsed, 1e-9):.1f} s of audio "
            f"per second of wall clock")

    if not no_summary:
        path, n = write_summary(scope.mirror_base)
        log(f"summary: {path} ({n} row(s))")

    return {"found": len(pairs), "done": done, "skipped": skipped,
            "failed": failed, "duplicates": copied, "elapsed": elapsed,
            "out_dir": scope.out_dir, "results": results}
