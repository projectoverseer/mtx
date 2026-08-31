"""Worker-count policy and the bounded thread pool the hot loops use.

Two kinds of parallelism live in mtx, and they are deliberately kept apart.

*Between files* the work is embarrassingly parallel and the unit is a process:
two tracks share nothing, and a process sidesteps both the GIL and the fact
that a decoded track is a large mutable array.  `mtx scan` owns that layer.

*Inside one file* the unit is a thread, and it is only worth using where the
primitive doing the work releases the GIL.  Measured on this codebase's hot
list (scipy 1.18, numpy 2.5):

    resample_poly / upfirdn      releases   ~3.8x on 4 threads
    ndimage.median_filter        releases   ~3.1x
    lfilter                      releases   ~2.7x
    sosfilt / sosfiltfilt        holds      ~1.2x
    welch / rfft                 holds      ~1.1x
    numpy reductions             holds      ~1.2x

So threading is applied to the true-peak oversampling pass and nowhere else:
everywhere else it would add contention and buy nothing.  The band split and
the per-frame Welch loops are GIL-bound, which is precisely why they scale on
the process layer instead.

The two layers must not both expand at once.  A scan running eight worker
processes gives each of them one thread; a single-file run gives that one
process every core.  `resolve_threads` is where that decision is made.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")

# Leave a little headroom so an interactive machine stays usable, and stop
# well short of the core count on large boxes: past about eight workers a scan
# is bound by memory bandwidth and decoded-signal footprint, not by cores.
MAX_DEFAULT_WORKERS = 8
RESERVED_CORES = 1

ENV_THREADS = "MTX_THREADS"


def cpu_count() -> int:
    """Usable logical processors, honouring an affinity mask where there is one."""
    try:
        sched = getattr(os, "sched_getaffinity", None)
        if sched is not None:
            return max(1, len(sched(0)))
    except (AttributeError, OSError):
        pass
    return max(1, os.cpu_count() or 1)


def physical_cores() -> int:
    """Cores that can actually run two analyses at once.

    This is the number that matters here, and it is not `os.cpu_count()`.  The
    work is dense numpy over arrays far larger than L3: two hyperthreads on one
    core share the load/store units and the cache these passes are bound by, so
    the second thread buys perhaps a fifth of a core, not a whole one.  Sizing
    the pool by logical processors just adds memory pressure and context
    switching for it.

    Falls back to half the logical count, which is right on every machine that
    has SMT and merely conservative on the ones that do not.
    """
    logical = cpu_count()
    try:
        if sys.platform == "win32":
            n = _windows_physical_cores()
        elif sys.platform == "darwin":
            import subprocess
            n = int(subprocess.run(["sysctl", "-n", "hw.physicalcpu"],
                                   capture_output=True, text=True,
                                   timeout=5).stdout.strip())
        else:
            ids = set()
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                phys = core = None
                for line in f:
                    key, _, val = line.partition(":")
                    key = key.strip()
                    if key == "physical id":
                        phys = val.strip()
                    elif key == "core id":
                        core = val.strip()
                    elif not line.strip() and phys is not None and core is not None:
                        ids.add((phys, core))
                        phys = core = None
            n = len(ids)
    except Exception:
        n = 0
    if not n or n > logical:
        n = max(1, logical // 2) if logical > 2 else logical
    return max(1, n)


def _windows_physical_cores() -> int:
    """Count processor cores from `GetLogicalProcessorInformation`."""
    import ctypes
    from ctypes import wintypes

    RELATION_PROCESSOR_CORE = 0

    class _Info(ctypes.Structure):
        _fields_ = [("ProcessorMask", ctypes.c_void_p),
                    ("Relationship", wintypes.DWORD),
                    ("Reserved", ctypes.c_ulonglong * 2)]

    kernel32 = ctypes.windll.kernel32
    length = wintypes.DWORD(0)
    kernel32.GetLogicalProcessorInformation(None, ctypes.byref(length))
    count = length.value // ctypes.sizeof(_Info)
    if count <= 0:
        return 0
    buf = (_Info * count)()
    if not kernel32.GetLogicalProcessorInformation(ctypes.byref(buf),
                                                   ctypes.byref(length)):
        return 0
    return sum(1 for i in buf if i.Relationship == RELATION_PROCESSOR_CORE)


def default_workers(cap: int = MAX_DEFAULT_WORKERS) -> int:
    """Worker count when the caller did not ask for one.

    One process per physical core, less a little headroom so the machine stays
    usable while a library scan runs for an hour.
    """
    cores = physical_cores()
    return max(1, min(cap, cores - RESERVED_CORES if cores > 4 else cores))


# What a scan leaves alone: the operating system, the file cache the decoding
# leans on, the torch process doing the separations, and whatever the user has
# open.  Held back as an amount rather than a share, because it is roughly
# constant -- taking a proportion would punish a small machine, where the
# reserve matters most, and leave a large one short of work.
#
# Overshooting is not a gentle degradation: lanes that page against each other
# lose more than the lane that would have been given up to avoid it.  A run
# that overcommitted this machine finished at a fifth of the rate a single
# process sustains on the same tracks, and killed workers outright.
#
# The reserve is not fitted to that run.  It is the OS, the file cache the
# decoding leans on, and the parent process, and 3 GB covers them with room --
# see PERFORMANCE.md "Finding 4a" for why a number fitted to that particular
# night would have been fitted to somebody else's memory leak.
MEMORY_RESERVE = 3_000_000_000


def total_memory_bytes() -> int:
    """Physical RAM, or 0 when it cannot be determined.

    Every caller treats 0 as "do not schedule by memory", so a platform this
    cannot read falls back to the behaviour it had before rather than to a
    guess that could be wrong in either direction.
    """
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class _Status(ctypes.Structure):
                _fields_ = [("dwLength", wintypes.DWORD),
                            ("dwMemoryLoad", wintypes.DWORD),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _Status()
            st.dwLength = ctypes.sizeof(_Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullTotalPhys)
        except Exception:
            return 0
        return 0
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 0


# A demucs process holds about this much host memory while it separates --
# measured at 2.47 GB peak on this bench, rounded up.  It is invisible to the
# byte budget, which counts only what the measuring workers decode, so every
# concurrent separation has to be held back here instead.  This is the direct
# reason the pipelined default is one stream: a second one costs about as much
# memory as a whole measuring lane, and measuring is the half that is slow.
SEPARATION_RESERVE = 3_000_000_000


def available_memory_bytes() -> int:
    """Physical memory not currently in use, or 0 when it cannot be read.

    Separate from `total_memory_bytes` because the two answer different
    questions: what the machine has, and what is left of it.  A scan sized
    against the first while another process holds most of the second is the
    failure this exists to notice.
    """
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class _Status(ctypes.Structure):
                _fields_ = [("dwLength", wintypes.DWORD),
                            ("dwMemoryLoad", wintypes.DWORD),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _Status()
            st.dwLength = ctypes.sizeof(_Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullAvailPhys)
        except Exception:
            return 0
        return 0
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 0


def memory_budget(streams: int = 0, reserve: int = MEMORY_RESERVE) -> int:
    """Bytes of decoded audio a scan may hold across all its workers.

    `streams` is the separations running beside the pool; each is held back
    because the budget cannot see it.  This is the direct reason the pipelined
    default is one stream: a second one costs about as much memory as a whole
    measuring lane, and measuring is the half that is slow.

    Zero when the machine's RAM cannot be read, which every caller takes as
    "schedule by worker count, as before" rather than as "no memory at all".
    """
    total = total_memory_bytes()
    if not total:
        return 0
    return max(0, total - reserve - max(0, int(streams)) * SEPARATION_RESERVE)


def resolve_threads(requested: int | None) -> int:
    """Threads this process may use inside a single file.

    `MTX_THREADS` is how `mtx scan` pins its workers to one thread each; an
    explicit `--jobs` still wins, so a deliberate single-file run in a shell
    that happens to export the variable behaves as asked.
    """
    if requested is not None and requested > 0:
        return int(requested)
    env = os.environ.get(ENV_THREADS)
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return default_workers()


def single_threaded_env() -> dict[str, str]:
    """Environment for a scan worker: one thread everywhere.

    The numeric stack underneath will happily start a thread pool per process,
    and eight processes each opening twelve threads is slower than eight
    processes with one, not faster.
    """
    return {
        ENV_THREADS: "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }


def apply_single_threaded_env() -> None:
    """Called at the top of a scan worker, before numpy is touched."""
    os.environ.update(single_threaded_env())


def ordered_window(fn: Callable[[T], R], items: Sequence[T], workers: int,
                   lookahead: int | None = None) -> Iterator[R]:
    """`map(fn, items)` over a thread pool, in order, with bounded memory.

    `ThreadPoolExecutor.map` submits every task at once, so a pass whose
    per-task result is a few megabytes of oversampled audio would hold the
    whole file in flight.  This keeps at most `lookahead` results alive and
    yields them in input order, which is what the scan state needs: an "over"
    that straddles a chunk boundary is only counted once if the chunks are
    folded in the order they occur.
    """
    if workers <= 1 or len(items) <= 1:
        for item in items:
            yield fn(item)
        return
    window = lookahead or (workers * 2)
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="mtx") as pool:
        pending: list[Any] = []
        nxt = 0
        while nxt < len(items) and len(pending) < window:
            pending.append(pool.submit(fn, items[nxt]))
            nxt += 1
        while pending:
            yield pending.pop(0).result()
            if nxt < len(items):
                pending.append(pool.submit(fn, items[nxt]))
                nxt += 1


def parallel_each(fn: Callable[[T], R], items: Iterable[T],
                  workers: int) -> list[R]:
    """Unordered-safe `map` where every result is small.  Order is preserved."""
    seq = list(items)
    if workers <= 1 or len(seq) <= 1:
        return [fn(i) for i in seq]
    with ThreadPoolExecutor(max_workers=min(workers, len(seq)),
                            thread_name_prefix="mtx") as pool:
        return list(pool.map(fn, seq))
