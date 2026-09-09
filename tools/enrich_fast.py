"""Enrich a whole corpus in parallel.  Same output as `mtx enrich`, ~4x faster.

    set LASTFM_API_KEY=...
    python tools/enrich_fast.py "E:\\Music\\_mtx_out" -j 8

`mtx enrich` walks one track at a time, and a track costs about 12 seconds:
nine HTTP requests, each carrying a mandated pause plus a round trip.  Almost
all of that is the process sitting still, so 1,321 tracks take three and a
half hours of mostly waiting.

The waiting is per host, and there are four of them.  While one track holds
the MusicBrainz clock, another can be talking to Deezer, a third to Last.fm,
a fourth parsing JSON off disk.  This runs a thread pool over folders to
overlap exactly that.

**It does not go faster by being ruder.**  The per-host pacing in
`mtx.online.http` is process-wide and lock-guarded, so MusicBrainz still sees
one request a second no matter how many workers are running -- that is what
makes this safe rather than a way to get a 503 and deserve it.  The floor is
therefore MusicBrainz: about three uncached requests a track at 1.1 s each,
so roughly an hour for a cold 1,321-track corpus however high `-j` goes.
Past `-j 8` there is nothing left to overlap.

Everything else matches `mtx enrich` exactly: same `enrich()` call, same
`online.json`, same cache. Interrupt it and run it again -- finished tracks
are skipped by default, and the HTTP cache makes a re-run of a done track
nearly free.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from env import load_env                        # noqa: E402
from mtx import __version__                     # noqa: E402
from mtx.cli import _enrich_targets             # noqa: E402
from mtx.online import ALL_PROVIDERS, DEFAULT_PROVIDERS, enrich  # noqa: E402
from mtx.split import load_analysis             # noqa: E402

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        try:
            print(f"[enrich] {msg}", file=sys.stderr, flush=True)
        except UnicodeEncodeError:
            enc = getattr(sys.stderr, "encoding", None) or "ascii"
            print(f"[enrich] {msg}".encode(enc, "replace").decode(enc, "replace"),
                  file=sys.stderr, flush=True)


def one(folder: str, cache: str, providers: list[str],
        refresh: bool, offline: bool) -> tuple[str, bool, str]:
    """Enrich a single folder.  Returns (folder, matched, note)."""
    try:
        res = load_analysis(os.path.join(folder, "analysis.json"))
    except (OSError, ValueError) as exc:
        return folder, False, f"cannot read analysis: {exc}"
    try:
        section = enrich(res, cache_dir=cache, providers=providers,
                         offline=offline, refresh=refresh, version=__version__)
    except Exception as exc:                       # one bad track must not
        return folder, False, f"{type(exc).__name__}: {exc}"   # end the run
    tmp = os.path.join(folder, "online.json.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(section, fh, indent=1, ensure_ascii=False)
        os.replace(tmp, os.path.join(folder, "online.json"))
    except OSError as exc:
        return folder, False, f"cannot write online.json: {exc}"
    genre = ((section.get("genres") or {}).get("primary")) or "no genre"
    return folder, bool(section.get("providers_available")), genre


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="the mtx output tree")
    ap.add_argument("-j", "--workers", type=int, default=8,
                    help="parallel tracks (default 8; more will not help, the "
                         "MusicBrainz rate limit is the floor)")
    ap.add_argument("--providers", default=",".join(DEFAULT_PROVIDERS),
                    help="comma-separated, or 'all'")
    ap.add_argument("--cache", help="default <root>/.mtx_cache")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore cached responses and re-fetch; this is how "
                         "you take a fresh popularity snapshot")
    ap.add_argument("--offline", action="store_true",
                    help="answer only from the cache, never the network")
    ap.add_argument("--force", action="store_true",
                    help="re-enrich folders that already have an online.json")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    load_env(args.root)

    providers = (list(ALL_PROVIDERS) if args.providers.strip().lower() == "all"
                 else [p.strip() for p in args.providers.split(",") if p.strip()])
    unknown = [p for p in providers if p not in ALL_PROVIDERS]
    if unknown:
        log(f"error: unknown provider(s): {', '.join(unknown)}")
        return 1
    if "lastfm" in providers and not os.environ.get("LASTFM_API_KEY"):
        log("warning: LASTFM_API_KEY is not set, so no playcount will be "
            "collected -- and playcount is the outcome variable. Get a free "
            "key at last.fm/api/account/create before a long run.")

    folders = _enrich_targets(args.root)
    if not folders:
        log(f"error: no analysis.json under {args.root}")
        return 1
    todo = folders if args.force else [
        f for f in folders if not os.path.isfile(os.path.join(f, "online.json"))]
    # Counted before --limit trims the queue, or the figure reports the tracks
    # this run is skipping rather than the ones already done.
    done_already = len(folders) - len(todo)
    if args.limit:
        todo = todo[:args.limit]

    cache = args.cache or os.path.join(args.root, ".mtx_cache")
    log(f"{len(todo)} to do ({done_already} already enriched), "
        f"{args.workers} workers, providers: {', '.join(providers)}")
    log(f"cache: {cache}{' (offline)' if args.offline else ''}")
    if not todo:
        return 0

    started = time.monotonic()
    done = matched = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, f, cache, providers, args.refresh,
                               args.offline): f for f in todo}
        try:
            for fut in as_completed(futures):
                folder, ok, note = fut.result()
                done += 1
                if ok:
                    matched += 1
                else:
                    failed += 1
                    log(f"  no match: {os.path.basename(folder)} -- {note}")
                if done % 25 == 0 or done == len(todo):
                    rate = done / max(time.monotonic() - started, 1e-6)
                    left = (len(todo) - done) / rate if rate else 0
                    log(f"[{done}/{len(todo)}] {matched} matched, {failed} not "
                        f"| {rate * 60:.0f}/min, ~{left / 60:.0f} min left")
        except KeyboardInterrupt:
            log("interrupted; finished tracks are written, re-run to continue")
            pool.shutdown(wait=False, cancel_futures=True)
            return 130

    elapsed = time.monotonic() - started
    log(f"done: {matched} matched, {failed} not, in {elapsed / 60:.1f} min "
        f"({elapsed / max(done, 1):.1f}s per track)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
