"""Delete cached stems for tracks that have already been measured.

Separation output is four uncompressed wavs a track -- about 165 MB -- and
`mtx scan` never evicts any of it.  That is fine for an album and impossible
for a library: 1 274 tracks want 210 GB of cache, which is more disk than most
machines have spare, and the run dies partway through with the measurements it
had already paid for still on disk but the scan not finished.

Once a track has a completed measurement its stems are no longer needed.  The
receipt `mtx scan` leaves beside each measurement records the source file's
sha256, and the stem cache is keyed on the first 24 characters of exactly that
hash -- so this reads the mirror tree and needs no access to the audio at all.

Reports by default.  Pass --apply to actually delete.

    python tools/prune_stems.py E:\\Music\\_mtx_out
    python tools/prune_stems.py E:\\Music\\_mtx_out --apply
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from mtx.metrics.stems import CACHE_DIR  # noqa: E402

LEDGER = "mtx_source.json"
ROW = "corpus_row.json"
KEY_CHARS = 24


def measured_keys(out_dir: str) -> tuple[set[str], int]:
    """Cache keys for every track under `out_dir` with a finished measurement.

    A ledger alone is not enough: it is written beside the outputs, and a run
    interrupted between the two would have this delete stems for a track that
    still has to be measured.  Requiring the corpus row as well means only a
    track that actually produced a measurement loses its stems.
    """
    keys, skipped = set(), 0
    for root, _, names in os.walk(out_dir):
        if LEDGER not in names:
            continue
        if ROW not in names:
            skipped += 1
            continue
        try:
            with open(os.path.join(root, LEDGER), encoding="utf-8") as f:
                led = json.load(f)
            digest = (led.get("source") or {}).get("sha256")
        except (OSError, ValueError):
            skipped += 1
            continue
        if isinstance(digest, str) and len(digest) >= KEY_CHARS:
            keys.add(digest[:KEY_CHARS])
        else:
            skipped += 1
    return keys, skipped


def entry_bytes(path: str) -> int:
    total = 0
    for root, _, names in os.walk(path):
        for n in names:
            try:
                total += os.path.getsize(os.path.join(root, n))
            except OSError:
                pass
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="a mirror tree written by `mtx scan`")
    ap.add_argument("--cache", default=CACHE_DIR,
                    help=f"stem cache to prune (default {CACHE_DIR})")
    ap.add_argument("--apply", action="store_true",
                    help="delete; without it nothing is removed")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.out_dir):
        print(f"no such mirror tree: {args.out_dir}")
        return 1
    if not os.path.isdir(args.cache):
        print(f"no stem cache at {args.cache}; nothing to do")
        return 0

    keys, skipped = measured_keys(args.out_dir)
    present = {d for d in os.listdir(args.cache)
               if os.path.isdir(os.path.join(args.cache, d))}
    stale = sorted(present & keys)
    keep = present - keys

    freed = sum(entry_bytes(os.path.join(args.cache, k)) for k in stale)
    held = sum(entry_bytes(os.path.join(args.cache, k)) for k in sorted(keep))

    print(f"cache            : {args.cache}")
    print(f"entries          : {len(present)}")
    print(f"measured already : {len(stale)}   {freed / 1e9:.2f} GB")
    print(f"still needed     : {len(keep)}   {held / 1e9:.2f} GB")
    if skipped:
        print(f"receipts skipped : {skipped} (no corpus row, or unreadable)")

    if not args.apply:
        print("\nreporting only; pass --apply to delete")
        return 0

    for k in stale:
        shutil.rmtree(os.path.join(args.cache, k), ignore_errors=True)
    print(f"\ndeleted {len(stale)} entry(s), freed {freed / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
