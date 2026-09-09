"""Re-point observation rows at the artist name the corpus now uses.

    python tools/notion/resync_artists.py <corpus root> [--dry-run]

`tools/identity.py` resolves a library folder to the name MusicBrainz uses --
"Red Hot Chilli Peppers" becomes "Red Hot Chili Peppers". Track pages pick
that up on the next push, because a push rewrites the whole page. Observation
rows do not: they are append-only, so a row written last week keeps the name
that was current last week, and the two tables stop agreeing.

That matters more than it looks. The Artist column is what a human filters on
when reading the log, and a join between the two tables on that column
silently drops every row spelt the old way. `audit.py` reports it as
`notion.artist_drift`; this repairs it.

Only the select value is touched. The reading, its date, its value and the
sha256 that actually identifies the track are all left exactly as they were --
this is a relabelling, not a correction of anything measured.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import identity                                  # noqa: E402
from client import Notion, NotionError           # noqa: E402
from push import State                           # noqa: E402


def log(msg: str) -> None:
    print(f"[resync] {msg}", file=sys.stderr, flush=True)


def artist_of(page: dict) -> str:
    props = page.get("properties") or {}
    return ((props.get("Artist") or {}).get("select") or {}).get("name") or ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument("--token")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    resolved = identity.load(args.root)
    if not resolved:
        log(f"no artists.json under {args.root}; run tools/identity.py first")
        return 1
    # Folder name -> the name the Corpus table shows.  Only entries that
    # actually changed are worth rewriting.
    rename = {folder: entry["notion_name"] for folder, entry in resolved.items()
              if entry.get("notion_name") and entry["notion_name"] != folder}
    if not rename:
        log("every folder already resolves to its own name; nothing to do")
        return 0
    log(f"{len(rename)} renamed artist(s): "
        + ", ".join(f"{k!r}->{v!r}" for k, v in sorted(rename.items())))

    state = State(os.path.join(args.root, ".notion_state.json"))
    obs_db = (state.data.get("databases") or {}).get("observations")
    if not obs_db:
        log("no observations database in the state file")
        return 1

    api = Notion(args.token, log=lambda _m: None)
    pages = api.query(obs_db)
    log(f"{len(pages)} observation row(s)")

    todo = [(p, rename[artist_of(p)]) for p in pages if artist_of(p) in rename]
    counts = collections.Counter(artist_of(p) for p, _ in todo)
    log(f"{len(todo)} row(s) to relabel: {dict(counts)}")
    if args.dry_run or not todo:
        return 0

    done = failed = 0
    for i, (page, name) in enumerate(todo, 1):
        try:
            api.request("PATCH", f"/pages/{page['id']}",
                        {"properties": {"Artist": {"select": {"name": name}}}})
            done += 1
        except NotionError as exc:
            failed += 1
            log(f"  {exc}")
        if i % 200 == 0 or i == len(todo):
            log(f"[{i}/{len(todo)}] {done} relabelled, {failed} failed")
    log(f"done: {done} relabelled, {failed} failed.  Run push.py with "
        f"--prune-options to drop the options nothing uses now.")
    return 1 if failed and not done else 0


if __name__ == "__main__":
    raise SystemExit(main())
