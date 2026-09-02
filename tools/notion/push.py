"""Push a measured corpus into Notion.

    python tools/notion/push.py E:\\Music\\_mtx_out --parent <page_id>
    python tools/notion/push.py E:\\Music\\_mtx_out --dry-run --limit 5

Two databases are created under the parent page and then kept in place:

* **Corpus** -- one page per analysed folder.  ~150 queryable properties,
  and the full 2,000-column row, section timeline, chord track and confidence
  notes in the page body.
* **Corpus Observations** -- append-only.  One row per time-varying figure per
  lookup, stamped with `observed_at`.  Re-running after a later `mtx enrich
  --refresh` adds rows; it never edits the old ones, which is what makes a
  trajectory recoverable instead of a snapshot that quietly goes stale.

The run is idempotent on tracks (matched by sha256) and additive on
observations, so it can be interrupted and resumed.  `--dry-run` writes every
payload to disk and sends nothing, which is how to check the shape without a
token.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))

from client import Notion, NotionError            # noqa: E402
from rows import (OBSERVATION_SCHEMA, body_blocks, database_schema,  # noqa: E402
                  load_folder, load_outcomes, observations_for,
                  properties_for)
from schema import PROPERTIES, TRAIT_VERSION, dig  # noqa: E402

from mtx.cli import _enrich_targets               # noqa: E402

TRACKS_DB = "Corpus"
OBSERVATIONS_DB = "Corpus Observations"


def log(msg: str) -> None:
    try:
        print(f"[push] {msg}", file=sys.stderr, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stderr, "encoding", None) or "ascii"
        print(f"[push] {msg}".encode(enc, "replace").decode(enc, "replace"),
              file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------


class State:
    """Which folders are already pushed, and the page each became.

    Kept beside the corpus rather than in Notion so a resumed run costs no
    requests to work out where it stopped.
    """

    def __init__(self, path: str):
        self.path = path
        self.data = {"tracks": {}, "observations": [], "databases": {}}
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    self.data.update(json.load(fh))
            except (OSError, ValueError):
                log(f"warning: unreadable state at {path}; starting fresh")

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=1, ensure_ascii=False)
        os.replace(tmp, self.path)


# --------------------------------------------------------------------------
# databases
# --------------------------------------------------------------------------


def ensure_databases(api: Notion, parent: str, state: State,
                     dry_run: bool) -> tuple[str, str]:
    if dry_run:
        return "dry-tracks", "dry-observations"

    known = state.data.get("databases") or {}
    if known.get("tracks") and known.get("observations"):
        return known["tracks"], known["observations"]

    existing = api.find_databases(parent)
    tracks = existing.get(TRACKS_DB)
    observations = existing.get(OBSERVATIONS_DB)

    if tracks:
        # Adding properties to a live database is safe and lets the schema
        # grow without a rebuild; removing them is not, so this only adds.
        api.update_database(tracks, database_schema())
        log(f"reusing {TRACKS_DB} ({tracks})")
    else:
        tracks = api.create_database(parent, TRACKS_DB, database_schema())["id"]
        log(f"created {TRACKS_DB} ({tracks})")

    if observations:
        log(f"reusing {OBSERVATIONS_DB} ({observations})")
    else:
        observations = api.create_database(parent, OBSERVATIONS_DB,
                                           OBSERVATION_SCHEMA)["id"]
        log(f"created {OBSERVATIONS_DB} ({observations})")

    state.data["databases"] = {"tracks": tracks, "observations": observations}
    state.save()
    return tracks, observations


# --------------------------------------------------------------------------
# push
# --------------------------------------------------------------------------


def push_track(api: Notion, db_id: str, doc: dict, page_id: str | None,
               with_body: bool) -> str:
    props = properties_for(doc)
    if page_id:
        api.update_page(page_id, props)
        return page_id
    blocks = body_blocks(doc) if with_body else []
    page = api.create_page(db_id, props, blocks[:100])
    if with_body and len(blocks) > 100:
        api.append_blocks(page["id"], blocks[100:])
    return page["id"]


def dump(out_dir: str, name: str, payload) -> None:
    os.makedirs(out_dir, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:120]
    with open(os.path.join(out_dir, f"{safe}.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="the mtx output tree, e.g. E:\\Music\\_mtx_out")
    ap.add_argument("--parent", help="Notion page id the databases live under")
    ap.add_argument("--token", help="Notion integration token (or NOTION_TOKEN)")
    ap.add_argument("--dry-run", action="store_true",
                    help="write payloads to --dump-dir and send nothing")
    ap.add_argument("--dump-dir", default="notion_payloads")
    ap.add_argument("--limit", type=int, help="only the first N folders")
    ap.add_argument("--state", help="resume file (default <root>/.notion_state.json)")
    ap.add_argument("--no-body", action="store_true",
                    help="properties only; skip the full row and section blocks")
    ap.add_argument("--skip-observations", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-push tracks already recorded in the state file")
    ap.add_argument("--archive-db", metavar="TITLE", action="append",
                    help=("archive a database of this title under --parent "
                          "once the push succeeds; repeatable"))
    args = ap.parse_args()

    if not args.dry_run and not args.parent:
        ap.error("--parent is required unless --dry-run")

    folders = _enrich_targets(args.root)
    if not folders:
        log(f"error: no analysis.json under {args.root}")
        return 1
    if args.limit:
        folders = folders[:args.limit]

    outcomes = load_outcomes(args.root)
    if not outcomes:
        log("note: no outcome.json; the within-artist outcome columns will "
            "be empty. Run tools/notion/outcome.py first.")

    state = State(args.state or os.path.join(args.root, ".notion_state.json"))
    api = Notion(args.token, dry_run=args.dry_run, log=log)
    tracks_db, obs_db = ensure_databases(api, args.parent, state, args.dry_run)

    log(f"{len(folders)} folder(s) | {len(PROPERTIES)} properties | "
        f"traits {TRAIT_VERSION} | {'DRY RUN' if args.dry_run else 'live'}")

    pushed = skipped = failed = obs_rows = 0
    started = time.monotonic()

    for i, folder in enumerate(folders, 1):
        name = os.path.basename(folder)
        try:
            doc = load_folder(folder, outcomes)
        except (OSError, ValueError) as exc:
            log(f"[{i}/{len(folders)}] {name}: cannot read analysis: {exc}")
            failed += 1
            continue

        sha = dig(doc, "file.sha256") or folder
        known = (state.data["tracks"] or {}).get(sha)
        if known and not args.force:
            skipped += 1
            continue

        try:
            if args.dry_run:
                dump(args.dump_dir, f"{i:04d}_{name}", {
                    "properties": properties_for(doc),
                    "blocks": len(body_blocks(doc)) if not args.no_body else 0,
                    "observations": observations_for(doc),
                })
                page_id = "dry-run"
            else:
                page_id = push_track(api, tracks_db, doc,
                                     known if args.force else None,
                                     not args.no_body)
                if not args.skip_observations:
                    for row in observations_for(doc):
                        api.create_page(obs_db, row)
                        obs_rows += 1
            state.data["tracks"][sha] = page_id
            pushed += 1
        except NotionError as exc:
            log(f"[{i}/{len(folders)}] {name}: {exc}")
            failed += 1
            continue

        if i % 25 == 0 or i == len(folders):
            rate = i / max(time.monotonic() - started, 1e-6)
            left = (len(folders) - i) / rate if rate else 0
            log(f"[{i}/{len(folders)}] {pushed} pushed, {skipped} skipped, "
                f"{failed} failed, {obs_rows} observations "
                f"| {rate * 60:.0f}/min, ~{left / 60:.0f} min left")
            state.save()

    state.save()

    # Only after a successful push: an archived database is recoverable from
    # the Notion trash, but retiring the old one before the new one exists
    # would leave nothing to read in between.
    if args.archive_db and not args.dry_run and pushed and not failed:
        existing = api.find_databases(args.parent)
        for title in args.archive_db:
            db_id = existing.get(title)
            if not db_id:
                log(f"archive: no database titled {title!r} under the parent")
            elif db_id in (tracks_db, obs_db):
                log(f"archive: refusing to archive {title!r} -- it is a "
                    f"database this run just wrote to")
            else:
                api.request("PATCH", f"/databases/{db_id}", {"archived": True})
                log(f"archived {title!r} ({db_id})")
    elif args.archive_db and failed:
        log(f"archive skipped: {failed} page(s) failed, so the old database "
            f"stays where it is")

    log(f"done: {pushed} pushed, {skipped} already present, {failed} failed, "
        f"{obs_rows} observation rows, {api.requests} API requests")
    if args.dry_run:
        log(f"payloads written to {os.path.abspath(args.dump_dir)}")
    return 1 if failed and not pushed else 0


if __name__ == "__main__":
    raise SystemExit(main())
