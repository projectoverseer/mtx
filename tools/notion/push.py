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
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))

from client import Notion, NotionError            # noqa: E402
from env import load_env                          # noqa: E402
from rows import (OBSERVATION_SCHEMA, body_blocks, database_schema,  # noqa: E402
                  load_cohorts, load_folder, load_identities,
                  load_outcomes,
                  observations_for,
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


class Lock:
    """Refuse to run while another push is running against the same corpus.

    Two concurrent pushes each read the set of observations already logged,
    each conclude the other's rows are not there yet, and each write the whole
    snapshot: 3,957 duplicate readings from one accidental double-launch.
    Nothing errored -- both runs reported success -- and the log quietly held
    two of everything.

    A stale lock from a killed run is taken over rather than treated as fatal:
    the failure this guards against is concurrency, and a crash leaves no
    concurrency behind.
    """

    STALE_AFTER_S = 3600.0

    def __init__(self, path: str) -> None:
        self.path = path
        self.held = False

    def acquire(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as fh:
                    other = json.load(fh)
            except (OSError, ValueError):
                other = {}
            age = time.time() - float(other.get("started", 0) or 0)
            if age < self.STALE_AFTER_S:
                raise SystemExit(
                    f"another push started {age / 60:.0f} min ago is still "
                    f"holding {self.path} (pid {other.get('pid')}).  Two at "
                    f"once duplicate every observation row.  Wait for it, or "
                    f"delete that file if you know it died.")
            log(f"taking over a stale lock from pid {other.get('pid')}")
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"pid": os.getpid(), "started": time.time()}, fh)
        self.held = True

    def release(self) -> None:
        if self.held:
            try:
                os.remove(self.path)
            except OSError:
                pass
            self.held = False


def new_properties(live: dict, wanted: dict) -> dict:
    """The properties `wanted` has and the live database does not.

    The whole schema used to be sent on every run, and `database_schema()`
    describes a select as `{"options": []}` because that is what creating one
    needs.  Notion reads an explicit empty list as *these are the options now*
    and deletes the rest -- and deleting an option blanks it on every page
    holding it.  So each run silently wiped every select column in the
    database.

    It went unnoticed for as long as it did because each run then rewrote all
    1,321 pages, which re-created every option on the way through: destroyed
    and restored inside one run, net zero, invisible.  The moment a run only
    wrote the pages that had changed, the restore covered 13 pages and 9
    artists, and 1,005 rows were left with an empty Artist.

    Sending only genuinely new properties keeps what this was for -- a column
    added to schema.py reaches the live table -- and cannot touch an existing
    one.
    """
    have = set((live.get("properties") or {}))
    return {name: spec for name, spec in wanted.items() if name not in have}


def add_new_properties(api: Notion, db_id: str, wanted: dict) -> list[str]:
    """Create any property the live database is missing.  Never edit one."""
    live = api.request("GET", f"/databases/{db_id}")
    add = new_properties(live, wanted)
    if add:
        api.update_database(db_id, add)
        log(f"  added {len(add)} new propert(y/ies): "
            f"{', '.join(sorted(add)[:6])}")
    return sorted(add)


def ensure_databases(api: Notion, parent: str, state: State,
                     dry_run: bool) -> tuple[str, str]:
    if dry_run:
        return "dry-tracks", "dry-observations"

    known = state.data.get("databases") or {}
    if known.get("tracks") and known.get("observations"):
        # Still push the schema: a property added to schema.py that never
        # reached the live database makes every page fail with "Could not find
        # property".  But send only the properties that are actually missing.
        add_new_properties(api, known["tracks"], database_schema())
        return known["tracks"], known["observations"]

    existing = api.find_databases(parent)
    tracks = existing.get(TRACKS_DB)
    observations = existing.get(OBSERVATIONS_DB)

    if tracks:
        # Adding properties to a live database is safe and lets the schema
        # grow without a rebuild; removing them is not, so this only adds.
        add_new_properties(api, tracks, database_schema())
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


# Files whose content ends up on a track's page.  `mtx_source.json` is in the
# list because it carries the schema and profile the page reports.
STAMPED = ("analysis.json", "online.json", "corpus_row.json", "mtx_source.json")
# Artefacts shared by every page: a change in any of them changes every row.
SHARED = ("cohort.json", "outcome.json", "artists.json")


def folder_stamp(folder: str) -> str:
    """A cheap fingerprint of everything this folder contributes to its page.

    `stat`, never a read: deciding what to push by parsing 1,321 analyses
    would cost nine minutes before the first page is written, which is the
    reason the skip check looked only at the state file in the first place.
    Size and modification time together are enough to notice an amendment --
    and an amendment is exactly what a transcription pass is.
    """
    parts = []
    for name in STAMPED:
        try:
            st = os.stat(os.path.join(folder, name))
            parts.append(f"{st.st_mtime_ns}.{st.st_size}")
        except OSError:
            parts.append("-")
    return ",".join(parts)


def shared_stamp(root: str) -> str:
    """A fingerprint of the artefacts every page reads.

    Hashed by content rather than by mtime: `cohort.json` is rewritten on
    every pipeline run whether or not a single number in it moved, and
    stamping it by mtime would force a full 1,321-page re-push daily for
    nothing.
    """
    h = hashlib.sha256()
    for name in SHARED:
        try:
            with open(os.path.join(root, name), "rb") as fh:
                h.update(hashlib.sha256(fh.read()).digest())
        except OSError:
            h.update(b"-")
    return h.hexdigest()[:16]


def needs_push(key: str, stamp: str, tracks: dict, stamps: dict,
               force: bool = False) -> bool:
    """Should this track be sent?  Three reasons, and the third was missing.

    Never pushed; pushed but the page is gone from the state; or pushed with
    content that has since changed.  Only the first two were checked, so an
    amended analysis reported `already present` for ever -- which is how a
    day's worth of transcripts, repaired lyrics and fresh percentiles came to
    sit on disk while the pipeline said every stage was clean.
    """
    if force:
        return True
    if not tracks.get(key):
        return True
    return stamps.get(key) != stamp


def source_sha(folder: str) -> str | None:
    """The track's sha256 from `mtx_source.json` -- 895 bytes, not 3 MB."""
    try:
        with open(os.path.join(folder, "mtx_source.json"), encoding="utf-8") as fh:
            return ((json.load(fh).get("source") or {}).get("sha256")) or None
    except (OSError, ValueError):
        return None


def observation_keys(api: Notion, db_id: str) -> set[tuple[str, str, str]]:
    """`(sha256, metric, day)` for every observation already logged.

    Append-only is a rule about *corrections* -- a later reading never edits an
    earlier one -- not a licence to write the same reading twice.  Re-pushing a
    corpus after fixing a column re-sends the same snapshot, and without this
    every row is duplicated: one forced re-push turned 3,785 observations into
    7,573.  A reading is identified by its track, its metric and the day it was
    taken, so a genuine later snapshot still lands as a new row.
    """
    keys: set[tuple[str, str, str]] = set()
    for page in api.query(db_id):
        props = page.get("properties") or {}
        rich = (props.get("Track sha256") or {}).get("rich_text") or []
        sha = rich[0]["text"]["content"] if rich else ""
        metric = ((props.get("Metric") or {}).get("select") or {}).get("name") or ""
        day = (((props.get("Observed at") or {}).get("date") or {})
               .get("start") or "")[:10]
        if sha and metric:
            keys.add((sha, metric, day))
    return keys


def observation_key(row: dict) -> tuple[str, str, str]:
    def text(name):
        rt = (row.get(name) or {}).get("rich_text") or []
        return rt[0]["text"]["content"] if rt else ""

    def select(name):
        return ((row.get(name) or {}).get("select") or {}).get("name") or ""

    day = ((row.get("Observed at") or {}).get("date") or {}).get("start", "")[:10]
    return (text("Track sha256"), select("Metric"), day)


def reconcile(api: Notion, db_id: str, state: State) -> int:
    """Rebuild the pushed-set from Notion itself, by sha256.

    The state file is an optimisation, not the record.  If it is lost, stale,
    or was last written 25 tracks before an interruption, a resumed run would
    create duplicate pages for everything it had forgotten -- and a duplicate
    is far more annoying to clean up than a re-push is to wait for.  So the
    live database is asked once at startup and believed over the file.
    """
    known = dict(state.data.get("tracks") or {})
    found = 0
    for page in api.query(db_id):
        prop = (page.get("properties") or {}).get("sha256") or {}
        rich = prop.get("rich_text") or []
        sha = rich[0]["text"]["content"] if rich else ""
        if sha:
            known[sha] = page["id"]
            found += 1
    state.data["tracks"] = known
    state.save()
    return found



def prune_options(api: Notion, db_id: str, log_fn=log) -> dict[str, int]:
    """Drop select and multi-select options no page uses any more.

    Notion keeps an option in the dropdown for ever once it has existed, even
    after the last page stops using it.  So correcting the data is not enough:
    re-pushing 1,321 tracks with a clean artist column leaves the 264 broken
    values sitting in the filter menu, where they still look like categories
    and still return nothing.  The options array on the database is
    authoritative, so writing back only the ones in use removes the rest.

    Run this only after a push that succeeded.  An option removed here is
    removed from any page still holding it.
    """
    used: dict[str, set[str]] = {}
    for page in api.query(db_id):
        for name, value in (page.get("properties") or {}).items():
            kind = value.get("type")
            if kind == "select" and value.get("select"):
                used.setdefault(name, set()).add(value["select"]["name"])
            elif kind == "multi_select":
                for opt in value.get("multi_select") or []:
                    used.setdefault(name, set()).add(opt["name"])

    db = api.request("GET", f"/databases/{db_id}")
    removed: dict[str, int] = {}
    payload: dict = {}
    for name, spec in (db.get("properties") or {}).items():
        kind = spec.get("type")
        if kind not in ("select", "multi_select"):
            continue
        options = spec[kind].get("options") or []
        keep = [o for o in options if o["name"] in used.get(name, set())]
        if len(keep) != len(options):
            removed[name] = len(options) - len(keep)
            payload[name] = {kind: {"options": [{"name": o["name"],
                                                 "color": o.get("color", "default")}
                                                for o in keep]}}
    if payload:
        api.update_database(db_id, payload)
        for name, count in sorted(removed.items(), key=lambda kv: -kv[1]):
            log_fn(f"  pruned {count} unused option(s) from {name!r}")
    return removed


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
    ap.add_argument("-j", "--workers", type=int, default=6,
                    help=("parallel pages (default 6). Pushing is latency-bound, "
                          "not throttle-bound, so this is most of the speed"))
    ap.add_argument("--state", help="resume file (default <root>/.notion_state.json)")
    ap.add_argument("--no-body", action="store_true",
                    help="properties only; skip the full row and section blocks")
    ap.add_argument("--skip-observations", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-push tracks already recorded in the state file")
    ap.add_argument("--prune-options", action="store_true",
                    help=("after a clean push, drop select options no page "
                          "uses; Notion keeps them for ever otherwise"))
    ap.add_argument("--archive-db", metavar="TITLE", action="append",
                    help=("archive a database of this title under --parent "
                          "once the push succeeds; repeatable"))
    args = ap.parse_args()
    load_env(args.root)

    if not args.dry_run and not args.parent:
        # Only needed to *create* the databases.  Once the state file knows
        # their ids the parent page is irrelevant, and demanding it turned
        # every routine re-push -- including the pipeline's -- into a usage
        # error, after the audit had already passed.
        known = State(args.state or os.path.join(args.root,
                                                 ".notion_state.json"))
        if not (known.data.get("databases") or {}).get("tracks"):
            ap.error("--parent is required the first time, to say which page "
                     "the databases should be created under")

    lock = Lock(os.path.join(args.root, ".notion_push.lock"))
    if not args.dry_run:
        lock.acquire()
    try:
        return _run(args)
    finally:
        lock.release()


def _run(args) -> int:
    folders = _enrich_targets(args.root)
    if not folders:
        log(f"error: no analysis.json under {args.root}")
        return 1
    if args.limit:
        folders = folders[:args.limit]

    outcomes = load_outcomes(args.root)
    identities = load_identities(args.root)
    cohorts = load_cohorts(args.root)
    if not outcomes:
        log("note: no outcome.json; the within-artist outcome columns will "
            "be empty. Run tools/notion/outcome.py first.")

    state = State(args.state or os.path.join(args.root, ".notion_state.json"))
    api = Notion(args.token, dry_run=args.dry_run, log=log)
    tracks_db, obs_db = ensure_databases(api, args.parent, state, args.dry_run)

    seen_observations: set[tuple[str, str, str]] = set()
    if not args.dry_run:
        found = reconcile(api, tracks_db, state)
        log(f"reconciled {found} existing page(s) from Notion")
        if not args.skip_observations:
            seen_observations = observation_keys(api, obs_db)
            log(f"{len(seen_observations)} observation(s) already logged")

    log(f"{len(folders)} folder(s) | {len(PROPERTIES)} properties | "
        f"traits {TRAIT_VERSION} | {args.workers} workers | "
        f"{'DRY RUN' if args.dry_run else 'live'}")

    # Decide what to skip from `mtx_source.json` (895 bytes) and a handful of
    # `stat` calls, never by loading the analysis.  Reading all 1,321
    # documents up front to look at one field would cost nine minutes before
    # the first page is written and hold every parsed document in memory at
    # once; the worker loads its own.
    #
    # The state used to record only *that* a sha had been pushed.  So an
    # amended analysis -- 1,306 new transcripts, 66 repaired lyrics, a fresh
    # set of cohort percentiles -- was invisible to it: `0 pushed, 1321
    # already present`, a clean exit, and 25 columns that stayed empty while
    # every stage reported success.  What it records now is what was pushed,
    # not merely that something was.
    shared = shared_stamp(args.root)
    stamps = state.data.setdefault("stamps", {})
    want: dict[str, str] = {}
    todo = []
    for f in folders:
        key = source_sha(f) or f
        want[key] = f"{shared}|{folder_stamp(f)}"
        if needs_push(key, want[key], state.data["tracks"], stamps,
                      args.force):
            todo.append(f)
    skipped = len(folders) - len(todo)
    changed = sum(1 for f in folders
                  if (source_sha(f) or f) in stamps
                  and stamps.get(source_sha(f) or f) != want.get(source_sha(f) or f))
    if changed and not args.force:
        log(f"{changed} track(s) changed since their last push")

    counts = {"pushed": 0, "failed": 0, "obs": 0}
    guard = threading.Lock()
    started = time.monotonic()

    def work(folder):
        name = os.path.basename(folder)
        try:
            doc = load_folder(folder, outcomes, args.root, identities,
                              cohorts)
            sha = dig(doc, "file.sha256") or folder
        except (OSError, ValueError) as exc:
            return folder, None, 0, f"{name}: cannot read analysis: {exc}"
        try:
            if args.dry_run:
                dump(args.dump_dir, name, {
                    "properties": properties_for(doc),
                    "blocks": 0 if args.no_body else len(body_blocks(doc)),
                    "observations": observations_for(doc),
                })
                return sha, "dry-run", 0, None
            # Always hand over the page this sha is already on.  `push_track`
            # creates when it gets None, and passing None only outside
            # `--force` was safe *only* while `todo` could not contain a sha
            # that had been pushed before.  It can now -- that is the point of
            # the stamp -- so a changed track would have arrived here with no
            # page id and been published a second time, once per amended
            # analysis, silently, alongside the original.
            page_id = push_track(api, tracks_db, doc,
                                 (state.data["tracks"] or {}).get(sha),
                                 not args.no_body)
            rows = 0
            if not args.skip_observations:
                for row in observations_for(doc):
                    key = observation_key(row)
                    with guard:
                        if key in seen_observations:
                            continue
                        seen_observations.add(key)
                    api.create_page(obs_db, row)
                    rows += 1
            return sha, page_id, rows, None
        except NotionError as exc:
            return sha, None, 0, f"{name}: {exc}"
        except Exception as exc:               # one bad page must not end the run
            return sha, None, 0, f"{name}: {type(exc).__name__}: {exc}"

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(work, folder) for folder in todo]
            try:
                for n, fut in enumerate(as_completed(futures), 1):
                    sha, page_id, rows, err = fut.result()
                    with guard:
                        if err:
                            counts["failed"] += 1
                            log(f"  {err}")
                        else:
                            state.data["tracks"][sha] = page_id
                            # Only on success: a failed page must be retried,
                            # and a stamp written regardless would mark it
                            # current for ever.
                            if sha in want:
                                stamps[sha] = want[sha]
                            counts["pushed"] += 1
                            counts["obs"] += rows
                        if n % 25 == 0 or n == len(todo):
                            rate = n / max(time.monotonic() - started, 1e-6)
                            left = (len(todo) - n) / rate if rate else 0
                            log(f"[{n}/{len(todo)}] {counts['pushed']} pushed, "
                                f"{skipped} already there, {counts['failed']} failed, "
                                f"{counts['obs']} observations "
                                f"| {rate * 60:.0f}/min, ~{left / 60:.0f} min left")
                            state.save()
            except KeyboardInterrupt:
                log("interrupted; re-run to continue (pages are reconciled by sha256)")
                pool.shutdown(wait=False, cancel_futures=True)
                state.save()
                return 130

    pushed, failed, obs_rows = counts["pushed"], counts["failed"], counts["obs"]

    state.save()

    # Only after a successful push: an archived database is recoverable from
    # the Notion trash, but retiring the old one before the new one exists
    # would leave nothing to read in between.
    # The gate is "the corpus is fully present and nothing failed this run",
    # not "this run wrote something": a re-run that finds everything already
    # pushed is exactly when archiving is safest, and requiring `pushed` meant
    # the retry that fixed the last failure could never trigger it.
    complete = (pushed + skipped) == len(folders) and not failed
    if args.prune_options and not args.dry_run and complete:
        for db_id, label in ((tracks_db, TRACKS_DB), (obs_db, OBSERVATIONS_DB)):
            log(f"pruning unused options in {label}")
            prune_options(api, db_id)
    if args.archive_db and not args.dry_run and complete:
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
    elif args.archive_db:
        log(f"archive skipped: {failed} failed and "
            f"{pushed + skipped}/{len(folders)} present, so the old "
            f"database stays where it is")

    log(f"done: {pushed} pushed, {skipped} already present, {failed} failed, "
        f"{obs_rows} observation rows, {api.requests} API requests")
    if args.dry_run:
        log(f"payloads written to {os.path.abspath(args.dump_dir)}")
    return 1 if failed and not pushed else 0


if __name__ == "__main__":
    raise SystemExit(main())
