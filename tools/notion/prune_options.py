"""Remove select options that no row uses.

    python tools/notion/prune_options.py <corpus root>            # report only
    python tools/notion/prune_options.py <corpus root> --apply

The genre vocabulary changed. `techno/house`, `rnb/swing` and `films/games`
were Discogs browse buckets filed as if they were genres; splitting them into
their parts moved every track off those spellings, and the options themselves
stayed behind. 67 of them, sitting in the filter menus of a table whose whole
purpose is to be filtered, each one offering a category that returns nothing.

That is the same defect as an empty column, wearing different clothes: it
looks like an answer until you pick it.

Why this is a separate tool, run deliberately, rather than part of the push
------------------------------------------------------------------------
Sending a select property's options is a full replacement, not a patch. The
schema sync once sent `{"options": []}` on every run, which deleted every
option in the table -- and with them the value on every page that held one.
1,005 rows went blank before anyone saw it, and it stayed hidden because the
same run immediately re-created the options it had just destroyed.

So this refuses to guess. It reads every page first, keeps every option any
page uses, and reports what it would drop unless `--apply` is passed. If the
page query fails or returns nothing, it stops: an empty read looks exactly
like "no option is used by anything", and acting on that would empty the
table.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from client import Notion, NotionError            # noqa: E402
from env import load_env                          # noqa: E402
from push import State                            # noqa: E402

SELECTS = ("select", "multi_select")

# Notion rejects a property update whose option list is longer than this, and
# the update is a full replacement -- so a column holding more than 100 options
# cannot have one removed.  `Genres all` holds 427 and `Cohort genres` 135.
# The only route left is to drop the column and let the push rebuild it, which
# is safe precisely because nothing in it is typed by hand: every value is
# derived from the analyses on disk.
MAX_OPTIONS = 100


def log(msg: str) -> None:
    print(f"[options] {msg}", file=sys.stderr, flush=True)


def used_options(pages: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Every option name any row actually holds, per column."""
    used: dict[str, set[str]] = collections.defaultdict(set)
    for page in pages:
        for name, prop in (page.get("properties") or {}).items():
            kind = prop.get("type")
            if kind == "select" and prop.get("select"):
                used[name].add(prop["select"]["name"])
            elif kind == "multi_select":
                for opt in prop.get("multi_select") or []:
                    used[name].add(opt["name"])
    return used


def orphans(schema: dict[str, Any],
            used: dict[str, set[str]]) -> dict[str, list[str]]:
    """`{column: [option, ...]}` for options no row holds."""
    out: dict[str, list[str]] = {}
    for name, prop in (schema.get("properties") or {}).items():
        kind = prop.get("type")
        if kind not in SELECTS:
            continue
        options = [o.get("name") for o in (prop.get(kind) or {}).get("options", [])]
        dead = [o for o in options if o and o not in used.get(name, set())]
        if dead:
            out[name] = dead
    return out


def retained(schema: dict[str, Any], column: str,
             drop: set[str]) -> dict[str, Any]:
    """The property payload that keeps everything except `drop`.

    Options are sent whole, so the retained list is rebuilt from what the live
    schema holds -- ids included, because an option re-sent without its id is
    a new option and the rows pointing at the old one lose their value.
    """
    prop = (schema.get("properties") or {})[column]
    kind = prop["type"]
    keep = [{"id": o["id"], "name": o["name"], "color": o.get("color")}
            for o in prop[kind].get("options", []) if o.get("name") not in drop]
    return {column: {kind: {"options": keep}}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="the corpus root, for mtx.env and the state file")
    ap.add_argument("--apply", action="store_true",
                    help="actually remove them (default: report only)")
    ap.add_argument("--rebuild", action="store_true",
                    help=f"for columns over {MAX_OPTIONS} options, where no "
                         f"single option can be removed: drop the column so a "
                         f"forced push rebuilds it from the corpus")
    ap.add_argument("--state", help="path to .notion_state.json")
    args = ap.parse_args()
    load_env(args.root)

    state = State(args.state or os.path.join(args.root, ".notion_state.json"))
    db_id = (state.data.get("databases") or {}).get("tracks")
    if not db_id:
        log("error: no tracks database in the state file; run push.py first")
        return 2

    api = Notion(log=lambda m: None)
    try:
        schema = api.request("GET", f"/databases/{db_id}")
        pages = api.query(db_id)
    except NotionError as exc:
        log(f"error: Notion unreachable: {exc}")
        return 2

    if not pages:
        # The guard that matters.  Every option is unused in an empty read,
        # so acting on one would delete the entire vocabulary and blank every
        # row that held it.
        log("error: the table returned no rows.  Refusing to treat that as "
            "'no option is used'; nothing was changed")
        return 2

    used = used_options(pages)
    dead = orphans(schema, used)
    total = sum(len(v) for v in dead.values())
    log(f"{len(pages)} row(s), {total} unused option(s) in {len(dead)} column(s)")
    for column, options in sorted(dead.items()):
        log(f"  {column}: {len(options)}")
        for opt in sorted(options):
            log(f"      {opt}")
    if not total:
        return 0
    if not args.apply:
        log("report only; pass --apply to remove them")
        return 0

    removed = 0
    oversize: list[str] = []
    for column, options in sorted(dead.items()):
        payload = retained(schema, column, set(options))
        kind = next(iter(payload[column]))
        if len(payload[column][kind]["options"]) > MAX_OPTIONS:
            oversize.append(column)
            continue
        try:
            api.request("PATCH", f"/databases/{db_id}",
                        {"properties": payload})
        except NotionError as exc:
            log(f"  {column}: failed, {exc}")
            continue
        removed += len(options)
        log(f"  {column}: removed {len(options)}")

    for column in oversize:
        kept = len((schema["properties"][column][
            schema["properties"][column]["type"]] or {}).get("options", []))
        if not args.rebuild:
            log(f"  {column}: {kept} options, over the {MAX_OPTIONS} an update "
                f"accepts, so no single option can be removed.  Pass --rebuild "
                f"to drop the column and re-push it")
            continue
        try:
            api.request("PATCH", f"/databases/{db_id}",
                        {"properties": {column: None}})
        except NotionError as exc:
            log(f"  {column}: could not be dropped, {exc}")
            continue
        removed += len(dead[column])
        log(f"  {column}: dropped ({kept} options).  Re-create it with "
            f"`python tools/notion/push.py <root> --force`")

    log(f"done: {removed} option(s) removed")
    if oversize and args.rebuild:
        log("the dropped column(s) are empty until a forced push rebuilds them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
