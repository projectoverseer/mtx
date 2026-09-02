"""Archive duplicate rows in the Observations log.

    python tools/notion/dedupe_observations.py <observations_db_id>

A reading is identified by `(track sha256, metric, day)`.  Before `push.py`
learned to check for those, a forced re-push wrote the whole snapshot again --
3,785 observations became 7,573, and the older copy of each pair carried the
broken artist names the re-push existed to fix.

Where a key has more than one row this keeps exactly one, preferring a clean
`Artist` value (no " / " separator, which is what the raw tag looked like) and
then the most recently created.  The rest are archived, not deleted: Notion
keeps an archived page recoverable, and a wrongly-archived observation should
cost a restore rather than a re-run of the enrichment.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import Notion, NotionError          # noqa: E402


def log(msg: str) -> None:
    print(f"[dedupe] {msg}", file=sys.stderr, flush=True)


def key_of(page: dict) -> tuple[str, str, str]:
    props = page.get("properties") or {}
    rich = (props.get("Track sha256") or {}).get("rich_text") or []
    sha = rich[0]["text"]["content"] if rich else ""
    metric = ((props.get("Metric") or {}).get("select") or {}).get("name") or ""
    day = (((props.get("Observed at") or {}).get("date") or {})
           .get("start") or "")[:10]
    return (sha, metric, day)


def artist_of(page: dict) -> str:
    props = page.get("properties") or {}
    return ((props.get("Artist") or {}).get("select") or {}).get("name") or ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("database")
    ap.add_argument("--token")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api = Notion(args.token, log=log)
    pages = api.query(args.database)
    log(f"{len(pages)} row(s) in the log")

    groups: dict[tuple[str, str, str], list[dict]] = collections.defaultdict(list)
    for page in pages:
        groups[key_of(page)].append(page)

    doomed: list[dict] = []
    for key, members in groups.items():
        if len(members) < 2:
            continue
        # A clean artist first, then the newest, so the survivor is the row
        # written by the corrected push.
        members.sort(key=lambda p: (" / " in artist_of(p),
                                    -_created(p)))
        doomed.extend(members[1:])

    log(f"{len(groups)} distinct reading(s); {len(doomed)} duplicate row(s) to archive")
    if args.dry_run or not doomed:
        for p in doomed[:5]:
            log(f"  would archive {key_of(p)} artist={artist_of(p)!r}")
        return 0

    archived = failed = 0
    for i, page in enumerate(doomed, 1):
        try:
            api.request("PATCH", f"/pages/{page['id']}", {"archived": True})
            archived += 1
        except NotionError as exc:
            failed += 1
            log(f"  {exc}")
        if i % 100 == 0 or i == len(doomed):
            log(f"[{i}/{len(doomed)}] {archived} archived, {failed} failed")
    log(f"done: {archived} archived, {failed} failed")
    return 1 if failed and not archived else 0


def _created(page: dict) -> float:
    stamp = page.get("created_time") or ""
    # ISO 8601 sorts lexically, so the string itself is the ordering key; the
    # numeric conversion only has to be monotonic.
    return float(int("".join(c for c in stamp if c.isdigit()) or 0))


if __name__ == "__main__":
    raise SystemExit(main())
