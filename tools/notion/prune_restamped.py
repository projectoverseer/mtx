"""Archive observations that were re-dated rather than re-read.

    python tools/notion/prune_restamped.py <observations_db_id> [--dry-run]

Before `observed_at` was taken from the provider's own `fetched_utc`, a run
answered from the HTTP cache stamped yesterday's numbers with today's date.
The log then held what looks like a second reading and is not one: same track,
same metric, same value, a day later, no request made.

Deleting every repeat would be wrong. "Still 24,652,445 plays a week later" is
a real observation, and an append-only log exists to hold exactly that. What
is not real is a row claiming to be a reading taken on a day nothing was read.

So this archives a row only when **all** of these hold against an earlier row
for the same track and metric:

  * the value is identical -- not merely close;
  * the earlier row is within `--within` days (default 2), so a genuinely
    unchanged figure months later still counts as a new observation;
  * the later row is the one archived, so the reading keeps the date it was
    actually taken on.

Rows whose value moved, and rows with no earlier counterpart at all, are left
alone -- which is what saves the ~89 tracks that only got a play count once
the Last.fm matcher was fixed.

Archived, never deleted: Notion keeps an archived page recoverable, and a
wrongly-archived reading should cost a restore rather than a re-run.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import Notion, NotionError          # noqa: E402


def log(msg: str) -> None:
    print(f"[restamp] {msg}", file=sys.stderr, flush=True)


def read(page: dict) -> tuple[str, str, str, float | None]:
    props = page.get("properties") or {}
    rich = (props.get("Track sha256") or {}).get("rich_text") or []
    sha = rich[0]["text"]["content"] if rich else ""
    metric = ((props.get("Metric") or {}).get("select") or {}).get("name") or ""
    day = (((props.get("Observed at") or {}).get("date") or {})
           .get("start") or "")[:10]
    value = (props.get("Value") or {}).get("number")
    return sha, metric, day, value


def days_between(a: str, b: str) -> int | None:
    try:
        return abs((datetime.date.fromisoformat(b)
                    - datetime.date.fromisoformat(a)).days)
    except (ValueError, TypeError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("database")
    ap.add_argument("--token")
    ap.add_argument("--within", type=int, default=2,
                    help="how close two identical readings have to be before "
                         "the later one counts as a re-stamp (days)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api = Notion(args.token, log=lambda _m: None)
    pages = api.query(args.database)
    log(f"{len(pages)} row(s) in the log")

    series: dict[tuple[str, str], list[tuple[str, float | None, dict]]] = \
        collections.defaultdict(list)
    for page in pages:
        sha, metric, day, value = read(page)
        if sha and metric and day:
            series[(sha, metric)].append((day, value, page))

    doomed: list[dict] = []
    exact_dupes = 0
    for (_sha, _metric), rows in series.items():
        rows.sort(key=lambda r: r[0])
        for i in range(1, len(rows)):
            day, value, page = rows[i]
            prev_day, prev_value, _prev = rows[i - 1]
            if day == prev_day:
                # Two rows for one day: a straight duplicate, whatever the
                # value.  The earlier-created one already survived a dedupe.
                exact_dupes += 1
                doomed.append(page)
                continue
            gap = days_between(prev_day, day)
            if value is not None and value == prev_value and \
                    gap is not None and gap <= args.within:
                doomed.append(page)

    log(f"{len(series)} series; {len(doomed)} row(s) to archive "
        f"({exact_dupes} same-day duplicates, "
        f"{len(doomed) - exact_dupes} re-stamped repeats)")
    if args.dry_run or not doomed:
        for page in doomed[:6]:
            log(f"  would archive {read(page)}")
        return 0

    archived = failed = 0
    for i, page in enumerate(doomed, 1):
        try:
            api.request("PATCH", f"/pages/{page['id']}", {"archived": True})
            archived += 1
        except NotionError as exc:
            failed += 1
            log(f"  {exc}")
        if i % 200 == 0 or i == len(doomed):
            log(f"[{i}/{len(doomed)}] {archived} archived, {failed} failed")
    log(f"done: {archived} archived, {failed} failed")
    return 1 if failed and not archived else 0


if __name__ == "__main__":
    raise SystemExit(main())
