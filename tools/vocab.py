"""The vocabulary the corpus actually speaks, and how big each cohort is.

    python tools/vocab.py <corpus root>
    python tools/vocab.py <corpus root> --min 8 --tags

Two uses, both practical.

Filling in a `declared.json` for your own mix, the question is "what do I put
in `cohort.genre`".  The answer is not what the track sounds like to you, it
is which label has enough released records behind it for a percentile to mean
anything.  Declaring `future house` when the corpus holds four of them buys a
comparison against four records; declaring `house` buys forty.

Reading a claim about the corpus, the question is "how many records is that
actually over".  A median PSR for a cohort of six is a number about six
records, and the count belongs next to it every time.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

# Below this a cohort statistic is a statement about a handful of records and
# should be read as one.
USABLE = 12


def log(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


def collect(root: str) -> dict:
    genres: collections.Counter = collections.Counter()
    umbrellas: collections.Counter = collections.Counter()
    tags: collections.Counter = collections.Counter()
    years: collections.Counter = collections.Counter()
    pairs: collections.Counter = collections.Counter()
    tracks = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith(".") and d != "stems")
        if "analysis.json" not in filenames:
            continue
        dirnames[:] = []
        tracks += 1
        path = os.path.join(dirpath, "online.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        voted = doc.get("genres") or {}
        primary, umbrella = voted.get("primary"), voted.get("umbrella")
        if primary:
            genres[primary] += 1
        if umbrella:
            umbrellas[umbrella] += 1
        for entry in voted.get("ranked") or []:
            if isinstance(entry, dict) and entry.get("name"):
                # Every genre that got a vote, not only the winner: a track can
                # legitimately sit in two cohorts.
                pairs[entry["name"]] += 1
        for tag in doc.get("descriptive_tags") or []:
            tags[str(tag)] += 1
        year = str(((doc.get("cross_checks") or {})
                    .get("release_date") or {}).get("earliest") or "")[:4]
        if year.isdigit():
            years[int(year)] += 1
    return {"tracks": tracks, "primary": genres, "umbrella": umbrellas,
            "any_vote": pairs, "tags": tags, "years": years}


def table(title: str, counter: collections.Counter, total: int,
          limit: int, floor: int) -> list[str]:
    rows = [f"", f"{title}  ({len(counter)} distinct)"]
    rows.append(f"  {'value':38s} {'tracks':>7s}  {'share':>6s}   usable")
    for name, count in counter.most_common(limit):
        share = 100.0 * count / total if total else 0.0
        mark = "yes" if count >= floor else "thin"
        rows.append(f"  {str(name)[:38]:38s} {count:7d}  {share:5.1f}%   {mark}")
    if len(counter) > limit:
        rest = sum(c for _n, c in counter.most_common()[limit:])
        rows.append(f"  {'(' + str(len(counter) - limit) + ' more)':38s} "
                    f"{rest:7d}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument("--min", type=int, default=USABLE,
                    help=f"tracks a cohort needs to be called usable "
                         f"(default {USABLE})")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--tags", action="store_true",
                    help="also list the descriptive tag vocabulary")
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args()

    got = collect(args.root)
    total = got["tracks"]
    log(f"{total} track(s) under {args.root}")
    for line in table("umbrella genres (broad cohorts)", got["umbrella"],
                      total, args.limit, args.min):
        log(line)
    for line in table("primary genres (specific cohorts)", got["primary"],
                      total, args.limit, args.min):
        log(line)
    for line in table("any genre that got a vote", got["any_vote"],
                      total, args.limit, args.min):
        log(line)
    if args.tags:
        for line in table("descriptive tags", got["tags"], total,
                          args.limit, args.min):
            log(line)

    years = got["years"]
    if years:
        log("")
        log(f"years  {min(years)}-{max(years)}, "
            f"{sum(c for y, c in years.items() if y >= max(years) - 5)} "
            f"track(s) in the last six")
        decade: collections.Counter = collections.Counter()
        for year, count in years.items():
            decade[year - year % 10] += count
        log("  " + "   ".join(f"{d}s:{c}" for d, c in sorted(decade.items())))

    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({k: (dict(v) if isinstance(v, collections.Counter) else v)
                       for k, v in got.items()}, fh, indent=1,
                      sort_keys=True, ensure_ascii=False)
            fh.write("\n")
        log(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
