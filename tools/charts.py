"""Load chart outcomes from a CSV into the corpus, one command for the lot.

    python tools/charts.py "E:/Music/_mtx_out" peaks.csv
    python tools/charts.py "E:/Music/_mtx_out" peaks.csv --dry-run
    python tools/charts.py "E:/Music/_mtx_out" --template peaks.csv

**Why this file exists.** The corpus can say what a record sounds like in
extraordinary detail and almost nothing about whether it worked. The one
outcome column it has is a Last.fm playcount, which counts scrobbling
listeners -- a proxy for a particular kind of enthusiast, not for commercial
success, and one that is biased by era, genre and platform in ways nothing
here corrects for. A 1998 house record and a 2024 pop single are not
comparable on it.

A chart peak is the closest thing to a hard outcome that exists, and no free
database carries it: MusicBrainz does not model charts, Discogs is about
pressings, Last.fm is about listening. So it has to be supplied, and the only
question is whether supplying it costs one file or 1,321.

**The CSV.** A header row, then one row per track. Match on any of `sha256`,
`isrc`, or `artist` + `title` -- whichever columns are present, in that order,
because a sha256 is exact and a title is a guess.

    artist,title,billboard_peak,weeks_on_chart,certification,chart
    Adele,Rolling in the Deep,1,65,9x platinum,Billboard Hot 100

Anything already in a track's `declared.json` is left alone unless
`--overwrite` is given: a hand-written sidecar is somebody's work, and a bulk
load should never quietly outrank it.

Writes `declared.json` beside each matched analysis, which is where
`tools/notion/schema.py` already reads `declared.outcome.*` from. Nothing
measured is touched.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from env import load_env                          # noqa: E402

FIELDS = ("billboard_peak", "weeks_on_chart", "certification", "chart")
NUMERIC = ("billboard_peak", "weeks_on_chart")

TEMPLATE_CSV = """artist,title,isrc,sha256,billboard_peak,weeks_on_chart,certification,chart
Adele,Rolling in the Deep,,,1,65,9x platinum,Billboard Hot 100
Daft Punk,Get Lucky,,,2,42,5x platinum,Billboard Hot 100
"""


def log(msg: str) -> None:
    print(f"[charts] {msg}", file=sys.stderr, flush=True)


def squash(text) -> str:
    import re
    import unicodedata
    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", stripped.lower())


def index_corpus(root: str) -> dict[str, list[str]]:
    """Three ways to find a folder, best first: sha256, ISRC, artist+title."""
    from mtx.online.match import primary_artist         # noqa: PLC0415
    from mtx.split import load_analysis                 # noqa: PLC0415

    keys: dict[str, list[str]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith(".") and d != "stems")
        if "analysis.json" not in filenames:
            continue
        dirnames[:] = []
        try:
            doc = load_analysis(os.path.join(dirpath, "analysis.json"),
                                want=["file", "tags"])
        except (OSError, ValueError):
            continue
        named = (doc.get("tags") or {}).get("named") or {}
        sha = (doc.get("file") or {}).get("sha256")
        online = os.path.join(dirpath, "online.json")
        isrc = None
        if os.path.isfile(online):
            try:
                with open(online, encoding="utf-8") as fh:
                    o = json.load(fh)
                o = o.get("online") or o
                isrc = ((o.get("identity") or {}).get("isrc"))
            except (OSError, ValueError):
                pass
        # Indexed under the credited artist *and* the lead artist alone.  A
        # chart lists "Daft Punk"; the file is tagged "Daft Punk ft. Pharrell
        # Williams", and without the second key the prefix match below never
        # gets as far as comparing titles.
        artist_keys = set()
        if named.get("artist"):
            artist_keys.add(squash(named["artist"]))
            lead = primary_artist(str(named["artist"]))
            if lead:
                artist_keys.add(squash(lead))
        title = squash(named.get("title")) if named.get("title") else ""
        entries = [f"sha:{sha}" if sha else None,
                   f"isrc:{squash(isrc)}" if isrc else None]
        entries += [f"at:{a}|{title}" for a in artist_keys if a and title]
        for key in entries:
            if key:
                keys.setdefault(key, []).append(dirpath)
    return keys


def lookup(index: dict[str, list[str]], row: dict) -> tuple[list[str], str]:
    for key, how in ((f"sha:{row.get('sha256', '').strip()}", "sha256"),
                     (f"isrc:{squash(row.get('isrc'))}", "isrc"),
                     (f"at:{squash(row.get('artist'))}"
                      f"|{squash(row.get('title'))}", "artist+title")):
        if key.split(":", 1)[1].strip("|") and key in index:
            return index[key], how

    # A chart lists "Get Lucky"; the library has "Get Lucky (Radio Edit -
    # feat. Pharrell Williams and Nile Rodgers)".  Same song.  Accepted only
    # within one artist and only when exactly one title starts with the given
    # one -- two candidates means the row is ambiguous, and a chart peak
    # attached to the wrong record is worse than one left off.
    artist, title = squash(row.get("artist")), squash(row.get("title"))
    if not (artist and title):
        return [], "no match"
    prefix = f"at:{artist}|{title}"
    # Counted in folders, not in keys: one folder is indexed under both its
    # credited artist and its lead artist, so counting keys called a single
    # unambiguous track ambiguous with itself.
    found = {f for k in index if k.startswith(prefix) for f in index[k]}
    if len(found) == 1:
        return sorted(found), "artist+title prefix"
    if len(found) > 1:
        return [], f"ambiguous: {len(found)} folders start with {title!r}"
    return [], "no match"


def apply_row(folder: str, row: dict, overwrite: bool) -> str:
    path = os.path.join(folder, "declared.json")
    doc: dict = {}
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            return "unreadable declared.json"
    outcome = doc.setdefault("outcome", {})
    if not isinstance(outcome, dict):
        return "declared.json has a non-object outcome"

    wrote = []
    for field in FIELDS:
        raw = (row.get(field) or "").strip()
        if not raw:
            continue
        if outcome.get(field) not in (None, "", []) and not overwrite:
            continue
        if field in NUMERIC:
            try:
                outcome[field] = int(float(raw))
            except ValueError:
                return f"{field}={raw!r} is not a number"
        else:
            outcome[field] = raw
        wrote.append(field)
    if not wrote:
        return "nothing new"
    doc["outcome"] = outcome
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)
    return "wrote " + ", ".join(wrote)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument("csv_path", nargs="?")
    ap.add_argument("--template", metavar="PATH",
                    help="write a starter CSV here and exit")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace outcome values already in a declared.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    load_env(args.root)

    if args.template:
        with open(args.template, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(TEMPLATE_CSV)
        log(f"wrote {args.template}")
        log("fill it in, delete the examples, then run this again with it")
        return 0
    if not args.csv_path:
        ap.error("give a CSV, or --template to start one")

    with open(args.csv_path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        log(f"{args.csv_path} has no rows")
        return 1
    unknown = set(rows[0]) - set(FIELDS) - {"artist", "title", "isrc", "sha256"}
    if unknown:
        log(f"ignoring unrecognised column(s): {', '.join(sorted(unknown))}")

    log(f"{len(rows)} row(s); indexing the corpus")
    index = index_corpus(args.root)
    log(f"{len(index)} lookup key(s) over the corpus")

    counts = {"matched": 0, "ambiguous": 0, "missed": 0, "wrote": 0}
    for row in rows:
        folders, how = lookup(index, row)
        who = f"{row.get('artist', '?')} - {row.get('title', '?')}"
        if not folders:
            counts["missed"] += 1
            log(f"  {how}: {who}")
            continue
        if len(folders) > 1:
            # Two folders for one row is the duplicate-recording case; both
            # are the same record and both should carry the outcome.
            counts["ambiguous"] += 1
        counts["matched"] += 1
        for folder in folders:
            if args.dry_run:
                log(f"  would set {who} ({how}) -> "
                    f"{os.path.relpath(folder, args.root)}")
                continue
            result = apply_row(folder, row, args.overwrite)
            if result.startswith("wrote"):
                counts["wrote"] += 1
            elif result not in ("nothing new",):
                log(f"  {who}: {result}")
    log(f"done: {counts['matched']} matched "
        f"({counts['ambiguous']} to more than one folder), "
        f"{counts['missed']} unmatched, {counts['wrote']} sidecar(s) written")
    if counts["wrote"]:
        log("run `python tools/pipeline.py --from identity` to publish them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
