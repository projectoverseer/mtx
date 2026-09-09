"""Write a `declared.json` template beside a track, or beside your own mix.

    python tools/declare.py "E:/Music/_mtx_out/Me/Demo/01. New Song"
    python tools/declare.py "E:/mixes/new song.flac" --out "E:/mtx_out/new song"
    python tools/declare.py <corpus root> --gaps       # where it would help most

`mtx enrich` answers "what is this record" by asking databases.  That only
works for records that have been released: a distributor issued the ISRC, an
editor typed the credits in, listeners tagged it.  Your own unreleased mix has
none of that and cannot be made to have it, so for your own work the facts are
not missing -- you are simply the source of them rather than a database.

That is the whole job of this sidecar, and it is why it matters more than it
looks.  Without it an unreleased master reaches `mtx cohort` with no genre and
no year, so it belongs to no cohort, so there is nothing to compare it to --
which is exactly the question you wanted answered.  Two fields, `cohort.genre`
and `cohort.year`, are the difference between "your PSR is 4.7 dB" and "your
PSR is 4.7 dB, which is the 12th percentile for club house since 2022".

A declared value never overwrites a measured one.  It travels with
`source: "declared"` and sits beside whatever the analysis found, so the two
can disagree in the open.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

TEMPLATE: dict = {
    "_note": (
        "Facts you are the source of.  Delete every key you cannot answer -- "
        "an empty string is a claim that the value is empty, a missing key is "
        "a claim that you did not say.  Nothing here overwrites a measurement."
    ),
    "_required_for_cohorts": (
        "cohort.genre and cohort.year.  Without them this track joins no "
        "cohort and `mtx cohort` can give it no percentile."
    ),
    "cohort": {
        "genre": "",
        "_genre_help": (
            "what this should be compared against, in the corpus's own "
            "vocabulary: run `python tools/vocab.py <root>` to see it.  Be "
            "specific -- 'house' is a better cohort than 'electronic'."
        ),
        "year": None,
        "_year_help": "the year you are competing in, normally this one.",
    },
    "title": "",
    "artist": "",
    "featured_artists": [],
    "version": "",
    "_version_help": (
        "radio_edit, extended, album_version, demo, instrumental, remix... "
        "two bounces of one song are two mixes, not two songs, and this is "
        "what says so."
    ),
    "work_key": "",
    "_work_key_help": (
        "the same string on every version of one song, so `mtx compare` and "
        "the corpus can tell a v3 mix from a different track."
    ),
    "sibling_versions": [],
    "release_date": "",
    "recording_date": "",
    "writers": [],
    "producers": [],
    "engineers": [],
    "performers": [],
    "publisher": "",
    "pro": "",
    "label": "",
    "isrc": "",
    "iswc": "",
    "upc": "",
    "explicit": None,
    "lyrics": "",
    "_lyrics_help": (
        "paste the sheet.  A declared lyric is exact; a transcript is an "
        "inference that mishears, and the two never merge."
    ),
    "lyrics_language": "",
    "samples": [],
    "interpolations": [],
    "origin": "",
    "notes": "",
    "outcome": {
        "_help": (
            "How the record actually did.  Nothing measures this and no free "
            "database carries it, so these are yours to supply -- and they "
            "are the only columns in the corpus that say whether a record "
            "worked, as opposed to what it sounds like.  A Last.fm playcount "
            "is scrobbling listeners, which is not the same question."
        ),
        "billboard_peak": None,
        "_billboard_peak_help": "highest chart position reached, 1 is best.",
        "weeks_on_chart": None,
        "certification": "",
        "_certification_help": "gold, platinum, 2x platinum, diamond...",
        "chart": "",
        "_chart_help": (
            "which chart the peak is from -- 'Billboard Hot 100', 'UK "
            "Singles', 'Vietnam Hot 100'.  A peak with no chart named is a "
            "number nobody can check."
        ),
    },
}


def log(msg: str) -> None:
    print(f"[declare] {msg}", file=sys.stderr, flush=True)


def target_folder(path: str, out: str | None) -> str:
    if out:
        return out
    if os.path.isdir(path):
        return path
    return os.path.dirname(os.path.abspath(path)) or "."


def prefill(folder: str) -> dict:
    """Start from what the file already says, so there is less to type."""
    doc = dict(TEMPLATE)
    doc["cohort"] = dict(TEMPLATE["cohort"])
    doc["outcome"] = dict(TEMPLATE["outcome"])
    analysis = os.path.join(folder, "analysis.json")
    if not os.path.isfile(analysis):
        return doc
    try:
        with open(analysis, encoding="utf-8") as fh:
            res = json.load(fh)
    except (OSError, ValueError):
        return doc
    named = (res.get("tags") or {}).get("named") or {}
    for key, tag in (("title", "title"), ("artist", "artist"),
                     ("label", "organization"), ("isrc", "isrc")):
        if named.get(tag):
            doc[key] = str(named[tag])
    if named.get("date"):
        doc["cohort"]["year"] = str(named["date"])[:4]
        doc["release_date"] = str(named["date"])
    if named.get("genre"):
        # The shop's own word, as a starting point and clearly not a vote.
        doc["cohort"]["genre"] = str(named["genre"]).lower()
    return doc


def gaps(root: str) -> int:
    """Where a sidecar would buy the most: tracks with no cohort at all."""
    missing = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith(".") and d != "stems")
        if "analysis.json" not in filenames:
            continue
        dirnames[:] = []
        if "declared.json" in filenames:
            continue
        online: dict = {}
        path = os.path.join(dirpath, "online.json")
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    online = json.load(fh)
            except (OSError, ValueError):
                online = {}
        genre = ((online.get("genres") or {}).get("umbrella")
                 or (online.get("genres") or {}).get("primary"))
        year = ((online.get("cross_checks") or {})
                .get("release_date") or {}).get("earliest")
        if not genre or not year:
            missing.append((os.path.relpath(dirpath, root),
                            "genre" if not genre else "",
                            "year" if not year else ""))
    log(f"{len(missing)} track(s) would join no cohort without a declared.json")
    for rel, g, y in missing[:40]:
        log(f"  {rel}  (missing {', '.join(filter(None, (g, y)))})")
    if len(missing) > 40:
        log(f"  ... {len(missing) - 40} more")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="an analysed folder, an audio file, or a "
                                 "corpus root with --gaps")
    ap.add_argument("--out", help="write the sidecar here instead")
    ap.add_argument("--gaps", action="store_true",
                    help="list the tracks a sidecar would help most")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing declared.json")
    args = ap.parse_args()

    if args.gaps:
        return gaps(args.path)

    folder = target_folder(args.path, args.out)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "declared.json")
    if os.path.exists(path) and not args.force:
        log(f"{path} already exists; pass --force to overwrite it")
        return 1
    doc = prefill(folder)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    log(f"wrote {path}")
    log("fill in cohort.genre and cohort.year first; they are the two that "
        "decide whether this track can be compared to anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
