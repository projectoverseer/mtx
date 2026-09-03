"""Resolve one canonical name and MBID per catalogue folder.

    python tools/identity.py <corpus root>          # writes <root>/artists.json

`mtx scan` mirrors the library tree, so the top-level folder is exactly one
name per artist and is therefore the right thing to *group* on.  It is the
wrong thing to *display*, because it is whatever the person who bought the
files typed:

    Red Hot Chilli Peppers   ->  Red Hot Chili Peppers
    Stepen Sanchez           ->  Stephen Sanchez
    TIESTO                   ->  Tiësto
    Fred again               ->  Fred again..
    Tyler, The Creator       ->  Tyler, The Creator   (but Notion rejects the
                                 comma, so the column has to hold a variant)

A misspelt folder is not only ugly.  It is a join key that fails: nothing
outside this machine knows an artist called "Red Hot Chilli Peppers", so no
chart table, no playlist export and no second corpus can ever be lined up
against it.

So the grouping stays on the folder and the *identity* is resolved separately,
by asking what MusicBrainz called the artist on each of that folder's tracks
and taking the majority.  One track can be a guest credit or a mis-entered row;
thirty tracks agreeing is the artist's name.  The MBID travels with it, which
is the key a machine should actually join on.

The result is a single file at the corpus root.  It is data, not a rename: no
folder is touched, `outcome.py` keeps normalising within the same groups, and
if the resolution is ever wrong the fix is to delete one line and re-run.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import os
import re
import sys
import unicodedata
from typing import Any

SCHEMA_VERSION = "1.0.0"
FILENAME = "artists.json"

# Below this share of a folder's tracks a name is a guest, not the artist.
MAJORITY = 0.5

# A folder with fewer tracks than this has too little evidence to overrule the
# name on disk: one soundtrack cue would be a majority of two.
MIN_TRACKS = 3


def log(msg: str) -> None:
    print(f"[identity] {msg}", file=sys.stderr, flush=True)


def squash(name: str) -> str:
    """`Fred again..` and `fred again` collapse to one key.

    Accents fold too, so `TIESTO` matches `Tiësto` -- which is the whole point:
    the folder is the accent-stripped, shift-locked version of the real name.
    """
    decomposed = unicodedata.normalize("NFKD", str(name or ""))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", stripped.lower())


def notion_safe(name: str) -> str:
    """A name Notion will store in a select without silently altering it.

    Notion rejects a comma in a select or multi-select option name.  That is a
    property of the destination, not of the artist, so the substitution is
    recorded next to the real name rather than replacing it.
    """
    return re.sub(r"\s*,\s*", "; ", str(name or "")).strip()


def scan(root: str) -> dict[str, list[dict[str, Any]]]:
    """`{folder: [track fact, ...]}` for every analysed folder under root."""
    by_folder: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith(".") and d != "stems")
        if "analysis.json" not in filenames:
            continue
        dirnames[:] = []
        rel = os.path.relpath(dirpath, root).replace("\\", "/").split("/")
        if not rel or rel[0] in (".", ".."):
            continue
        fact: dict[str, Any] = {"folder": dirpath}
        online_path = os.path.join(dirpath, "online.json")
        if os.path.isfile(online_path):
            try:
                with open(online_path, encoding="utf-8") as fh:
                    doc = json.load(fh)
            except (OSError, ValueError):
                doc = {}
            mb = doc.get("musicbrainz") or {}
            artists = [a for a in (mb.get("artists") or []) if isinstance(a, dict)]
            if artists:
                fact["mb_name"] = artists[0].get("name")
                fact["mb_mbid"] = artists[0].get("mbid")
            fact["score"] = ((mb.get("match") or {}).get("score"))
        by_folder[rel[0]].append(fact)
    return by_folder


def resolve_one(folder: str, facts: list[dict[str, Any]]) -> dict[str, Any]:
    """The canonical name and MBID for one catalogue folder."""
    out: dict[str, Any] = {
        "folder": folder,
        "name": folder,
        "notion_name": notion_safe(folder),
        "mbid": None,
        "source": "folder",
        "tracks": len(facts),
        "votes": 0,
        "agreement": None,
        "renamed": False,
    }
    votes = collections.Counter()
    mbids: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for fact in facts:
        name = str(fact.get("mb_name") or "").strip()
        if not name:
            continue
        votes[name] += 1
        if fact.get("mb_mbid"):
            mbids[name][fact["mb_mbid"]] += 1
    if not votes:
        return out

    # The lead credit on most of the folder's tracks.  Ties break on the name
    # closest to the folder, so a duo folder does not flip between runs.
    target = squash(folder)
    top, count = max(votes.items(),
                     key=lambda kv: (kv[1], squash(kv[0]) == target, -len(kv[0])))
    out["votes"] = count
    out["agreement"] = round(count / len(facts), 4)
    if mbids[top]:
        out["mbid"] = mbids[top].most_common(1)[0][0]

    same = squash(top) == target
    # A one-track folder cannot vote, but "Stepen Sanchez" against "Stephen
    # Sanchez" needs no vote: one dropped letter is a typo, not a different
    # person.  Anything this close is the same name spelt badly.
    near = difflib.SequenceMatcher(None, squash(top), target).ratio() >= 0.88
    enough = count >= MIN_TRACKS and count / len(facts) >= MAJORITY
    out["folder_similarity"] = round(
        difflib.SequenceMatcher(None, squash(top), target).ratio(), 4)
    if same or near or enough:
        # `same` covers the case this exists for: the folder is the right
        # artist spelt wrong, and the database has the spelling.  `enough`
        # covers a folder named after a project or a misspelling far enough
        # from the real name to not squash equal.
        out["name"] = top
        out["notion_name"] = notion_safe(top)
        out["source"] = "musicbrainz"
        out["renamed"] = top != folder
    else:
        out["note"] = (f"kept the folder name: the most common credit {top!r} "
                       f"covers {count}/{len(facts)} track(s)")
    return out


def build(root: str) -> dict[str, Any]:
    by_folder = scan(root)
    artists = {folder: resolve_one(folder, facts)
               for folder, facts in sorted(by_folder.items())}

    # Two folders that resolve to one artist would silently merge two
    # catalogues in every within-artist comparison.  Report, never merge.
    collisions: dict[str, list[str]] = collections.defaultdict(list)
    for folder, entry in artists.items():
        collisions[squash(entry["name"])].append(folder)
    duplicates = {k: v for k, v in collisions.items() if len(v) > 1}

    return {
        "schema_version": SCHEMA_VERSION,
        "root": os.path.abspath(root),
        "artists": artists,
        "folders": len(artists),
        "renamed": sum(1 for a in artists.values() if a["renamed"]),
        "unresolved": sorted(f for f, a in artists.items()
                             if a["source"] == "folder"),
        "collisions": duplicates,
    }


def load(root: str) -> dict[str, dict[str, Any]]:
    """`{folder: entry}`, or empty when identity has not been resolved yet."""
    path = os.path.join(root, FILENAME)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("artists") or {}
    except (OSError, ValueError):
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", help="the corpus root that `mtx scan` wrote")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    doc = build(args.root)
    log(f"{doc['folders']} catalogue folder(s), {doc['renamed']} renamed by "
        f"MusicBrainz, {len(doc['unresolved'])} unresolved")
    for folder, entry in sorted(doc["artists"].items()):
        if entry["renamed"]:
            log(f"  {folder!r} -> {entry['name']!r} "
                f"({entry['votes']}/{entry['tracks']} tracks, {entry['mbid']})")
    for folder in doc["unresolved"]:
        log(f"  unresolved: {folder!r} "
            f"-- {doc['artists'][folder].get('note') or 'no MusicBrainz match'}")
    for key, folders in doc["collisions"].items():
        log(f"  collision: {folders} all resolve to one artist")

    if args.dry_run:
        return 0
    path = os.path.join(args.root, FILENAME)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    log(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
