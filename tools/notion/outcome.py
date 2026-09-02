"""Derive an outcome variable the corpus can actually learn from.

    python tools/notion/outcome.py E:\\Music\\_mtx_out

Writes `outcome.json` at the root of the analysed tree, keyed by track sha256.
`push.py` reads it and adds the derived columns; nothing here touches
`analysis.json`, and nothing here is a measurement.

**Why raw playcount is not the answer.**  Drake's worst album track outstreams
almost any independent hit.  Sort a corpus of 55 famous artists by playcount
and you have ranked artist fame, then catalogue age -- never songwriting or
mixing.  Every pattern found against that target is a pattern about who is in
the corpus.

**What this computes instead** is each track's playcount against *the same
artist's other tracks*.  That holds fame, budget, label push, era, producer
and mastering engineer roughly constant, so what is left varies with the
record.  The natural experiment is already on disk: about 1,100 of these 1,321
tracks are album cuts by artists whose singles are here too.

**Everything here is a snapshot with a date on it.**  Playcount is a
current-value figure with no history -- Heat Waves took 59 weeks to reach
number one, and no API will tell you what it read in month six.  So every
derived number carries `observed_at` and the window it was computed over, and
re-running after a later `mtx enrich --refresh` produces a *new* set rather
than a correction to this one.  The Observations log keeps them all.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))

from mtx.cli import _enrich_targets              # noqa: E402

# Below this many tracks by one artist, a within-artist position is noise: the
# median is being set by two or three records.  Reported as null with a reason
# rather than as a number that looks like the others.
MIN_TRACKS_FOR_Z = 5
MIN_TRACKS_FOR_TERCILE = 6

OUTCOME_VERSION = "1.0.0"


def log(msg: str) -> None:
    try:
        print(f"[outcome] {msg}", file=sys.stderr, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stderr, "encoding", None) or "ascii"
        print(f"[outcome] {msg}".encode(enc, "replace").decode(enc, "replace"),
              file=sys.stderr, flush=True)


def _read(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def catalogue_artist(root: str, folder: str, row: dict) -> str:
    """The artist whose catalogue this track belongs to.

    Deliberately the top-level folder under the scan root rather than the
    track's artist tag.  `mtx scan` mirrors the library, so that folder is the
    album artist, while the tag carries features: "Calvin Harris", "Calvin
    Harris / Dua Lipa" and "Calvin Harris feat. Rag'n'Bone Man" are one
    catalogue and three tags.  Grouping on the tag turned 55 artists into 264
    and left most of them with too few tracks to position anything against --
    which is exactly the comparison this file exists to make.
    """
    rel = os.path.relpath(os.path.abspath(folder), os.path.abspath(root))
    head = rel.replace("\\", "/").split("/")[0]
    if head and head not in (".", ".."):
        return head
    return str(row.get("Artist") or "")


def collect(root: str) -> list[dict]:
    """One light record per analysed folder.

    Reads `corpus_row.json` (about 1 KB) rather than `analysis.json` (about
    3 MB): artist, title and sha256 are all it needs, and the difference over
    1,321 folders is seconds against minutes.
    """
    rows = []
    for folder in _enrich_targets(root):
        row = _read(os.path.join(folder, "corpus_row.json"))
        online = _read(os.path.join(folder, "online.json"))
        if not row:
            continue
        sha = ((row.get("_source") or {}).get("sha256") or "")
        pop = online.get("popularity") or {}
        rg = ((online.get("musicbrainz") or {}).get("release_group") or {})
        rows.append({
            "sha256": sha,
            "recording_mbid": (online.get("identity") or {}).get("recording_mbid"),
            "isrc": (online.get("identity") or {}).get("isrc"),
            "folder": folder,
            "artist": catalogue_artist(root, folder, row),
            "credited_artist": row.get("Artist") or "",
            "title": row.get("Title") or "",
            "playcount": pop.get("lastfm_playcount"),
            "listeners": pop.get("lastfm_listeners"),
            "deezer_rank": pop.get("deezer_rank"),
            "release_type": rg.get("primary_type"),
            "duration_s": (((online.get("cross_checks") or {}).get("duration")
                            or {}).get("local_s")),
            "observed_at": online.get("queried_utc"),
        })
    return rows


def _log10(value) -> float | None:
    """Playcounts span four orders of magnitude, so compare them in logs.

    On a linear scale one runaway hit sets the artist's mean and every other
    track reads as a failure by the same amount.
    """
    if not isinstance(value, (int, float)) or value <= 0:
        return None
    return math.log10(float(value))


def _z(value: float, values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    sd = statistics.pstdev(values)
    if sd <= 0:
        return None
    return round((value - statistics.fmean(values)) / sd, 4)


def _percentile_rank(value: float, values: list[float]) -> float:
    below = sum(1 for v in values if v < value)
    equal = sum(1 for v in values if v == value)
    return round(100.0 * (below + 0.5 * equal) / len(values), 2)


def mark_duplicates(rows: list[dict]) -> int:
    """Flag rows that are the same recording as another row.

    Eight recordings appear twice in this corpus -- a single and its album,
    usually -- with different sha256s, because they are different masters of
    one performance.  Worth keeping: two masters of the same recording is the
    only A/B in the library where the song is held constant.  But they are one
    recording, and counted twice they double-vote in every percentile and in
    the artist's own median.

    `recording_duplicates` is how many rows share the recording, so a query
    can ask for 1 and get a deduplicated corpus. `recording_primary` marks
    one row per group -- the longest, which is the album cut rather than a
    radio edit -- so the other queries can keep exactly one.
    """
    groups: dict[str, list[dict]] = {}
    for r in rows:
        key = r.get("recording_mbid") or r.get("isrc")
        if key:
            groups.setdefault(key, []).append(r)
    dupes = 0
    for key, members in groups.items():
        n = len(members)
        if n > 1:
            dupes += n
        # Longest first: the album cut over the radio edit.  Ties break on
        # sha256 so the choice is stable between runs.
        members.sort(key=lambda r: (-(r.get("duration_s") or 0), r["sha256"]))
        for i, r in enumerate(members):
            r["recording_duplicates"] = n
            r["recording_primary"] = (i == 0)
            r["recording_key"] = key
    for r in rows:
        r.setdefault("recording_duplicates", 1)
        r.setdefault("recording_primary", True)
    return dupes


def derive(rows: list[dict]) -> dict:
    mark_duplicates(rows)
    by_artist: dict[str, list[dict]] = {}
    for r in rows:
        by_artist.setdefault(r["artist"], []).append(r)

    corpus_logs = [x for x in (_log10(r["playcount"]) for r in rows) if x is not None]
    out: dict[str, dict] = {}
    covered = 0

    for artist, tracks in by_artist.items():
        logs = {r["sha256"]: _log10(r["playcount"]) for r in tracks}
        usable = [v for v in logs.values() if v is not None]
        median = statistics.median(usable) if usable else None

        terciles = None
        if len(usable) >= MIN_TRACKS_FOR_TERCILE:
            ordered = sorted(usable)
            n = len(ordered)
            terciles = (ordered[n // 3], ordered[2 * n // 3])

        for r in tracks:
            value = logs[r["sha256"]]
            entry: dict = {
                "artist": artist,
                "recording_duplicates": r.get("recording_duplicates", 1),
                "recording_primary": r.get("recording_primary", True),
                "recording_key": r.get("recording_key"),
                "artist_track_count": len(tracks),
                "artist_tracks_with_playcount": len(usable),
                "playcount": r["playcount"],
                "observed_at": r["observed_at"],
                "release_type": r["release_type"],
                "is_single": (None if not r["release_type"]
                              else r["release_type"].lower() == "single"),
            }
            if value is None:
                entry["reason"] = ("no playcount for this track; "
                                   "run mtx enrich with LASTFM_API_KEY")
            elif len(usable) < MIN_TRACKS_FOR_Z:
                entry["reason"] = (
                    f"only {len(usable)} track(s) by this artist carry a "
                    f"playcount; a within-artist position needs "
                    f"{MIN_TRACKS_FOR_Z}")
            else:
                covered += 1
                entry["playcount_z_within_artist"] = _z(value, usable)
                entry["playcount_vs_artist_median_db"] = round(
                    10.0 * (value - median), 2) if median is not None else None
                entry["percentile_within_artist"] = _percentile_rank(value, usable)
                if corpus_logs:
                    entry["percentile_in_corpus"] = _percentile_rank(value, corpus_logs)
                if terciles:
                    entry["outcome_tercile"] = (
                        "bottom" if value <= terciles[0]
                        else "top" if value > terciles[1] else "middle")
            out[r["sha256"]] = entry

    observed = sorted(r["observed_at"] for r in rows if r["observed_at"])
    return {
        "outcome_version": OUTCOME_VERSION,
        "definition":
            "playcount_z_within_artist is the track's log10 Last.fm playcount "
            "as a z-score against the same artist's other tracks in this "
            "corpus. It is a position among that artist's catalogue at the "
            "moment of the lookup, not a measure of quality and not a "
            "prediction. Raw playcount is dominated by artist fame and "
            "catalogue age; this is the cheapest way to hold both roughly "
            "constant.",
        "caveats": [
            "A snapshot. Playcount has no history: what a track read six "
            "months after release cannot be recovered, only captured going "
            "forward.",
            "Streaming reflects playlisting, sync and virality far more than "
            "it reflects mixing or songwriting. Treat a correlation here as a "
            "question worth asking, never as a cause.",
            "Corpus artists are famous. Within-artist position says nothing "
            "about how an unknown artist's record would perform.",
        ],
        "params": {"min_tracks_for_z": MIN_TRACKS_FOR_Z,
                   "min_tracks_for_tercile": MIN_TRACKS_FOR_TERCILE,
                   "scale": "log10"},
        "window": {"earliest_observation": observed[0] if observed else None,
                   "latest_observation": observed[-1] if observed else None},
        "coverage": {"tracks": len(rows), "with_z": covered,
                     "artists": len(by_artist)},
        "tracks": out,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--out", help="default <root>/outcome.json")
    args = ap.parse_args()

    rows = collect(args.root)
    if not rows:
        log(f"error: no corpus_row.json under {args.root}")
        return 1
    result = derive(rows)
    path = args.out or os.path.join(args.root, "outcome.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False)

    cov = result["coverage"]
    log(f"{cov['tracks']} tracks, {cov['artists']} artists, "
        f"{cov['with_z']} with a within-artist position")
    singles = sum(1 for t in result["tracks"].values() if t.get("is_single"))
    dupes = sum(1 for t in result["tracks"].values()
                if (t.get("recording_duplicates") or 1) > 1)
    log(f"{dupes} row(s) share a recording with another row "
        f"(filter recording_primary to deduplicate)")
    log(f"{singles} single(s), "
        f"{sum(1 for t in result['tracks'].values() if t.get('is_single') is False)} "
        f"album cut(s) -- the contrast set")
    log(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
