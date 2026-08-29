"""Last.fm: listener tags and real play counts.  Optional, needs a free key.

Where MusicBrainz genres are edited by a few hundred contributors, Last.fm
tags are applied by millions of listeners, so the two disagree in useful ways:
MusicBrainz will call a record `alternative pop`, Last.fm will also call it
`sad`, `nocturnal` and `2019`.  The genre vote takes the former; the mood and
era words land in a separate tag list.

`listeners` and `playcount` are the only hard popularity numbers any of these
providers expose -- Deezer's `rank` is a rescaled internal score, Apple gives
none -- which makes them the honest axis for "did this record actually land".

Set LASTFM_API_KEY to enable.  Skipped silently when unset.
"""

from __future__ import annotations

import os
from typing import Any

from .http import Client, build_url

BASE = "http://ws.audioscrobbler.com/2.0/"


def api_key() -> str:
    return os.environ.get("LASTFM_API_KEY", "").strip()


def _tags(node: dict[str, Any]) -> list[dict[str, Any]]:
    raw = (node.get("toptags") or {}).get("tag") or []
    if isinstance(raw, dict):
        raw = [raw]
    out = []
    for t in raw:
        if isinstance(t, dict) and t.get("name"):
            try:
                count = float(t.get("count") or 0)
            except (TypeError, ValueError):
                count = 0.0
            out.append({"name": t["name"], "count": count})
    return out


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def lookup(client: Client, local: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"available": False, "errors": [], "requests": 0}
    key = api_key()
    if not key:
        result["errors"].append("LASTFM_API_KEY not set")
        return result
    artist, title = local.get("artist") or "", local.get("title") or ""
    if not (artist and title):
        result["errors"].append("need both artist and title")
        return result

    url = build_url(BASE, method="track.getInfo", api_key=key, artist=artist,
                    track=title, autocorrect=1, format="json")
    body, err = client.get_json(url)
    result["requests"] += 1
    if err:
        result["errors"].append(f"track.getInfo: {err}")
    track = (body or {}).get("track") or {}
    if track:
        result["available"] = True
        result["track"] = {
            "name": track.get("name"),
            "artist": (track.get("artist") or {}).get("name"),
            "album": (track.get("album") or {}).get("title"),
            "listeners": _int(track.get("listeners")),
            "playcount": _int(track.get("playcount")),
            "duration_s": (_int(track.get("duration")) or 0) / 1000.0 or None,
            "url": track.get("url"),
            "mbid": track.get("mbid") or None,
        }
        result["tags_track"] = _tags(track)
        wiki = (track.get("wiki") or {}).get("summary")
        if wiki:
            result["wiki_summary"] = wiki
    else:
        result["errors"].append((body or {}).get("message") or "track not found")

    url = build_url(BASE, method="artist.getInfo", api_key=key, artist=artist,
                    autocorrect=1, format="json")
    body, err = client.get_json(url)
    result["requests"] += 1
    if err:
        result["errors"].append(f"artist.getInfo: {err}")
    art = (body or {}).get("artist") or {}
    if art:
        result["artist"] = {
            "name": art.get("name"),
            "listeners": _int((art.get("stats") or {}).get("listeners")),
            "playcount": _int((art.get("stats") or {}).get("playcount")),
            "url": art.get("url"),
        }
        result["tags_artist"] = _tags(art)
        result["similar_artists"] = [
            a.get("name") for a in ((art.get("similar") or {}).get("artist") or [])
            if isinstance(a, dict) and a.get("name")][:10]
    return result
