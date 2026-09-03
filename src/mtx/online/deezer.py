"""Deezer: exact ISRC addressing, plus a published tempo and a popularity rank.

The one provider here that resolves an ISRC to a single track in a single
request -- no candidate list, no ambiguity -- which makes it the cheapest way
to confirm that the file on disk is the commercial release it claims to be.

Two of its fields are worth more than the genre it returns:

  `bpm`   an independently published tempo.  mtx estimates tempo from an onset
          envelope and marks the estimate `low` confidence more often than not;
          a second opinion that agrees turns a soft number into a firm one, and
          one that disagrees by exactly a factor of two exposes the octave
          error beat trackers are prone to.
  `rank`  Deezer's popularity score.  A corpus assembled to learn what
          successful records do should be able to sort by how successful they
          actually were.

No key, no registration.
"""

from __future__ import annotations

from typing import Any

from . import match
from .http import Client, build_url

BASE = "https://api.deezer.com"


def _track_summary(t: dict[str, Any]) -> dict[str, Any]:
    dur = t.get("duration")
    return {
        "id": t.get("id"),
        "title": t.get("title"),
        "duration_s": float(dur) if isinstance(dur, (int, float)) else None,
        "artist": ", ".join(c.get("name", "") for c in (t.get("contributors") or [])
                            ) or (t.get("artist") or {}).get("name", ""),
    }


def lookup(client: Client, local: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"available": False, "errors": [], "requests": 0}
    isrc = (local.get("isrc") or "").strip().upper().replace("-", "")

    track = None
    by_isrc = False
    if isrc:
        body, err = client.get_json(f"{BASE}/track/isrc:{isrc}")
        result["requests"] += 1
        if body and not body.get("error") and body.get("id"):
            track, by_isrc = body, True
        elif err and "404" not in err:
            result["errors"].append(f"isrc: {err}")

    if track is None and local.get("title"):
        query = f'track:"{match.search_title(local["title"])}"'
        lead = match.primary_artist(local.get("artist", ""))
        if lead:
            query = f'artist:"{lead}" ' + query
        url = build_url(f"{BASE}/search", q=query, limit=10)
        body, err = client.get_json(url)
        result["requests"] += 1
        if err:
            result["errors"].append(f"search: {err}")
        cands = [_track_summary(t) for t in (body or {}).get("data") or []]
        winner, scored = match.best(local, cands, by_isrc=False, floor=0.6)
        result["candidates"] = scored[:5]
        if winner:
            body, err = client.get_json(f"{BASE}/track/{winner['id']}")
            result["requests"] += 1
            if body and not body.get("error"):
                track = body
            elif err:
                result["errors"].append(f"track: {err}")

    if not track:
        result["errors"].append("no match")
        return result

    dur = track.get("duration")
    remote = {"title": track.get("title"),
              "duration_s": float(dur) if isinstance(dur, (int, float)) else None,
              "artist": ", ".join(c.get("name", "")
                                  for c in (track.get("contributors") or []))}
    result["match"] = match.score_candidate(local, remote, by_isrc=by_isrc)
    result["available"] = True
    result["fetched_utc"] = client.last_fetched_utc

    bpm = track.get("bpm")
    result["track"] = {
        "id": track.get("id"),
        "title": track.get("title"),
        "duration_s": remote["duration_s"],
        # Deezer reports 0 for "not measured"; a real record is never 0 BPM.
        "bpm": float(bpm) if isinstance(bpm, (int, float)) and bpm > 0 else None,
        "gain_db": track.get("gain"),
        "rank": track.get("rank"),
        "explicit": track.get("explicit_lyrics"),
        "release_date": track.get("release_date"),
        "isrc": track.get("isrc"),
        "disk_number": track.get("disk_number"),
        "track_position": track.get("track_position"),
        "url": track.get("link"),
        "preview_url": track.get("preview"),
    }
    result["contributors"] = [{"name": c.get("name"), "role": c.get("role"),
                               "id": c.get("id")}
                              for c in (track.get("contributors") or [])]

    album_id = (track.get("album") or {}).get("id")
    if album_id:
        alb, err = client.get_json(f"{BASE}/album/{album_id}")
        result["requests"] += 1
        if err:
            result["errors"].append(f"album: {err}")
        elif alb and not alb.get("error"):
            result["album"] = {
                "id": alb.get("id"), "title": alb.get("title"),
                "label": alb.get("label"), "upc": alb.get("upc"),
                "track_count": alb.get("nb_tracks"),
                "release_date": alb.get("release_date"),
                "record_type": alb.get("record_type"),
                "rank": alb.get("fans"),
                "url": alb.get("link"),
            }
            result["genres"] = [{"name": g.get("name")} for g in
                                (alb.get("genres") or {}).get("data") or []
                                if g.get("name")]
    return result
