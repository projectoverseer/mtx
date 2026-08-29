"""Apple/iTunes Search: a third genre taxonomy and a firm release date.

Apple has no ISRC endpoint, so this is a search plus the same candidate
scoring the other providers use.  It earns its request for two reasons: its
genre vocabulary is independent of both MusicBrainz's community tags and
Deezer's shop categories, which makes agreement between them meaningful; and
`trackTimeMillis` is exact, so a match here is verifiable rather than assumed.

No key, no registration.
"""

from __future__ import annotations

from typing import Any

from . import match
from .http import Client, build_url

BASE = "https://itunes.apple.com/search"


def lookup(client: Client, local: dict[str, Any],
           country: str = "US") -> dict[str, Any]:
    result: dict[str, Any] = {"available": False, "errors": [], "requests": 0}
    title = (local.get("title") or "").strip()
    if not title:
        result["errors"].append("no title to search on")
        return result

    term = (f"{match.search_artist(local.get('artist', ''))} "
            f"{match.search_title(title)}").strip()
    url = build_url(BASE, term=term, media="music", entity="song",
                    limit=15, country=country)
    body, err = client.get_json(url)
    result["requests"] += 1
    if err:
        result["errors"].append(f"search: {err}")
        return result

    cands = []
    for row in (body or {}).get("results") or []:
        ms = row.get("trackTimeMillis")
        cands.append({
            "title": row.get("trackName"),
            "artist": row.get("artistName"),
            "duration_s": (ms / 1000.0) if isinstance(ms, (int, float)) else None,
            "_row": row,
        })
    if not cands:
        result["errors"].append("no results")
        return result

    winner, scored = match.best(local, cands, by_isrc=False, floor=0.6)
    result["candidates"] = [{k: v for k, v in c.items() if k != "_row"}
                            for c in scored[:5]]
    if not winner:
        result["errors"].append(
            f"best candidate scored {scored[0]['match']['score']:.2f}, below 0.60")
        return result

    row = winner["_row"]
    result["available"] = True
    result["match"] = winner["match"]
    result["track"] = {
        "id": row.get("trackId"),
        "title": row.get("trackName"),
        "artist": row.get("artistName"),
        "album": row.get("collectionName"),
        "duration_s": winner["duration_s"],
        "release_date": (row.get("releaseDate") or "")[:10] or None,
        "explicit": row.get("trackExplicitness"),
        "track_number": row.get("trackNumber"),
        "track_count": row.get("trackCount"),
        "country": row.get("country"),
        "url": row.get("trackViewUrl"),
        "preview_url": row.get("previewUrl"),
    }
    genre = row.get("primaryGenreName")
    result["genres"] = [{"name": genre}] if genre else []
    return result
