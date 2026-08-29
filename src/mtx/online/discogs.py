"""Discogs: engineer credits and a genre/style split.  Optional, needs a token.

Discogs is the only provider here that reliably names the people in the room.
Its release pages carry `Mixed By`, `Mastered By`, `Recorded By`, `Producer`
and `Written-By` as structured credits, transcribed from physical sleeves --
which is exactly the information a purchased download strips out.

It is also the only one that separates coarse genre from fine style
(`Electronic` / `Synth-pop, Downtempo`), so the two feed the vote at different
weights rather than competing.

The cost is that Discogs indexes releases, not recordings: a search returns
the album, and a track's credits may live on the release rather than the
track.  The credits collected here are therefore release-level unless the
tracklist names them, and are labelled as such.

Set DISCOGS_TOKEN to enable.  Skipped silently when unset.
"""

from __future__ import annotations

import os
import re
from typing import Any

from . import match
from .http import Client, build_url

BASE = "https://api.discogs.com"

# Discogs role strings are free text with a controlled core.  These are the
# roles worth lifting; anything else is kept verbatim under its own name.
ROLE_MAP = {
    "producer": "producer",
    "co-producer": "producer",
    "executive producer": "executive producer",
    "mixed by": "mixing engineer",
    "remix": "remixer",
    "remixed by": "remixer",
    "mastered by": "mastering engineer",
    "recorded by": "recording engineer",
    "engineer": "engineer",
    "engineer [additional]": "assistant engineer",
    "engineer [assistant]": "assistant engineer",
    "assistant engineer": "assistant engineer",
    "written-by": "writer",
    "composed by": "composer",
    "lyrics by": "lyricist",
    "arranged by": "arranger",
    "vocals": "vocals",
    "keyboards": "keyboards",
    "drums": "drums",
    "bass": "bass",
    "guitar": "guitar",
}


def token() -> str:
    return os.environ.get("DISCOGS_TOKEN", "").strip()


def _clean_name(name: str) -> str:
    """Discogs disambiguates duplicate artists with a trailing `(2)`."""
    return re.sub(r"\s*\(\d+\)$", "", str(name or "")).strip()


def _credits(nodes: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    out = []
    for node in nodes or []:
        raw_role = str(node.get("role") or "").strip()
        for part in re.split(r",(?![^\[]*\])", raw_role):
            part = part.strip()
            if not part:
                continue
            key = part.lower()
            role = ROLE_MAP.get(key)
            if role is None:
                # `Mixed By [Additional]` -> match on the bracket-free head.
                head = re.sub(r"\s*\[.*?\]", "", key).strip()
                role = ROLE_MAP.get(head, head or "credit")
            out.append({"role": role, "raw_role": part,
                        "name": _clean_name(node.get("name")),
                        "scope": scope})
    return out


def lookup(client: Client, local: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"available": False, "errors": [], "requests": 0}
    tok = token()
    if not tok:
        result["errors"].append("DISCOGS_TOKEN not set")
        return result

    headers = {"Authorization": f"Discogs token={tok}"}
    params: dict[str, Any] = {"type": "release", "per_page": 10}
    if local.get("barcode"):
        params["barcode"] = local["barcode"]
    else:
        params["artist"] = local.get("artist") or ""
        params["track"] = local.get("title") or ""
        params["release_title"] = local.get("album") or ""
    url = build_url(f"{BASE}/database/search", **params)
    body, err = client.get_json(url, headers=headers)
    result["requests"] += 1
    if err:
        result["errors"].append(f"search: {err}")
        return result

    hits = [h for h in (body or {}).get("results") or [] if h.get("id")]
    if not hits:
        result["errors"].append("no results")
        return result

    # Search hits carry no duration, so rank on title text and prefer the
    # earliest year -- the original pressing over a later compilation.
    def rank(hit: dict[str, Any]) -> tuple[float, str]:
        text = str(hit.get("title") or "")
        score = match.title_score(
            f"{local.get('artist', '')} {local.get('album') or local.get('title', '')}",
            text)
        return (-score, str(hit.get("year") or "9999"))

    hits.sort(key=rank)
    hit = hits[0]

    rel, err = client.get_json(f"{BASE}/releases/{hit['id']}", headers=headers)
    result["requests"] += 1
    if err or not rel:
        result["errors"].append(f"release {hit['id']}: {err or 'empty'}")
        return result

    result["available"] = True
    result["release"] = {
        "id": rel.get("id"), "title": rel.get("title"),
        "year": rel.get("year"), "country": rel.get("country"),
        "labels": [lb.get("name") for lb in (rel.get("labels") or [])],
        "formats": [f.get("name") for f in (rel.get("formats") or [])],
        "url": rel.get("uri"),
        "notes": (rel.get("notes") or "")[:2000] or None,
    }
    result["genres"] = [{"name": g} for g in (rel.get("genres") or [])]
    result["styles"] = [{"name": s} for s in (rel.get("styles") or [])]

    credits = _credits(rel.get("extraartists") or [], "release")

    # A tracklist entry that matches the file gets its own, more specific,
    # credits appended.
    want = match.simplify_title(local.get("title") or "")
    for entry in rel.get("tracklist") or []:
        if want and match.simplify_title(entry.get("title") or "") == want:
            credits += _credits(entry.get("extraartists") or [], "track")
            result["track_position"] = entry.get("position")
            break
    result["credits"] = credits
    return result
