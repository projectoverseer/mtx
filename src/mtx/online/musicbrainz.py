"""MusicBrainz: the granular genre vote, plus credits the file tags never carry.

Four requests per track, each one earning its place:

  1. ISRC  -> candidate recordings.  The ISRC resource refuses `inc=genres`,
     which is why this cannot be a single call.
  2. the winning recording, with genres, tags, releases and artist relations.
  3. its release group, where MusicBrainz genre voting actually concentrates
     (a recording usually has a handful of votes, an album has many).
  4. the work behind the recording, whose artist relations are the songwriting
     credits -- composer and lyricist, the one thing a purchased file's tags
     almost never carry and the reason this is worth four requests.

The artist's own genres are fetched too when the first three leave the vote
thin, since an artist page is the best-populated genre surface MusicBrainz has.
"""

from __future__ import annotations

from typing import Any

from . import match
from .http import Client, build_url

BASE = "https://musicbrainz.org/ws/2"

# Relation types that name a person who shaped the record.  MusicBrainz spells
# the mixing credit `mix` and the mastering credit `mastering`; `engineer` is
# the generic fallback used when a release does not distinguish.
CREDIT_ROLES = {
    "producer": "producer",
    "engineer": "engineer",
    "recording": "recording engineer",
    "mix": "mixing engineer",
    "mastering": "mastering engineer",
    "programming": "programming",
    "instrument": "instrument",
    "vocal": "vocals",
    "performer": "performer",
    "arranger": "arranger",
    "orchestrator": "orchestrator",
    "composer": "composer",
    "lyricist": "lyricist",
    "writer": "writer",
    "remixer": "remixer",
}


def _rec_summary(rec: dict[str, Any]) -> dict[str, Any]:
    length = rec.get("length")
    return {
        "id": rec.get("id"),
        "title": rec.get("title"),
        "duration_s": (length / 1000.0) if isinstance(length, (int, float)) else None,
        "artist": _credit_string(rec.get("artist-credit") or []),
        "disambiguation": rec.get("disambiguation") or "",
        "first_release_date": rec.get("first-release-date"),
        "video": bool(rec.get("video")),
    }


def _credit_string(credit: list[Any]) -> str:
    out = []
    for part in credit:
        if isinstance(part, dict):
            out.append(str(part.get("name") or (part.get("artist") or {}).get("name") or ""))
            out.append(str(part.get("joinphrase") or ""))
        elif isinstance(part, str):
            out.append(part)
    return "".join(out).strip()


def _genres(node: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"name": g.get("name"), "count": g.get("count", 1)}
            for g in (node.get("genres") or []) if g.get("name")]


def _tags(node: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"name": t.get("name"), "count": t.get("count", 1)}
            for t in (node.get("tags") or []) if t.get("name")]


def _relations(node: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for rel in node.get("relations") or []:
        if rel.get("target-type") != "artist":
            continue
        artist = rel.get("artist") or {}
        rtype = str(rel.get("type") or "")
        role = CREDIT_ROLES.get(rtype, rtype)
        attrs = [str(a) for a in (rel.get("attributes") or [])]
        if attrs and rtype in ("instrument", "vocal"):
            role = f"{role} ({', '.join(attrs)})"
        out.append({"role": role, "type": rtype, "name": artist.get("name"),
                    "artist_mbid": artist.get("id"), "attributes": attrs})
    return out


def lookup(client: Client, local: dict[str, Any]) -> dict[str, Any]:
    """Resolve one track.  Never raises; every failure lands in `errors`."""
    result: dict[str, Any] = {"available": False, "errors": [], "requests": 0}
    isrc = (local.get("isrc") or "").strip().upper().replace("-", "")

    candidates: list[dict[str, Any]] = []
    by_isrc = False

    if isrc:
        body, err = client.get_json(f"{BASE}/isrc/{isrc}?fmt=json")
        result["requests"] += 1
        if err and "404" not in err:
            result["errors"].append(f"isrc lookup: {err}")
        for rec in (body or {}).get("recordings") or []:
            candidates.append(_rec_summary(rec))
        by_isrc = bool(candidates)
        result["isrc_candidates"] = len(candidates)

    if not candidates and local.get("title"):
        # No ISRC, or an ISRC MusicBrainz has never seen.  Fall back to a
        # field-qualified search, which is far more precise than a bare query.
        query = f'recording:"{_esc(match.search_title(local["title"]))}"'
        # MusicBrainz's artist field holds one name; a tag reading
        # "Frank Sinatra / Count Basie" matches nothing as written.
        lead = match.primary_artist(local.get("artist", ""))
        if lead:
            query += f' AND artist:"{_esc(lead)}"'
        url = build_url(f"{BASE}/recording/?fmt=json", query=query, limit=12)
        body, err = client.get_json(url)
        result["requests"] += 1
        if err:
            result["errors"].append(f"search: {err}")
        for rec in (body or {}).get("recordings") or []:
            candidates.append(_rec_summary(rec))
        result["search_candidates"] = len(candidates)

    if not candidates:
        result["errors"].append("no candidate recording")
        return result

    winner, scored = match.best(local, candidates, by_isrc=by_isrc, floor=0.5)
    result["candidates"] = scored[:6]
    if not winner:
        result["errors"].append(
            f"best candidate scored {scored[0]['match']['score']:.2f}, below 0.50")
        return result

    result["match"] = winner["match"]
    mbid = winner["id"]

    inc = ("artist-credits+releases+release-groups+genres+tags+isrcs"
           "+artist-rels+work-rels")
    rec, err = client.get_json(f"{BASE}/recording/{mbid}?fmt=json&inc={inc}")
    result["requests"] += 1
    if err or not rec:
        result["errors"].append(f"recording {mbid}: {err or 'empty'}")
        return result

    result["available"] = True
    length = rec.get("length")
    result["recording"] = {
        "mbid": mbid,
        "title": rec.get("title"),
        "duration_s": (length / 1000.0) if isinstance(length, (int, float)) else None,
        "disambiguation": rec.get("disambiguation") or "",
        "artist": _credit_string(rec.get("artist-credit") or []),
        "isrcs": rec.get("isrcs") or [],
        "url": f"https://musicbrainz.org/recording/{mbid}",
    }
    result["genres_recording"] = _genres(rec)
    result["tags_recording"] = _tags(rec)
    result["credits"] = _relations(rec)

    artists = [{"name": (p.get("artist") or {}).get("name"),
                "mbid": (p.get("artist") or {}).get("id")}
               for p in (rec.get("artist-credit") or []) if isinstance(p, dict)
               and p.get("artist")]
    result["artists"] = artists

    # Earliest release wins: it is the one that dates the recording rather than
    # the compilation that happened to be listed first.
    releases = sorted(
        [r for r in (rec.get("releases") or []) if isinstance(r, dict)],
        key=lambda r: (r.get("date") or "9999"))
    if releases:
        rel = releases[0]
        rg = rel.get("release-group") or {}
        result["release"] = {
            "mbid": rel.get("id"), "title": rel.get("title"),
            "date": rel.get("date"), "country": rel.get("country"),
            "status": rel.get("status"),
        }
        result["release_group"] = {
            "mbid": rg.get("id"), "title": rg.get("title"),
            "primary_type": rg.get("primary-type"),
            "secondary_types": rg.get("secondary-types") or [],
            "first_release_date": rg.get("first-release-date"),
        }
        result["release_count"] = len(releases)
        if rg.get("id"):
            rgb, rgerr = client.get_json(
                f"{BASE}/release-group/{rg['id']}?fmt=json&inc=genres+tags")
            result["requests"] += 1
            if rgerr:
                result["errors"].append(f"release-group: {rgerr}")
            else:
                result["genres_release_group"] = _genres(rgb or {})
                result["tags_release_group"] = _tags(rgb or {})

    # The work carries the songwriting.  A recording without a work relation is
    # common for recent releases; that is a gap in the database, not an error.
    work_id = None
    for rel in rec.get("relations") or []:
        if rel.get("target-type") == "work" and (rel.get("work") or {}).get("id"):
            work_id = rel["work"]["id"]
            break
    if work_id:
        wb, werr = client.get_json(
            f"{BASE}/work/{work_id}?fmt=json&inc=artist-rels+genres+tags")
        result["requests"] += 1
        if werr:
            result["errors"].append(f"work: {werr}")
        elif wb:
            result["work"] = {"mbid": work_id, "title": wb.get("title"),
                              "iswcs": wb.get("iswcs") or []}
            result["writers"] = [c for c in _relations(wb)
                                 if c["type"] in ("composer", "lyricist", "writer",
                                                  "arranger", "orchestrator")]

    # Only reach for the artist page when the track and album vote is thin --
    # artist genres describe a career, and they drown a single record.
    have = len(result.get("genres_recording") or []) + \
        len(result.get("genres_release_group") or [])
    if have < 3 and artists and artists[0].get("mbid"):
        ab, aerr = client.get_json(
            f"{BASE}/artist/{artists[0]['mbid']}?fmt=json&inc=genres+tags")
        result["requests"] += 1
        if aerr:
            result["errors"].append(f"artist: {aerr}")
        else:
            result["genres_artist"] = _genres(ab or {})
            result["tags_artist"] = _tags(ab or {})

    return result


def _esc(text: str) -> str:
    """Escape the Lucene syntax MusicBrainz search uses."""
    out = str(text)
    for ch in '+-&|!(){}[]^"~*?:\\/':
        out = out.replace(ch, " ")
    return " ".join(out.split())
