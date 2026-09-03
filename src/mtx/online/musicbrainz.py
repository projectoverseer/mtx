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


# Secondary types that mean "this package is not where the record came from".
# A soundtrack is deliberately absent: for a song written for a film the
# soundtrack *is* the original release.
REPACKAGE = frozenset({"Compilation", "Live", "Remix", "DJ-mix",
                       "Mixtape/Street", "Demo", "Interview", "Audiobook",
                       "Spokenword", "Audio drama"})

# A bootleg carries a real date about as often as it carries a real title.
STATUS_RANK = {"Official": 0, "Promotion": 1, "Pseudo-Release": 2,
               "Withdrawn": 3, "Cancelled": 4, "Bootleg": 5}
PRIMARY_RANK = {"Album": 0, "EP": 1, "Single": 2, "Broadcast": 3, "Other": 4}


_pad_date = match.pad_date


def _all_releases(client: Client, mbid: str, rec: dict[str, Any],
                  result: dict[str, Any]) -> list[dict[str, Any]]:
    """Every release carrying this recording, not the first 25 of them."""
    url = (f"{BASE}/release?recording={mbid}&fmt=json"
           f"&inc=release-groups&limit=100")
    body, err = client.get_json(url)
    result["requests"] += 1
    if err:
        result["errors"].append(f"releases browse: {err}")
    rows = [r for r in ((body or {}).get("releases") or []) if isinstance(r, dict)]
    if rows:
        total = (body or {}).get("release-count")
        if isinstance(total, int) and total > len(rows):
            # Beyond 100 the tail is reissues; record that it was cut rather
            # than let a truncated list read as a complete one.
            result["releases_truncated_at"] = len(rows)
            result["releases_total"] = total
        return rows
    # The embedded copy is capped and unordered, but it beats nothing.
    return [r for r in (rec.get("releases") or []) if isinstance(r, dict)]


def _has_album(releases: list[dict[str, Any]], album: str) -> bool:
    """Does any of these releases carry the album name the file claims?"""
    want = match.fold(album)
    if not want:
        return False
    for rel in releases:
        rg = rel.get("release-group") or {}
        for title in (rg.get("title"), rel.get("title")):
            folded = match.fold(title or "")
            if folded and (folded == want or want in folded or folded in want):
                return True
    return False


def _pick_release(releases: list[dict[str, Any]],
                  local_album: str) -> dict[str, Any]:
    """The release this file is a copy of, best effort, deterministically."""
    want = match.fold(local_album)

    def key(rel: dict[str, Any]) -> tuple:
        rg = rel.get("release-group") or {}
        titles = (match.fold(rg.get("title") or ""), match.fold(rel.get("title") or ""))
        # The album the file is filed under is the strongest evidence there is:
        # it says which package this copy was ripped from.  Exact before
        # containing, because `Heat Waves` is a substring of `Heat Waves
        # (expansion pack)` and the remix EP is not the record.
        if want and want in titles:
            album_match = 0
        elif want and any(want in t for t in titles if t):
            album_match = 1
        else:
            album_match = 2
        return (
            album_match,
            STATUS_RANK.get(rel.get("status"), 3),
            0 if not (REPACKAGE & set(rg.get("secondary-types") or [])) else 1,
            PRIMARY_RANK.get(rg.get("primary-type"), 5),
            _pad_date(rg.get("first-release-date") or rel.get("date")),
            # Within one release group the pressings are the same record; take
            # the first one issued so the country and date describe the
            # original rather than a later territory's reprint.
            _pad_date(rel.get("date")),
            str(rel.get("id") or ""),
        )

    return min(releases, key=key)


def _release_groups(releases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every distinct release group, so the packaging history stays auditable."""
    seen: dict[str, dict[str, Any]] = {}
    for rel in releases:
        rg = rel.get("release-group") or {}
        rgid = rg.get("id")
        if not rgid or rgid in seen:
            continue
        seen[rgid] = {
            "mbid": rgid, "title": rg.get("title"),
            "primary_type": rg.get("primary-type"),
            "secondary_types": rg.get("secondary-types") or [],
            "first_release_date": rg.get("first-release-date"),
        }
    return sorted(seen.values(),
                  key=lambda g: _pad_date(g.get("first_release_date")))[:40]


def _song_first_release(releases: list[dict[str, Any]]) -> str | None:
    """When the song came out, as distinct from when this package came out.

    Kept separate from `release.date` because they answer different questions.
    A 2011 remaster of a 1979 record is correctly dated 2011 as a *release*
    and belongs in the 1979 cohort as a *song*.
    """
    # Compilations are excluded, not deprioritised.  A compilation's own
    # first-release-date is when that *series* started, which has nothing to do
    # with this song: `Skyfall` appears on `Best of Bond… James Bond`, a
    # release group first issued in 1992, and was dated twenty years early.
    dates = [str(rg.get("first-release-date"))
             for rel in releases
             if (rel.get("status") or "Official") != "Bootleg"
             for rg in [rel.get("release-group") or {}]
             if rg.get("first-release-date")
             and not (REPACKAGE & set(rg.get("secondary-types") or []))]
    if not dates:
        return None
    return match.earliest_date(dates)


def _issued_as_single(releases: list[dict[str, Any]], title: str) -> bool | None:
    """Was this song released as a single in its own right?

    Not "does it appear on some single" -- a B-side does that.  The test is a
    single-type release group named after the song, which is what a single is.
    Returns None when nothing was found to judge from, never False by default.
    """
    if not releases:
        return None
    want = match.simplify_title(title)
    if not want:
        return None
    for rel in releases:
        rg = rel.get("release-group") or {}
        if rg.get("primary-type") != "Single":
            continue
        got = match.simplify_title(rg.get("title") or "")
        if got and (got == want or want in got or got in want):
            return True
    return False


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
        # Which albums this candidate appears on.  A duplicate recording entry
        # that exists only because someone catalogued a compilation is not the
        # album cut, and the file's own album tag is what separates them.
        "release_titles": [r.get("title") for r in (rec.get("releases") or [])
                           if isinstance(r, dict) and r.get("title")][:25],
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
        # `inc=artist-credits` is not decoration.  Without it every recording
        # an ISRC returns has an empty artist, so `artist_score` is 0.0 for all
        # of them, every candidate ties at the same score, and the winner is
        # whichever row the API happened to list first.  That is how a Olivia
        # Dean track ended up credited to an unrelated artist called `OLIVIA`:
        # two candidates, both scoring 1.00, and the wrong one listed first.
        body, err = client.get_json(
            f"{BASE}/isrc/{isrc}?fmt=json&inc=artist-credits+releases")
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

    # An ISRC match that lands on a recording credited to somebody else is the
    # signature of a mis-entered database row, and it is silent otherwise: the
    # duration and title agree, so the score stays at 1.00.  Say so in the file.
    credited = result["recording"]["artist"]
    if credited and local.get("artist"):
        agreement = match.artist_score(local["artist"], credited)
        result["artist_agreement"] = round(agreement, 4)
        if agreement < 0.5:
            # Deliberately not phrased as "the match is wrong".  Across this
            # corpus most of these are a feature credit the tag omits, or a
            # tag holding a writer's name -- `Giorgio by Moroder` is tagged
            # `Thomas Bangalter`.  It says the two sources disagree, which is
            # a fact, and leaves which one is wrong to whoever looks.
            result["errors"].append(
                f"credited artist {credited!r} disagrees with the file's "
                f"{local['artist']!r}; the match rests on duration and title")

    # Which of a recording's releases is *the* release is the single most
    # consequential choice in this module: it decides the release date, the
    # release type, the label, and whether the track counts as a single.  Two
    # defects used to make it almost arbitrary.
    #
    # First, `inc=releases` on a recording lookup is capped at 25 rows.  On a
    # 1,321-track corpus 465 tracks -- 35% -- hit that cap, so for a third of
    # the library the list being chosen from was an arbitrary quarter of the
    # truth.  The browse endpoint pages properly.
    #
    # Second, "earliest wins" was implemented as a string sort, and `"1999"`
    # sorts before `"1999-06-08"`.  A year-only bootleg therefore beat the
    # dated album every time: `Scar Tissue` was dated from a German bootleg
    # compilation called `Best` rather than from `Californication`.
    releases = _all_releases(client, mbid, rec, result)

    # MusicBrainz routinely holds more than one recording entity per ISRC: the
    # one an editor created from the CD, and one a bot created from the digital
    # reissue.  The rights holder says they are the same recording -- that is
    # what an ISRC is -- but only one of them inherits the packaging history.
    # `Californication` matched the digital entity, whose only releases are
    # 2020s compilations, and was dated 2022 as a result.  When the winner's
    # releases do not include the album the file says it came from, and a
    # sibling under the same ISRC might, ask about the sibling too.
    if by_isrc and local.get("album") and not _has_album(releases, local["album"]):
        for sibling in (result.get("candidates") or [])[1:4]:
            sid = sibling.get("id")
            if not sid or sid == mbid or sibling["match"]["score"] < 0.75:
                continue
            extra = _all_releases(client, sid, {}, result)
            if _has_album(extra, local["album"]):
                result["release_siblings_used"] = [sid]
                releases = releases + extra
                break

    if releases:
        rel = _pick_release(releases, local.get("album") or "")
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
        result["release_groups"] = _release_groups(releases)
        result["first_release_date"] = _song_first_release(releases)
        result["issued_as_single"] = _issued_as_single(
            releases, rec.get("title") or "")
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
