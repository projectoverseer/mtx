"""Optional metadata enrichment from the public music databases.

mtx measures a file.  It cannot know what the record is called, who mixed it,
what genre a listener would file it under, or whether the tempo it estimated
from an onset envelope is the tempo the artist counted.  Those facts exist,
they are public, and none of them are in the file: a purchased download keeps
the ISRC and throws away the credits.

This subpackage looks them up and writes them into an `online` section, under
three rules that keep it compatible with what mtx is for:

  * Off unless asked.  `mtx analyze` never touches the network; enrichment is
    `mtx enrich`, or `--online` passed deliberately.
  * Everything is sourced.  Each fact records which provider said it and how
    well that provider's row matched the file, so a wrong answer is traceable
    rather than mysterious.
  * Nothing measured is overwritten.  Where an outside number can be compared
    with one mtx derived -- tempo, duration -- both are kept side by side in
    `cross_checks`, and the disagreement is the output.

That last one is the point.  mtx marks its tempo estimate low-confidence on
most of a pop corpus; an independent published BPM that agrees settles it, and
one that is exactly double exposes the octave error beat trackers make.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from . import deezer, discogs, genre, itunes, lastfm, match, musicbrainz
from .http import Client

ONLINE_SCHEMA_VERSION = "1.0.0"

# Keyless providers run by default.  The two that need credentials are enabled
# by setting them; naming them without a key is an error worth reporting, not
# a silent skip, so they stay in the list and record why they did nothing.
DEFAULT_PROVIDERS = ("musicbrainz", "deezer", "itunes")
KEYED_PROVIDERS = ("lastfm", "discogs")
ALL_PROVIDERS = DEFAULT_PROVIDERS + KEYED_PROVIDERS

USER_AGENT_TEMPLATE = "mtx/{version} ( https://github.com/projectoverseer/mtx )"

# Tempo agreement bands.  0.02 is tighter than any beat tracker's own spread
# and loose enough for the rounding a shop applies to a published BPM.
TEMPO_EXACT = 0.02
TEMPO_OCTAVE = 0.04


def _first(*values: Any) -> Any:
    for v in values:
        if v not in (None, "", [], {}):
            return v
    return None


def local_facts(analysis: dict[str, Any]) -> dict[str, Any]:
    """What the file already claims about itself, for matching and searching."""
    tags = analysis.get("tags") or {}
    named = tags.get("named") or {}
    every = {str(k).lower(): v for k, v in (tags.get("all") or {}).items()}
    audio = analysis.get("audio") or {}
    return {
        "isrc": str(_first(named.get("isrc"), every.get("isrc")) or "").strip(),
        "title": str(_first(named.get("title"), every.get("title")) or "").strip(),
        "artist": str(_first(named.get("artist"), named.get("albumartist"),
                             every.get("artist")) or "").strip(),
        "album": str(_first(named.get("album"), every.get("album")) or "").strip(),
        "barcode": str(_first(named.get("barcode"), every.get("upc"),
                              every.get("barcode")) or "").strip(),
        "date": str(_first(named.get("date"), every.get("date")) or "").strip(),
        "genre_tag": str(named.get("genre") or "").strip(),
        "duration_s": audio.get("duration_s"),
    }


def _tempo_check(analysis: dict[str, Any], published: float | None,
                 source: str) -> dict[str, Any]:
    """Compare mtx's estimated tempo with a published one."""
    tempo = ((analysis.get("structure") or {}).get("tempo") or {})
    local = tempo.get("bpm")
    out: dict[str, Any] = {
        "local_bpm": local,
        "local_confidence": tempo.get("confidence"),
        "published_bpm": published,
        "published_source": source if published else None,
    }
    if not local or not published:
        out["verdict"] = "unavailable"
        out["resolved_bpm"] = local
        out["resolved_confidence"] = tempo.get("confidence")
        return out

    ratio = float(published) / float(local)
    out["ratio"] = round(ratio, 5)
    out["delta_bpm"] = round(float(published) - float(local), 3)

    if abs(ratio - 1.0) <= TEMPO_EXACT:
        out["verdict"] = "agree"
        # Two independent methods landing on the same number is stronger than
        # either alone, whatever the beat tracker thought of itself.
        out["resolved_bpm"] = round(float(published), 3)
        out["resolved_confidence"] = "high"
        out["note"] = "independent sources agree; local estimate confirmed"
    elif abs(ratio - 2.0) <= TEMPO_OCTAVE * 2 or abs(ratio - 0.5) <= TEMPO_OCTAVE:
        out["verdict"] = "octave"
        out["resolved_bpm"] = round(float(published), 3)
        # The relationship is certain, the number is not: both readings describe
        # the same grid, and which one a musician would call "the tempo" is a
        # judgment neither source has made.  Medium, with both values kept.
        out["resolved_confidence"] = "medium"
        out["alternate_bpm"] = round(float(local), 3)
        out["note"] = ("half/double-time disagreement: the beat tracker locked "
                       "to a different metrical level, not a different tempo; "
                       "both readings are kept")
    elif abs(ratio - 1.5) <= 0.03 or abs(ratio - (2 / 3)) <= 0.02:
        out["verdict"] = "triplet"
        out["resolved_bpm"] = round(float(published), 3)
        out["resolved_confidence"] = "medium"
        out["note"] = "3:2 relationship; one source is counting a triplet feel"
    else:
        out["verdict"] = "disagree"
        out["resolved_bpm"] = local
        out["resolved_confidence"] = "low"
        out["note"] = "sources disagree by more than a metrical relationship"
    return out


def _duration_check(local_s: float | None,
                    found: dict[str, float | None]) -> dict[str, Any]:
    deltas = {k: round(v - local_s, 3) for k, v in found.items()
              if v is not None and local_s is not None}
    worst = max((abs(d) for d in deltas.values()), default=None)
    return {
        "local_s": local_s,
        "providers_s": found,
        "deltas_s": deltas,
        "max_abs_delta_s": worst,
        "verdict": ("unavailable" if worst is None else
                    "exact" if worst <= match.EXACT_S else
                    "close" if worst <= match.CLOSE_S else "differs"),
    }


def _merge_credits(results: dict[str, Any],
                   analysis: dict[str, Any]) -> dict[str, Any]:
    """One role -> names map, built from every source that named anybody."""
    people: dict[str, dict[str, list[str]]] = {}

    def add(role: str, name: str, source: str) -> None:
        role = (role or "").strip().lower()
        name = (name or "").strip()
        if not role or not name:
            return
        slot = people.setdefault(role, {})
        who = slot.setdefault(name, [])
        if source not in who:
            who.append(source)

    mb = results.get("musicbrainz") or {}
    for c in (mb.get("credits") or []) + (mb.get("writers") or []):
        add(c.get("role"), c.get("name"), "musicbrainz")
    dc = results.get("discogs") or {}
    for c in dc.get("credits") or []:
        add(c.get("role"), c.get("name"), f"discogs:{c.get('scope', 'release')}")
    # Deezer's contributor role is `main` or `featured`, which name a billing
    # position rather than a job in the room.
    deezer_roles = {"main": "main artist", "featured": "featured artist"}
    for c in (results.get("deezer") or {}).get("contributors") or []:
        raw = str(c.get("role") or "performer").lower()
        add(deezer_roles.get(raw, raw), c.get("name"), "deezer")

    # The file's own tags are a source like any other, and often the only one
    # that names the mix engineer on a recent release.
    tag_roles = {
        "producer": "producer", "mixing": "mixing engineer",
        "mixing engineer": "mixing engineer", "mixer": "mixing engineer",
        "mastering engineer": "mastering engineer", "mastering": "mastering engineer",
        "engineer": "engineer", "second engineer": "assistant engineer",
        "assistant engineer": "assistant engineer", "composer": "composer",
        "lyricist": "lyricist", "writer": "writer", "arranger": "arranger",
    }
    every = {str(k).lower(): v for k, v in
             ((analysis.get("tags") or {}).get("all") or {}).items()}
    for key, role in tag_roles.items():
        value = every.get(key)
        if not value:
            continue
        for part in str(value).replace(";", "/").replace("\x00", "/").split("/"):
            add(role, part.strip(), "file:tag")

    return {role: [{"name": n, "sources": s} for n, s in sorted(names.items())]
            for role, names in sorted(people.items())}


def enrich(analysis: dict[str, Any], cache_dir: str | None = None,
           providers: tuple[str, ...] | list[str] = DEFAULT_PROVIDERS,
           offline: bool = False, refresh: bool = False,
           log: Callable[[str], None] | None = None,
           version: str = "0.0.0") -> dict[str, Any]:
    """Look one analysed track up and return its `online` section."""
    log = log or (lambda _m: None)
    t0 = time.time()
    local = local_facts(analysis)
    client = Client(cache_dir=cache_dir,
                    user_agent=USER_AGENT_TEMPLATE.format(version=version),
                    log=log, offline=offline, refresh=refresh)

    out: dict[str, Any] = {
        "schema_version": ONLINE_SCHEMA_VERSION,
        "queried_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query": local,
        "providers_requested": list(providers),
        "providers_available": [],
        "errors": [],
    }

    runners = {"musicbrainz": musicbrainz.lookup, "deezer": deezer.lookup,
               "itunes": itunes.lookup, "lastfm": lastfm.lookup,
               "discogs": discogs.lookup}

    for name in providers:
        runner = runners.get(name)
        if runner is None:
            out["errors"].append(f"unknown provider: {name}")
            continue
        try:
            res = runner(client, local)
        except Exception as exc:  # a provider outage must not lose the analysis
            res = {"available": False, "errors": [f"{type(exc).__name__}: {exc}"]}
        out[name] = res
        if res.get("available"):
            out["providers_available"].append(name)
        for err in res.get("errors") or []:
            out["errors"].append(f"{name}: {err}")
        log(f"{name}: {'ok' if res.get('available') else 'no match'}"
            f" ({res.get('requests', 0)} request(s))")

    # -- genre vote ----------------------------------------------------------

    mb = out.get("musicbrainz") or {}
    lf = out.get("lastfm") or {}
    dg = out.get("discogs") or {}
    genre_sources = {
        "musicbrainz:recording": mb.get("genres_recording"),
        "musicbrainz:release-group": mb.get("genres_release_group"),
        "musicbrainz:artist": mb.get("genres_artist"),
        "deezer:album": (out.get("deezer") or {}).get("genres"),
        "itunes:track": (out.get("itunes") or {}).get("genres"),
        "lastfm:track": lf.get("tags_track"),
        "lastfm:artist": lf.get("tags_artist"),
        "discogs:style": dg.get("styles"),
        "discogs:genre": dg.get("genres"),
        "file:tag": [local["genre_tag"]] if local.get("genre_tag") else None,
    }
    out["genres"] = genre.collect({k: v for k, v in genre_sources.items() if v})
    # Every name attached to this record, so that "billie eilish" cannot end
    # up in a mood vocabulary alongside "nocturnal" and "party".
    artist_names = [local.get("artist") or ""]
    artist_names += [a.get("name") for a in (mb.get("artists") or [])
                     if isinstance(a, dict)]
    artist_names += [p.get("name") for p in
                     ((out.get("credits") or {}).get("main artist") or [])
                     if isinstance(p, dict)]
    out["descriptive_tags"] = genre.collect_tags({
        "musicbrainz:recording": mb.get("tags_recording"),
        "musicbrainz:release-group": mb.get("tags_release_group"),
        "lastfm:track": lf.get("tags_track"),
        "lastfm:artist": lf.get("tags_artist"),
    }, exclude=artist_names)

    # -- cross-checks --------------------------------------------------------

    dz_track = (out.get("deezer") or {}).get("track") or {}
    out["cross_checks"] = {
        "tempo": _tempo_check(analysis, dz_track.get("bpm"), "deezer"),
        "duration": _duration_check(local.get("duration_s"), {
            "musicbrainz": (mb.get("recording") or {}).get("duration_s"),
            "deezer": dz_track.get("duration_s"),
            "itunes": ((out.get("itunes") or {}).get("track") or {}).get("duration_s"),
        }),
    }

    dates = {
        "file_tag": local.get("date") or None,
        "musicbrainz_release": (mb.get("release") or {}).get("date"),
        "musicbrainz_first": (mb.get("release_group") or {}).get("first_release_date"),
        "deezer": dz_track.get("release_date"),
        "itunes": ((out.get("itunes") or {}).get("track") or {}).get("release_date"),
    }
    known = sorted(d for d in dates.values() if d)
    out["cross_checks"]["release_date"] = {
        "sources": dates,
        "earliest": known[0] if known else None,
        "agree": len({d[:10] for d in known}) <= 1 if known else None,
    }

    # -- rolled up -----------------------------------------------------------

    out["credits"] = _merge_credits(out, analysis)
    out["popularity"] = {
        "deezer_rank": dz_track.get("rank"),
        "deezer_album_fans": ((out.get("deezer") or {}).get("album") or {}).get("rank"),
        "lastfm_listeners": (lf.get("track") or {}).get("listeners"),
        "lastfm_playcount": (lf.get("track") or {}).get("playcount"),
        "lastfm_artist_listeners": (lf.get("artist") or {}).get("listeners"),
    }
    out["identity"] = {
        "isrc": local.get("isrc") or None,
        "recording_mbid": (mb.get("recording") or {}).get("mbid"),
        "release_mbid": (mb.get("release") or {}).get("mbid"),
        "release_group_mbid": (mb.get("release_group") or {}).get("mbid"),
        "work_mbid": (mb.get("work") or {}).get("mbid"),
        "iswcs": (mb.get("work") or {}).get("iswcs") or [],
        "deezer_id": dz_track.get("id"),
        "itunes_id": ((out.get("itunes") or {}).get("track") or {}).get("id"),
        "discogs_release_id": (dg.get("release") or {}).get("id"),
        "label": _first(((out.get("deezer") or {}).get("album") or {}).get("label"),
                        ((dg.get("release") or {}).get("labels") or [None])[0]),
    }

    scores = [r["match"]["score"] for r in
              (out.get(p) or {} for p in providers)
              if isinstance(r, dict) and isinstance(r.get("match"), dict)]
    out["match_confidence"] = round(sum(scores) / len(scores), 4) if scores else 0.0
    out["cache"] = dict(client.stats)
    out["elapsed_seconds"] = round(time.time() - t0, 3)
    return out


__all__ = ["enrich", "local_facts", "ALL_PROVIDERS", "DEFAULT_PROVIDERS",
           "KEYED_PROVIDERS", "ONLINE_SCHEMA_VERSION"]
