"""Declared metadata, and version identity.

For a published record, writer splits, the publisher, the producer credits and
whether this is the radio edit are *discoverable* -- `enrich` goes and looks
them up.  For your own unreleased work none of it is missing: you are the
source of it rather than a database.  So the gap is not a measurement gap, it
is an input gap, and it is closed by a sidecar.

The rule the whole module exists to enforce: **a declared value is passed
through with `source: "declared"` and is never merged into a measured or a
database-sourced field.**  `online.credits` already carries a `sources` array
per claim; `"declared"` is one more origin alongside `"file:tag"` and
`"musicbrainz"`, not a way to overwrite them.

Version identity is the other half and is the piece worth having first: two
bounces of one song currently have no way to say they are two mixes rather than
two songs, so no comparison between them is expressible.  It is derived from
tags, offline, and needs nobody's database.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Any

SIDECAR_NAME = "declared.json"

# Version markers, longest first so "radio edit" wins over "edit".
VERSION_MARKERS: tuple[tuple[str, str], ...] = (
    ("radio edit", "radio_edit"),
    ("extended mix", "extended"),
    ("extended version", "extended"),
    ("clean version", "clean"),
    ("album version", "album_version"),
    ("single version", "single_version"),
    ("sped up", "sped_up"),
    ("speed up", "sped_up"),
    ("slowed down", "slowed"),
    ("slowed + reverb", "slowed"),
    ("instrumental", "instrumental"),
    ("a cappella", "acappella"),
    ("acappella", "acappella"),
    ("acoustic", "acoustic"),
    ("remaster", "remaster"),
    ("remastered", "remaster"),
    ("explicit", "explicit"),
    ("clean", "clean"),
    ("remix", "remix"),
    ("rework", "remix"),
    ("bootleg", "remix"),
    ("live", "live"),
    ("demo", "demo"),
    ("edit", "edit"),
    ("mix", "alternate_mix"),
)

# Keys the sidecar recognises.  Anything else is kept but reported as unknown:
# a typo that silently vanished would be worse than one that is listed.
KNOWN_FIELDS: tuple[str, ...] = (
    "lyrics", "lyrics_language", "title", "artist", "featured_artists",
    "genre", "release_year", "recording_date", "release_date",
    "writers", "publisher", "pro", "producers", "engineers", "performers",
    "samples", "interpolations", "origin", "version", "sibling_versions",
    "work_key", "isrc", "iswc", "upc", "label", "explicit",
    "cohort", "notes",
)


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.casefold()


def strip_markers(text: str) -> tuple[str, list[str]]:
    """Remove every recognised version marker from a title.

    Returns the cleaned title and the markers that were found, so the same pass
    produces both the version identity and the work key.
    """
    found: list[str] = []
    low = _fold(text)
    # Longest phrase first, and each match is blanked out of the haystack
    # before the shorter ones are tried: "radio edit" must not also report
    # "edit", and "slowed down" must not also report "slowed".
    for phrase, name in sorted(VERSION_MARKERS, key=lambda pn: -len(pn[0])):
        pattern = rf"\b{re.escape(phrase)}\b"
        if re.search(pattern, low):
            low = re.sub(pattern, " " * len(phrase), low)
            if name not in found:
                found.append(name)
    # Drop bracketed and dashed suffixes that contain a marker.
    cleaned = text
    for pattern in (r"\s*[\(\[][^)\]]*[\)\]]", r"\s+-\s+[^-]*$"):
        while True:
            m = None
            for cand in re.finditer(pattern, cleaned):
                inner = _fold(cand.group(0))
                if any(re.search(rf"\b{re.escape(p)}\b", inner)
                       for p, _ in VERSION_MARKERS):
                    m = cand
                    break
            if m is None:
                break
            cleaned = (cleaned[:m.start()] + cleaned[m.end():]).strip()
    return cleaned.strip(), found


def work_key(artist: str | None, title: str | None) -> str | None:
    """A stable identity for the song, independent of which version this is."""
    if not title:
        return None
    base, _ = strip_markers(title)
    parts = []
    for value in (artist or "", base):
        v = _fold(value)
        v = re.sub(r"\b(feat|ft|featuring|with)\b.*$", "", v)
        v = re.sub(r"[^a-z0-9 ]+", " ", v)
        parts.append(" ".join(v.split()))
    key = "|".join(p for p in parts if p)
    return key or None


def version_identity(tags: dict[str, Any]) -> dict[str, Any]:
    """Which version of the song this file is, from its tags alone."""
    named = (tags or {}).get("named") or {}
    allt = (tags or {}).get("all") or {}
    title = named.get("title")
    artist = named.get("artist")
    haystack_parts = [str(title or "")]
    for key, value in allt.items():
        if any(k in str(key).lower() for k in
               ("subtitle", "version", "tset", "tit3", "comment", "grouping")):
            haystack_parts.append(str(value))
    haystack = " ; ".join(p for p in haystack_parts if p)
    base, markers = strip_markers(haystack)
    clean_title, _ = strip_markers(title) if title else (None, [])
    return {
        "available": bool(title or markers),
        "source": "file:tag",
        "title_as_tagged": title,
        "title_without_version_markers": clean_title,
        "markers": markers,
        "is_primary_version": bool(not markers),
        "work_key": work_key(artist, title),
        "work_key_rule": "case-folded, accent-stripped, punctuation-stripped "
                         "artist and title with every version marker and any "
                         "'feat.' clause removed",
        "sibling_versions": {
            "available": False,
            "reason": "how many other versions of this work exist is a database "
                      "question; `mtx enrich` fills it in, and an unreleased "
                      "track has none to find",
        },
        "note": "two files that agree on work_key and differ on markers are two "
                "versions of one song; the comparison between them is what "
                "`mtx compare` measures",
    }


def sidecar_paths(audio_path: str, out_dir: str | None = None,
                  explicit: str | None = None) -> list[str]:
    """Where a declared.json may live, in the order it is looked for."""
    out: list[str] = []
    if explicit:
        out.append(explicit)
    base = os.path.splitext(os.path.abspath(audio_path))[0]
    out.append(base + ".declared.json")
    out.append(os.path.join(os.path.dirname(os.path.abspath(audio_path)),
                            SIDECAR_NAME))
    if out_dir:
        out.append(os.path.join(out_dir, SIDECAR_NAME))
    seen: set[str] = set()
    return [p for p in out if not (p in seen or seen.add(p))]


def load(audio_path: str, collector: Any = None, out_dir: str | None = None,
         explicit: str | None = None) -> dict[str, Any]:
    """Read the sidecar, if there is one, and label every value it carries."""
    candidates = sidecar_paths(audio_path, out_dir, explicit)
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        return {
            "available": False,
            "reason": "no declared.json found",
            "searched": candidates,
            "schema": list(KNOWN_FIELDS),
            "note": "a declared sidecar is how an unreleased track supplies the "
                    "facts a database would otherwise hold: the lyric, the "
                    "splits, the intended genre and year. Nothing in it is "
                    "measured and nothing measured is overwritten by it.",
        }
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        if collector is not None:
            collector.warn("declared", f"could not read {path}: {exc!r}")
        return {"available": False, "reason": f"unreadable sidecar: {exc!r}",
                "path": path}
    if not isinstance(data, dict):
        if collector is not None:
            collector.warn("declared", f"{path} is not a JSON object")
        return {"available": False, "reason": "sidecar is not a JSON object",
                "path": path}
    unknown = sorted(k for k in data if k not in KNOWN_FIELDS)
    if unknown and collector is not None:
        collector.warn("declared",
                       f"{path} carries field(s) mtx does not recognise and "
                       f"passes through unchanged: {', '.join(unknown)}")
    fields = {k: {"value": v, "source": "declared"} for k, v in data.items()}
    return {
        "available": True,
        "path": path,
        "source": "declared",
        "fields": fields,
        "unknown_fields": unknown,
        "rule": "every value here was stated by whoever wrote the sidecar. It "
                "is reported with source=declared and is never merged into a "
                "measured field or into online.*",
    }


def declared_value(declared: dict[str, Any], key: str) -> Any:
    """The raw value of one declared field, or None."""
    if not declared or not declared.get("available"):
        return None
    entry = (declared.get("fields") or {}).get(key)
    return entry.get("value") if isinstance(entry, dict) else None
