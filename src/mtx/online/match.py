"""Deciding whether a database row is really this recording.

An ISRC lookup is not proof.  Labels reuse an ISRC across a radio edit and the
album cut, reissues carry the original code, and a compilation can point at a
different master of the same performance.  `bad guy` is the worked example:
its ISRC returns three MusicBrainz recordings and the first one listed is a
175 s radio edit, while the file on disk is 194 s.  Taking `recordings[0]`
would have attached the wrong duration, the wrong release and the wrong
credits to every track in the corpus.

So every candidate is scored against what mtx already measured -- duration
above all, since that is the one field the local analysis knows exactly -- and
the score travels with the result.  A consumer that wants only certain matches
can filter on it; nothing here silently discards a candidate.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any

# Duration agreement, in seconds, and what each band means.  Two seconds is
# wide enough for the ragged fade-outs and leading silence that differ between
# a store's copy and a CD rip, and narrow enough to separate a radio edit.
EXACT_S = 2.0
CLOSE_S = 5.0

# Bracketed suffixes that mark the same performance under a different package.
_NOISE = re.compile(
    r"\s*[\(\[]\s*(?:"
    r"feat\.?|ft\.?|featuring|with|prod\.?|"
    r"official|audio|video|music video|visualizer|lyric[s]?|"
    r"remaster(?:ed)?(?:\s*\d{4})?|\d{4}\s*remaster(?:ed)?|"
    r"single|album|radio|explicit|clean|bonus|deluxe|expanded|"
    r"mono|stereo|dolby\s*atmos|spatial|hi-?res|"
    r"from\b.*|theme\b.*"
    r")[^\)\]]*[\)\]]",
    re.IGNORECASE)
_DASH_SUFFIX = re.compile(
    r"\s+-\s+(?:.*\bremaster(?:ed)?\b.*|.*\bversion\b.*|.*\bedit\b.*|"
    r".*\bmix\b.*|single|radio edit)$", re.IGNORECASE)


def pad_date(value) -> str:
    """`1999` and `1999-06-08` compared on the same scale.

    ISO 8601 sorts lexically only when every part is present.  Left short, a
    year-only date sorts *before* every dated release in that year, which is
    how a bootleg dated `1999` beat an album dated `1999-06-08`.  Filled with
    the earliest the date could mean, so the ordering is a lower bound rather
    than an invention.
    """
    text = str(value or "").strip()
    if not text:
        return "9999-99-99"
    parts = text.split("-")
    while len(parts) < 3:
        parts.append("01")
    return "-".join(p.zfill(4) if i == 0 else p.zfill(2)
                    for i, p in enumerate(parts[:3]))


def earliest_date(values) -> str | None:
    """The earliest of several dates, at the best precision available.

    `2020` and `2020-06-29` are the same claim at two resolutions, not two
    dates one of which is earlier.  Taking the string minimum picks the vague
    one and throws the day away; 365 of 1,321 corpus releases were dated to
    the year for exactly that reason.  So: earliest year, then the most
    precise reading inside it.
    """
    dates = [str(v).strip() for v in (values or []) if v]
    if not dates:
        return None
    year = pad_date(min(dates, key=pad_date))[:4]
    same_year = [d for d in dates if pad_date(d)[:4] == year]
    precise = [d for d in same_year if len(d) >= 7]
    return min(precise or same_year, key=pad_date)


def fold(text: str) -> str:
    """Lowercase, strip accents and punctuation: a key for loose comparison."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(text))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    stripped = stripped.replace("&", " and ")
    stripped = re.sub(r"[^\w\s]", " ", stripped, flags=re.UNICODE)
    return re.sub(r"\s+", " ", stripped).strip().lower()


def simplify_title(title: str) -> str:
    """Drop the packaging so `bad guy (Official Audio)` matches `bad guy`."""
    t = _NOISE.sub("", str(title or ""))
    t = _DASH_SUFFIX.sub("", t)
    return fold(t)


def search_title(title: str) -> str:
    """The title with packaging removed but still readable, for a search box.

    `simplify_title` folds case and strips punctuation, which is right for
    comparing two strings and wrong for asking a database a question.  A
    reissue titled `Fly Me To The Moon (Dolby Atmos)` finds nothing under its
    full name and everything under its real one.
    """
    t = _NOISE.sub("", str(title or ""))
    t = _DASH_SUFFIX.sub("", t)
    t = re.sub(r"\s+", " ", t).strip(" -")
    return t or str(title or "").strip()


def search_artist(artist: str) -> str:
    """Tag multi-value separators turned into something a search box accepts."""
    a = re.sub(r"\s*[/;\x00]\s*", ", ", str(artist or ""))
    return re.sub(r"\s+", " ", a).strip(" ,")


# A feature credit wearing brackets.  `Ariana Grande (ft. Pharell Willians)`
# has no whitespace before the `ft.`, so a word-boundary split leaves the
# string whole and every name search then asks for an artist who does not
# exist -- which is what 91 missing play counts turned out to be.
_BRACKET_FEAT = re.compile(
    r"\s*[\(\[]\s*(?:feat\.?|ft\.?|featuring|with|prod\.?|w/)\b[^\)\]]*[\)\]]?",
    re.IGNORECASE)


def primary_artist(artist: str) -> str:
    """The first credited name, for databases whose artist field holds one."""
    text = _BRACKET_FEAT.sub("", str(artist or ""))
    # A comma separates two artists in "Drake, 21 Savage" and separates
    # nothing at all in "Tyler, The Creator".  Splitting the second one asks
    # Last.fm about an artist called "Tyler", which exists, and comes back
    # with a play count off by four orders of magnitude.
    text = re.sub(r",\s+(?=(?:the|los|las|die|le|la)\b)", "\x01", text,
                  flags=re.IGNORECASE)
    parts = re.split(r"\s*[/;,\x00]\s*|\s+(?:feat\.?|ft\.?|featuring|with)\s+",
                     text, flags=re.IGNORECASE)
    return next((p.replace("\x01", ", ").strip(" ([-")
                 for p in parts if p.strip(" ([-\x01")), "")


def ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def title_score(local: str, remote: str) -> float:
    """1.0 for the same title once packaging is discounted."""
    a, b = simplify_title(local), simplify_title(remote)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # A title that contains the other is usually a feature credit spelled out
    # in one source and not the other.
    if a in b or b in a:
        return 0.94
    return ratio(a, b)


def artist_score(local: str, remote: str) -> float:
    """Compare artist lists as sets: collaborators are ordered differently."""
    a = {p for p in re.split(r"[,;/&]|\bfeat\.?\b|\bft\.?\b|\band\b",
                             fold(local)) if p.strip()}
    b = {p for p in re.split(r"[,;/&]|\bfeat\.?\b|\bft\.?\b|\band\b",
                             fold(remote)) if p.strip()}
    a = {p.strip() for p in a if p.strip()}
    b = {p.strip() for p in b if p.strip()}
    if not a or not b:
        return 0.0
    if a & b:
        # Any shared credited artist is strong evidence; scale by overlap so a
        # full match still outranks a partial one.
        return 0.85 + 0.15 * (len(a & b) / max(len(a), len(b)))
    return ratio(" ".join(sorted(a)), " ".join(sorted(b)))


def duration_score(local_s: float | None, remote_s: float | None) -> float:
    if local_s is None or remote_s is None:
        return 0.0
    delta = abs(float(local_s) - float(remote_s))
    if delta <= EXACT_S:
        return 1.0
    if delta <= CLOSE_S:
        return 0.75
    if delta <= 15.0:
        return 0.3
    return 0.0


def score_candidate(local: dict[str, Any], remote: dict[str, Any],
                    by_isrc: bool = False) -> dict[str, Any]:
    """Rank one provider row against the local measurement.

    `local` carries duration_s / title / artist as mtx measured and read them.
    `remote` carries the same three from the provider, any of them optional.
    """
    d = duration_score(local.get("duration_s"), remote.get("duration_s"))
    t = title_score(local.get("title", ""), remote.get("title", ""))
    a = artist_score(local.get("artist", ""), remote.get("artist", ""))

    delta = None
    if local.get("duration_s") is not None and remote.get("duration_s") is not None:
        delta = round(float(remote["duration_s"]) - float(local["duration_s"]), 3)

    # Duration carries the most weight because it is the only one of the three
    # that mtx knows exactly rather than reads from a tag someone typed.
    have = []
    if remote.get("duration_s") is not None:
        have.append((d, 0.55))
    if remote.get("title"):
        have.append((t, 0.30))
    if remote.get("artist"):
        have.append((a, 0.15))
    total_weight = sum(w for _s, w in have)
    score = sum(s * w for s, w in have) / total_weight if total_weight else 0.0

    # An ISRC match is a claim by the rights holder that this is the same
    # recording, so it lifts a candidate that agrees on everything it exposed.
    if by_isrc:
        score = min(1.0, 0.6 + 0.4 * score)

    notes = []
    if delta is not None and abs(delta) > CLOSE_S:
        notes.append(f"duration differs by {delta:+.1f} s")
    if remote.get("title") and t < 0.8:
        notes.append(f"title differs: {remote.get('title')!r}")
    if remote.get("artist") and a < 0.5:
        notes.append(f"artist differs: {remote.get('artist')!r}")

    return {"score": round(score, 4), "duration_delta_s": delta,
            "title_score": round(t, 4), "artist_score": round(a, 4),
            "duration_score": round(d, 4), "matched_by": "isrc" if by_isrc else "search",
            "notes": notes}


def _album_rank(local_album: Any, release_titles: Any) -> int:
    """0 when this candidate appears on the album the file says it is from.

    Databases carry duplicate recording entries -- one for the album cut, one
    somebody created while cataloguing a compilation -- and an ISRC returns
    both.  They agree on title, artist and duration, so nothing in the score
    separates them, but only one of them appears on the record the file was
    ripped from.
    """
    want = fold(local_album or "")
    titles = [fold(t) for t in (release_titles or []) if t]
    if not want or not titles:
        return 1
    return 0 if any(want == t or want in t or t in want for t in titles) else 1


def best(local: dict[str, Any], candidates: list[dict[str, Any]],
         by_isrc: bool = False, floor: float = 0.5
         ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Pick the highest-scoring candidate above `floor`.

    Returns (winner, all_scored).  The full scored list is kept so the written
    record shows what was rejected and why -- a lookup that silently picked one
    of three plausible rows would be impossible to audit later.

    Ties are broken explicitly rather than by API order.  Two rows that agree
    on duration and title score identically, and the older one is the original
    release: the newer duplicate is a compilation entry, a re-upload or a
    mis-credited stub.  Leaving that to `sort` stability means the answer
    depends on the order a database chose to serialise its rows in, which is
    not a property of the record.
    """
    scored = []
    for cand in candidates:
        s = score_candidate(local, cand, by_isrc=by_isrc)
        row = dict(cand)
        row["match"] = s
        scored.append(row)
    scored.sort(key=lambda r: (-r["match"]["score"],
                               _album_rank(local.get("album"),
                                           r.get("release_titles")),
                               str(r.get("first_release_date") or "9999"),
                               bool(r.get("video")),
                               str(r.get("id") or "")))
    if scored and scored[0]["match"]["score"] >= floor:
        return scored[0], scored
    return None, scored
