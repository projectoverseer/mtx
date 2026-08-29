"""Turning several databases' disagreeing genre labels into one ranked list.

The genre tag embedded in a purchased file is a shop's shelf label: the corpus
this was built against carries `Pop` thirteen times, plus `Miscellaneous`,
`Film Soundtracks` and `Alternative & Indie` -- categories chosen to fill a
store menu, not to describe a record.  Thirty of sixty-four files carry no
genre at all.

The public databases are better, and they are better in different ways, so the
useful move is not to pick one but to collect all of them with their votes
attached.  MusicBrainz returns community-counted genres at three levels of
specificity; Last.fm returns listener tags; Discogs separates coarse genre
from fine style; Deezer and Apple each return one shop category.

Two deliberate limits:

  * Normalisation only repairs spelling.  `hip-hop` and `Hip-Hop/Rap` become
    `hip hop`, but `alternative pop` is never folded into `pop` -- the
    granularity is the entire value of going online in the first place.
  * The coarse bucket is offered alongside the ranked list, never instead of
    it, so a query can filter on `pop` and still read `avant-garde pop`.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

# How much a source's opinion counts.  A genre attached to this exact
# recording beats one attached to everything the artist ever released.
SOURCE_WEIGHT = {
    "musicbrainz:recording": 1.00,
    "lastfm:track": 0.90,
    "musicbrainz:release-group": 0.80,
    "discogs:style": 0.80,
    "lastfm:album": 0.55,
    "deezer:album": 0.60,
    "itunes:track": 0.60,
    "discogs:genre": 0.55,
    "musicbrainz:artist": 0.50,
    "lastfm:artist": 0.45,
    "file:tag": 0.40,
}

# Spelling variants only.  Anything that would merge two genres a listener can
# tell apart belongs in UMBRELLA, not here.
ALIAS = {
    "hip-hop": "hip hop", "hiphop": "hip hop", "hip hop/rap": "hip hop",
    "hip-hop/rap": "hip hop", "rap/hip hop": "hip hop", "rap/hip-hop": "hip hop",
    "hip hop rap": "hip hop", "hip-hop & rap": "hip hop",
    "rnb": "r&b", "r'n'b": "r&b", "r n b": "r&b", "r and b": "r&b",
    "rhythm and blues": "r&b", "rhythm & blues": "r&b", "r&b/soul": "r&b",
    "r&b / soul": "r&b", "randb": "r&b",
    "singer/songwriter": "singer-songwriter", "singer songwriter": "singer-songwriter",
    "drum & bass": "drum and bass", "drum'n'bass": "drum and bass",
    "dnb": "drum and bass", "d&b": "drum and bass", "drum n bass": "drum and bass",
    "synthpop": "synth-pop", "synth pop": "synth-pop",
    "dance pop": "dance-pop", "electro pop": "electropop",
    "trip-hop": "trip hop", "lofi": "lo-fi", "lo fi": "lo-fi",
    "nu disco": "nu-disco", "post punk": "post-punk", "post rock": "post-rock",
    "kpop": "k-pop", "j-pop": "j-pop", "jpop": "j-pop",
    "film soundtracks": "soundtrack", "film soundtrack": "soundtrack",
    "soundtracks": "soundtrack", "original score": "score",
    "alternative/indie": "alternative", "alternative & indie": "alternative",
    "indie/alternative": "alternative", "alt": "alternative",
    "electronica/dance": "electronic", "dance/electronic": "electronic",
    "electronic/dance": "electronic",
    "world music": "world", "hard rock/metal": "metal",
    "contemporary rnb": "contemporary r&b",
}

# Labels that name a shelf rather than a sound.  Dropped outright.
JUNK = {"miscellaneous", "misc", "other", "unknown", "music", "general",
        "various", "untagged", "genre", "none", "n/a", "audio", "seen live",
        "favorites", "favourite", "favourites", "albums i own", "spotify"}

# Coarse buckets, tried in order.  English genre compounds are head-final --
# `alternative pop` is a pop record, `pop rock` is a rock record -- so the
# rules are ordered by how strongly a word claims the head, not alphabetically.
UMBRELLA = [
    (r"\b(hip ?hop|rap|trap|drill|grime|boom bap|crunk)\b", "hip hop"),
    (r"\b(reggaeton|salsa|bachata|cumbia|latin|bossa nova|samba|tango)\b", "latin"),
    (r"\b(reggae|dancehall|ska)\b", "reggae"),
    (r"\bmetal\b", "metal"),
    (r"(house|techno|trance|dubstep|drum and bass|breakbeat|jungle|hardstyle|"
     r"eurodance|synthwave|vaporwave|\bidm\b|\bedm\b|\bambient\b|\bgarage\b|"
     r"\bdowntempo\b|\btrip hop\b|\bglitch\b|\bdub\b|\bfuture bass\b|"
     r"\belectronic\b|\belectronica\b|\bbig beat\b|\bdisco\b|wave\b)", "electronic"),
    (r"\b(jazz|bebop|swing|big band|ragtime|dixieland)\b", "jazz"),
    (r"\b(classical|orchestral|opera|baroque|chamber|symphon\w*)\b", "classical"),
    (r"\b(country|americana|bluegrass|honky tonk)\b", "country"),
    (r"\bblues\b", "blues"),
    (r"\b(gospel|worship|ccm)\b", "gospel"),
    (r"\b(soundtrack|score|musical)\b", "soundtrack"),
    (r"\b(r&b|soul|funk|motown|new jack swing)\b", "r&b/soul"),
    (r"\b(punk|grunge|rock|shoegaze|emo|hardcore|britpop)\b", "rock"),
    (r"\b(folk|singer-songwriter|acoustic)\b", "folk"),
    # No leading boundary: `electropop` and `dream pop` are both pop records,
    # and only one of them spells it as a separate word.
    (r"pop\b", "pop"),
    (r"\bdance\b", "electronic"),
    (r"\b(alternative|indie)\b", "alternative"),
    (r"\bworld\b", "world"),
]

# Tag noise: a listener's private filing system rather than a description of
# the record.  Bare years, review-site handles and star ratings all show up in
# Last.fm and MusicBrainz tag lists.
TAG_NOISE = re.compile(
    r"^\d{4}$|^\d{2}s$|charts?\b|"
    r"\.(de|com|net|org|co\.uk)\b|^ph[ _]|\bstars?\b|^my |^i |\balbums?\b|"
    r"\bcheck out\b|\bradio\b|^under \d|^top \d", re.IGNORECASE)


def normalise(name: str) -> str:
    """Lowercase, tidy separators, apply the alias table.  Empty if junk."""
    if not name:
        return ""
    n = str(name).strip().lower()
    n = n.replace("’", "'").replace("_", " ")
    n = re.sub(r"\s*/\s*", "/", n)
    n = re.sub(r"\s+", " ", n).strip(" -/&")
    n = ALIAS.get(n, n)
    # `Alternative & Indie` style pairs survive as the first half once the
    # alias table has had its chance at the whole string.
    if n in JUNK or len(n) < 2:
        return ""
    return n


def umbrella(name: str) -> str:
    """The coarse bucket a normalised genre falls in, or "" if none fits.

    Two passes, because English genre compounds are head-final: the first only
    accepts a rule that matches at the end of the name, so `ambient pop` is
    filed under pop rather than under electronic on the strength of its
    modifier.  The unanchored pass then catches names whose head is a word no
    rule knows.
    """
    for pattern, bucket in UMBRELLA:
        if re.search(pattern + r"\s*$", name):
            return bucket
    for pattern, bucket in UMBRELLA:
        if re.search(pattern, name):
            return bucket
    return ""


def _votes(items: Iterable[Any]) -> list[tuple[str, float]]:
    """(name, count) pairs from either bare strings or {name, count} dicts."""
    out = []
    for item in items or []:
        if isinstance(item, dict):
            name = item.get("name") or item.get("genre") or ""
            count = item.get("count")
        else:
            name, count = str(item), None
        name = normalise(name)
        if not name:
            continue
        try:
            weight = float(count) if count is not None else 1.0
        except (TypeError, ValueError):
            weight = 1.0
        out.append((name, max(weight, 0.0)))
    return out


def collect(by_source: dict[str, Iterable[Any]], top: int = 12) -> dict[str, Any]:
    """Merge every source's labels into one ranked, sourced list.

    Within a source the votes are scaled against that source's own top vote,
    not against their sum.  Sharing the total instead would punish exactly the
    sources worth having: MusicBrainz spreads nine genres over a record, so
    each would land near a ninth, while a shop that returns the single word
    `Alternative` would collect its full weight and win.  Scaling to the
    source's maximum lets each provider cast one full-strength vote for its
    own top pick and weaker ones behind it, so detail is no longer penalised.
    """
    scores: dict[str, float] = {}
    sources: dict[str, list[str]] = {}
    raw: dict[str, list[dict[str, Any]]] = {}

    for source, items in by_source.items():
        votes = _votes(items)
        if not votes:
            continue
        raw[source] = [{"name": n, "votes": v} for n, v in votes]
        peak_vote = max((v for _n, v in votes), default=0.0) or 1.0
        trust = SOURCE_WEIGHT.get(source, 0.5)
        for name, vote in votes:
            share = min(vote / peak_vote, 1.0)
            scores[name] = scores.get(name, 0.0) + share * trust
            sources.setdefault(name, [])
            if source not in sources[name]:
                sources[name].append(source)

    if not scores:
        return {"available": False, "primary": None, "umbrella": None,
                "ranked": [], "umbrella_ranked": [], "by_source": raw}

    peak = max(scores.values())
    ranked = [
        {"name": name,
         "score": round(score, 5),
         # Relative to the winner, so "how much weaker is the second guess"
         # reads off directly.
         "confidence": round(score / peak, 4),
         "umbrella": umbrella(name),
         "sources": sources[name]}
        for name, score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    ][:top]

    buckets: dict[str, float] = {}
    for row in ranked:
        if row["umbrella"]:
            buckets[row["umbrella"]] = buckets.get(row["umbrella"], 0.0) + row["score"]
    umbrella_ranked = [
        {"name": n, "score": round(s, 5)}
        for n, s in sorted(buckets.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    return {
        "available": True,
        "primary": ranked[0]["name"],
        "umbrella": umbrella_ranked[0]["name"] if umbrella_ranked else None,
        "ranked": ranked,
        "umbrella_ranked": umbrella_ranked,
        "agreement": round(
            len(ranked[0]["sources"]) / max(1, len(raw)), 4),
        "source_count": len(raw),
        "by_source": raw,
    }


def collect_tags(by_source: dict[str, Iterable[Any]], top: int = 20
                 ) -> list[dict[str, Any]]:
    """Descriptive tags that are not genres -- `dark`, `nocturnal`, `party`.

    Kept separate from the genre vote because they answer a different question,
    and because a mood word scoring alongside `electropop` would make the
    ranked genre list useless for filtering.
    """
    scores: dict[str, float] = {}
    sources: dict[str, list[str]] = {}
    for source, items in by_source.items():
        votes = _votes(items)
        peak_vote = max((v for _n, v in votes), default=0.0) or 1.0
        for name, vote in votes:
            if umbrella(name) or TAG_NOISE.search(name):
                continue  # a genre, or somebody's private shelf label
            scores[name] = scores.get(name, 0.0) + min(vote / peak_vote, 1.0)
            sources.setdefault(name, [])
            if source not in sources[name]:
                sources[name].append(source)
    return [{"name": n, "score": round(s, 5), "sources": sources[n]}
            for n, s in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))][:top]
