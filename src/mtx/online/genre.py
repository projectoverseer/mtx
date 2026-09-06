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
    "2 step": "2-step", "2step": "2-step", "g funk": "g-funk",
    "singer song writer": "singer-songwriter",
    "singer & songwriter": "singer-songwriter",
    "singer and songwriter": "singer-songwriter",
    "break up": "breakup", "trapsoul": "trap soul",
    "children s music": "children's music", "childrens music": "children's music",
    "r b": "r&b", "discoth que": "discothèque",
    # Latin-script is the vocabulary this table speaks; a Cyrillic vote for
    # the same genre is the same vote and has to land on the same option.
    "\u043f\u043e\u043f": "pop", "\u0440\u043e\u043a": "rock",
    "\u043c\u0435\u0436\u0434\u0443\u043d\u0430\u0440\u043e\u0434\u043d\u0430\u044f "
    "\u043f\u043e\u043f \u043c\u0443\u0437\u044b\u043a\u0430": "pop",
    "kpop": "k-pop", "k pop": "k-pop", "j-pop": "j-pop", "jpop": "j-pop",
    "j pop": "j-pop", "lo fi": "lo-fi", "nu disco": "nu-disco",
    "post punk": "post-punk", "post rock": "post-rock",
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
    # Bare numbers of any length -- years, decades, track positions, and the
    # 13-digit barcode somebody pasted into a tag field.
    r"^\d+$|^\d{2,4}s$|"
    # Durations and dates.
    r"^\d+[:.]\d+$|^\d{4}-\d{2}|"
    # Ranges and counts: "1-4 wochen", "5+ wochen" are shelf labels in any
    # language, and a number with a unit is never a description of a sound.
    r"^\d+\s*[-\u2013]\s*\d+\b|^\d+\s*\+|\bwochen?\b|\bmonate?\b|"
    # Review sites, star ratings, and a listener's own filing system.
    r"charts?\b|\bbest of\b|\bsession\d|\btrack\d|\bvol\.? ?\d|"
    r"\.(de|com|net|org|co\.uk)\b|^ph[ _]|\bstars?\b|^my |^i |"
    # `^top \d` missed "eurohit top 40": a chart is a chart wherever the words
    # naming it happen to sit in the string.
    r"\balbums?\b|\bcheck out\b|\bradio\b|^under \d|\btop \d|\bhit(s)? \d|"
    # Nothing but punctuation: "<3" and friends.
    r"^[^\w\s]+$",
    re.IGNORECASE)


def normalise(name: str) -> str:
    """Lowercase, tidy separators, apply the alias table.  Empty if junk."""
    if not name:
        return ""
    n = str(name).strip().lower()
    n = n.replace("’", "'").replace("_", " ")
    n = re.sub(r"\s*/\s*", "/", n)
    n = re.sub(r"\s+", " ", n).strip(" -/&")
    n = ALIAS.get(n, n)
    # Hyphen and space are the same word in tag data: `neo-soul` and `neo soul`
    # are one genre voted for twice, and left alone they become two options in
    # every categorical filter built from this.  Fold to the space form and let
    # ALIAS decide which spelling is canonical, so the table stays the single
    # authority and no rule here has to guess which genres keep their hyphen.
    spaced = re.sub(r"\s*-\s*", " ", n)
    n = ALIAS.get(spaced, spaced)
    # `Alternative & Indie` style pairs survive as the first half once the
    # alias table has had its chance at the whole string.
    # Two letters minimum.  "<3" and "3:45" survive every pattern above --
    # the digit is a word character, so a punctuation test does not catch them
    # -- and neither is the name of a sound.
    if n in JUNK or len(n) < 2 or len(re.findall(r"[^\W\d_]", n)) < 2:
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


def _squash(name: str) -> str:
    """`Billie Eilish` and `billieeilish` are one string once squashed."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# Discogs files everything under fifteen top-level buckets, several of which
# are lists wearing a single label: `Folk, World, & Country` is not a genre
# any record is in, it is three of them stapled together for a shop's browse
# menu.  Notion rejects a comma in a select option outright, so the push
# silently swapped it for a semicolon -- which kept the table working and left
# `folk; world; & country` sitting in the filter menu as a category matching
# 38 tracks and describing none of them.
# The ampersand must have whitespace on both sides.  A bucket is written
# `Folk, World, & Country`; a genre is written `R&B`, and splitting on a bare
# `&` turned `Contemporary R&B` into `contemporary r` on 283 tracks -- a
# category that reads like a real one and is a fragment of a word.
_BUCKET = re.compile(r"\s*,\s*|\s+&\s+|\s+/\s+")


def split_bucket(name: str) -> list[str]:
    """One label per genre, so a browse-menu bucket votes for its parts.

    Only splits on the separators a bucket is built from, and only when what
    falls out is more than one word long -- `Drum & Bass` and `Rock & Roll`
    are single genres that happen to contain an ampersand, and splitting them
    would invent `drum`, `bass` and `roll`.
    """
    text = str(name or "").strip()
    if not text:
        return []
    if _squash(text) in _INDIVISIBLE:
        return [text]
    parts = [p.strip(" -/&") for p in _BUCKET.split(text)]
    parts = [p for p in parts if len(p) > 2]
    return parts or [text]


# Genres whose own name contains a separator.  Kept as a list rather than a
# rule because there is no rule: `Funk / Soul` is Discogs' bucket for two
# genres and `Drum & Bass` is one genre, and only knowing the music tells
# them apart.
# Spelt the way `_squash` leaves them: it strips the ampersand rather than
# expanding it, so `Drum & Bass` arrives here as `drumbass`.
_INDIVISIBLE = {
    "drumbass", "drumandbass", "drumnbass",
    "rockroll", "rockandroll",
    "rhythmblues", "rhythmandblues",
    "bluesrock", "folkrock", "poprock", "jazzfunk", "souljazz",
}


def collect(by_source: dict[str, Iterable[Any]], top: int = 12,
            exclude: Iterable[str] | None = None) -> dict[str, Any]:
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
    # Listeners tag a record with the artist who made it, and a proper name in
    # a genre list is a category matching one artist that describes nothing.
    # Compared squashed, so "billieeilish" is caught as well as "Billie Eilish".
    banned = {_squash(x) for x in (exclude or []) if x}
    banned.discard("")

    for source, items in by_source.items():
        votes = _votes(items)
        if not votes:
            continue
        raw[source] = [{"name": n, "votes": v} for n, v in votes]
        peak_vote = max((v for _n, v in votes), default=0.0) or 1.0
        trust = SOURCE_WEIGHT.get(source, 0.5)
        for whole, vote in votes:
          for name in split_bucket(whole):
            # The tag noise filter belongs here too.  It only ever ran on the
            # descriptive tags, so a shelf label that `umbrella()` failed to
            # classify fell through into the genre list instead -- which is
            # how "best of 2026" became a genre.
            if _squash(name) in banned or TAG_NOISE.search(name):
                continue
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


def collect_tags(by_source: dict[str, Iterable[Any]], top: int = 20,
                 exclude: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Descriptive tags that are not genres -- `dark`, `nocturnal`, `party`.

    Kept separate from the genre vote because they answer a different question,
    and because a mood word scoring alongside `electropop` would make the
    ranked genre list useless for filtering.
    """
    scores: dict[str, float] = {}
    sources: dict[str, list[str]] = {}
    # Listeners tag a record with the artist who made it.  True, and useless as
    # a description -- and it drops a proper name into a mood vocabulary, where
    # it then looks like every other value in the column.
    banned = {_squash(x) for x in (exclude or []) if x}
    banned.discard("")
    for source, items in by_source.items():
        votes = _votes(items)
        peak_vote = max((v for _n, v in votes), default=0.0) or 1.0
        for name, vote in votes:
            if (umbrella(name) or TAG_NOISE.search(name)
                    or _squash(name) in banned):
                continue  # a genre, a shelf label, or the artist's own name
            scores[name] = scores.get(name, 0.0) + min(vote / peak_vote, 1.0)
            sources.setdefault(name, [])
            if source not in sources[name]:
                sources[name].append(source)
    return [{"name": n, "score": round(s, 5), "sources": sources[n]}
            for n, s in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))][:top]
