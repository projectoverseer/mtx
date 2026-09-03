"""The checking step: refuse to publish a corpus that is quietly wrong.

    python tools/audit.py <corpus root>                 # on-disk corpus
    python tools/audit.py <corpus root> --notion        # and the live tables
    python tools/audit.py <corpus root> --json out.json

Every defect this file checks for was found the hard way, in data that had
already been published and looked fine:

  * an Olivia Dean track credited to an unrelated artist called `OLIVIA`,
    because an ISRC returned two recordings, both scored exactly 1.00, and the
    wrong one was listed first;
  * `Scar Tissue` dated from a German bootleg compilation, because `"1999"`
    sorts before `"1999-06-08"`;
  * 264 distinct artist values for 55 artists, because the tag was the key;
  * `best of 2016` filed as a genre.

None of those raise an exception.  A pipeline with no audit stage reports
success on all of them, which is the failure mode that matters: the corpus is
the evidence base, and evidence that is confidently wrong is worse than
evidence that is missing.

Severities
----------
`error`   the value is wrong, and analysis built on it will be wrong.
`warn`    the value is suspect, or a field that should be there is not.
`info`    a property of the corpus worth knowing before drawing conclusions.

Exit code is 1 when any `error` fired, so this can gate a push.  Nothing here
writes to the corpus: an audit that repairs things cannot be trusted to report
honestly on the next run.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import unicodedata
from typing import Any, Callable, Iterable

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "notion"))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import identity                                        # noqa: E402

AUDIT_VERSION = "1.0.0"

# A match below this is a coin flip dressed as a fact.
MATCH_FLOOR = 0.75
# Two seconds is the band `match.py` calls exact; past five something differs.
DURATION_DRIFT_S = 5.0
# A release date more than a year from the file's own tag is a different record.
DATE_DRIFT_YEARS = 1
# One artist past this share and every corpus-wide pattern is about them.
DOMINANCE_PCT = 12.0

REPACKAGE = {"Compilation", "Live", "Remix", "DJ-mix", "Mixtape/Street", "Demo"}


# --------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------


def log(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


def squash(text: Any) -> str:
    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", stripped.lower())


def dig(doc: Any, path: str, default: Any = None) -> Any:
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur if cur is not None else default


def _days_between(a: str, b: str) -> int | None:
    try:
        import datetime                                # noqa: PLC0415
        return abs((datetime.date.fromisoformat(b)
                    - datetime.date.fromisoformat(a)).days)
    except (ValueError, TypeError):
        return None


def year_of(value: Any) -> int | None:
    text = str(value or "").strip()[:4]
    return int(text) if text.isdigit() else None


class Finding:
    """One defect, with enough context to act on it without re-deriving it."""

    def __init__(self, check: str, severity: str, summary: str,
                 fix: str) -> None:
        self.check = check
        self.severity = severity
        self.summary = summary
        self.fix = fix
        self.hits: list[dict[str, Any]] = []

    def hit(self, where: str, **detail: Any) -> None:
        self.hits.append({"where": where, **detail})

    def as_dict(self) -> dict[str, Any]:
        return {"check": self.check, "severity": self.severity,
                "summary": self.summary, "fix": self.fix,
                "count": len(self.hits), "examples": self.hits[:12]}


class Report:
    def __init__(self) -> None:
        self.findings: list[Finding] = []
        self.facts: dict[str, Any] = {}

    def check(self, name: str, severity: str, summary: str, fix: str) -> Finding:
        f = Finding(name, severity, summary, fix)
        self.findings.append(f)
        return f

    def fact(self, key: str, value: Any) -> None:
        self.facts[key] = value

    def failed(self) -> list[Finding]:
        return [f for f in self.findings if f.hits]

    def errors(self) -> list[Finding]:
        return [f for f in self.failed() if f.severity == "error"]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_corpus(root: str) -> list[dict[str, Any]]:
    """One light record per analysed folder.  Deliberately not the full doc.

    `analysis.json` is 4,000 features and the audit needs about thirty of
    them; loading all of it 1,300 times to check a date is how a check that
    should take four seconds takes four minutes, and a check nobody runs
    catches nothing.
    """
    tracks: list[dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith(".") and d != "stems")
        if "analysis.json" not in filenames:
            continue
        dirnames[:] = []
        rel = os.path.relpath(dirpath, root).replace("\\", "/")
        parts = rel.split("/")
        rec: dict[str, Any] = {
            "folder": dirpath,
            "rel": rel,
            "catalogue": parts[0] if parts and parts[0] not in (".", "..") else "",
            "album_folder": parts[1] if len(parts) > 2 else "",
            "online": {},
        }
        try:
            with open(os.path.join(dirpath, "mtx_source.json"), encoding="utf-8") as fh:
                rec["source"] = json.load(fh)
        except (OSError, ValueError):
            rec["source"] = {}
        online_path = os.path.join(dirpath, "online.json")
        if os.path.isfile(online_path):
            try:
                with open(online_path, encoding="utf-8") as fh:
                    rec["online"] = json.load(fh)
            except (OSError, ValueError):
                rec["online_unreadable"] = True
        rec["has_online"] = bool(rec["online"])
        tracks.append(rec)
    return tracks


def headline(rec: dict[str, Any]) -> dict[str, Any]:
    """The measured headline, from the two small files rather than the big one.

    `analysis.json` is 3 MB per track: parsing 1,321 of them to read a
    loudness costs four gigabytes and four minutes.  `corpus_row.json` is
    1.2 kB and carries every number this file judges, and `mtx_source.json`
    carries the run that produced it.  An audit nobody runs because it is slow
    catches nothing.
    """
    if "_row" in rec:
        return rec["_row"]
    row: dict[str, Any] = {}
    path = os.path.join(rec["folder"], "corpus_row.json")
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                row = json.load(fh)
        except (OSError, ValueError):
            row = {}
    rec["_row"] = row
    return row


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------


def check_identity(rep: Report, root: str, tracks: list[dict[str, Any]]) -> None:
    resolved = identity.load(root)

    unresolved = rep.check(
        "identity.unresolved", "warn",
        "a catalogue folder no database could put a name and an MBID to",
        "run tools/identity.py; if it stays unresolved the artist is genuinely "
        "absent from MusicBrainz and the folder name is the best key there is")
    stale = rep.check(
        "identity.stale", "error",
        "artists.json is missing or does not cover every folder, so the "
        "Artist column will fall back to whatever the folder was named",
        "python tools/identity.py <root>")
    collision = rep.check(
        "identity.collision", "error",
        "two folders resolve to one artist, which would silently merge two "
        "catalogues in every within-artist comparison",
        "merge the folders on disk and re-scan, or correct one folder name")
    mismatch = rep.check(
        "identity.credit_mismatch", "error",
        "the database credits this recording to somebody who is neither the "
        "catalogue artist nor anyone in the file's own artist tag",
        "the ISRC matched a mis-entered row; check online.json's candidates, "
        "and if MusicBrainz is wrong, fix it there -- the corpus follows it")

    folders = {t["catalogue"] for t in tracks if t["catalogue"]}
    for folder in sorted(folders):
        entry = resolved.get(folder)
        if not entry:
            stale.hit(folder)
            continue
        if entry.get("source") == "folder":
            unresolved.hit(folder, note=entry.get("note"))
    for key, group in (identity.build(root)["collisions"] if resolved else {}).items():
        collision.hit(", ".join(group), resolves_to=key)

    for t in tracks:
        mb = t["online"].get("musicbrainz") or {}
        credited = [str((a or {}).get("name") or "") for a in (mb.get("artists") or [])]
        if not credited:
            continue
        # A feature or a side project is a legitimate different credit, so the
        # test is deliberately weak: *nothing* in common with either the folder
        # or the tag on the file.
        known = {squash(t["catalogue"])}
        entry = resolved.get(t["catalogue"]) or {}
        known.add(squash(entry.get("name")))
        for name in re.split(r"[;/,&]| feat\.? | ft\.? ",
                             str(dig(t["online"], "query.artist") or "")):
            if name.strip():
                known.add(squash(name))
        known.discard("")
        if not known:
            continue
        hit = any(k and (k in squash(c) or squash(c) in k)
                  for c in credited for k in known)
        if not hit:
            mismatch.hit(t["rel"], credited=credited,
                         tag=dig(t["online"], "query.artist"),
                         score=dig(t["online"], "musicbrainz.match.score"))


def check_release(rep: Report, tracks: list[dict[str, Any]]) -> None:
    repack = rep.check(
        "release.repackage_source", "warn",
        "the release this track was dated from is a compilation, bootleg or "
        "remix package rather than the record it came off",
        "re-run enrichment: the release picker prefers an official, "
        "non-repackaged release group that matches the album tag")
    drift = rep.check(
        "release.date_conflict", "warn",
        "the resolved release date is more than a year from the date in the "
        "file's own tags",
        "usually a reissue matched instead of the original; check "
        "online.cross_checks.release_date.sources")
    nodate = rep.check(
        "release.no_date", "warn",
        "no release date from any source, so this track sits in no era cohort",
        "supply it in declared.json, or accept that the track is undateable")
    truncated = rep.check(
        "release.truncated", "info",
        "more than 100 releases carry this recording, so the packaging "
        "history was cut; the date and the single flag are still sound",
        "nothing to do -- recorded so a count of release groups is not read "
        "as complete")
    bootleg = rep.check(
        "release.bootleg", "error",
        "the chosen release is a bootleg, which carries a real date about as "
        "often as it carries a real title",
        "re-run enrichment; the picker now ranks Official above Bootleg")

    for t in tracks:
        if not t["has_online"]:
            continue
        mb = t["online"].get("musicbrainz") or {}
        if not mb.get("available"):
            continue
        rg = mb.get("release_group") or {}
        rel = mb.get("release") or {}
        album = squash(dig(t["online"], "query.album"))
        titles = {squash(rg.get("title")), squash(rel.get("title"))}
        if REPACKAGE & set(rg.get("secondary_types") or []):
            if not (album and album in titles):
                repack.hit(t["rel"], release=rg.get("title"),
                           types=rg.get("secondary_types"),
                           album_tag=dig(t["online"], "query.album"))
        if rel.get("status") == "Bootleg":
            bootleg.hit(t["rel"], release=rel.get("title"))
        if mb.get("releases_truncated_at"):
            truncated.hit(t["rel"], total=mb.get("releases_total"))

        checks = dig(t["online"], "cross_checks.release_date", {}) or {}
        package = checks.get("consensus") or checks.get("earliest")
        tag = dig(t["online"], "query.date")
        if not package:
            nodate.hit(t["rel"])
        else:
            # Against the *package* date, not the song's.  A 2015 soundtrack
            # legitimately carries a 2003 recording, and flagging that as a
            # conflict buries the real ones under 90 correct rows.
            a, b = year_of(package), year_of(tag)
            if a and b and abs(a - b) > DATE_DRIFT_YEARS:
                drift.hit(t["rel"], package=package, file_tag=tag,
                          song=checks.get("song_first_release"))


def check_match(rep: Report, tracks: list[dict[str, Any]]) -> None:
    missing = rep.check(
        "match.none", "warn",
        "no MusicBrainz match at all: no credits, no genre vote, no release "
        "date, and no cross-check on the tempo",
        "usually a track too new or too obscure to be catalogued; add it to "
        "MusicBrainz, or supply the facts in declared.json")
    weak = rep.check(
        "match.low_score", "warn",
        f"the winning candidate scored below {MATCH_FLOOR}, so every online "
        "field on this track is a guess",
        "check online.json's candidate list; a wrong ISRC in the file's tags "
        "is the usual cause")
    drift = rep.check(
        "match.duration_drift", "warn",
        f"the matched recording differs from the file by more than "
        f"{DURATION_DRIFT_S:.0f}s, which usually means a different edit",
        "an edit, a radio cut or a remaster with a different fade; the "
        "credits are probably right and the release may not be")
    noonline = rep.check(
        "coverage.not_enriched", "error",
        "analysed but never enriched, so it reaches Notion with no identity, "
        "no genre and no popularity",
        "python tools/enrich_fast.py <root> -j 8 --providers all")
    unreadable = rep.check(
        "coverage.online_unreadable", "error",
        "online.json exists but will not parse -- almost always a run killed "
        "part-way through a write",
        "delete the file and re-enrich that folder")

    stale = rep.check(
        "observation.stale_stamp", "warn",
        "the popularity figures were read more than a week before this "
        "enrichment ran, so an observation logged now would carry an old "
        "number under today's date",
        "re-enrich with --refresh to take a genuinely fresh reading; without "
        "it the Observations log gains a row that only looks new")

    for t in tracks:
        online = t.get("online") or {}
        read, ran = (online.get("popularity_observed_utc"),
                     online.get("queried_utc"))
        if read and ran and str(read)[:10] != str(ran)[:10]:
            days = _days_between(str(read)[:10], str(ran)[:10])
            if days is not None and days > 7:
                stale.hit(t["rel"], read=str(read)[:10], enriched=str(ran)[:10],
                          days=days)

    for t in tracks:
        if t.get("online_unreadable"):
            unreadable.hit(t["rel"])
            continue
        if not t["has_online"]:
            noonline.hit(t["rel"])
            continue
        mb = t["online"].get("musicbrainz") or {}
        if not mb.get("available"):
            missing.hit(t["rel"], errors=(mb.get("errors") or [])[:2])
            continue
        m = mb.get("match") or {}
        if (m.get("score") or 0) < MATCH_FLOOR:
            weak.hit(t["rel"], score=m.get("score"), notes=m.get("notes"))
        delta = m.get("duration_delta_s")
        if isinstance(delta, (int, float)) and abs(delta) > DURATION_DRIFT_S:
            drift.hit(t["rel"], delta_s=delta,
                      matched=(mb.get("recording") or {}).get("title"))


def check_coverage(rep: Report, tracks: list[dict[str, Any]]) -> None:
    fields: list[tuple[str, str, str, Callable[[dict], bool]]] = [
        ("coverage.no_genre", "no voted genre, so this track joins no genre "
         "cohort and answers no 'what should my house track sound like'",
         "genres come from MusicBrainz, Deezer and Last.fm; a track none of "
         "them know needs declared.json",
         lambda t: not dig(t["online"], "genres.primary")),
        ("coverage.no_playcount", "no Last.fm playcount, so this track has no "
         "outcome variable and drops out of every popularity comparison",
         "set LASTFM_API_KEY and re-enrich; multi-artist query strings are "
         "the usual cause of a miss",
         lambda t: dig(t["online"], "popularity.lastfm_playcount") is None),
        ("coverage.no_credits", "no producer, engineer or writer credits",
         "MusicBrainz credits are editor-entered and thin on recent pop; "
         "Discogs fills some of it in",
         lambda t: not (t["online"].get("credits") or {})),
        ("coverage.no_isrc", "no ISRC, so the strongest identity key is "
         "missing and matching falls back to title and duration",
         "the file's tags never carried one; nothing to fix locally",
         lambda t: not dig(t["online"], "identity.isrc")),
        ("coverage.no_discogs", "no Discogs release, so the label, catalogue "
         "number and pressing credits are unavailable",
         "set DISCOGS_TOKEN; digital-only releases are often genuinely absent",
         lambda t: not dig(t["online"], "identity.discogs_release_id")),
    ]
    checks = [(rep.check(name, "info", summary, fix), test)
              for name, summary, fix, test in fields]
    for t in tracks:
        if not t["has_online"]:
            continue
        for finding, test in checks:
            try:
                if test(t):
                    finding.hit(t["rel"])
            except Exception:                       # a check must never abort
                finding.hit(t["rel"], note="check raised")


def check_vocabulary(rep: Report, tracks: list[dict[str, Any]]) -> None:
    """Categorical values, judged by what a machine can group on.

    A categorical column exists to be grouped by.  Two spellings of one value
    are two groups, so every near-duplicate is a silent halving of a cohort.
    """
    case = rep.check(
        "vocab.case_collision", "error",
        "two spellings of one value differing only in case or punctuation; "
        "Notion stores the first casing it sees and folds the rest onto it, "
        "so the column ends up displaying a name nothing agrees with",
        "extend ALIAS in src/mtx/online/genre.py, re-enrich, then re-push "
        "with --prune-options")
    comma = rep.check(
        "vocab.comma", "warn",
        "a value containing a comma, which Notion rejects outright in a "
        "select or multi-select option name",
        "the push replaces it with a semicolon; the exact value stays in the "
        "rich_text twin of the column")
    junk = rep.check(
        "vocab.not_a_genre", "error",
        "a shelf label, a chart position or a year filed as a genre",
        "extend TAG_NOISE in src/mtx/online/genre.py and re-enrich")

    seen: dict[str, dict[str, set[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(set))
    noise = re.compile(
        r"^\d+$|^\d{2,4}s$|^\d{4}-|\bbest of\b|\bcharts?\b|\btop \d|"
        r"\bvol\.? ?\d|\bwochen?\b|\bplaylist\b|\bfavou?rites?\b", re.I)
    for t in tracks:
        values = {
            "genre": [g.get("name") for g in
                      (dig(t["online"], "genres.ranked", []) or [])],
            "tag": list(dig(t["online"], "descriptive_tags", []) or []),
        }
        for kind, names in values.items():
            for name in names:
                if not isinstance(name, str) or not name.strip():
                    continue
                seen[kind][squash(name)].add(name)
                if "," in name:
                    comma.hit(t["rel"], kind=kind, value=name)
                if kind == "genre" and noise.search(name):
                    junk.hit(t["rel"], value=name)
    for kind, groups in seen.items():
        for key, spellings in sorted(groups.items()):
            if len(spellings) > 1:
                case.hit(f"{kind}:{key}", spellings=sorted(spellings))


def check_measurement(rep: Report, tracks: list[dict[str, Any]]) -> None:
    """Sanity on the audio itself: a corrupt file measures, it does not fail."""
    silent = rep.check(
        "audio.near_silent", "error",
        "integrated loudness below -30 LUFS: an empty file, a lead-in, or a "
        "bounce that never got its master bus",
        "check the source file; it is almost certainly not the record")
    hot = rep.check(
        "audio.impossible_loudness", "error",
        "integrated loudness above -3 LUFS, which no released master reaches",
        "check the source file")
    short = rep.check(
        "audio.not_a_song", "info",
        "under 60 seconds: an interlude, a skit or a score cue, which drags "
        "every duration and structure statistic it is averaged into",
        "keep it if you want it, but exclude it from cohorts -- `mtx cohort` "
        "and outcome.py both read this list")
    stale = rep.check(
        "analysis.stale_schema", "warn",
        "analysed by an older mtx than the newest in the corpus, so newer "
        "fields are absent and cohort percentiles mix two definitions",
        "re-scan those folders with `mtx scan --force`")
    partial = rep.check(
        "analysis.quick_profile", "warn",
        "measured with --profile quick, so the full battery never ran",
        "re-scan with the default full profile")
    nostems = rep.check(
        "analysis.no_stems", "info",
        "no stem separation, so there is no vocal-versus-instrumental "
        "masking, no per-stem loudness and no melody surface",
        "re-scan with --stems; it is the expensive half of a scan")

    schema_seen: collections.Counter = collections.Counter()
    tool_seen: collections.Counter = collections.Counter()
    for t in tracks:
        run = (t.get("source") or {}).get("run") or {}
        if run:
            schema_seen[run.get("schema_version")] += 1
            tool_seen[run.get("tool_version")] += 1
            if run.get("profile") and run["profile"] != "full":
                partial.hit(t["rel"], profile=run["profile"])
            if run.get("stems") is False:
                nostems.hit(t["rel"])
        row = headline(t)
        lufs = row.get("LUFS-I")
        if isinstance(lufs, (int, float)):
            if lufs < -30:
                silent.hit(t["rel"], lufs_i=round(lufs, 2))
            elif lufs > -3:
                hot.hit(t["rel"], lufs_i=round(lufs, 2))
        # mtx measured this duration exactly; it is echoed into the enrichment
        # query, which is the cheap place to read it back from.
        dur = dig(t["online"], "query.duration_s")
        if isinstance(dur, (int, float)) and dur < 60:
            short.hit(t["rel"], duration_s=round(dur, 1))

    newest = max((k for k in schema_seen if k), default=None)
    if newest and len(schema_seen) > 1:
        for t in tracks:
            got = ((t.get("source") or {}).get("run") or {}).get("schema_version")
            if got and got != newest:
                stale.hit(t["rel"], schema=got, newest=newest)
    rep.fact("analysis_schema_versions", dict(schema_seen))
    rep.fact("analysis_tool_versions", dict(tool_seen))


def check_deep(rep: Report, tracks: list[dict[str, Any]]) -> None:
    """Checks that need the full analysis, so they cost minutes not seconds.

    Off by default and worth running weekly.  The defect this exists for is the
    quietest kind there is: a field that is populated, labelled with a source,
    and holding the wrong thing.
    """
    credit = rep.check(
        "lyrics.credit_not_lyric", "error",
        "the lyric field holds a songwriter credit rather than a lyric -- one "
        "line, a handful of words, `source: file:tag`.  A substring match on "
        "the tag key meant `composerlyricist` was read as a lyric, and Apple "
        "puts that key on most commercial files",
        "the matcher is fixed, but these analyses predate the fix: re-scan "
        "with `mtx scan --force`, or transcribe, or declare the text")
    empty = rep.check(
        "lyrics.absent", "info",
        "no lyric from any source, so nothing about the writing is measurable "
        "on this track: no rhyme scheme, no repetition, no delivery rate",
        "run the pipeline with --transcribe, or paste the sheet into "
        "declared.json")
    warned = rep.check(
        "analysis.warnings", "info",
        "the analysis recorded a warning about itself",
        "read the warnings block; most are benign, none are invented")

    for t in tracks:
        path = os.path.join(t["folder"], "analysis.json")
        try:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        for warning in (doc.get("warnings") or [])[:1]:
            warned.hit(t["rel"], warning=str(warning)[:120])
        lyrics = doc.get("lyrics") or {}
        if not lyrics.get("available"):
            empty.hit(t["rel"])
            continue
        stats = lyrics.get("statistics") or {}
        lines = (stats.get("lines") or {}).get("count")
        chars = (stats.get("characters") or {}).get("count")
        if (lyrics.get("source") == "file:tag"
                and isinstance(lines, int) and lines <= 2
                and isinstance(chars, int) and chars < 200):
            credit.hit(t["rel"], lines=lines, characters=chars)


def check_hygiene(rep: Report, tracks: list[dict[str, Any]]) -> None:
    """Properties of the corpus as a sample, not of any one row."""
    dominance = rep.check(
        "corpus.artist_dominance", "info",
        f"one artist is more than {DOMINANCE_PCT:.0f}% of the corpus, so an "
        "unconditioned corpus-wide pattern is a pattern about them",
        "keep buying breadth; until then condition on artist, which is what "
        "the outcome z-score already does")
    dupe = rep.check(
        "corpus.duplicate_recording", "warn",
        "one recording appears in more than one folder, so it is counted "
        "twice in every corpus-wide statistic",
        "outcome.py marks duplicates; keep the primary and exclude the rest, "
        "or delete the redundant copy")
    thin = rep.check(
        "corpus.thin_catalogue", "info",
        "fewer than 5 tracks for this artist, too few for a within-artist "
        "z-score to mean anything",
        "buy more of that artist, or accept that they contribute only to "
        "genre-level cohorts")

    per_artist = collections.Counter(t["catalogue"] for t in tracks)
    total = sum(per_artist.values()) or 1
    for artist, count in per_artist.most_common():
        share = 100.0 * count / total
        if share > DOMINANCE_PCT:
            dominance.hit(artist, tracks=count, share_pct=round(share, 1))
        if count < 5:
            thin.hit(artist, tracks=count)

    by_recording: dict[str, list[str]] = collections.defaultdict(list)
    for t in tracks:
        mbid = dig(t["online"], "identity.recording_mbid")
        if mbid:
            by_recording[mbid].append(t["rel"])
    for mbid, where in by_recording.items():
        if len(where) > 1:
            dupe.hit(where[0], mbid=mbid, copies=where)

    rep.fact("tracks", len(tracks))
    rep.fact("artists", len(per_artist))
    rep.fact("enriched", sum(1 for t in tracks if t["has_online"]))
    rep.fact("largest_catalogue_share_pct",
             round(100.0 * max(per_artist.values()) / total, 1) if per_artist else 0)


# --------------------------------------------------------------------------
# the Notion side
# --------------------------------------------------------------------------


def check_notion(rep: Report, root: str) -> None:
    """What actually landed, as opposed to what was sent.

    A push reports success per page.  It cannot see that two pages disagree
    about an artist's name, or that a select option nothing uses any more is
    still cluttering the filter menu.
    """
    from client import Notion, NotionError          # noqa: PLC0415

    state_path = os.path.join(root, ".notion_state.json")
    try:
        with open(state_path, encoding="utf-8") as fh:
            state = json.load(fh)
    except (OSError, ValueError):
        log(f"  no Notion state at {state_path}; skipping the live checks")
        return
    dbs = state.get("databases") or {}
    if not dbs.get("tracks"):
        return

    orphan = rep.check(
        "notion.orphan_option", "warn",
        "a select option no row uses, left behind by a corrected value; "
        "Notion never garbage-collects these and they clutter every filter",
        "python tools/notion/push.py <root> --prune-options")
    collide = rep.check(
        "notion.case_collision", "error",
        "two option names differing only by case; Notion matches options "
        "case-insensitively and keeps the first casing forever, so one of "
        "these is displaying under the other's spelling",
        "delete the wrong option from the database schema, then re-push")
    drift = rep.check(
        "notion.artist_drift", "error",
        "the Artist values in the Observations log are not the same set as "
        "in the Corpus, so a join between the two tables silently loses rows",
        "re-push both tables from the same artists.json")
    missing = rep.check(
        "notion.missing_rows", "error",
        "the corpus on disk has tracks the database does not",
        "python tools/notion/push.py <root>")

    api = Notion(log=lambda m: None)
    try:
        tracks_db = api.request("GET", f"/databases/{dbs['tracks']}")
    except NotionError as exc:
        log(f"  Notion unreachable: {exc}")
        return

    used: dict[str, set[str]] = collections.defaultdict(set)
    pages = api.query(dbs["tracks"])
    for page in pages:
        for name, prop in (page.get("properties") or {}).items():
            if prop.get("type") == "select" and prop.get("select"):
                used[name].add(prop["select"]["name"])
            elif prop.get("type") == "multi_select":
                for opt in prop.get("multi_select") or []:
                    used[name].add(opt["name"])
    for name, prop in (tracks_db.get("properties") or {}).items():
        kind = prop.get("type")
        if kind not in ("select", "multi_select"):
            continue
        options = [o["name"] for o in prop[kind].get("options", [])]
        for opt in options:
            if opt not in used.get(name, set()):
                orphan.hit(f"{name}: {opt}")
        by_case: dict[str, list[str]] = collections.defaultdict(list)
        for opt in options:
            by_case[opt.casefold()].append(opt)
        for key, group in by_case.items():
            if len(group) > 1:
                collide.hit(f"{name}: {group}")

    rep.fact("notion_rows", len(pages))
    on_disk = {t["rel"] for t in load_corpus(root)}
    if len(pages) < len(on_disk):
        missing.hit(f"{len(on_disk) - len(pages)} track(s) not in Notion",
                    on_disk=len(on_disk), in_notion=len(pages))

    if dbs.get("observations"):
        obs_artists = set()
        for page in api.query(dbs["observations"]):
            sel = ((page.get("properties") or {}).get("Artist") or {}).get("select")
            if sel:
                obs_artists.add(sel["name"])
        corpus_artists = used.get("Artist", set())
        for name in sorted(obs_artists ^ corpus_artists):
            drift.hit(name,
                      in_corpus=name in corpus_artists,
                      in_observations=name in obs_artists)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def run(root: str, notion: bool = False, deep: bool = False) -> Report:
    rep = Report()
    tracks = load_corpus(root)
    if not tracks:
        raise ValueError(f"no analysed folder under {root}")
    check_identity(rep, root, tracks)
    check_match(rep, tracks)
    check_release(rep, tracks)
    check_coverage(rep, tracks)
    check_vocabulary(rep, tracks)
    check_measurement(rep, tracks)
    check_hygiene(rep, tracks)
    if deep:
        check_deep(rep, tracks)
    if notion:
        check_notion(rep, root)
    return rep


ORDER = {"error": 0, "warn": 1, "info": 2}
MARK = {"error": "FAIL", "warn": "WARN", "info": "note"}


def render(rep: Report, verbose: bool = False) -> str:
    lines = ["", "corpus", "------"]
    for key, value in rep.facts.items():
        lines.append(f"  {key:32s} {value}")
    lines += ["", "checks", "------"]
    clean = [f for f in rep.findings if not f.hits]
    for f in sorted(rep.failed(), key=lambda f: (ORDER[f.severity], f.check)):
        lines.append(f"  {MARK[f.severity]}  {f.check}  ({f.count if False else len(f.hits)})")
        lines.append(f"        {f.summary}")
        for hit in f.hits[:5 if not verbose else 40]:
            detail = ", ".join(f"{k}={v!r}" for k, v in hit.items() if k != "where")
            lines.append(f"          - {hit['where']}" + (f"  [{detail}]" if detail else ""))
        if len(f.hits) > (5 if not verbose else 40):
            lines.append(f"          ... {len(f.hits) - (5 if not verbose else 40)} more")
        lines.append(f"        fix: {f.fix}")
        lines.append("")
    lines.append(f"  {len(clean)} check(s) clean: "
                 + ", ".join(sorted(f.check for f in clean)))
    errors = rep.errors()
    lines += ["", f"{len(errors)} error(s), "
              f"{len([f for f in rep.failed() if f.severity == 'warn'])} warning(s)"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument("--deep", action="store_true",
                    help="also read every analysis.json: catches a lyric field "
                         "holding a songwriter credit.  Minutes, not seconds")
    ap.add_argument("--notion", action="store_true",
                    help="also check the live database this corpus pushes to")
    ap.add_argument("--json", metavar="PATH",
                    help="write the full report (default <root>/audit.json)")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--warn-is-error", action="store_true",
                    help="exit non-zero on warnings too")
    args = ap.parse_args()

    try:
        rep = run(args.root, notion=args.notion, deep=args.deep)
    except ValueError as exc:
        log(f"error: {exc}")
        return 2
    log(render(rep, verbose=args.verbose))

    path = args.json or os.path.join(args.root, "audit.json")
    doc = {"audit_version": AUDIT_VERSION, "root": os.path.abspath(args.root),
           "facts": rep.facts,
           "findings": [f.as_dict() for f in rep.findings]}
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    log(f"  wrote {path}")

    if rep.errors():
        return 1
    if args.warn_is_error and rep.failed():
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
