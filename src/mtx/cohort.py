"""`mtx cohort`: where a track sits among comparable records.

`-7.77 LUFS` and `tilt -4.79 dB/oct` mean nothing on their own.  A consumer has
no way to ask where a value sits among comparable records, and that is the
feature that makes an unfinished mix legible.

**This is deliberately not part of `analyze`.**  A per-track measurement must
not depend on what else happens to be in the folder; that would break
reproducibility, which is property one.  So this is a corpus-level command that
reads a directory of analyses and writes a *separate* file of relative
positions.  The absolute numbers are never touched.

Cohorts are defined by genre and year.  For the published reference records
those labels come from `enrich`, which is online; for an unreleased candidate
they are **declared**, because an unreleased track cannot look them up.  That
asymmetry is the whole point: a mix in progress can be positioned against the
released records it is competing with, provided you state what it should be
compared to.

The corpus hygiene report is part of the output rather than a footnote.  With
n=64 and 193 columns, any pattern found is a pattern about whichever artist
dominates the folder, and the command says so in the file it writes.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np

from .params import PARAMS
from .split import load_analysis

# The metrics a cohort position is computed for.  Deliberately a fixed,
# readable list rather than every column: a percentile is only useful where the
# quantity is comparable between records.
METRICS: tuple[tuple[str, str], ...] = (
    ("headline.lufs_i", "Integrated loudness (LUFS)"),
    ("headline.lra_lu", "Loudness range (LU)"),
    ("headline.true_peak_dbtp_16x", "True peak (dBTP)"),
    ("headline.plr_db", "PLR (dB)"),
    ("headline.psr_min_db", "PSR minimum (dB)"),
    ("headline.psr_median_db", "PSR median (dB)"),
    ("headline.dr14", "DR14"),
    ("headline.crest_whole_db", "Crest, whole file (dB)"),
    ("headline.crest_loudest_10s_db", "Crest, loudest 10 s (dB)"),
    ("headline.spectral_tilt_db_per_oct", "Spectral tilt (dB/oct)"),
    ("headline.air_band_pct", "Air band (%)"),
    ("headline.sub_band_pct", "Sub band (%)"),
    ("headline.side_minus_mid_db", "Side minus mid (dB)"),
    ("headline.side_minus_mid_below_120hz_db", "Side minus mid below 120 Hz (dB)"),
    ("headline.correlation_mean", "Mean correlation"),
    ("headline.hf_cutoff_hz", "HF cutoff (Hz)"),
    ("headline.tempo_bpm", "Tempo (BPM)"),
    ("headline.duration_s", "Duration (s)"),
    ("headline.section_count", "Section count"),
    ("stereo.mono_sum_damage.broadband_loss_db", "Mono-sum loss (dB)"),
    ("dynamics.flat_top.total_flat_samples", "Flat-top samples"),
    ("processing.pumping.depth_db", "Pumping depth (dB)"),
    ("harmony.harmonic_rhythm.changes_per_bar", "Chord changes per bar"),
    ("harmony.vocabulary.distinct_chords", "Distinct chords"),
    ("harmony.degrees.diatonic_time_pct", "Diatonic time (%)"),
    ("rhythm.grid.deviation.std_ms", "Grid deviation std (ms)"),
    ("rhythm.syncopation.mean_per_bar", "Syncopation per bar"),
    ("rhythm.swing.offbeat_position_median", "Off-beat position"),
    ("form.chorus_share_pct", "Chorus share (%)"),
    ("form.time_to_first_chorus_s", "Time to first chorus (s)"),
    ("form.time_to_vocal_entry_s", "Time to vocal entry (s)"),
    ("stems.melody.vocals.range.p5_p95_semitones", "Vocal range p5-p95 (semitones)"),
    ("stems.melody.vocals.notes_per_second_of_voicing", "Notes per second"),
    ("stems.arrangement.density.mean", "Mean concurrent sources"),
    ("lyrics.statistics.type_token_ratio", "Lyric type-token ratio"),
    ("lyrics.statistics.compression_ratio", "Lyric compression ratio"),
)


def dig(res: dict[str, Any], path: str) -> Any:
    node: Any = res
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    if isinstance(node, bool):
        return None
    if isinstance(node, (int, float)) and math.isfinite(float(node)):
        return float(node)
    return None


def _sidecar(folder: str, name: str) -> dict[str, Any]:
    p = os.path.join(folder, name)
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _year(value: Any) -> int | None:
    if value is None:
        return None
    s = str(value)
    for i in range(len(s) - 3):
        chunk = s[i:i + 4]
        if chunk.isdigit() and 1900 <= int(chunk) <= 2100:
            return int(chunk)
    return None


def labels_for(res: dict[str, Any], folder: str,
               catalogue: str | None = None) -> dict[str, Any]:
    """Genre and year for one track, with where each came from.

    Declared beats online beats the file tag, and the origin travels with the
    value: a cohort built from tags is a different object from one built from a
    voted genre, and the reader has to be able to tell.
    """
    tags = (res.get("tags") or {}).get("named") or {}
    online = _sidecar(folder, "online.json")
    declared = _sidecar(folder, "declared.json")
    o = online.get("online") if isinstance(online.get("online"), dict) else online

    genre = source = None
    d_cohort = declared.get("cohort") if isinstance(declared.get("cohort"), dict) else {}
    if d_cohort.get("genre") or declared.get("genre"):
        genre, source = (d_cohort.get("genre") or declared.get("genre")), "declared"
    elif isinstance(o, dict):
        # `online.json` writes the vote under `genres`, plural.  Reading
        # `genre` found nothing on every enriched track in the corpus, so
        # every cohort silently fell through to the shop's own genre tag --
        # which is the string the vote exists to replace.
        voted = o.get("genres") if isinstance(o.get("genres"), dict) else {}
        if voted.get("umbrella") or voted.get("primary"):
            genre, source = (voted.get("umbrella") or voted.get("primary")), "online"
    if genre is None and tags.get("genre"):
        genre, source = tags["genre"], "file:tag"

    year = year_source = None
    if d_cohort.get("year") or declared.get("release_year"):
        year = _year(d_cohort.get("year") or declared.get("release_year"))
        year_source = "declared"
    if year is None and isinstance(o, dict):
        # Same defect as the genre: `release` is nested under the provider that
        # said it, and the resolved answer is the cross-check.  Ordered from
        # "when the song came out" to "when this pressing came out", because an
        # era cohort is about the song.
        mb = o.get("musicbrainz") if isinstance(o.get("musicbrainz"), dict) else {}
        checks = o.get("cross_checks") if isinstance(o.get("cross_checks"), dict) else {}
        rel = mb.get("release") if isinstance(mb.get("release"), dict) else {}
        group = mb.get("release_group") if isinstance(mb.get("release_group"), dict) else {}
        year = _year((checks.get("release_date") or {}).get("earliest")
                     or mb.get("first_release_date")
                     or group.get("first_release_date")
                     or rel.get("date"))
        if year is not None:
            year_source = "online"
    if year is None:
        year = _year(tags.get("date"))
        if year is not None:
            year_source = "file:tag"

    # The library folder, when the caller knows it.  The hygiene report counts
    # distinct artists, and on the tag "Tyler, The Creator" and "Tyler, The
    # Creator / Daniel Caesar" count as two -- which understates dominance,
    # the one thing the report exists to state.
    artist = (declared.get("artist") or catalogue or tags.get("artist")
              or tags.get("albumartist") or "(unknown artist)")
    title = declared.get("title") or tags.get("title") or os.path.basename(folder)
    return {"artist": str(artist), "title": str(title),
            "genre": (str(genre).strip().lower() if genre else None),
            "genre_source": source, "year": year, "year_source": year_source,
            "genres": _voted_genres(o, genre, d_cohort, declared)}


def _voted_genres(online: Any, primary: Any, d_cohort: dict,
                  declared: dict) -> list[str]:
    """Every genre this record credibly belongs to, not only the winner.

    The vote already ranks them with a confidence, and a record is genuinely
    several things at once: filed under its winner alone a club track lands in
    `electronic` and never in `house`, and `house` is the cohort somebody
    mixing a club track is asking about.
    """
    P = PARAMS["cohort"]
    out: list[str] = []
    for value in (primary, d_cohort.get("genre"), declared.get("genre")):
        if value and str(value).strip().lower() not in out:
            out.append(str(value).strip().lower())
    for extra in (d_cohort.get("genres") or declared.get("genres") or []):
        if str(extra).strip().lower() not in out:
            out.append(str(extra).strip().lower())
    if isinstance(online, dict):
        voted = online.get("genres") if isinstance(online.get("genres"), dict) else {}
        for entry in (voted.get("ranked") or []):
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip().lower()
            confidence = entry.get("confidence")
            if (name and name not in out
                    and isinstance(confidence, (int, float))
                    and confidence >= float(P["secondary_genre_confidence"])):
                out.append(name)
    return out[:int(P["max_genres_per_track"])]


def _percentile_of(value: float, pool: list[float]) -> float | None:
    if not pool:
        return None
    below = sum(1 for v in pool if v < value)
    equal = sum(1 for v in pool if v == value)
    return 100.0 * (below + 0.5 * equal) / len(pool)


def _z(value: float, pool: list[float]) -> float | None:
    if len(pool) < 2:
        return None
    sd = float(np.std(pool))
    return (value - float(np.mean(pool))) / sd if sd > 0 else None


def _hygiene(rows: list[dict[str, Any]], cohorts: dict[str, list[int]]
             ) -> dict[str, Any]:
    artists: dict[str, int] = {}
    for r in rows:
        artists[r["artist"]] = artists.get(r["artist"], 0) + 1
    n = len(rows)
    top = sorted(artists.items(), key=lambda kv: -kv[1])
    biggest = (top[0][1] / n) if (top and n) else None
    P = PARAMS["cohort"]
    problems: list[str] = []
    if n < P["min_corpus_for_statistics"]:
        problems.append(
            f"{n} tracks is below the {P['min_corpus_for_statistics']} this "
            "command treats as a usable corpus; percentiles over a set this "
            "small describe the set, not the music")
    if biggest is not None and biggest > P["max_single_artist_share"]:
        problems.append(
            f"{top[0][0]} is {100 * biggest:.0f}% of the corpus (limit "
            f"{100 * P['max_single_artist_share']:.0f}%); any pattern found is "
            "a pattern about that artist, and cohorts need artist stratification")
    small = {k: len(v) for k, v in cohorts.items()
             if len(v) < P["min_cohort_size"] and k != "all"}
    if small:
        problems.append(
            f"{len(small)} cohort(s) hold fewer than {P['min_cohort_size']} "
            "tracks; their percentiles are reported with the count next to them")
    return {
        "tracks": n,
        "distinct_artists": len(artists),
        "largest_artist": top[0][0] if top else None,
        "largest_artist_share": biggest,
        "artist_counts": dict(top[:40]),
        "cohort_sizes": {k: len(v) for k, v in sorted(cohorts.items())},
        "problems": problems,
        "clean": not problems,
        "note": "a reference corpus is not a corpus because it is large; it is "
                "a corpus because no one artist or era dominates it",
    }


def _neighbours(rows: list[dict[str, Any]], zmat: np.ndarray,
                embeddings: list[np.ndarray | None], k: int) -> None:
    """Nearest neighbours, from embeddings where they exist and z-space where not."""
    have_emb = [i for i, e in enumerate(embeddings) if e is not None]
    if len(have_emb) >= 2 and len({embeddings[i].size for i in have_emb}) == 1:
        M = np.vstack([embeddings[i] for i in have_emb])
        M = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
        sim = M @ M.T
        np.fill_diagonal(sim, -np.inf)
        for pos, i in enumerate(have_emb):
            order = np.argsort(sim[pos])[::-1][:k]
            rows[i]["neighbours"] = {
                "basis": "embedding cosine",
                "list": [{"artist": rows[have_emb[j]]["artist"],
                          "title": rows[have_emb[j]]["title"],
                          "similarity": float(sim[pos, j])} for j in order],
            }
    finite = np.where(np.isfinite(zmat), zmat, 0.0)
    valid = np.isfinite(zmat)
    for i in range(len(rows)):
        if "neighbours" in rows[i]:
            continue
        both = valid & valid[i]
        counts = both.sum(axis=1)
        d = np.sqrt(np.sum(((finite - finite[i]) ** 2) * both, axis=1)
                    / np.maximum(counts, 1))
        d[i] = np.inf
        d[counts < 5] = np.inf
        order = np.argsort(d)[:k]
        rows[i]["neighbours"] = {
            "basis": "mean per-metric z-space distance over shared metrics",
            "list": [{"artist": rows[j]["artist"], "title": rows[j]["title"],
                      "distance": float(d[j])} for j in order
                     if np.isfinite(d[j])],
        }


def build(root: str, neighbours: int = 5, log=None) -> dict[str, Any]:
    """Read a folder of analyses and compute every track's relative position."""
    from .export import find_analyses

    paths = find_analyses(root)
    if not paths:
        raise ValueError(f"no analysis.json found under {root}")
    rows: list[dict[str, Any]] = []
    values: list[dict[str, float | None]] = []
    embeddings: list[np.ndarray | None] = []
    for p in paths:
        try:
            res = load_analysis(p)
        except (OSError, ValueError) as exc:
            if log:
                log(f"  skipped {p}: {exc}")
            continue
        folder = os.path.dirname(os.path.abspath(p))
        rel_parts = os.path.relpath(folder, os.path.abspath(root)).split(os.sep)
        catalogue = rel_parts[0] if rel_parts[0] not in (".", "..") else None
        lab = labels_for(res, folder, catalogue)
        lab["folder"] = os.path.basename(folder)
        lab["analysis_path"] = os.path.abspath(p)
        lab["sha256"] = (res.get("file") or {}).get("sha256")
        rows.append(lab)
        values.append({key: dig(res, key) for key, _ in METRICS})
        emb = (res.get("embedding") or {})
        vec = emb.get("vector") if emb.get("available") else None
        embeddings.append(np.asarray(vec, dtype=float) if vec else None)
        if log:
            log(f"  {lab['artist']} - {lab['title']}")
    if not rows:
        raise ValueError("no readable analyses")

    P = PARAMS["cohort"]
    win = int(P["year_window"])
    cohorts: dict[str, list[int]] = {"all": list(range(len(rows)))}
    for i, r in enumerate(rows):
        window = (f"year={r['year'] - win}-{r['year'] + win}"
                  if r["year"] is not None else None)
        names = r.get("genres") or ([r["genre"]] if r["genre"] else [])
        # Most specific first, and within a tier the record's own strongest
        # genre first, so the ladder below falls back in a defensible order
        # rather than in whatever order the keys happened to be appended.
        keys = [f"genre={n}|{window}" for n in names if window]
        keys += [f"genre={n}" for n in names]
        if window:
            keys.append(window)
        r["cohort_keys"] = keys
    # Year cohorts are windows, so membership is by overlap, not by equality.
    for i, r in enumerate(rows):
        for key in r["cohort_keys"]:
            members = cohorts.setdefault(key, [])
            if key.startswith("year=") or "|year=" in key:
                continue
            members.append(i)
    for i, r in enumerate(rows):
        if r["year"] is None:
            continue
        for key in r["cohort_keys"]:
            if not (key.startswith("year=") or "|year=" in key):
                continue
            members = cohorts.setdefault(key, [])
            genre = None
            if "|" in key:
                genre = key.split("|", 1)[0].split("=", 1)[1]
            lo, hi = key.rsplit("=", 1)[1].split("-")
            for j, other in enumerate(rows):
                if other["year"] is None or not (int(lo) <= other["year"] <= int(hi)):
                    continue
                # Membership is "voted for this genre", not "won it": the same
                # rule the un-windowed keys use, or a track would appear in
                # `genre=house` and vanish from `genre=house|year=2022-2026`.
                if genre is not None and genre not in (
                        other.get("genres") or [other.get("genre")]):
                    continue
                if j not in members:
                    members.append(j)

    keys = [k for k, _ in METRICS]
    zmat = np.full((len(rows), len(keys)), np.nan)
    floor = int(P["min_cohort_size"])
    for i, r in enumerate(rows):
        r["metrics"] = {}
        # Most specific first, then fall back until the cohort is big enough to
        # say anything.  A track alone in its own genre-and-year cohort would
        # otherwise have every percentile and every z-score come back null,
        # which is the one case where the answer matters most: a new mix is
        # exactly the track nothing else in the folder shares a label with.
        ladder = list(r["cohort_keys"]) + ["all"]
        primary = "all"
        for cand in ladder:
            if len(cohorts.get(cand, [])) >= floor:
                primary = cand
                break
        else:
            # Nothing cleared the floor: take the most specific non-trivial one.
            for cand in ladder:
                if len(cohorts.get(cand, [])) >= 2:
                    primary = cand
                    break
        r["primary_cohort"] = primary
        r["primary_cohort_size"] = len(cohorts.get(primary, cohorts["all"]))
        # `cohort_keys` is ordered most specific first, so the ideal answer is
        # the head of the list.  Anything else means the ladder fell back and
        # the percentile is against a broader pool than the labels asked for.
        r["primary_cohort_is_fallback"] = bool(
            r["cohort_keys"] and primary != r["cohort_keys"][0])
        r["cohort_choice_rule"] = (
            f"the most specific cohort holding at least {floor} tracks, falling "
            "back to a broader one and finally to the whole corpus")
        same_artist = [j for j, o in enumerate(rows) if o["artist"] == r["artist"]]
        for m, (key, label) in enumerate(METRICS):
            v = values[i][key]
            if v is None:
                r["metrics"][key] = {"value": None}
                continue
            entry: dict[str, Any] = {"value": v, "label": label}
            for cohort_name, idxs in (("corpus", cohorts["all"]),
                                      ("cohort", cohorts.get(primary, cohorts["all"])),
                                      ("artist", same_artist)):
                pool = [values[j][key] for j in idxs if values[j][key] is not None]
                entry[f"{cohort_name}_n"] = len(pool)
                entry[f"{cohort_name}_percentile"] = _percentile_of(v, pool)
                entry[f"{cohort_name}_z"] = _z(v, pool)
                if cohort_name == "cohort" and pool:
                    entry["cohort_median"] = float(np.median(pool))
            r["metrics"][key] = entry
            z = entry.get("cohort_z")
            if z is not None:
                zmat[i, m] = z
        finite = zmat[i][np.isfinite(zmat[i])]
        r["typicality"] = {
            "mean_abs_z": float(np.mean(np.abs(finite))) if finite.size else None,
            "metrics_used": int(finite.size),
            "definition": "mean absolute z-score against the primary cohort. "
                          "A distance, not a judgement: a low value means this "
                          "record sits where the cohort sits, which is neither "
                          "good nor bad.",
        }
        r["distance_to_cohort_centroid"] = (
            float(np.sqrt(np.mean(finite ** 2))) if finite.size else None)

    if neighbours > 0:
        _neighbours(rows, zmat, embeddings, neighbours)

    return {
        "cohort_schema_version": "1.0.0",
        "root": os.path.abspath(root),
        "metrics": [{"key": k, "label": l} for k, l in METRICS],
        "params": PARAMS["cohort"],
        "cohorts": {k: sorted(v) for k, v in sorted(cohorts.items())},
        "hygiene": _hygiene(rows, cohorts),
        "tracks": rows,
        "rule": "this file is written beside the analyses and never into them: "
                "a per-track measurement must not depend on what else is in "
                "the folder",
    }


def render(doc: dict[str, Any], top: int = 12) -> str:
    """A short human-readable summary of the cohort file."""
    h = doc["hygiene"]
    out = ["# mtx cohort", "",
           f"root: {doc['root']}",
           f"tracks: {h['tracks']}  artists: {h['distinct_artists']}  "
           f"cohorts: {len(doc['cohorts'])}", ""]
    out.append("## CORPUS HYGIENE" + "")
    out.append("")
    if h["problems"]:
        for p in h["problems"]:
            out.append(f"- {p}")
    else:
        out.append("- no problems found")
    out.append(f"- largest artist: {h['largest_artist']} "
               f"({100 * (h['largest_artist_share'] or 0):.0f}% of the corpus)")
    out.append("")
    out.append("## MOST AND LEAST TYPICAL")
    out.append("")
    rows = [r for r in doc["tracks"]
            if (r.get("typicality") or {}).get("mean_abs_z") is not None]
    rows.sort(key=lambda r: r["typicality"]["mean_abs_z"])
    missing = len(doc["tracks"]) - len(rows)
    if missing:
        out.append(f"_{missing} track(s) had no cohort with enough members to "
                   "position them against; see primary_cohort_size._")
        out.append("")
    out.append("| mean |z| | artist | title | cohort | n |")
    out.append("| --- | --- | --- | --- | --- |")
    half = max(1, top // 2)
    shown = rows if len(rows) <= top else rows[:half] + rows[-half:]
    for r in shown:
        out.append(f"| {r['typicality']['mean_abs_z']:.2f} | {r['artist']} | "
                   f"{r['title']} | {r['primary_cohort']} | "
                   f"{r['primary_cohort_size']} |")
    out.append("")
    out.append("_mean |z| is a distance from the cohort centre, not a rating. "
               "mtx measures; it does not score._")
    out.append("")
    return "\n".join(out)
