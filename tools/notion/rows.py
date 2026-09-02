"""Turn one analysed folder into a Notion page: properties plus body blocks.

The split is the whole design.  **Properties** are the ~150 fields you would
ever filter, sort or benchmark across tracks -- they come back with every
database query, so they have to stay a shortlist.  **Body blocks** carry
everything else at full fidelity: the complete 2,000-column flattened row, the
section table, the chord track, the confidence notes, and a pointer to the
`analysis.json` on disk that still holds the arrays none of this reaches.

So nothing is discarded.  The only thing the shortlist decides is what is
cheap to reach, and what costs one more request.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mtx.export import flatten, track_row          # noqa: E402
from mtx.split import load_analysis                # noqa: E402

from schema import (PROPERTIES, TRAIT_VERSION, dig,   # noqa: E402
                    trait_documentation, trait_states)

TEXT_LIMIT = 2000          # Notion's per-rich-text-object cap
BLOCK_LINES = 45           # lines per code block, keeps each well under the cap


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_outcomes(root: str) -> dict:
    """`{sha256: entry}` from `outcome.json`, or empty when it has not been run."""
    path = os.path.join(root, "outcome.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("tracks") or {}
    except (OSError, ValueError):
        return {}


def load_folder(folder: str, outcomes: dict | None = None) -> dict:
    """`analysis.json` with `online.json` mounted at `online.`.

    The sidecar is mounted rather than merged because that is what it is:
    `mtx analyze` promises byte-identical output for the same input, and a
    block built from whatever MusicBrainz said this morning cannot live inside
    that promise.  Mounting keeps the boundary visible in every path.
    """
    doc = load_analysis(os.path.join(folder, "analysis.json"))
    online_path = os.path.join(folder, "online.json")
    if os.path.isfile(online_path):
        with open(online_path, encoding="utf-8") as fh:
            doc["online"] = json.load(fh)
    else:
        doc["online"] = {}
    declared_path = os.path.join(folder, "declared.json")
    if os.path.isfile(declared_path):
        with open(declared_path, encoding="utf-8") as fh:
            doc.setdefault("declared", {}).update(json.load(fh))
    doc["mtx"] = {"analysis_path": os.path.abspath(folder),
                  "folder": os.path.basename(folder)}
    sha = dig(doc, "file.sha256")
    doc["outcome"] = (outcomes or {}).get(sha) or {}
    return doc


# --------------------------------------------------------------------------
# property coercion
# --------------------------------------------------------------------------


def _text(value: Any, limit: int = TEXT_LIMIT) -> str:
    return str(value)[:limit]


def _option(value: Any) -> str | None:
    """A select/multi-select option name.

    Notion rejects commas in option names outright, so they are replaced
    rather than allowed to fail the whole page.  100 characters is the cap.
    """
    text = str(value).replace(",", ";").strip()
    return text[:100] or None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None                       # NaN and infinities are not JSON
    return round(float(value), 6)


def _date(value: Any) -> dict | None:
    if not value:
        return None
    text = str(value).strip()
    if len(text) == 4 and text.isdigit():
        text = f"{text}-01-01"            # a bare year is a valid Notion date
    elif len(text) == 7:
        text = f"{text}-01"
    return {"start": text[:10] if "T" not in text else text}


def notion_value(kind: str, value: Any) -> dict | None:
    if value is None or value == "":
        return None
    if kind == "title":
        return {"title": [{"type": "text", "text": {"content": _text(value)}}]}
    if kind == "rich_text":
        return {"rich_text": [{"type": "text", "text": {"content": _text(value)}}]}
    if kind == "number":
        n = _number(value)
        return {"number": n} if n is not None else None
    if kind == "select":
        opt = _option(value)
        return {"select": {"name": opt}} if opt else None
    if kind == "multi_select":
        if not isinstance(value, list):
            value = [value]
        opts = [{"name": o} for o in
                dict.fromkeys(filter(None, (_option(v) for v in value)))]
        return {"multi_select": opts} if opts else None
    if kind == "checkbox":
        return {"checkbox": bool(value)}
    if kind == "date":
        d = _date(value)
        return {"date": d} if d else None
    if kind == "url":
        return {"url": _text(value, 2000)}
    raise ValueError(f"unknown property kind: {kind}")


def properties_for(doc: dict) -> dict:
    """The Tier-1 payload.  A field that is absent is simply omitted.

    Never write a placeholder for a missing measurement: an empty Notion
    property reads as "not measured", and a zero would read as a reading.
    """
    out: dict[str, Any] = {}
    for prop in PROPERTIES:
        try:
            raw = prop.read(doc)
        except Exception:
            raw = None
        built = notion_value(prop.kind, raw)
        if built is not None:
            out[prop.name] = built
    # A title is mandatory; Notion rejects a page without one.
    if "Title" not in out:
        out["Title"] = notion_value("title", dig(doc, "mtx.folder") or "untitled")
    return out


def database_schema() -> dict:
    """The property definitions for `POST /databases`."""
    out: dict[str, Any] = {}
    for prop in PROPERTIES:
        if prop.kind == "title":
            out[prop.name] = {"title": {}}
        elif prop.kind in ("select", "multi_select"):
            out[prop.name] = {prop.kind: {"options": []}}
        else:
            out[prop.name] = {prop.kind: {}}
    return out


# --------------------------------------------------------------------------
# body blocks -- Tier 2 and 3
# --------------------------------------------------------------------------


def _heading(text: str) -> dict:
    return {"object": "block", "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text",
                                         "text": {"content": text}}]}}


def _paragraph(text: str) -> dict:
    return {"object": "block", "type": "paragraph",
            "paragraph": {"rich_text": [{"type": "text",
                                         "text": {"content": _text(text)}}]}}


def _code_blocks(lines: list[str], language: str = "plain text") -> list[dict]:
    """One code block per `BLOCK_LINES` lines, each under the 2,000-char cap."""
    out = []
    for i in range(0, len(lines), BLOCK_LINES):
        chunk = "\n".join(lines[i:i + BLOCK_LINES])
        while chunk:
            piece, chunk = chunk[:TEXT_LIMIT], chunk[TEXT_LIMIT:]
            out.append({
                "object": "block", "type": "code",
                "code": {"language": language,
                         "rich_text": [{"type": "text",
                                        "text": {"content": piece}}]},
            })
    return out


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def full_row_blocks(doc: dict) -> list[dict]:
    """Tier 2: every column `mtx export` emits, grouped, as `path = value`."""
    row = track_row(doc, dig(doc, "mtx.analysis_path") or "")
    by_group: dict[str, list[str]] = {}
    for key in sorted(row):
        if key.startswith("mtx."):
            continue
        group = key.split(".", 1)[0]
        value = row[key]
        if value is None or value == "":
            continue
        by_group.setdefault(group, []).append(f"{key} = {_fmt(value)}")

    blocks = [_heading("FULL ROW"),
              _paragraph(f"{sum(len(v) for v in by_group.values())} populated "
                         f"columns from mtx export, grouped. Everything the "
                         f"flat table holds; arrays stay in analysis.json.")]
    for group in sorted(by_group):
        blocks.append(_paragraph(f"— {group} ({len(by_group[group])})"))
        blocks.extend(_code_blocks(by_group[group]))
    return blocks


def section_blocks(doc: dict) -> list[dict]:
    """Tier 3: the arrangement timeline, one line per section.

    Stored as a block and not a related database on purpose: 1,321 tracks x
    ~31 sections is 41,000 pages, which is hours of API calls for data you can
    only use once you have already chosen a track.
    """
    sections = dig(doc, "structure.sections", []) or []
    if not sections:
        return []
    # `form.sections` carries the functional label per section index; the
    # measured vectors live in `structure.sections` under the same index.
    labels = {s.get("index"): (s.get("label") or "")
              for s in (dig(doc, "form.sections", []) or [])
              if isinstance(s, dict)}
    masking = {r.get("index"): r
               for r in (dig(doc, "stems.masking.per_section", []) or [])
               if isinstance(r, dict)}
    vocal = dig(doc, "form.vocal_presence.present", []) or []

    head = (f"{'#':>3} {'label':<12} {'start':>8} {'len':>7} {'LUFS':>7} "
            f"{'crest':>6} {'tilt':>6} {'s/m':>7} {'onset':>6} {'voc':>4} {'v-i':>6}")
    lines = [head, "-" * len(head)]
    for i, sec in enumerate(sections):
        if not isinstance(sec, dict):
            continue
        m = masking.get(i, {})
        lines.append(
            f"{i:>3} {str(labels.get(i, '')):<12} "
            f"{_fmt(sec.get('start') or ''):>8} "
            f"{_num(sec.get('duration_s')):>7} {_num(sec.get('lufs_i')):>7} "
            f"{_num(sec.get('crest_db')):>6} {_num(sec.get('tilt_db_per_oct')):>6} "
            f"{_num(sec.get('side_minus_mid_db')):>7} "
            f"{_num(sec.get('onset_rate_per_s')):>6} "
            f"{('yes' if (i < len(vocal) and vocal[i]) else 'no'):>4} "
            f"{_num(m.get('vocal_minus_instrumental_lu')):>6}")
    return [_heading("SECTIONS"),
            _paragraph(f"{len(sections)} sections. 'v-i' is vocal minus "
                       f"instrumental in LU, from stems.masking.per_section -- "
                       f"a list, so no flat export reaches it."),
            *_code_blocks(lines)]


def _num(value: Any, width: int = 0) -> str:
    n = _number(value)
    return "-" if n is None else f"{n:.1f}"


def chord_blocks(doc: dict) -> list[dict]:
    chords = dig(doc, "harmony.chords", []) or []
    if not chords:
        return []
    lines, row = [], []
    for ch in chords:
        if not isinstance(ch, dict):
            row.append(f"{str(ch):<9}{'':<7}")
            continue
        # `slash_label` when the measured bass note is not the root: an
        # inversion is a different chord to play, so it is worth keeping.
        name = ch.get("slash_label") or ch.get("label") or "?"
        row.append(f"{str(name):<11}{str(ch.get('degree') or ''):<6}")
        if len(row) == 4:
            lines.append("".join(row))
            row = []
    if row:
        lines.append("".join(row))
    conf = dig(doc, "harmony.confidence")
    match = _number(dig(doc, "harmony.mean_template_match"))
    return [_heading("CHORDS"),
            _paragraph(f"{len(chords)} beat-synchronous segments, "
                       f"{dig(doc, 'harmony.vocabulary.distinct_chords')} distinct. "
                       f"Confidence {conf}; mean template match "
                       f"{match if match is not None else 'n/a'}. "
                       f"Chord name then scale degree."),
            *_code_blocks(lines)]


def confidence_blocks(doc: dict) -> list[dict]:
    notes = doc.get("confidence_notes") or []
    warnings = doc.get("warnings") or []
    if not notes and not warnings:
        return []
    lines = []
    for n in notes:
        if isinstance(n, dict):
            lines.append(f"[{n.get('confidence','?'):>6}] {n.get('metric','')}: "
                         f"{n.get('reason','')}")
    for w in warnings:
        lines.append(f"[  warn] {w}")
    return [_heading("CONFIDENCE"),
            _paragraph("Verbatim from the analysis, so a claim's uncertainty "
                       "travels with the claim rather than being re-derived."),
            *_code_blocks(lines)]


def pointer_blocks(doc: dict) -> list[dict]:
    traits = trait_states(doc)
    unmeasured = [k for k, v in traits.items() if v == "not-measured"]
    lines = [
        f"analysis.json   {dig(doc, 'mtx.analysis_path')}",
        f"sha256          {dig(doc, 'file.sha256')}",
        f"mtx run         mtx {dig(doc, 'run.tool_version')} / schema "
        f"{dig(doc, 'run.schema_version')} / profile {dig(doc, 'run.profile')}",
        f"stems           {dig(doc, 'run.stems_model')}",
        f"enriched        {dig(doc, 'online.queried_utc') or 'not enriched'}",
        f"providers       {', '.join(dig(doc, 'online.providers_available', []) or []) or '-'}",
        "",
        f"traits not measured on this track: "
        f"{', '.join(unmeasured) if unmeasured else 'none'}",
        "",
        trait_documentation(),
    ]
    return [_heading("POINTERS"),
            _paragraph("analysis.json holds ~4,261 fields including the arrays "
                       "no flat table reaches: LTAS, F0 contour, beat times, "
                       "the correlation timeline. Read it from disk by path."),
            *_code_blocks(lines)]


def body_blocks(doc: dict) -> list[dict]:
    return [*section_blocks(doc), *chord_blocks(doc),
            *confidence_blocks(doc), *pointer_blocks(doc),
            *full_row_blocks(doc)]


# --------------------------------------------------------------------------
# observations -- the append-only log
# --------------------------------------------------------------------------


# Track-level figures only.  `deezer_album_fans` is a property of the album
# and `lastfm_artist_listeners` of the artist, so logging either per track
# writes the same number 137 times for Drake and adds no information while
# making the log 1.7x larger.  Both stay on the track row as `Latest ...`
# caches; if artist momentum over time becomes interesting, it wants its own
# per-artist log rather than a column repeated across a catalogue.
OBSERVATION_METRICS = [
    ("deezer_rank", "online.popularity.deezer_rank"),
    ("lastfm_listeners", "online.popularity.lastfm_listeners"),
    ("lastfm_playcount", "online.popularity.lastfm_playcount"),
]


def observations_for(doc: dict) -> list[dict]:
    """One row per time-varying figure, stamped with when it was true.

    Popularity is an observation, not a property.  Heat Waves took 59 weeks to
    reach number one; a single stored scalar cannot tell that record apart from
    one that debuted at the top and fell away, because today they read the
    same.  Nor can it be recovered later: these are current-value endpoints
    with no history, so a figure not captured this month is gone.
    """
    observed = dig(doc, "online.queried_utc")
    if not observed:
        return []
    sha = dig(doc, "file.sha256") or ""
    title = dig(doc, "tags.named.title") or dig(doc, "mtx.folder") or "untitled"
    artist = dig(doc, "tags.named.artist") or ""
    out = []
    for metric, path in OBSERVATION_METRICS:
        value = _number(dig(doc, path))
        if value is None:
            continue
        out.append({
            "Observation": notion_value("title", f"{title} · {metric} · {observed[:10]}"),
            "Track sha256": notion_value("rich_text", sha),
            "Artist": notion_value("select", artist),
            "Metric": notion_value("select", metric),
            "Value": notion_value("number", value),
            "Observed at": notion_value("date", observed),
            "Source": notion_value("select", metric.split("_")[0]),
        })
    return out


OBSERVATION_SCHEMA = {
    "Observation": {"title": {}},
    "Track sha256": {"rich_text": {}},
    "Artist": {"select": {"options": []}},
    "Metric": {"select": {"options": []}},
    "Value": {"number": {}},
    "Observed at": {"date": {}},
    "Source": {"select": {"options": []}},
}
