"""Tidy tabular export: `mtx export`.

`batch --csv` writes headline metrics per track, and everything richer reaches
a human only as database properties -- a browsing surface, not a dataset.  The
per-section vectors are the most valuable part of the dump and were the hardest
to get at, because they are nested three levels down inside a document that has
to be parsed per track.

This writes two flat tables over a folder of analyses:

* **track** -- one row per track, every scalar the document holds, under its
  full dotted path.
* **section** -- one row per (track, section), joining what
  `structure.sections` measured to the form label, the per-section masking
  index and the per-section pulse rate.

CSV always; parquet when `pyarrow` is installed, because a 900-column CSV is a
transport format and not a working one.
"""

from __future__ import annotations

import csv
import os
from typing import Any

from .split import load_analysis

# Bulk arrays never belong in a flat table: they are what analysis.json is for.
SKIP_KEYS = ("times_s", "beat_times_s", "downbeat_times_s", "boundaries_s",
             "per_third_octave", "notes_list", "per_bar", "vector",
             "accent_components", "per_window", "bar_labels", "words",
             "arc_per_line", "concurrent_sources", "post_offset_envelope_db",
             "features", "sections", "chords", "progression", "all",
             "ffprobe_raw", "per_section", "excerpts", "list", "instances",
             "per_stem_series")
SKIP_TOP = ("params", "coverage")
MAX_DEPTH = 6


def flatten(node: Any, prefix: str = "", depth: int = 0,
            out: dict[str, Any] | None = None) -> dict[str, Any]:
    """Every scalar in the document, keyed by its dotted path."""
    if out is None:
        out = {}
    if depth > MAX_DEPTH:
        return out
    if isinstance(node, dict):
        for key, value in node.items():
            if depth == 0 and key in SKIP_TOP:
                continue
            if any(h == str(key) or str(key).endswith(h) for h in SKIP_KEYS):
                continue
            flatten(value, f"{prefix}.{key}" if prefix else str(key), depth + 1, out)
    elif isinstance(node, (list, tuple)):
        if node and all(isinstance(v, (int, float, str, bool, type(None))) for v in node):
            out[prefix + "_count"] = len(node)
    elif isinstance(node, (int, float, str, bool)) or node is None:
        out[prefix] = node
    return out


def find_analyses(root: str) -> list[str]:
    """Every analysis.json under `root`, deepest-first stable order."""
    if os.path.isfile(root):
        return [root]
    found: list[str] = []
    for dirpath, _, names in os.walk(root):
        if "analysis.json" in names:
            found.append(os.path.join(dirpath, "analysis.json"))
    return sorted(found)


def track_row(res: dict[str, Any], path: str) -> dict[str, Any]:
    row = flatten(res)
    row["mtx.analysis_path"] = os.path.abspath(path)
    row["mtx.folder"] = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return row


def section_rows(res: dict[str, Any], path: str) -> list[dict[str, Any]]:
    """One row per section, with everything else that is indexed by section."""
    structure = res.get("structure") or {}
    sections = structure.get("sections") or []
    if not sections:
        return []
    tags = (res.get("tags") or {}).get("named") or {}
    ident = {
        "mtx.folder": os.path.basename(os.path.dirname(os.path.abspath(path))),
        "mtx.analysis_path": os.path.abspath(path),
        "file.sha256": (res.get("file") or {}).get("sha256"),
        "tags.title": tags.get("title"),
        "tags.artist": tags.get("artist"),
        "audio.duration_s": (res.get("audio") or {}).get("duration_s"),
    }
    form = res.get("form") or {}
    by_index_form = {s.get("index"): s for s in (form.get("sections") or [])}
    masking = ((res.get("stems") or {}).get("masking") or {}).get("per_section") or []
    by_index_mask = {m.get("index"): m for m in masking}
    pulse = (((res.get("rhythm") or {}).get("pulse_rate") or {}).get("per_section") or [])
    by_index_pulse = {p.get("section_index"): p for p in pulse}
    contour = (((res.get("stems") or {}).get("melody") or {}).get("vocals") or {})
    by_index_melody = {c.get("section_index"): c
                       for c in (contour.get("contour_per_section") or [])}
    arrangement = ((res.get("stems") or {}).get("arrangement") or {}).get("per_section") or []
    by_index_arr = {a.get("section_index"): a for a in arrangement}

    rows: list[dict[str, Any]] = []
    for s in sections:
        idx = s.get("index")
        row = dict(ident)
        row.update(flatten(s, "section"))
        f = by_index_form.get(idx) or {}
        for key in ("letter", "label", "label_confidence", "vocal_present"):
            row[f"form.{key}"] = f.get(key)
        m = by_index_mask.get(idx) or {}
        for key, value in (m.get("masking_index_db") or {}).items():
            row[f"masking.{key}_db"] = value
        for key in ("vocal_lufs", "instrumental_lufs", "vocal_minus_instrumental_lu"):
            if key in m:
                row[f"masking.{key}"] = m[key]
        p = by_index_pulse.get(idx) or {}
        for key in ("onsets_per_beat", "relative_pulse_rate", "beats", "onsets"):
            row[f"rhythm.{key}"] = p.get(key)
        mel = by_index_melody.get(idx) or {}
        for key in ("median_midi", "max_midi", "notes"):
            row[f"melody.{key}"] = mel.get(key)
        a = by_index_arr.get(idx) or {}
        row["arrangement.mean_concurrent_sources"] = a.get("mean_concurrent_sources")
        row["arrangement.stems_present"] = ",".join(a.get("stems_present") or []) or None
        rows.append(row)
    return rows


def _columns(rows: list[dict[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for r in rows:
        for k in r:
            seen.setdefault(k, None)
    lead = [k for k in ("mtx.folder", "tags.title", "tags.artist",
                        "section.index", "section.start_s") if k in seen]
    rest = sorted(k for k in seen if k not in lead)
    return lead + rest


def _write_csv(rows: list[dict[str, Any]], path: str) -> str:
    cols = _columns(rows)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    return path


def _write_parquet(rows: list[dict[str, Any]], path: str) -> str | None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        return None
    cols = _columns(rows)
    table = pa.table({c: [r.get(c) for r in rows] for c in cols})
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    pq.write_table(table, path)
    return path


def export(root: str, out_dir: str, level: str = "both",
           fmt: str = "csv", log=None) -> dict[str, Any]:
    """Write the track and/or section table over every analysis under `root`."""
    paths = find_analyses(root)
    if not paths:
        raise ValueError(f"no analysis.json found under {root}")
    tracks: list[dict[str, Any]] = []
    sections: list[dict[str, Any]] = []
    failed: list[str] = []
    for p in paths:
        try:
            res = load_analysis(p)
        except (OSError, ValueError) as exc:
            failed.append(f"{p}: {exc}")
            continue
        if level in ("track", "both"):
            tracks.append(track_row(res, p))
        if level in ("section", "both"):
            sections.extend(section_rows(res, p))
        if log:
            log(f"  {os.path.basename(os.path.dirname(p))}")

    written: dict[str, str] = {}
    for name, rows in (("tracks", tracks), ("sections", sections)):
        if not rows:
            continue
        base = os.path.join(out_dir, f"mtx_{name}")
        written[f"{name}.csv"] = _write_csv(rows, base + ".csv")
        if fmt in ("parquet", "both"):
            got = _write_parquet(rows, base + ".parquet")
            if got:
                written[f"{name}.parquet"] = got
            elif log:
                log("  parquet skipped: pyarrow is not installed "
                    "(pip install pyarrow)")
    return {
        "analyses_found": len(paths),
        "tracks": len(tracks),
        "section_rows": len(sections),
        "failed": failed,
        "written": written,
        "track_columns": len(_columns(tracks)) if tracks else 0,
        "section_columns": len(_columns(sections)) if sections else 0,
    }
