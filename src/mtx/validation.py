"""The DR14 validation record.

DR14 is the one headline metric mtx cannot check against its own reference:
the tool ships no copyrighted audio, so the implementation is only verified
against analytically known synthetic cases.  A single track whose published DR
rating is already known converts it from corpus-relative to published --- but
that check has to be *recorded*, or every future run repeats the disclaimer.

This module owns that record.  `mtx validate-dr <file> --published N` measures
one file, stores the pair, and every later run reads the store and says how
many tracks the implementation agrees with and by how much.  Nothing here
judges: the store holds measured value, published value and the difference.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

# Agreement tolerance for the summary line.  Published DR ratings are integers
# and the reference meter rounds, so a whole-DR difference is the natural unit.
AGREEMENT_TOLERANCE_DR = 1.0

ENV_VAR = "MTX_DR14_VALIDATION"


def store_path() -> str:
    """Where the record lives.  `MTX_DR14_VALIDATION` overrides it."""
    override = os.environ.get(ENV_VAR)
    if override:
        return override
    base = os.environ.get("APPDATA") or os.path.join(
        os.path.expanduser("~"), ".config")
    return os.path.join(base, "mtx", "dr14_validation.json")


def load(path: str | None = None) -> dict[str, Any]:
    p = path or store_path()
    if not os.path.isfile(p):
        return {"entries": []}
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {"entries": [], "unreadable": p}
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        return {"entries": [], "unreadable": p}
    return data


def add_entry(entry: dict[str, Any], path: str | None = None) -> str:
    """Append one measured-vs-published pair, replacing any earlier one for the
    same file.  Returns the store path."""
    p = path or store_path()
    data = load(p)
    entries = [e for e in data["entries"]
               if e.get("sha256") != entry.get("sha256")]
    entries.append(entry)
    entries.sort(key=lambda e: (e.get("title") or "", e.get("sha256") or ""))
    os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"entries": entries,
                   "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                  f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    return p


def summary(path: str | None = None) -> dict[str, Any]:
    """What the current record says.  Shape is stable whether or not it exists."""
    p = path or store_path()
    data = load(p)
    entries = data.get("entries", [])
    deltas = [float(e["delta"]) for e in entries
              if isinstance(e.get("delta"), (int, float))]
    n = len(deltas)
    worst = max((abs(d) for d in deltas), default=None)
    agree = [d for d in deltas if abs(d) <= AGREEMENT_TOLERANCE_DR]
    return {
        "store_path": p,
        "tracks_checked": n,
        "tracks_within_tolerance": len(agree),
        "tolerance_dr": AGREEMENT_TOLERANCE_DR,
        "max_abs_delta_dr": worst,
        "mean_delta_dr": (sum(deltas) / n) if n else None,
        "entries": [{k: e.get(k) for k in
                     ("title", "artist", "published_dr", "measured_dr", "delta",
                      "sha256", "checked_utc", "tool_version")}
                    for e in entries],
        # Validated means: at least one track measured against a published
        # rating, and every checked track inside the tolerance.
        "validated": bool(n) and len(agree) == n,
    }
