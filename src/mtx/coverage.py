"""The missingness and confidence mask.

Confidences are computed and reported per metric group, and presence is
implicit in whether a field is null -- which means any consumer has to
rediscover both by walking the whole document.  This module walks it once and
writes down the answer: one uniform vector saying which features are present
and how far each is trusted.

It derives the mask from the result rather than from a hand-kept list of field
names, so a metric added tomorrow appears in the mask without anyone
remembering to register it.
"""

from __future__ import annotations

import math
from typing import Any

# Blocks that are provenance or configuration rather than measurement.
SKIP_TOP = ("params", "run", "warnings", "confidence_notes", "coverage")
# Keys whose contents are bulk data: recorded as one series entry, not walked.
SERIES_HINT = ("times_s", "_timeline", "per_third_octave", "beat_times_s",
               "downbeat_times_s", "boundaries_s", "vector", "per_window",
               "notes_list", "per_bar", "arc_per_line", "words",
               "post_offset_envelope_db", "bar_labels", "progression",
               "concurrent_sources", "accent_components")
MAX_INLINE_LIST = 8
MAX_DEPTH = 7


def _kind(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "series"
    if isinstance(value, dict):
        return "block"
    return "other"


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _walk(node: Any, path: str, out: dict[str, dict[str, Any]],
          confidence: str | None, depth: int) -> None:
    if depth > MAX_DEPTH:
        return
    if isinstance(node, dict):
        conf = node.get("confidence") if isinstance(node.get("confidence"), str) else confidence
        if node.get("available") is False:
            out[path] = {"present": False, "kind": "block",
                         "reason": node.get("reason"), "confidence": None}
            return
        for key, value in node.items():
            if key in ("confidence", "confidence_reason", "method", "definition",
                       "note", "notes", "rule", "basis", "caveat", "reason",
                       "params"):
                continue
            child = f"{path}.{key}" if path else str(key)
            if any(h in str(key) for h in SERIES_HINT) and isinstance(value, (list, tuple)):
                out[child] = {"present": bool(len(value)), "kind": "series",
                              "length": len(value), "confidence": conf}
                continue
            _walk(value, child, out, conf, depth + 1)
        return
    if isinstance(node, (list, tuple)):
        if len(node) > MAX_INLINE_LIST or not node or not isinstance(node[0], dict):
            out[path] = {"present": bool(len(node)), "kind": "series",
                         "length": len(node), "confidence": confidence}
            return
        for i, item in enumerate(node):
            _walk(item, f"{path}[{i}]", out, confidence, depth + 1)
        return
    out[path] = {"present": node is not None and _finite(node),
                 "kind": _kind(node), "confidence": confidence}


def build(res: dict[str, Any]) -> dict[str, Any]:
    """The mask for one analysis result."""
    mask: dict[str, dict[str, Any]] = {}
    for key, value in res.items():
        if key in SKIP_TOP:
            continue
        _walk(value, key, mask, None, 0)

    groups: dict[str, dict[str, Any]] = {}
    for path, entry in mask.items():
        g = path.split(".", 1)[0].split("[", 1)[0]
        row = groups.setdefault(g, {"features": 0, "present": 0,
                                    "low_confidence": 0, "unavailable_blocks": 0})
        row["features"] += 1
        row["present"] += int(bool(entry["present"]))
        if entry.get("confidence") == "low":
            row["low_confidence"] += 1
        if entry["kind"] == "block" and not entry["present"]:
            row["unavailable_blocks"] += 1
    for row in groups.values():
        row["present_pct"] = (100.0 * row["present"] / row["features"]
                              if row["features"] else None)

    total = len(mask)
    present = sum(1 for e in mask.values() if e["present"])
    notes = {n.get("metric"): n.get("confidence")
             for n in (res.get("confidence_notes") or []) if isinstance(n, dict)}
    return {
        "feature_count": total,
        "present_count": present,
        "present_pct": (100.0 * present / total) if total else None,
        "by_group": dict(sorted(groups.items())),
        "declared_confidences": notes,
        "features": dict(sorted(mask.items())),
        "definition": "one entry per leaf field of this document outside "
                      "params/run/warnings. `present` is false when the value "
                      "is null, non-finite, an empty series, or inside a block "
                      "whose `available` is false; `confidence` is the nearest "
                      "enclosing block's confidence, where one was stated.",
        "why": "so a consumer can ask which of N features exist and how far "
               "each is trusted without walking the document itself",
    }
