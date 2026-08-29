"""The prediction sheet and the calibration check.

Committing a number before seeing it is the whole point of `--blind`, and a
prediction made with the answer one `cat` away is not a commitment.  So this
module renders a sheet that carries the field list, the units and the method --
everything that makes a guess *informed* -- and none of the values.

`mtx predict --check` is arithmetic only: signed error, absolute error, and
whether the stated interval contained the true value.  It never says whether a
prediction was good.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable

from .digest import BLANK, PREDICT_FIELDS, _flags, _method, n

# `LUFS-I               = ____  +/- ____   conf ____`
FILL_RE = re.compile(
    r"^\s*(?P<label>[^=]+?)\s*=\s*(?P<value>[^\s]+)"
    r"(?:\s*\+/-\s*(?P<range>[^\s]+))?"
    r"(?:\s*conf\s*(?P<conf>[^\s%]+)%?)?\s*$")

NUM_RE = re.compile(r"^[-+]?\d+(?:\.\d+)?$")


def _float(text: str | None) -> float | None:
    if text is None:
        return None
    t = text.strip().rstrip("%")
    return float(t) if NUM_RE.match(t) else None


def render_predict_sheet(res: dict[str, Any]) -> str:
    """The redacted headline, as a form to fill in.

    METHOD stays visible on purpose: knowing *how* a number is derived is fair
    information for a prediction; knowing the number is not.  DETAIL and CORPUS
    ROW are absent, because both restate the headline values and a sheet that
    leaks them is not blind.
    """
    tags = res.get("tags", {}).get("named", {})
    width = max(len(label) for label, _, _, _ in PREDICT_FIELDS)
    rows = []
    for label, _key, unit, _nd in PREDICT_FIELDS:
        u = f" {unit}" if unit else ""
        rows.append(f"{label.ljust(width)} = {BLANK}{u}  +/- {BLANK}   conf {BLANK}%")
    return (
        "# mtx prediction sheet\n\n"
        f"file: {res.get('file', {}).get('filename')}\n"
        f"title: {tags.get('title') or '(no title tag)'}\n"
        f"artist: {tags.get('artist') or '(no artist tag)'}\n"
        f"sha256: {(res.get('file', {}).get('sha256') or '')[:16]}...\n"
        f"tool: mtx {res['run']['tool_version']} / schema "
        f"{res['run']['schema_version']} / profile {res['run']['profile']}\n"
        f"audio: {res['audio']['sample_rate_hz']} Hz, {res['audio']['channels']} ch, "
        f"{res['audio']['subtype']}, {res['audio']['duration_s']} s\n\n"
        "The digest for this file has been written and not printed. Fill in the\n"
        "fields you want to commit to -- a value, a +/- range, and how confident\n"
        "you are that the true value falls inside that range -- then:\n\n"
        "    mtx predict --check <this file> <the digest.md or analysis.json>\n\n"
        "Leave a field as ____ to skip it. An interval, not a point guess.\n\n"
        "## PREDICTIONS\n\n```\n" + "\n".join(rows) + "\n```\n\n"
        + _flags(res) + "\n" + _method(res))


def parse_predictions(text: str) -> dict[str, dict[str, float | None]]:
    """Read a filled-in sheet.  Unfilled and unparseable lines are skipped."""
    labels = {label.lower(): label for label, _, _, _ in PREDICT_FIELDS}
    out: dict[str, dict[str, float | None]] = {}
    for line in text.splitlines():
        m = FILL_RE.match(line)
        if not m:
            continue
        label = labels.get(m.group("label").strip().lower())
        if label is None:
            continue
        value = _float(m.group("value"))
        if value is None:
            continue
        out[label] = {"value": value,
                      "range": _float(m.group("range")),
                      "confidence": _float(m.group("conf"))}
    return out


def _actuals_from_json(path: str) -> dict[str, float | None]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    h = data.get("headline", {})
<<<<<<< HEAD
    if data.get("split") and (not isinstance(h, dict) or h.get("mtx_moved")):
        # A split analysis keeps the headline in the index, so this is the rare
        # case; rejoining is still cheaper than telling the caller to do it.
        from .split import load_analysis
        h = load_analysis(path).get("headline", {})
=======
>>>>>>> 425d1b1c98d36da4d8be6bf9a20bfab8da99db3a
    return {label: h.get(key) for label, key, _, _ in PREDICT_FIELDS}


def _actuals_from_digest(path: str) -> dict[str, float | None]:
    """Read the HEADLINE block back out of a digest.

    The labels come from the same table the block is printed from, so this
    cannot drift; a label mtx did not print is simply absent.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if "## HEADLINE" not in text:
        raise ValueError(f"{path} has no HEADLINE section")
    block = text.split("## HEADLINE", 1)[1].split("## ", 1)[0]
    out: dict[str, float | None] = {}
    for label, _key, _unit, _nd in PREDICT_FIELDS:
        for line in block.splitlines():
            if not line.startswith(label):
                continue
            rest = line[len(label):].strip()
            if not rest or rest[0].isalnum() and not rest[0].isdigit():
                continue  # a longer label that merely starts the same way
            m = re.match(r"^([-+]?\d+(?:\.\d+)?)", rest)
            out[label] = float(m.group(1)) if m else None
            break
    return out


def read_actuals(path: str) -> dict[str, float | None]:
    if path.lower().endswith(".json"):
        return _actuals_from_json(path)
    return _actuals_from_digest(path)


def check(predictions: dict[str, dict[str, float | None]],
          actuals: dict[str, float | None]) -> dict[str, Any]:
    """Signed error, absolute error, and whether the interval held.

    Rows are ordered by |error| x confidence, largest first: that is the order
    the errors are worth reading in, and it is a sort, not a verdict.
    """
    rows = []
    for label, _key, unit, nd in PREDICT_FIELDS:
        if label not in predictions:
            continue
        p = predictions[label]
        actual = actuals.get(label)
        pred = p["value"]
        err = (pred - actual) if (actual is not None and pred is not None) else None
        inside = None
        if err is not None and p.get("range") is not None:
            inside = abs(err) <= abs(p["range"])
        conf = p.get("confidence")
        rows.append({
            "field": label, "unit": unit, "decimals": nd,
            "predicted": pred, "range": p.get("range"), "confidence": conf,
            "actual": actual, "error": err,
            "abs_error": abs(err) if err is not None else None,
            "interval_held": inside,
            "rank_score": (abs(err) * (conf / 100.0 if conf is not None else 1.0))
            if err is not None else None,
        })
    rows.sort(key=lambda r: (r["rank_score"] is None,
                             -(r["rank_score"] or 0.0)))
    scored = [r for r in rows if r["abs_error"] is not None]
    held = [r for r in rows if r["interval_held"] is not None]
    return {
        "rows": rows,
        "fields_predicted": len(rows),
        "fields_scored": len(scored),
        "mean_abs_error": (sum(r["abs_error"] for r in scored) / len(scored))
        if scored else None,
        "intervals_stated": len(held),
        "intervals_held": sum(1 for r in held if r["interval_held"]),
    }


def render_check(result: dict[str, Any], pred_path: str, truth_path: str) -> str:
    from .digest import table

    rows: Iterable[dict[str, Any]] = result["rows"]
    body = []
    for r in rows:
        nd = r["decimals"]
        rng = f" +/-{n(r['range'], nd)}" if r["range"] is not None else ""
        body.append([
            r["field"],
            n(r["predicted"], nd) + rng,
            n(r["actual"], nd),
            (f"{r['error']:+.{nd}f}" if r["error"] is not None else "n/a"),
            n(r["abs_error"], nd),
            ("hit" if r["interval_held"] else "miss") if r["interval_held"] is not None else "-",
            n(r["confidence"], 0) if r["confidence"] is not None else "-",
            n(r["rank_score"], 2),
        ])
    head = ["field", "predicted", "actual", "error", "|error|", "interval",
            "conf%", "|err|xconf"]
    summary = (
        f"fields predicted {result['fields_predicted']}, scored "
        f"{result['fields_scored']}, mean |error| "
        f"{n(result['mean_abs_error'], 2)} (mixed units -- read per field)\n"
        f"intervals stated {result['intervals_stated']}, held "
        f"{result['intervals_held']}")
    return ("# mtx calibration check\n\n"
            f"predictions: {os.path.basename(pred_path)}\n"
            f"measured:    {os.path.basename(truth_path)}\n\n"
            "Sorted by |error| x confidence: the most confident miss first.\n"
            "Errors are predicted minus measured. This block states the gap and\n"
            "nothing about whether it is a good one.\n\n```\n"
            + table(head, body) + "\n\n" + summary + "\n```\n")
