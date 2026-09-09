"""Writing `analysis.json` as parts small enough to upload somewhere.

`analysis.json` is the exhaustive dump, and on a four-minute track with the
full profile it is comfortably past 5 MB -- which is exactly the per-file cap
Notion (and several other places a measurement archive ends up) enforces on an
upload.  A file that cannot be uploaded is a file that stays on one machine.

So when the document does not fit under the cap, it is written as an index plus
numbered parts instead of one file:

    analysis.json            the index: every section small enough to stay
                             inline (headline, run, params, warnings, ...),
                             plus `split` -- the manifest naming the parts
    analysis.part01.json     one fragment of the document, with the path it
    analysis.part02.json     belongs at
    ...

Every file is valid JSON on its own and each one is under the cap, so the whole
set uploads.  `load_analysis()` reads the index and returns the whole document
again; `mtx join` does the same thing from the command line and writes it out.

Nothing is split when the document already fits: the common case stays exactly
one `analysis.json`, byte for byte what it was before.

A fragment is `(path, slice, value)`:

- `path` -- the key sequence the fragment sits at, `["processing",
  "multiband_timeline"]`.  An empty path means the document root.
- `slice` -- `[start, stop]` when the fragment is a chunk of a list at that
  path, `null` when it is a (possibly partial) dict.
- `value` -- the fragment itself.

Rejoining is a deep merge of the fragments in part order: dict fragments set
their keys, list chunks concatenate.  Nothing is lost and nothing is rounded --
the split is a transport detail, not a second representation.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterable
from typing import Any

# Notion refuses an upload over 5 MB.  The default part size sits below that
# with room to spare, because the cap is on the file and a part carries a small
# header on top of its fragment.
NOTION_UPLOAD_LIMIT = 5 * 1024 * 1024
DEFAULT_PART_BYTES = 4_500_000

# What the index keeps free for the manifest and the moved-section markers.
_INDEX_RESERVE = 64 * 1024

# Below this a split cannot mean anything useful: the manifest alone would not
# fit, and the caller has almost certainly passed bytes where they meant KB.
MIN_PART_BYTES = 128 * 1024

_PART_RE = re.compile(r"^(?P<stem>.+)\.part(?P<n>\d+)\.json$", re.IGNORECASE)


def encode(obj: Any) -> str:
    """The one JSON encoding mtx writes: stable key order, no NaN, LF."""
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False,
                      allow_nan=False) + "\n"


def _size(obj: Any) -> int:
    """Encoded size in bytes -- what a file size limit is actually counted in."""
    return len(encode(obj).encode("utf-8"))


def _write(path: str, obj: Any) -> int:
    text = encode(obj)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return len(text.encode("utf-8"))


# --------------------------------------------------------------------------
# fragmenting


def _chunk_list(path: list[str], value: list, max_bytes: int) -> list[tuple]:
    """Cut a list into consecutive chunks that each fit under the cap."""
    step = max(1, math.ceil(len(value) / max(1, math.ceil(_size(value) / (max_bytes * 0.9)))))
    while step > 1:
        if all(_size(value[i:i + step]) <= max_bytes
               for i in range(0, len(value), step)):
            break
        step = max(1, step // 2)
    return [(list(path), [i, min(i + step, len(value))], value[i:i + step])
            for i in range(0, len(value), step)]


def _fragments(path: list[str], value: Any, max_bytes: int) -> list[tuple]:
    """Break `value` into fragments that each encode to under `max_bytes`.

    A dict is broken up by grouping its children into buckets rather than one
    file per key: a section with two hundred small keys and one huge one should
    produce two files, not two hundred and one.  A child that is oversize on
    its own recurses.  A list is cut into consecutive chunks.  A scalar that is
    somehow over the cap on its own is emitted whole and flagged by the caller
    -- truncating a measurement to fit an upload limit is not on the table.
    """
    if _size(value) <= max_bytes:
        return [(list(path), None, value)]

    if isinstance(value, dict) and value:
        frags: list[tuple] = []
        bucket: dict[str, Any] = {}
        bucket_size = 2  # the enclosing braces
        for key in sorted(value):
            child = value[key]
            cost = _size(child) + len(key.encode("utf-8")) + 8  # quotes, colon, comma
            if cost > max_bytes:
                if bucket:
                    frags.append((list(path), None, bucket))
                    bucket, bucket_size = {}, 2
                frags.extend(_fragments(path + [key], child, max_bytes))
            elif bucket_size + cost > max_bytes:
                frags.append((list(path), None, bucket))
                bucket, bucket_size = {key: child}, 2 + cost
            else:
                bucket[key] = child
                bucket_size += cost
        if bucket:
            frags.append((list(path), None, bucket))
        return frags

    if isinstance(value, list) and len(value) > 1:
        return _chunk_list(path, value, max_bytes)

    return [(list(path), None, value)]


def _part_object(stem: str, index: int, of: int, path: list[str], sl,
                 value: Any) -> dict[str, Any]:
    return {
        "mtx_part": {
            "stem": stem,
            "index": index,
            "of": of,
            "path": path,
            "slice": sl,
            "index_file": f"{stem}.json",
            "note": "one fragment of a split mtx analysis; `mtx join` "
                    "rebuilds the whole document from the index file",
        },
        "data": value,
    }


def _part_size(stem: str, path: list[str], sl, value: Any) -> int:
    """What the fragment costs *as a file*, header and all.

    The cap is on the file, not on the fragment: the header is a few hundred
    bytes and nesting the fragment under `data` indents every line of it by one
    more space, which on a timeline of a hundred thousand numbers is not a
    rounding error.  Sizing the wrapper is the only way to be sure.
    """
    return _size(_part_object(stem, 99, 99, path, sl, value))


def _shrink(stem: str, path: list[str], sl, value: Any, max_bytes: int,
            budget: int, depth: int = 0) -> list[tuple]:
    """Re-split a fragment whose file would still be over the cap."""
    if _part_size(stem, path, sl, value) <= max_bytes or depth >= 8:
        return [(path, sl, value)]
    smaller = max(4096, int(budget * 0.75))
    sub = _fragments(path, value, smaller)
    if len(sub) == 1 and sub[0][1] is None:
        return [(path, sl, value)]  # indivisible: emitted whole and flagged
    out: list[tuple] = []
    for sub_path, sub_slice, sub_value in sub:
        if sl is not None:
            # Chunking a chunk: slices stay absolute, so a reader can place a
            # part without walking the ones before it.
            sub_slice = ([sl[0] + sub_slice[0], sl[0] + sub_slice[1]]
                         if sub_slice is not None else sl)
        out.extend(_shrink(stem, sub_path, sub_slice, sub_value, max_bytes,
                           smaller, depth + 1))
    return out


def _section_fragments(stem: str, key: str, value: Any,
                       max_bytes: int) -> list[tuple]:
    budget = max(4096, int(max_bytes * 0.95) - 1024)  # header and indent room
    out: list[tuple] = []
    for path, sl, val in _fragments([key], value, budget):
        out.extend(_shrink(stem, path, sl, val, max_bytes, budget))
    return out


# --------------------------------------------------------------------------
# writing


def _moved_marker(files: list[str]) -> dict[str, Any]:
    return {
        "mtx_moved": True,
        "parts": files,
        "note": "this section did not fit under the part size limit and lives "
                "in the part file(s) named here; `mtx join` puts it back",
    }


def _prune_parts(out_dir: str, stem: str, keep: dict[str, str]) -> list[str]:
    """Delete part files of `stem` that this run did not write.

    Re-analysing into a folder that already holds a result can produce fewer
    parts than last time -- a shorter track, a larger cap, a section that got
    smaller.  The index names the parts it owns, so an orphan is never *read*,
    but it sits in the folder looking exactly like current data.  Removing it
    keeps the folder equal to the result rather than to the union of every
    result ever written there.
    """
    removed: list[str] = []
    try:
        names = os.listdir(out_dir)
    except OSError:
        return removed
    for name in names:
        m = _PART_RE.match(name)
        if m and m.group("stem") == stem and name not in keep:
            try:
                os.remove(os.path.join(out_dir, name))
                removed.append(name)
            except OSError:
                pass
    return removed


def write_analysis(res: dict[str, Any], out_dir: str, stem: str = "analysis", *,
                   max_bytes: int | None = DEFAULT_PART_BYTES,
                   log=None) -> dict[str, str]:
    """Write `<stem>.json`, split into parts if it does not fit under the cap.

    `max_bytes=None` disables splitting entirely and always writes one file.
    Returns the written paths, keyed by filename, index first.
    """
    doc = dict(res)
    written: dict[str, str] = {}
    index_path = os.path.join(out_dir, f"{stem}.json")

    whole = _size(doc)
    if max_bytes is None or whole <= max_bytes:
        _write(index_path, doc)
        written[f"{stem}.json"] = index_path
        stale = _prune_parts(out_dir, stem, written)
        if stale and log:
            log(f"  removed {len(stale)} part file(s) from an earlier run")
        return written

    # Move whole top-level sections out, largest first, until what is left fits
    # in the index with room for the manifest.  Sections are moved whole where
    # possible so the parts stay readable on their own: one file that is all of
    # `processing` beats two that are arbitrary halves of the document.
    sizes = {k: _size(v) for k, v in doc.items()}
    order = sorted(sizes, key=lambda k: (-sizes[k], k))
    # The reserve scales with the cap: a quarter of a small cap is still room
    # for a manifest, where a fixed 64 KB would be larger than the whole part
    # and would push even `headline` out of an index that had space for it.
    reserve = min(_INDEX_RESERVE, max(1024, max_bytes // 4))
    moved: list[str] = []
    for key in order:
        trial = {k: (_moved_marker([]) if k in moved else v)
                 for k, v in doc.items()}
        if _size(trial) + reserve <= max_bytes:
            break
        moved.append(key)
    moved.sort()

    frags: list[tuple] = []
    for key in moved:
        frags.extend(_section_fragments(stem, key, doc[key], max_bytes))

    manifest: list[dict[str, Any]] = []
    per_section: dict[str, list[str]] = {k: [] for k in moved}
    oversize = 0
    for i, (path, sl, value) in enumerate(frags, 1):
        name = f"{stem}.part{i:02d}.json"
        part = _part_object(stem, i, len(frags), path, sl, value)
        size = _write(os.path.join(out_dir, name), part)
        written[name] = os.path.join(out_dir, name)
        if size > max_bytes:
            oversize += 1
        manifest.append({"file": name, "path": path, "slice": sl, "bytes": size})
        if path:
            per_section.setdefault(path[0], []).append(name)

    for key in moved:
        doc[key] = _moved_marker(per_section[key])
    doc["split"] = {
        "note": "the full document does not fit under the part size limit, so "
                "it was written as this index plus the part files below. Every "
                "file here is valid JSON and under the limit; `mtx join "
                f"{stem}.json` rebuilds the whole document.",
        "whole_bytes": whole,
        "part_max_bytes": max_bytes,
        "part_count": len(manifest),
        "sections_in_parts": moved,
        "parts": manifest,
        "rejoin": f"mtx join {stem}.json",
    }
    if oversize:
        doc["split"]["oversize_parts"] = (
            f"{oversize} part(s) are still over the limit: the value they hold "
            "cannot be divided any further. Raise the limit or accept that "
            "those parts do not upload; nothing was truncated.")

    index_size = _write(index_path, doc)
    written = {f"{stem}.json": index_path, **written}
    stale = _prune_parts(out_dir, stem, written)
    if stale and log:
        log(f"  removed {len(stale)} part file(s) from an earlier run")
    if log:
        log(f"  {stem}.json: {whole / 1e6:.1f} MB over the {max_bytes / 1e6:.1f} MB "
            f"part limit -> index ({index_size / 1e6:.2f} MB) + "
            f"{len(manifest)} part file(s)")
        if oversize:
            log(f"  warning: {oversize} part(s) are still over the limit "
                "(indivisible value); nothing was truncated")
        if index_size > max_bytes:
            log(f"  warning: the index is {index_size / 1e6:.2f} MB, over the "
                "limit: the manifest for this many parts does not fit in it")
    return written


# --------------------------------------------------------------------------
# reading


def _merge(root: dict[str, Any], path: list[str], sl, value) -> None:
    if not path:
        if not isinstance(value, dict):
            raise ValueError("a root fragment must be an object")
        root.update(value)
        return
    node: Any = root
    for key in path[:-1]:
        nxt = node.get(key)
        if not isinstance(nxt, dict) or nxt.get("mtx_moved"):
            nxt = {}
            node[key] = nxt
        node = nxt
    leaf = path[-1]
    current = node.get(leaf)
    if sl is not None:
        if not isinstance(current, list):
            current = []
        node[leaf] = list(current) + list(value)
    elif isinstance(value, dict) and isinstance(current, dict) \
            and not current.get("mtx_moved"):
        current.update(value)
    else:
        node[leaf] = value


def _wanted(part_path: list[str], want: frozenset[str] | None) -> bool:
    """Does this part hold anything under one of the requested prefixes?

    A part is merged when its path and a wanted prefix lie on the same branch
    in either direction: the part `["processing"]` carries `processing.pumping`
    and is needed for it, while `["processing", "multiband_timeline",
    "rms_db"]` does not and is not.
    """
    if want is None:
        return True
    here = ".".join(str(p) for p in (part_path or []))
    if not here:
        return True
    for prefix in want:
        if here == prefix or here.startswith(prefix + ".") or                 prefix.startswith(here + "."):
            return True
    return False


def load_analysis(path: str, want: Iterable[str] | None = None) -> dict[str, Any]:
    """Read an analysis back, whether it was written whole or split.

    `path` is the index (`analysis.json`); an unsplit file is simply returned.
    A missing part is an error rather than a silently short document.

    `want` names the dotted sections the caller actually reads, and parts on
    other branches are then skipped.  This is not a micro-optimisation: on this
    corpus the index is 3 MB and the parts add 14 more, almost all of it two
    timeline arrays.  `mtx cohort` reads about forty scalars and was parsing
    18 GB of JSON to find them.  Omit it and everything is loaded, which is
    what `mtx join` needs and what every existing caller gets.
    """
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    split = doc.pop("split", None)
    if not split:
        return doc
    keep = frozenset(want) if want is not None else None
    base = os.path.dirname(os.path.abspath(path))
    for key in list(doc):
        if isinstance(doc[key], dict) and doc[key].get("mtx_moved"):
            del doc[key]
    for entry in split.get("parts", []):
        if not _wanted(entry.get("path") or [], keep):
            continue
        part_path = os.path.join(base, entry["file"])
        if not os.path.isfile(part_path):
            raise FileNotFoundError(
                f"{path} names part {entry['file']}, which is not in {base}; "
                "the whole set has to travel together")
        with open(part_path, encoding="utf-8") as f:
            part = json.load(f)
        meta = part.get("mtx_part") or {}
        _merge(doc, meta.get("path", entry.get("path", [])),
               meta.get("slice", entry.get("slice")), part.get("data"))
    return doc


def join(path: str, out_path: str | None = None) -> str:
    """Rebuild the whole document from an index (or a directory holding one)."""
    if os.path.isdir(path):
        candidates = sorted(n for n in os.listdir(path)
                            if n.endswith(".json") and not _PART_RE.match(n)
                            and n in ("analysis.json", "comparison.json"))
        if not candidates:
            raise FileNotFoundError(f"no analysis.json or comparison.json in {path}")
        path = os.path.join(path, candidates[0])
    doc = load_analysis(path)
    if out_path is None:
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(os.path.dirname(os.path.abspath(path)),
                                f"{stem}.full.json")
    _write(out_path, doc)
    return out_path
