"""Add a learned embedding to analyses that already exist, without re-measuring.

    python tools/embed.py "E:/Music/_mtx_out"
    python tools/embed.py "E:/Music/_mtx_out" --limit 50
    python tools/embed.py "E:/Music/_mtx_out" --force     # redo embeddings

Why this file exists, and not `mtx scan --embed`: `--embed` is a flag on a
scan, and a scan of an already-measured corpus re-runs the whole battery
including demucs -- eleven minutes a track, **240 hours** for 1,321 of them.
The embedding itself is a few seconds. So this loads each analysis, computes
the one missing block, and writes it back, exactly as `tools/transcribe.py`
does for lyrics.

**What the vectors are for.** 193 hand-engineered scalars are ideal for
interpretability and poor at timbre similarity: two records can agree on every
column in this tool and sound nothing alike. One vector per track gives
nearest-neighbour search, which answers a question none of the scalars can --
*which released records actually sound like this mix* -- and answers it
without depending on a genre label being right. `mtx cohort --neighbours`
consumes them.

**On the backend.** `laion_clap` is the obvious choice and is not installed
here on purpose: pip resolves it by downgrading numpy 2.5 to 1.26, librosa
1.0 to 0.11 and scipy 1.18 to 1.17 -- the core of the measurement stack. A
library that changes what `mtx analyze` measures, silently, across a corpus
built over months, is a far worse outcome than a different embedding model.
`transformers:MERT` installs without moving a single pinned version, and MERT
is a music-representation model rather than an audio-text one, which is
closer to the question being asked anyway.

Every amended analysis records what changed and with which model under
`run.amendments`, because editing an analysis in place breaks the
reproducibility promise unless the edit is on the record.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from mtx import __version__ as MTX_VERSION            # noqa: E402
from mtx.audio import AudioSource                     # noqa: E402
from mtx.metrics import embedding as m_embedding      # noqa: E402
from mtx.split import load_analysis                   # noqa: E402
from mtx.util import Collector                        # noqa: E402
from transcribe import _save, folders, source_path    # noqa: E402

AMENDMENT = "embedding"


def log(msg: str) -> None:
    print(f"[embed] {msg}", file=sys.stderr, flush=True)


def already_done(doc: dict) -> bool:
    """Is there a vector here?  Not: was one ever asked for.

    The same distinction that cost 95 tracks a pointless re-run in the
    transcription pass.  A block that says `available: false` with a reason is
    a recorded failure and must be retried; a block with a vector in it is
    done, whatever else the document says about it.
    """
    emb = doc.get("embedding") or {}
    return bool(emb.get("available") and emb.get("vector"))


def embed_one(folder: str, force: bool) -> tuple[str, str]:
    """Returns (status, detail).  Never raises: one bad file is not a run."""
    path = os.path.join(folder, "analysis.json")
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        return "error", f"unreadable analysis: {exc}"

    # Reading the index keeps the split manifest and the parts untouched; if
    # `embedding` has been moved out to a part, an inline write would be
    # shadowed, so the whole document is re-split instead.
    moved = bool(isinstance(doc.get("embedding"), dict)
                 and doc["embedding"].get("mtx_moved"))
    if moved:
        doc = load_analysis(path)

    if already_done(doc) and not force:
        return "skip", "already embedded"
    audio = source_path(doc)
    if not audio:
        return "skip", "source audio is not where the analysis says it is"

    collector = Collector()
    try:
        src = AudioSource(audio, collector)
    except Exception as exc:
        return "fail", f"could not read audio: {exc!r}"

    got = m_embedding.analyse(src, [], collector, enabled=True)
    if not got.get("available"):
        reason = str(got.get("reason") or "no embedding")
        # Record the attempt.  A failure that writes nothing leaves an
        # analysis byte-identical to one nobody ever asked to embed, and the
        # audit then reports "no vector" -- a finding about the track -- when
        # the finding is about the run.
        try:
            block = doc.setdefault("embedding", {})
            if isinstance(block, dict):
                block["available"] = False
                block["reason"] = reason
                block["attempted_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                       time.gmtime())
                _save(folder, path, doc, moved)
        except (OSError, ValueError):
            pass
        return "fail", reason

    doc["embedding"] = got
    headline = doc.setdefault("headline", {})
    headline["embedding_backend"] = got.get("backend")
    headline["embedding_dimensions"] = got.get("dimensions")

    run = doc.setdefault("run", {})
    amendments = [a for a in (run.get("amendments") or [])
                  if not (isinstance(a, dict) and a.get("what") == AMENDMENT)]
    amendments.append({
        "what": AMENDMENT,
        "when_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool_version": MTX_VERSION,
        "backend": got.get("backend"),
        "model": got.get("model"),
        "dimensions": got.get("dimensions"),
        "note": "the embedding block and the two headline fields naming it "
                "were computed; no measured value was touched, and none is "
                "ever derived from a vector",
    })
    run["amendments"] = amendments

    try:
        _save(folder, path, doc, moved)
    except (OSError, ValueError) as exc:
        return "error", f"could not write: {exc}"
    return "ok", (f"{got.get('dimensions')}-d vector from "
                  f"{got.get('backend')}, |v|={got.get('l2_norm', 0.0):.1f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument("--limit", type=int, help="stop after this many folders")
    ap.add_argument("--force", action="store_true",
                    help="re-embed folders that already have a vector")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    available = [name for name, fn in (("laion_clap", m_embedding._try_laion_clap),
                                       ("openl3", m_embedding._try_openl3),
                                       ("transformers:MERT", m_embedding._try_mert))
                 if _importable(name)]
    log(f"backends available: {available or 'none'}")
    if not available:
        log("no embedding backend installed.  `pip install transformers` "
            "adds MERT without moving numpy, librosa or scipy; laion_clap "
            "downgrades all three and must not go in this environment.")
        return 1

    todo = folders(args.root)
    log(f"{len(todo)} analysed folder(s)")
    if args.dry_run:
        for folder in todo[:20]:
            log(f"  would embed {folder}")
        return 0

    # Serial: one GPU holds one model, and a second worker would queue behind
    # it while doubling the memory it needs.
    counts = {"ok": 0, "skip": 0, "fail": 0, "error": 0}
    started = time.time()
    for i, folder in enumerate(todo, 1):
        if args.limit and counts["ok"] >= args.limit:
            break
        status, detail = embed_one(folder, args.force)
        counts[status] += 1
        if status != "skip":
            rel = os.path.relpath(folder, args.root)
            log(f"[{i}/{len(todo)}] {status}: {rel} -- {detail}")
        if counts["ok"] and counts["ok"] % 25 == 0 and status == "ok":
            rate = counts["ok"] / max(time.time() - started, 1e-9)
            left = (len(todo) - i) / max(rate, 1e-9) / 60.0
            log(f"  {counts['ok']} done, {rate * 60:.1f}/min, ~{left:.0f} min left")
    log(f"done: {counts['ok']} embedded, {counts['skip']} skipped, "
        f"{counts['fail']} no vector, {counts['error']} error(s), "
        f"in {(time.time() - started) / 60:.1f} min")
    return 1 if counts["error"] and not counts["ok"] else 0


def _importable(name: str) -> bool:
    import importlib.util
    root = name.split(":", 1)[0]
    try:
        return importlib.util.find_spec(root) is not None
    except (ImportError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
