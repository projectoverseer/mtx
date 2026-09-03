"""Add a transcript to analyses that already exist, without re-measuring them.

    python tools/transcribe.py "E:/Music/_mtx_out"
    python tools/transcribe.py "E:/Music/_mtx_out" --limit 50
    python tools/transcribe.py "E:/Music/_mtx_out" --force     # redo transcripts

The obvious way to add lyrics to a scanned corpus is `mtx scan --force`.  That
re-runs the whole battery including demucs, which is eleven minutes a track:
**240 hours** for 1,321 tracks.  Transcription itself is thirty seconds a
track on the GPU, so the whole job is **eleven hours** if nothing else re-runs.
That is the entire reason this file exists.

What it does per folder: read `analysis.json`, run the transcription backend
over the source audio, re-run the lyric battery on the result, and write the
`lyrics` block and the two `headline` fields that summarise it back.  Nothing
else in the document is touched.

`mtx analyze` promises byte-identical output for the same input, and editing
its output in place would break that promise silently.  So it is not silent:
every amended analysis records what changed, when, and with which model, under
`run.amendments`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from mtx import __version__ as MTX_VERSION            # noqa: E402
from mtx.metrics import lyrics as m_lyrics            # noqa: E402
from mtx.params import PARAMS                         # noqa: E402
from mtx.util import Collector                        # noqa: E402

AMENDMENT = "lyrics.transcript"


def log(msg: str) -> None:
    print(f"[transcribe] {msg}", file=sys.stderr, flush=True)


def folders(root: str) -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith(".") and d != "stems")
        if "analysis.json" in filenames:
            dirnames[:] = []
            out.append(dirpath)
    return sorted(out)


def already_done(doc: dict) -> bool:
    return (doc.get("lyrics") or {}).get("source") == "transcript"


def source_path(doc: dict) -> str | None:
    path = (doc.get("file") or {}).get("path_absolute")
    return path if path and os.path.isfile(path) else None


def transcribe_one(folder: str, force: bool) -> tuple[str, str]:
    """Returns (status, detail).  Never raises: one bad file is not a run."""
    path = os.path.join(folder, "analysis.json")
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        return "error", f"unreadable analysis: {exc}"

    if already_done(doc) and not force:
        return "skip", "already transcribed"
    audio = source_path(doc)
    if not audio:
        return "skip", "source audio is not where the analysis says it is"

    collector = Collector()
    got = m_lyrics.analyse(
        doc.get("tags") or {}, doc.get("declared"), doc.get("stems"),
        doc.get("structure"), collector, want_transcript=True, mix_path=audio)

    transcript = got.get("transcript") or {}
    if not transcript.get("available"):
        return "fail", str(transcript.get("reason") or "no transcript")

    doc["lyrics"] = got
    headline = doc.setdefault("headline", {})
    headline["lyric_source"] = got.get("source")
    headline["lyric_word_count"] = (got.get("statistics") or {}).get("words")

    # Editing an analysis in place breaks the reproducibility promise unless
    # the edit is on the record.  This is the record.
    run = doc.setdefault("run", {})
    amendments = run.setdefault("amendments", [])
    amendments = [a for a in amendments
                  if not (isinstance(a, dict) and a.get("what") == AMENDMENT)]
    amendments.append({
        "what": AMENDMENT,
        "when_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool_version": MTX_VERSION,
        "backend": transcript.get("backend"),
        "model": transcript.get("model"),
        "device": transcript.get("device"),
        "input": transcript.get("input"),
        "note": "the lyrics block and the two headline fields that summarise "
                "it were recomputed; nothing else in this document changed",
    })
    run["amendments"] = amendments

    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True, ensure_ascii=False,
                      allow_nan=False)
            fh.write("\n")
        os.replace(tmp, path)
    except (OSError, ValueError) as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        return "error", f"could not write: {exc}"

    words = (got.get("statistics") or {}).get("words") or 0
    lines = (got.get("statistics") or {}).get("lines") or 0
    return "ok", f"{words} words over {lines} line(s), from the {transcript.get('input')}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument("--limit", type=int, help="stop after this many folders")
    ap.add_argument("--force", action="store_true",
                    help="re-transcribe folders that already have one")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    P = PARAMS["lyrics"]["transcript"]
    model = os.environ.get("MTX_WHISPER_MODEL") or P.get("model")
    log(f"model {model!r}, device {P.get('device')!r}, vad {P.get('vad')}")

    todo = folders(args.root)
    log(f"{len(todo)} analysed folder(s)")
    if args.dry_run:
        for folder in todo[:20]:
            log(f"  would transcribe {folder}")
        return 0

    # Serial on purpose.  One GPU holds one model, and a second worker would
    # queue behind it while doubling the VRAM.
    counts = {"ok": 0, "skip": 0, "fail": 0, "error": 0}
    started = time.time()
    for i, folder in enumerate(todo, 1):
        if args.limit and counts["ok"] >= args.limit:
            break
        status, detail = transcribe_one(folder, args.force)
        counts[status] += 1
        if status != "skip":
            rel = os.path.relpath(folder, args.root)
            log(f"[{i}/{len(todo)}] {status}: {rel} -- {detail}")
        if counts["ok"] and counts["ok"] % 25 == 0 and status == "ok":
            rate = counts["ok"] / max(time.time() - started, 1e-9)
            left = (len(todo) - i) / max(rate, 1e-9) / 60.0
            log(f"  {counts['ok']} done, {rate * 60:.1f}/min, "
                f"~{left:.0f} min left")
    log(f"done: {counts['ok']} transcribed, {counts['skip']} skipped, "
        f"{counts['fail']} no transcript, {counts['error']} error(s), "
        f"in {(time.time() - started) / 60:.1f} min")
    return 1 if counts["error"] and not counts["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
