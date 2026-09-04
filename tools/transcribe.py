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
    """Has this folder been transcribed?  Not: is the transcript being used.

    The distinction cost 95 tracks a pointless re-run on every pass.  A track
    whose file tags carry a real lyric sheet keeps `lyrics.source` at
    `file:tag`, because a sheet written by the label beats a transcription of
    a separated stem and should win.  The transcript is still made, still
    stored, and still what the alignment and prosody measurements read -- but
    a check that asked "is the transcript the chosen source" saw `file:tag`,
    concluded nothing had been done, and spent thirty seconds re-deriving a
    transcript identical to the one already on disk.  Every run.  Forever.

    So ask whether a transcript exists, not whether it won.
    """
    lyrics = doc.get("lyrics") or {}
    if lyrics.get("source") == "transcript":
        return True
    transcript = lyrics.get("transcript") or {}
    if transcript.get("available") and transcript.get("source") == "transcript":
        return True
    # A transcript that heard nothing is still an answer about this track, and
    # re-running it every night would cost thirty seconds each time to learn
    # the same thing.
    return bool(transcript.get("rejected_as_lyric"))


# How far below the mix a separated vocal has to sit before the track counts
# as instrumental.  Sung tracks in this corpus cluster around -5 LU; the one
# score cue in a 70-track sample was -33.5.
INSTRUMENTAL_LU = -25.0


def vocal_level(folder: str) -> float | None:
    """How loud the separated vocal is against the mix, in LU.

    Whisper hallucinates on music with no voice in it -- confidently, in
    fluent sentences -- and the result would land in the corpus as a lyric
    with a source attached.  The separation already measured whether there is
    a voice here, so ask it before spending thirty seconds inventing one.
    """
    from mtx.split import load_analysis                # noqa: PLC0415
    try:
        doc = load_analysis(os.path.join(folder, "analysis.json"))
    except (OSError, ValueError):
        return None
    vocals = ((doc.get("stems") or {}).get("stems") or {}).get("vocals") or {}
    level = (vocals.get("level_vs_mix") or {}).get("lufs_delta")
    return float(level) if isinstance(level, (int, float)) else None


def source_path(doc: dict) -> str | None:
    path = (doc.get("file") or {}).get("path_absolute")
    return path if path and os.path.isfile(path) else None


def _save(folder: str, path: str, doc: dict, whole: bool) -> None:
    """Persist the amended analysis, keeping whatever shape it arrived in.

    The common case rewrites only the index, atomically.  When `lyrics` has
    been moved out to a part, the document has to be re-split instead --
    `write_analysis` recomputes which sections move, rewrites the manifest and
    deletes the parts the new layout does not use, so the folder ends up equal
    to the result rather than to the union of two of them.
    """
    if whole:
        from mtx.split import write_analysis           # noqa: PLC0415
        write_analysis(doc, folder)
        return
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True, ensure_ascii=False,
                      allow_nan=False)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _record_attempt(folder: str, path: str, doc: dict, whole: bool,
                    reason: str, transcript: dict) -> None:
    """Note that transcription was tried here and did not work.

    Deliberately not enough to count as done: `available` stays false and no
    rejection is recorded, so the next run picks the track up again.  What it
    adds is the reason and the devices that were tried, which is what turns
    "this track has no lyrics" into "this track failed on a 4 GB card at
    05:12" -- one of those is a finding about the music and the other is a
    finding about the run.
    """
    lyrics = doc.setdefault("lyrics", {})
    if not isinstance(lyrics, dict):
        return
    note = lyrics.setdefault("transcript", {})
    if not isinstance(note, dict):
        return
    note["available"] = False
    note["reason"] = reason
    note["attempted_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if transcript.get("attempted"):
        note["attempted_devices"] = transcript["attempted"]
    _save(folder, path, doc, whole)


def transcribe_one(folder: str, force: bool) -> tuple[str, str]:
    """Returns (status, detail).  Never raises: one bad file is not a run."""
    path = os.path.join(folder, "analysis.json")
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        return "error", f"unreadable analysis: {exc}"

    # Reading the *index* rather than the merged document is deliberate: it
    # keeps the `split` manifest and the part files untouched, so writing back
    # rewrites one 3 MB index instead of re-emitting 17 MB of timelines.  It
    # only holds while `lyrics` actually lives in the index.  A long enough
    # transcript could push it out to a part, and then an inline write is
    # shadowed on the next read by the part that still holds the old value --
    # a silent revert, with the log still saying `ok`.
    moved_lyrics = bool(isinstance(doc.get("lyrics"), dict)
                        and doc["lyrics"].get("mtx_moved"))
    if moved_lyrics:
        from mtx.split import load_analysis            # noqa: PLC0415
        doc = load_analysis(path)

    if already_done(doc) and not force:
        return "skip", "already transcribed"
    audio = source_path(doc)
    if not audio:
        return "skip", "source audio is not where the analysis says it is"
    level = vocal_level(folder)
    if level is not None and level < INSTRUMENTAL_LU:
        return "skip", (f"instrumental: the vocal stem is {level:.1f} LU below "
                        f"the mix, and a transcript here would be invention")

    collector = Collector()
    got = m_lyrics.analyse(
        doc.get("tags") or {}, doc.get("declared"), doc.get("stems"),
        doc.get("structure"), collector, want_transcript=True, mix_path=audio)

    transcript = got.get("transcript") or {}
    if not transcript.get("available"):
        reason = str(transcript.get("reason") or "no transcript")
        # Record the attempt.  A failure that writes nothing leaves an
        # analysis byte-identical to one nobody ever asked to transcribe, so
        # 78 tracks killed by an out-of-memory sat behind a clean audit
        # looking like tracks with no words in them.  The distinction is the
        # difference between "buy a lyric sheet" and "re-run the job".
        try:
            _record_attempt(folder, path, doc, moved_lyrics, reason, transcript)
        except (OSError, ValueError):
            pass                        # the transcript matters, the note does not
        return "fail", reason

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

    try:
        _save(folder, path, doc, moved_lyrics)
    except (OSError, ValueError) as exc:
        return "error", f"could not write: {exc}"

    if transcript.get("rejected_as_lyric"):
        # Written all the same: the empty transcript is the finding, and
        # recording it stops the next run spending thirty seconds re-learning
        # that this track has no words in it.
        return "thin", transcript["rejected_as_lyric"]
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
    counts = {"ok": 0, "thin": 0, "skip": 0, "fail": 0, "error": 0}
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
    log(f"done: {counts['ok']} transcribed, {counts['thin']} with no words "
        f"heard, {counts['skip']} skipped, {counts['fail']} no transcript, "
        f"{counts['error']} error(s), in {(time.time() - started) / 60:.1f} min")
    return 1 if counts["error"] and not counts["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
