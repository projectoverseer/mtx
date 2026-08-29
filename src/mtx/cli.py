"""Command-line interface.

Progress goes to stderr; stdout carries only what a caller might want to pipe.
Exit codes: 0 success, 1 bad input, 2 a self-test assertion failed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import unicodedata
from typing import Any

from . import __version__

AUDIO_EXTENSIONS = (".flac", ".wav", ".aif", ".aiff", ".w64", ".caf", ".ogg",
                    ".opus", ".mp3", ".m4a", ".aac", ".wv", ".ape")


def _log(msg: str) -> None:
    print(f"[mtx] {msg}", file=sys.stderr, flush=True)


_BAD_FS_CHARS = re.compile('[<>:"/\\\\|?*\\x00-\\x1f]')


def _sanitise_component(s: str) -> str:
    """Make a tag value usable as a single path component on Windows and POSIX."""
    s = _BAD_FS_CHARS.sub("_", s)
    return re.sub(r"\s+", " ", s).strip()


def _join_artists(raw: str) -> str:
    """`ROSE;Bruno Mars;Rose` -> `ROSE, Bruno Mars` (multi-value tags, deduped)."""
    parts, seen = [], set()
    for chunk in re.split("[;" + chr(0) + "]| / ", raw):
        chunk = chunk.strip()
        # Fold accents for the duplicate check so "ROSE" and "ROSE" (accented) match.
        key = "".join(c for c in unicodedata.normalize("NFKD", chunk.lower())
                      if not unicodedata.combining(c))
        if chunk and key not in seen:
            seen.add(key)
            parts.append(chunk)
    return ", ".join(parts)


def _folder_name(path: str, res: dict[str, Any] | None = None) -> str:
    """`Artist - Title` from the embedded tags, falling back to the filename."""
    named = ((res or {}).get("tags") or {}).get("named") or {}
    title = _sanitise_component(str(named.get("title") or ""))
    artist = _sanitise_component(_join_artists(
        str(named.get("artist") or "") or str(named.get("albumartist") or "")))
    if title and artist:
        name = f"{artist} - {title}"
    elif title:
        name = title
    else:
        name = ""
    name = name[:150].strip().rstrip(". ")   # Windows: no trailing dot or space
    return name or os.path.splitext(os.path.basename(path))[0]


def _default_out(base_out: str | None, path: str,
                 res: dict[str, Any] | None = None) -> str:
    return os.path.join(base_out or "mtx_out", _folder_name(path, res))


def _check_readable(path: str) -> None:
    if not os.path.isfile(path):
        _log(f"error: not a file: {path}")
        raise SystemExit(1)
    try:
        import soundfile as sf
        sf.info(path)
    except Exception as exc:
        _log(f"error: cannot read audio from {path}: {exc}")
        raise SystemExit(1)


def _parse_bytes(text: str, flag: str, example: str) -> int:
    """`20k`, `20kb`, `4.5m`, `20480` -> bytes."""
    t = text.strip().lower().replace("_", "")
    mult = 1
    for suffix, m in (("kb", 1024), ("k", 1024), ("mb", 1024 * 1024), ("m", 1024 * 1024)):
        if t.endswith(suffix):
            t, mult = t[: -len(suffix)], m
            break
    try:
        return int(round(float(t) * mult))
    except ValueError:
        _log(f"error: cannot read {flag} {text!r}; try {example}")
        raise SystemExit(1)


def parse_part_size(args: argparse.Namespace) -> int | None:
    """The per-file cap the JSON output is split to fit under.

    `None` means "never split".  The default exists because the exhaustive dump
    of a normal track is past the 5 MB per-file limit an upload usually has,
    and a measurement that cannot leave the machine is half a measurement.
    """
    from .split import DEFAULT_PART_BYTES, MIN_PART_BYTES, NOTION_UPLOAD_LIMIT

    if getattr(args, "no_split", False):
        return None
    text = getattr(args, "max_part_size", None)
    if text is None:
        return DEFAULT_PART_BYTES
    value = _parse_bytes(text, "--max-part-size", "4.5m or 4500000")
    if value < MIN_PART_BYTES:
        _log(f"error: --max-part-size {text!r} is below the "
             f"{MIN_PART_BYTES // 1024} KB floor; below that the index and its "
             "manifest do not fit in a part of their own")
        raise SystemExit(1)
    if value > NOTION_UPLOAD_LIMIT:
        _log(f"note: --max-part-size {value / 1e6:.1f} MB is above the 5 MB "
             "limit Notion enforces per upload; the parts will be too big for it")
    return value


def _add_part_size_args(p: argparse.ArgumentParser) -> None:
    from .split import DEFAULT_PART_BYTES

    p.add_argument("--max-part-size", metavar="BYTES",
                   help="size a single JSON output file may reach before it is "
                        f"written as an index plus parts (default "
                        f"{DEFAULT_PART_BYTES / 1e6:.1f}m, under the 5 MB "
                        "per-file upload limit); e.g. 4.5m or 2m")
    p.add_argument("--no-split", action="store_true",
                   help="always write one JSON file, however large")


def cmd_join(args: argparse.Namespace) -> int:
    """Put a split analysis back together."""
    from .split import join

    if not os.path.exists(args.path):
        _log(f"error: no such file or directory: {args.path}")
        return 1
    try:
        out = join(args.path, args.out)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        _log(f"error: {exc}")
        return 1
    _log(f"rejoined -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    print(out)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from .analyze import analyze_file, write_outputs

    _check_readable(args.file)
    part_size = parse_part_size(args)
    t0 = time.time()
    res = analyze_file(args.file, profile=args.profile, want_stems=args.stems,
                       log=lambda s: _log(f"  {s}"))
    out_dir = (args.out if args.out and args.single_out
               else _default_out(args.out, args.file, res))
    written = write_outputs(res, out_dir, json_only=args.json_only,
                            plots=args.plots, src_path=args.file,
                            max_part_bytes=part_size, log=_log)
    _log(f"done in {time.time() - t0:.1f} s -> {out_dir}")
    for k, v in written.items():
        _log(f"  {k}: {v}")
    if not args.json_only:
        size = os.path.getsize(written["digest.md"])
        _log(f"  digest size: {size} bytes ({size / 1024:.1f} KB)")
    n_warn = len(res.get("warnings", []))
    if n_warn:
        _log(f"  {n_warn} warning(s) in FLAGS")
    print(out_dir)
    return 0


CSV_FIELDS = [
    "filename", "title", "artist", "date", "duration_s", "sample_rate_hz",
    "channels", "subtype", "lufs_i", "lra_lu", "true_peak_dbtp_16x",
    "sample_peak_dbfs", "plr_db", "psr_min_db", "psr_median_db", "dr14",
    "crest_whole_db", "crest_loudest_10s_db", "spectral_tilt_db_per_oct",
    "air_band_pct", "sub_band_pct", "side_minus_mid_db",
    "side_minus_mid_below_120hz_db", "mono_crossover_hz", "correlation_mean",
    "correlation_min", "flat_top_sample_count", "flat_top_longest_run_ms",
    "hf_cutoff_hz", "effective_bit_depth", "tempo_bpm", "key", "section_count",
    "warnings",
]


def cmd_batch(args: argparse.Namespace) -> int:
    from .analyze import analyze_file, write_outputs

    if not os.path.isdir(args.dir):
        _log(f"error: not a directory: {args.dir}")
        return 1
    files: list[str] = []
    if args.recursive:
        for root, _, names in os.walk(args.dir):
            for nm in sorted(names):
                if nm.lower().endswith(AUDIO_EXTENSIONS):
                    files.append(os.path.join(root, nm))
    else:
        for nm in sorted(os.listdir(args.dir)):
            p = os.path.join(args.dir, nm)
            if os.path.isfile(p) and nm.lower().endswith(AUDIO_EXTENSIONS):
                files.append(p)
    files.sort()
    if not files:
        _log(f"error: no audio files found in {args.dir}")
        return 1
    _log(f"{len(files)} file(s) to analyse")

    part_size = parse_part_size(args)
    base_out = args.out or "mtx_out"
    rows: list[dict[str, Any]] = []
    failures = 0
    for i, path in enumerate(files, 1):
        _log(f"[{i}/{len(files)}] {os.path.basename(path)}")
        try:
            res = analyze_file(path, profile=args.profile, want_stems=args.stems,
                               log=lambda s: _log(f"  {s}"))
        except Exception as exc:
            failures += 1
            _log(f"  failed: {exc!r}")
            continue
        out_dir = _default_out(base_out, path, res)
        write_outputs(res, out_dir, json_only=args.json_only, plots=args.plots,
                      src_path=path, max_part_bytes=part_size, log=_log)
        h = res["headline"]
        t = res.get("tags", {}).get("named", {})
        row = {k: h.get(k) for k in CSV_FIELDS if k in h}
        row.update({
            "filename": os.path.basename(path),
            "title": t.get("title"), "artist": t.get("artist"), "date": t.get("date"),
            "sample_rate_hz": res["audio"]["sample_rate_hz"],
            "channels": res["audio"]["channels"],
            "subtype": res["audio"]["subtype"],
            "warnings": len(res.get("warnings", [])),
        })
        rows.append(row)

    csv_path = args.csv or os.path.join(base_out, "summary.csv")
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    _log(f"summary: {csv_path} ({len(rows)} row(s), {failures} failure(s))")
    print(csv_path)
    return 1 if failures and not rows else 0


def cmd_compare(args: argparse.Namespace) -> int:
    from .compare import compare_files

    _check_readable(args.file_a)
    _check_readable(args.file_b)
    out_dir = args.out or os.path.join(
        "mtx_out", "compare_" +
        os.path.splitext(os.path.basename(args.file_a))[0] + "__vs__" +
        os.path.splitext(os.path.basename(args.file_b))[0])
    paths = compare_files(args.file_a, args.file_b, out_dir,
                          null_test=args.null_test, profile=args.profile,
                          max_part_bytes=parse_part_size(args), log=_log)
    for k, v in paths.items():
        _log(f"  {k}: {v}")
    print(out_dir)
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    from .selftest import run_selftest
    return run_selftest(verbose=not args.quiet)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mtx",
        description="master extractor: an exhaustive, reproducible measurement "
                    "dump for lossless audio. Measures; never interprets.")
    p.add_argument("--version", action="version", version=f"mtx {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="measure one file")
    a.add_argument("file")
    a.add_argument("--out", help="output directory (default ./mtx_out/<basename>/)")
    a.add_argument("--single-out", action="store_true",
                   help="write directly into --out instead of --out/<basename>/")
    a.add_argument("--profile", choices=("quick", "full"), default="full")
    a.add_argument("--plots", action="store_true", help="also write plots/*.png")
    a.add_argument("--stems", action="store_true",
                   help="separate stems with demucs and measure each")
    a.add_argument("--json-only", action="store_true", help="skip digest.md")
    _add_part_size_args(a)
    a.set_defaults(func=cmd_analyze)

    b = sub.add_parser("batch", help="measure every audio file in a folder")
    b.add_argument("dir")
    b.add_argument("--out", help="base output directory (default ./mtx_out/)")
    b.add_argument("--recursive", action="store_true")
    b.add_argument("--csv", help="summary CSV path (default <out>/summary.csv)")
    b.add_argument("--profile", choices=("quick", "full"), default="full")
    b.add_argument("--plots", action="store_true")
    b.add_argument("--stems", action="store_true")
    b.add_argument("--json-only", action="store_true")
    _add_part_size_args(b)
    b.set_defaults(func=cmd_batch)

    c = sub.add_parser("compare", help="level-matched comparison of two files")
    c.add_argument("file_a")
    c.add_argument("file_b")
    c.add_argument("--out")
    c.add_argument("--null-test", action="store_true",
                   help="align, invert and sum; report the residual")
    c.add_argument("--profile", choices=("quick", "full"), default="full")
    _add_part_size_args(c)
    c.set_defaults(func=cmd_compare)

    j = sub.add_parser("join", help="rebuild a split analysis.json from its "
                                    "index and part files")
    j.add_argument("path", help="the index file (analysis.json) or the "
                                "directory holding it")
    j.add_argument("--out", help="output path (default <name>.full.json "
                                 "next to the index)")
    j.set_defaults(func=cmd_join)

    s = sub.add_parser("selftest", help="synthetic signals with known answers")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_selftest)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        _log("interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
