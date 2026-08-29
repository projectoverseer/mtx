"""Command-line interface.

Progress goes to stderr; stdout carries only what a caller might want to pipe.
Exit codes: 0 success, 1 bad input, 2 a self-test assertion failed.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Any

from . import __version__

AUDIO_EXTENSIONS = (".flac", ".wav", ".aif", ".aiff", ".w64", ".caf", ".ogg",
                    ".opus", ".mp3", ".m4a", ".aac", ".wv", ".ape")


def _log(msg: str) -> None:
    print(f"[mtx] {msg}", file=sys.stderr, flush=True)


def _default_out(base_out: str | None, path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(base_out or "mtx_out", stem)


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


def cmd_analyze(args: argparse.Namespace) -> int:
    from .analyze import analyze_file, write_outputs

    _check_readable(args.file)
    t0 = time.time()
    res = analyze_file(args.file, profile=args.profile, want_stems=args.stems,
                       log=lambda s: _log(f"  {s}"))
    out_dir = args.out if args.out and args.single_out else _default_out(args.out, args.file)
    written = write_outputs(res, out_dir, json_only=args.json_only,
                            plots=args.plots, src_path=args.file, log=_log)
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
        out_dir = _default_out(base_out, path)
        write_outputs(res, out_dir, json_only=args.json_only, plots=args.plots,
                      src_path=path, log=_log)
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
                          log=_log)
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
    b.set_defaults(func=cmd_batch)

    c = sub.add_parser("compare", help="level-matched comparison of two files")
    c.add_argument("file_a")
    c.add_argument("file_b")
    c.add_argument("--out")
    c.add_argument("--null-test", action="store_true",
                   help="align, invert and sum; report the residual")
    c.add_argument("--profile", choices=("quick", "full"), default="full")
    c.set_defaults(func=cmd_compare)

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
