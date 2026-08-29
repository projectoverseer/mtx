"""Command-line interface.

Progress goes to stderr; stdout carries only what a caller might want to pipe.
Exit codes: 0 success, 1 bad input, 2 a self-test assertion failed.
"""

from __future__ import annotations

import argparse
import csv
import json
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


def parse_budget(text: str | None) -> int | None:
    """`20k`, `20kb`, `20480` -> bytes."""
    if text is None:
        return None
    t = text.strip().lower().replace("_", "")
    mult = 1
    for suffix, m in (("kb", 1024), ("k", 1024), ("mb", 1024 * 1024), ("m", 1024 * 1024)):
        if t.endswith(suffix):
            t, mult = t[: -len(suffix)], m
            break
    try:
        value = int(round(float(t) * mult))
    except ValueError:
        _log(f"error: cannot read --digest-budget {text!r}; try 20k or 20480")
        raise SystemExit(1)
    if value < 6144:
        # The never-dropped sections (header, HEADLINE, FLAGS, CORPUS ROW,
        # METHOD) are around 5 KB on their own; a lower cap could only be met
        # by dropping something the digest promises to always carry.
        _log(f"error: --digest-budget {text!r} is below the 6 KB floor; the "
             "sections that are never dropped do not fit under it")
        raise SystemExit(1)
    return value


def _split_sections(text: str | None) -> list[str] | None:
    if not text:
        return None
    return [part for part in (p.strip() for p in text.split(",")) if part]


def cmd_analyze(args: argparse.Namespace) -> int:
    from .analyze import analyze_file, write_outputs

    _check_readable(args.file)
    sections = _split_sections(getattr(args, "sections", None))
    budget = parse_budget(getattr(args, "digest_budget", None))
    if sections:
        from .digest import resolve_sections
        try:
            resolve_sections(sections)
        except ValueError as exc:
            _log(f"error: {exc}")
            return 1
    blind = bool(getattr(args, "blind", False))
    if blind and args.json_only:
        _log("error: --blind and --json-only are contradictory; the prediction "
             "sheet is a digest rendering")
        return 1
    t0 = time.time()
    res = analyze_file(args.file, profile=args.profile, want_stems=args.stems,
                       log=lambda s: _log(f"  {s}"))
    out_dir = args.out if args.out and args.single_out else _default_out(args.out, args.file)
    written = write_outputs(res, out_dir, json_only=args.json_only,
                            plots=args.plots, src_path=args.file,
                            digest_budget=budget, sections=sections,
                            blind=blind, log=_log)
    _log(f"done in {time.time() - t0:.1f} s -> {out_dir}")
    for k, v in written.items():
        _log(f"  {k}: {v}")
    if not args.json_only:
        size = os.path.getsize(written["digest.md"])
        _log(f"  digest size: {size} bytes ({size / 1024:.1f} KB)")
    n_warn = len(res.get("warnings", []))
    if n_warn:
        _log(f"  {n_warn} warning(s) in FLAGS")
    if blind:
        # Only the prediction sheet reaches stdout: a prediction made with the
        # digest one `cat` away is not a commitment.
        _log("blind mode: digest.md was written and is NOT printed; commit the "
             "prediction first, then read it")
        print(written["predict.md"])
    else:
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
    "mtx_run", "warnings",
]

# `--csv-schema corpus` renames the columns a corpus database already has a
# property for, and leaves the rest under their internal names.  A folder of
# records then imports as a populated table instead of a mapping exercise;
# breadth across genres and eras is what makes a reference atlas teach craft
# rather than one style, so the cost of breadth is worth lowering.
CSV_CORPUS_NAMES = {
    "title": "Title",
    "artist": "Artist",
    "date": "Year",
    "lufs_i": "LUFS-I",
    "true_peak_dbtp_16x": "True peak",
    "lra_lu": "LRA",
    "plr_db": "PLR",
    "psr_min_db": "PSR min",
    "psr_median_db": "PSR median",
    "dr14": "DR14",
    "crest_loudest_10s_db": "Crest (loudest 10s)",
    "mtx_run": "mtx run",
}


def cmd_batch(args: argparse.Namespace) -> int:
    from .analyze import analyze_file, write_outputs
    from .digest import run_provenance

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
    if args.profile == "quick" and getattr(args, "csv_schema", "internal") != "internal":
        _log("warning: --profile quick skips the 16x true peak, so the "
             "'True peak' column will be empty in every row; use the full "
             "profile for a corpus the rows are meant to be kept in")

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
            "mtx_run": run_provenance(res),
            "warnings": len(res.get("warnings", [])),
        })
        rows.append(row)

    csv_path = args.csv or os.path.join(base_out, "summary.csv")
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    schema = getattr(args, "csv_schema", "internal")
    if schema == "masters":  # the name the corpus database uses at the far end
        schema = "corpus"
    for r in rows:  # a stored row is read by people; full float precision is not
        for k, v in r.items():
            if isinstance(v, float):
                r[k] = round(v, 3)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if schema == "corpus":
            w.writerow({k: CSV_CORPUS_NAMES.get(k, k) for k in CSV_FIELDS})
        else:
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


def cmd_predict(args: argparse.Namespace) -> int:
    from .predict import check, parse_predictions, read_actuals, render_check

    for p in (args.predictions, args.measured):
        if not os.path.isfile(p):
            _log(f"error: not a file: {p}")
            return 1
    with open(args.predictions, encoding="utf-8") as f:
        preds = parse_predictions(f.read())
    if not preds:
        _log(f"error: no filled-in predictions found in {args.predictions}. "
             "Lines look like `LUFS-I = -8.5 +/- 1.0 conf 70%`.")
        return 1
    try:
        actuals = read_actuals(args.measured)
    except (ValueError, OSError) as exc:
        _log(f"error: cannot read measurements from {args.measured}: {exc}")
        return 1
    result = check(preds, actuals)
    _log(f"{result['fields_scored']} of {result['fields_predicted']} predicted "
         "field(s) had a measured value to compare against")
    print(render_check(result, args.predictions, args.measured), end="")
    return 0


def cmd_validate_dr(args: argparse.Namespace) -> int:
    """Record one measured-vs-published DR pair, or show the record."""
    from . import __version__ as tool_version
    from .validation import add_entry, summary

    if args.show or args.file is None:
        rec = summary(args.store)
        _log(f"store: {rec['store_path']}")
        print(json.dumps(rec, indent=1, sort_keys=True, ensure_ascii=False))
        return 0
    if args.published is None:
        _log("error: --published <DR> is required; that published rating is the "
             "only thing mtx cannot measure for itself")
        return 1
    _check_readable(args.file)
    from .analyze import analyze_file
    res = analyze_file(args.file, profile="quick", log=lambda s: _log(f"  {s}"))
    measured = res["headline"]["dr14"]
    if measured is None:
        _log("error: DR14 came back null for this file; see warnings")
        return 1
    tags = res.get("tags", {}).get("named", {})
    delta = float(measured) - float(args.published)
    entry = {
        "title": tags.get("title"),
        "artist": tags.get("artist"),
        "filename": res.get("file", {}).get("filename"),
        "sha256": res.get("file", {}).get("sha256"),
        "published_dr": float(args.published),
        "measured_dr": round(float(measured), 2),
        "delta": round(delta, 2),
        "source": args.source,
        "tool_version": tool_version,
        "schema_version": res["run"]["schema_version"],
        "checked_utc": res["run"]["generated_utc"],
    }
    path = add_entry(entry, args.store)
    rec = summary(args.store)
    _log(f"measured DR {measured:.2f} vs published {float(args.published):.1f} "
         f"-> delta {delta:+.2f}")
    _log(f"record now: {rec['tracks_checked']} track(s), "
         f"{rec['tracks_within_tolerance']} within {rec['tolerance_dr']} DR, "
         f"validated={rec['validated']}")
    print(path)
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
    a.add_argument("--blind", action="store_true",
                   help="also write predict.md (the headline redacted to a form) "
                        "and print only its path, so a prediction can be "
                        "committed before the numbers are read")
    a.add_argument("--sections", metavar="A,B,C",
                   help="restrict DETAIL to these groups or blocks "
                        "(groups: loudness, dynamics, spectrum, stereo, "
                        "forensics, structure, processing)")
    a.add_argument("--digest-budget", metavar="BYTES",
                   help="raise or lower the digest size cap, e.g. 20k "
                        "(default 12k, plus 4k when --stems renders)")
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
    b.add_argument("--csv-schema", choices=("internal", "corpus", "masters"),
                   default="internal",
                   help="'corpus' names the columns after the corpus database's "
                        "properties so the CSV imports as a populated table")
    b.set_defaults(func=cmd_batch)

    c = sub.add_parser("compare", help="level-matched comparison of two files")
    c.add_argument("file_a")
    c.add_argument("file_b")
    c.add_argument("--out")
    c.add_argument("--null-test", action="store_true",
                   help="align, invert and sum; report the residual")
    c.add_argument("--profile", choices=("quick", "full"), default="full")
    c.set_defaults(func=cmd_compare)

    pr = sub.add_parser("predict", help="score a filled-in prediction sheet")
    pr.add_argument("--check", dest="predictions", required=True,
                    metavar="PREDICTIONS",
                    help="the filled-in predict.md (or any file of "
                         "`field = value +/- range conf N%%` lines)")
    pr.add_argument("measured", help="the digest.md or analysis.json to score against")
    pr.set_defaults(func=cmd_predict)

    v = sub.add_parser("validate-dr",
                       help="record this implementation's DR14 against a "
                            "published rating for a track you own")
    v.add_argument("file", nargs="?")
    v.add_argument("--published", type=float,
                   help="the published DR rating for this track")
    v.add_argument("--source", help="where the published rating came from")
    v.add_argument("--store", help="validation record path "
                                   "(default: platform config dir; "
                                   "MTX_DR14_VALIDATION overrides)")
    v.add_argument("--show", action="store_true", help="print the record and exit")
    v.set_defaults(func=cmd_validate_dr)

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
