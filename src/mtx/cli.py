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
from .parallel import cpu_count, default_workers
from .scan import AUDIO_EXTENSIONS


def _log(msg: str) -> None:
    try:
        print(f"[mtx] {msg}", file=sys.stderr, flush=True)
    except UnicodeEncodeError:
        # A legacy console codepage must not be able to end a library scan
        # halfway through because one album is called "÷".
        enc = getattr(sys.stderr, "encoding", None) or "ascii"
        safe = f"[mtx] {msg}".encode(enc, errors="replace").decode(enc, "replace")
        print(safe, file=sys.stderr, flush=True)


# Characters no Windows path component may contain (POSIX only bars "/").
_BAD_FS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _sanitise_component(s: str) -> str:
    """Make a tag value usable as a single path component."""
    return re.sub(r"\s+", " ", _BAD_FS_CHARS.sub("_", s)).strip()


def _join_artists(raw: str) -> str:
    """`ROSE;Bruno Mars;Rose` -> `ROSE, Bruno Mars`: multi-value tags, deduped.

    Tags arrive joined by ";", NUL or " / " depending on the container and on
    how mutagen flattened a list.
    """
    parts: list[str] = []
    seen: set[str] = set()
    for chunk in re.split(r"[;\x00]| / ", raw):
        chunk = chunk.strip()
        # Fold accents for the duplicate test so "ROSÉ" and "Rose" count as one.
        key = "".join(c for c in unicodedata.normalize("NFKD", chunk.lower())
                      if not unicodedata.combining(c))
        if chunk and key not in seen:
            seen.add(key)
            parts.append(chunk)
    return ", ".join(parts)


def _folder_name(path: str, res: dict[str, Any] | None = None) -> str:
    """`Artist - Title` from the embedded tags; the filename when untagged."""
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
    name = name[:150].strip().rstrip(". ")  # Windows: no trailing dot or space
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


def parse_budget(text: str | None) -> int | None:
    """The digest size cap."""
    if text is None:
        return None
    value = _parse_bytes(text, "--digest-budget", "20k or 20480")
    if value < 6144:
        # The never-dropped sections (header, HEADLINE, FLAGS, CORPUS ROW,
        # METHOD) are around 5 KB on their own; a lower cap could only be met
        # by dropping something the digest promises to always carry.
        _log(f"error: --digest-budget {text!r} is below the 6 KB floor; the "
             "sections that are never dropped do not fit under it")
        raise SystemExit(1)
    return value


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


def _enrich_targets(path: str) -> list[str]:
    """The analysed folders under `path`: itself, or its immediate children."""
    if os.path.isfile(path):
        return [os.path.dirname(os.path.abspath(path)) or "."]
    if os.path.isfile(os.path.join(path, "analysis.json")):
        return [path]
    out = []
    for name in sorted(os.listdir(path)):
        sub = os.path.join(path, name)
        if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "analysis.json")):
            out.append(sub)
    return out


def cmd_enrich(args: argparse.Namespace) -> int:
    """Attach public-database metadata to folders `mtx analyze` already wrote.

    The result is a sidecar `online.json`, never a rewrite of `analysis.json`:
    `mtx analyze` promises byte-identical output for the same input, and a
    section whose content depends on what MusicBrainz looked like this morning
    cannot live inside that promise.
    """
    from .online import ALL_PROVIDERS, KEYED_PROVIDERS, enrich
    from .split import load_analysis

    if not os.path.exists(args.path):
        _log(f"error: no such file or directory: {args.path}")
        return 1
    targets = _enrich_targets(args.path)
    if not targets:
        _log(f"error: no analysis.json under {args.path}")
        return 1

    if args.providers.strip().lower() == "all":
        providers = list(ALL_PROVIDERS)
    else:
        providers = [p for p in (x.strip() for x in args.providers.split(",")) if p]
        unknown = [p for p in providers if p not in ALL_PROVIDERS]
        if unknown:
            _log(f"error: unknown provider(s): {', '.join(unknown)}; "
                 f"choose from {', '.join(ALL_PROVIDERS)}")
            return 1
    for name in providers:
        if name in KEYED_PROVIDERS:
            env = "LASTFM_API_KEY" if name == "lastfm" else "DISCOGS_TOKEN"
            if not os.environ.get(env):
                _log(f"warning: {name} needs {env}; it will report no match")

    cache = args.cache if args.cache is not None else os.path.join(
        args.path if os.path.isdir(args.path) else ".", ".mtx_cache")
    _log(f"{len(targets)} folder(s), providers: {', '.join(providers)}")
    _log(f"cache: {cache}{' (offline)' if args.offline else ''}")

    ok = matched = 0
    for i, folder in enumerate(targets, 1):
        name = os.path.basename(os.path.abspath(folder))
        try:
            res = load_analysis(os.path.join(folder, "analysis.json"))
        except (OSError, ValueError) as exc:
            _log(f"[{i}/{len(targets)}] {name}: cannot read analysis: {exc}")
            continue
        _log(f"[{i}/{len(targets)}] {name}")
        section = enrich(res, cache_dir=cache, providers=providers,
                         offline=args.offline, refresh=args.refresh,
                         log=lambda s: _log(f"    {s}"), version=__version__)
        out_path = args.out if (args.out and len(targets) == 1) else \
            os.path.join(folder, "online.json")
        with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(section, fh, indent=1, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
        ok += 1
        g = section.get("genres") or {}
        if section.get("providers_available"):
            matched += 1
        _log(f"    genre: {g.get('primary') or '-'} "
             f"({g.get('umbrella') or '-'}), "
             f"match {section.get('match_confidence', 0):.2f} -> {out_path}")
        tempo = (section.get("cross_checks") or {}).get("tempo") or {}
        if tempo.get("verdict") in ("octave", "triplet", "disagree"):
            _log(f"    tempo {tempo['verdict']}: local {tempo.get('local_bpm')} "
                 f"vs published {tempo.get('published_bpm')}")

    _log(f"enriched {ok} folder(s); {matched} matched at least one provider")
    if args.print_json and ok == 1:
        print(json.dumps(section, indent=1, ensure_ascii=False))
    return 0


def _split_sections(text: str | None) -> list[str] | None:
    if not text:
        return None
    return [part for part in (p.strip() for p in text.split(",")) if part]


def cmd_analyze(args: argparse.Namespace) -> int:
    from .analyze import analyze_file, write_outputs

    _check_readable(args.file)
    sections = _split_sections(getattr(args, "sections", None))
    budget = parse_budget(getattr(args, "digest_budget", None))
    part_size = parse_part_size(args)
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
                       log=lambda s: _log(f"  {s}"),
                       threads=getattr(args, "jobs", None),
                       stems_model=getattr(args, "stems_model", None),
                       declared_path=getattr(args, "declared", None),
                       want_transcript=bool(getattr(args, "transcribe", False)),
                       want_embedding=bool(getattr(args, "embed", False)))
    out_dir = args.out if args.out and args.single_out else _default_out(args.out, args.file, res)
    written = write_outputs(res, out_dir, json_only=args.json_only,
                            plots=args.plots, src_path=args.file,
                            digest_budget=budget, sections=sections,
                            max_part_bytes=part_size, blind=blind, log=_log)
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

    part_size = parse_part_size(args)
    base_out = args.out or "mtx_out"
    rows: list[dict[str, Any]] = []
    failures = 0
    for i, path in enumerate(files, 1):
        _log(f"[{i}/{len(files)}] {os.path.basename(path)}")
        try:
            res = analyze_file(path, profile=args.profile, want_stems=args.stems,
                               log=lambda s: _log(f"  {s}"),
                               stems_model=getattr(args, "stems_model", None),
                               want_transcript=bool(getattr(args, "transcribe", False)),
                               want_embedding=bool(getattr(args, "embed", False)))
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


def cmd_scan(args: argparse.Namespace) -> int:
    """Measure a library, an artist or an album -- whatever the path covers."""
    from .scan import NoRootRegistered, run_scan

    if not os.path.exists(args.path):
        _log(f"error: no such file or directory: {args.path}")
        return 1
    # A scan's workers are separate processes, and both of these have to reach
    # the demucs call inside each of them; the environment is what a child
    # inherits, where a parsed flag would have to be threaded through analyze.
    device = getattr(args, "stems_device", "auto")
    segment = getattr(args, "stems_segment", None)
    if device not in ("auto", None) or segment:
        from .metrics.stems import ENV_DEVICE, ENV_SEGMENT
        if device not in ("auto", None):
            os.environ[ENV_DEVICE] = device
        if segment:
            os.environ[ENV_SEGMENT] = str(segment)
    try:
        stats = run_scan(
            args.path, out=args.out, library_root=args.library_root,
            profile=args.profile, jobs=args.jobs,
            stems_jobs=getattr(args, "stems_jobs", None), force=args.force,
            recheck=args.recheck, stems=args.stems, plots=args.plots,
            json_only=args.json_only, max_part_bytes=parse_part_size(args),
            stems_model=getattr(args, "stems_model", None),
            transcribe=bool(getattr(args, "transcribe", False)),
            embed=bool(getattr(args, "embed", False)),
            dry_run=args.dry_run, no_summary=args.no_summary,
            dedup=not getattr(args, "no_dedup", False), log=_log)
    except NoRootRegistered as exc:
        _log(f"error: {exc}")
        return 1
    except ValueError as exc:
        _log(f"error: {exc}")
        return 1
    print(stats["out_dir"])
    if stats["failed"] and not stats["done"]:
        return 1
    return 0


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


def cmd_cohort(args: argparse.Namespace) -> int:
    """Where each track sits among comparable records.

    A separate command over a separate file, on purpose: a per-track
    measurement must not change because of what else is in the folder.
    """
    from .cohort import build, render

    if not os.path.exists(args.path):
        _log(f"error: no such file or directory: {args.path}")
        return 1
    try:
        doc = build(args.path, neighbours=args.neighbours, log=_log)
    except ValueError as exc:
        _log(f"error: {exc}")
        return 1
    out_dir = args.out or (args.path if os.path.isdir(args.path) else ".")
    os.makedirs(out_dir, exist_ok=True)
    from .util import jsonable
    j = os.path.join(out_dir, "cohort.json")
    with open(j, "w", encoding="utf-8", newline="\n") as f:
        json.dump(jsonable(doc), f, indent=1, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        f.write("\n")
    m = os.path.join(out_dir, "cohort.md")
    with open(m, "w", encoding="utf-8", newline="\n") as f:
        f.write(render(doc))
    h = doc["hygiene"]
    _log(f"{h['tracks']} track(s), {h['distinct_artists']} artist(s), "
         f"{len(doc['cohorts'])} cohort(s)")
    for problem in h["problems"]:
        _log(f"  corpus hygiene: {problem}")
    _log(f"  cohort.json: {j}")
    _log(f"  cohort.md: {m}")
    print(out_dir)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Flat tables at track and track x section level."""
    from .export import export

    if not os.path.exists(args.path):
        _log(f"error: no such file or directory: {args.path}")
        return 1
    out_dir = args.out or (args.path if os.path.isdir(args.path) else ".")
    try:
        stats = export(args.path, out_dir, level=args.level, fmt=args.format,
                       log=_log)
    except ValueError as exc:
        _log(f"error: {exc}")
        return 1
    _log(f"{stats['analyses_found']} analysis file(s): "
         f"{stats['tracks']} track row(s) x {stats['track_columns']} column(s), "
         f"{stats['section_rows']} section row(s) x "
         f"{stats['section_columns']} column(s)")
    for name, path in stats["written"].items():
        _log(f"  {name}: {path}")
    for fail in stats["failed"]:
        _log(f"  failed: {fail}")
    print(out_dir)
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    from .selftest import run_selftest
    return run_selftest(verbose=not args.quiet)


def build_parser() -> argparse.ArgumentParser:
    from .online import DEFAULT_PROVIDERS as DEFAULT_PROVIDER_NAMES

    p = argparse.ArgumentParser(
        prog="mtx",
        description="master extractor: an exhaustive, reproducible measurement "
                    "dump for lossless audio. Measures; never interprets.")
    p.add_argument("--version", action="version", version=f"mtx {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="measure one file")
    a.add_argument("file")
    a.add_argument("--out", help="output directory (default ./mtx_out/<artist - title>/)")
    a.add_argument("--single-out", action="store_true",
                   help="write directly into --out instead of --out/<basename>/")
    a.add_argument("--profile", choices=("quick", "full"), default="full")
    a.add_argument("--plots", action="store_true", help="also write plots/*.png")
    a.add_argument("--stems-model", metavar="NAME",
                   help="demucs model to separate with (default htdemucs); "
                        "htdemucs_6s splits guitar and piano out of `other`")
    a.add_argument("--declared", metavar="FILE",
                   help="a declared.json sidecar; its values are reported with "
                        "source=declared and never merged into a measured field")
    a.add_argument("--transcribe", action="store_true",
                   help="transcribe the vocal stem for a time-aligned lyric "
                        "(optional backend, needs --stems)")
    a.add_argument("--embed", action="store_true",
                   help="compute a learned embedding vector (optional backend)")
    a.add_argument("--stems", action="store_true",
                   help="separate stems with demucs and measure each")
    a.add_argument("--json-only", action="store_true", help="skip digest.md")
    a.add_argument("--jobs", "-j", type=int, metavar="N",
                   help="threads to use inside this one file (default "
                        f"{default_workers()} here); only the true-peak "
                        "oversampling pass can spend them")
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
    b.add_argument("--stems-model", metavar="NAME")
    b.add_argument("--transcribe", action="store_true")
    b.add_argument("--embed", action="store_true")
    b.add_argument("--json-only", action="store_true")
    b.add_argument("--csv-schema", choices=("internal", "corpus", "masters"),
                   default="internal",
                   help="'corpus' names the columns after the corpus database's "
                        "properties so the CSV imports as a populated table")
    _add_part_size_args(b)
    b.set_defaults(func=cmd_batch)

    sc = sub.add_parser("scan",
                        help="measure every unmeasured file under a path "
                             "(album, artist or whole library), in parallel")
    sc.add_argument("path", nargs="?", default=".",
                    help="what to measure: an album, an artist, or the library "
                         "root (default: the current directory)")
    sc.add_argument("--out", metavar="DIR",
                    help="root of the mirror tree results are written to; "
                         "remembered per library root, so it only has to be "
                         "given once (default ./mtx_out on a first scan)")
    sc.add_argument("--library-root", metavar="DIR",
                    help="what the mirror tree's paths are relative to "
                         "(default: the path being scanned, on a first scan)")
    sc.add_argument("--jobs", "-j", type=int, metavar="N",
                    help=f"parallelism budget (default {default_workers()} here: "
                         f"{cpu_count()} core(s) less headroom, capped at 8). "
                         "Spent on worker processes first, then on threads "
                         "within a file when fewer files than workers remain")
    sc.add_argument("--force", action="store_true",
                    help="re-measure everything, including files that already "
                         "have a current result")
    sc.add_argument("--recheck", action="store_true",
                    help="decide staleness by hashing each source file instead "
                         "of trusting its size and modification time")
    sc.add_argument("--dry-run", action="store_true",
                    help="list what would be measured, and why, then stop")
    sc.add_argument("--no-summary", action="store_true",
                    help="skip rewriting summary.csv over the scanned subtree")
    sc.add_argument("--no-dedup", action="store_true",
                    help="measure every copy of a file separately, instead of "
                         "copying the result across files with identical bytes")
    sc.add_argument("--profile", choices=("quick", "full"), default="full")
    sc.add_argument("--plots", action="store_true")
    sc.add_argument("--stems", action="store_true")
    sc.add_argument("--stems-model", metavar="NAME",
                    help="demucs model (default htdemucs); htdemucs_6s splits "
                         "guitar and piano out of `other`")
    sc.add_argument("--stems-device", choices=("auto", "cuda", "cpu"),
                    default="auto",
                    help="where to separate (default auto: the GPU if torch "
                         "can see one). On a GPU the separations are done "
                         "up front, a few at a time, before the pool starts")
    sc.add_argument("--stems-jobs", type=int, metavar="N",
                    help="separations to run on the card at once (default: "
                         "what its memory holds, about 900 MiB each, capped "
                         "at 4). One stream leaves a card two thirds busy; "
                         "the rest of a track is decode and wav writing")
    sc.add_argument("--stems-segment", type=int, metavar="SECONDS",
                    help="seconds of audio demucs holds on the device at once. "
                         "Lower it if a small card runs out of memory; the "
                         "default tries 7.8 and steps down only on a failure")
    sc.add_argument("--transcribe", action="store_true",
                    help="transcribe the vocal stem for a time-aligned lyric "
                         "(optional backend, needs --stems)")
    sc.add_argument("--embed", action="store_true",
                    help="compute a learned embedding vector (optional backend)")
    sc.add_argument("--json-only", action="store_true", help="skip digest.md")
    _add_part_size_args(sc)
    sc.set_defaults(func=cmd_scan)

    c = sub.add_parser("compare", help="level-matched comparison of two files")
    c.add_argument("file_a")
    c.add_argument("file_b")
    c.add_argument("--out")
    c.add_argument("--null-test", action="store_true",
                   help="align, invert and sum; report the residual")
    c.add_argument("--profile", choices=("quick", "full"), default="full")
    _add_part_size_args(c)
    c.set_defaults(func=cmd_compare)

    e = sub.add_parser("enrich",
                       help="look analysed folders up in the public music "
                            "databases and write online.json beside each")
    e.add_argument("path", help="an analysed folder, or a folder of them")
    e.add_argument("--providers", default=",".join(DEFAULT_PROVIDER_NAMES),
                   metavar="A,B,C",
                   help="comma-separated, or 'all'; keyless: musicbrainz, "
                        "deezer, itunes; keyed: lastfm (LASTFM_API_KEY), "
                        "discogs (DISCOGS_TOKEN)")
    e.add_argument("--cache", metavar="DIR",
                   help="response cache (default <path>/.mtx_cache); a second "
                        "run over the same folder makes no requests")
    e.add_argument("--offline", action="store_true",
                   help="answer only from the cache, never the network")
    e.add_argument("--refresh", action="store_true",
                   help="ignore cached responses and re-fetch")
    e.add_argument("--out", metavar="FILE",
                   help="write here instead of online.json (single folder only)")
    e.add_argument("--print", dest="print_json", action="store_true",
                   help="also print the section to stdout (single folder only)")
    e.set_defaults(func=cmd_enrich)

    j = sub.add_parser("join", help="rebuild a split analysis.json from its "
                                    "index and part files")
    j.add_argument("path", help="the index file (analysis.json) or the "
                                "directory holding it")
    j.add_argument("--out", help="output path (default <name>.full.json "
                                 "next to the index)")
    j.set_defaults(func=cmd_join)

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

    ch = sub.add_parser("cohort",
                        help="where each track sits among comparable records")
    ch.add_argument("path", help="a folder of analysed folders")
    ch.add_argument("--out", metavar="DIR",
                    help="where cohort.json and cohort.md go (default: PATH)")
    ch.add_argument("--neighbours", type=int, default=5, metavar="N",
                    help="nearest neighbours per track (0 to skip)")
    ch.set_defaults(func=cmd_cohort)

    ex = sub.add_parser("export",
                        help="flat track and track x section tables over a "
                             "folder of analyses")
    ex.add_argument("path", help="a folder of analysed folders")
    ex.add_argument("--out", metavar="DIR", help="output directory (default: PATH)")
    ex.add_argument("--level", choices=("track", "section", "both"), default="both")
    ex.add_argument("--format", choices=("csv", "parquet", "both"), default="csv",
                    help="parquet needs pyarrow; CSV is always written")
    ex.set_defaults(func=cmd_export)

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
