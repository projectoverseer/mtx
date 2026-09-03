"""One command, from new FLACs to a Notion row you can trust.

    python tools/pipeline.py                    # everything, using mtx.env
    python tools/pipeline.py --from enrich      # resume part-way
    python tools/pipeline.py --only audit
    python tools/pipeline.py --dry-run

The corpus grows by twenty-odd tracks a day.  At that rate the thing that
decides whether the data stays trustworthy is not how good any one stage is,
it is whether the stages are ever run in the wrong order or quietly skipped.
So they live here, in order, with the reason each one has to come after the
last:

    scan      measure the audio                    (no network, hours)
    enrich    look the tracks up                   (network, minutes)
    identity  resolve one name per artist folder   -- needs enrich
    outcome   normalise plays within each artist   -- needs enrich + identity
    cohort    percentiles within genre and era     -- needs enrich
    audit     refuse to publish a corpus that is wrong
    push      Notion                               -- gated by audit

`audit` is a gate, not a report.  A stage that finds an error stops the run
before anything reaches Notion, because a wrong row is worse than a missing
one: the missing one gets noticed.

Secrets come from the environment, or from a `mtx.env` file beside the corpus
(`KEY=value` a line, `#` comments).  It never enters git -- it lives with the
music, not with the code.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from typing import Any, Callable

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

DEFAULT_LIBRARY = r"E:\Music"
DEFAULT_OUT = r"E:\Music\_mtx_out"
ENV_FILE = "mtx.env"

# Which keys each stage needs, so a missing one is reported before an hour of
# work rather than as a column of empty cells afterwards.
NEEDS = {
    "enrich": ("LASTFM_API_KEY", "DISCOGS_TOKEN"),
    "push": ("NOTION_TOKEN",),
}


def log(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


def load_env(root: str) -> list[str]:
    """`mtx.env` beside the corpus, for the keys that would otherwise be typed.

    Returns the names it set, never the values: a log line that echoes a token
    has published it to every terminal scrollback on the machine.
    """
    path = os.path.join(root, ENV_FILE)
    if not os.path.isfile(path):
        return []
    loaded = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and value and not os.environ.get(key):
                os.environ[key] = value
                loaded.append(key)
    return loaded


class Stage:
    def __init__(self, name: str, why: str,
                 build: Callable[[argparse.Namespace], list[str]],
                 gate: bool = False) -> None:
        self.name = name
        self.why = why
        self.build = build
        self.gate = gate


def python() -> str:
    return sys.executable


def stages() -> list[Stage]:
    return [
        Stage("scan", "measure every audio file that has no analysis yet",
              lambda a: [python(), "-m", "mtx", "scan", a.library,
                         "--out", a.root, "--stems",
                         *(["--jobs", str(a.jobs)] if a.jobs else []),
                         *(["--transcribe"] if a.transcribe else []),
                         *(["--embed"] if a.embed else []),
                         *(["--force"] if a.force else [])]),
        Stage("enrich", "look every analysed folder up in the databases",
              lambda a: [python(), os.path.join(HERE, "enrich_fast.py"), a.root,
                         "-j", str(a.net_jobs), "--providers", "all",
                         *(["--force"] if a.force else []),
                         *(["--refresh"] if a.refresh else [])]),
        Stage("identity", "one canonical name and MBID per artist folder",
              lambda a: [python(), os.path.join(HERE, "identity.py"), a.root]),
        Stage("outcome", "position within the artist's own catalogue",
              lambda a: [python(), os.path.join(HERE, "notion", "outcome.py"),
                         a.root]),
        Stage("cohort", "percentiles within genre and era",
              lambda a: [python(), "-m", "mtx", "cohort", a.root,
                         "--neighbours", str(a.neighbours)]),
        Stage("audit", "refuse to publish a corpus that is quietly wrong",
              lambda a: [python(), os.path.join(HERE, "audit.py"), a.root,
                         *(["--notion"] if not a.no_notion_audit else []),
                         *(["--warn-is-error"] if a.strict else [])],
              gate=True),
        Stage("push", "send the tracks and the observations to Notion",
              lambda a: [python(), os.path.join(HERE, "notion", "push.py"),
                         a.root, "-j", str(a.net_jobs),
                         *(["--parent", a.parent] if a.parent else []),
                         *(["--force"] if a.force else []),
                         *(["--prune-options"] if a.prune else [])]),
    ]


def run(stage: Stage, args: argparse.Namespace) -> int:
    cmd = stage.build(args)
    log("")
    log(f"=== {stage.name} :: {stage.why}")
    log(f"    {' '.join(shlex.quote(c) for c in cmd)}")
    if args.dry_run:
        return 0
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [os.path.join(REPO, "src"), env.get("PYTHONPATH")]))
    t0 = time.time()
    code = subprocess.call(cmd, cwd=REPO, env=env)
    log(f"    {stage.name}: exit {code} in {time.time() - t0:,.0f}s")
    return code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--library", default=DEFAULT_LIBRARY,
                    help="the folder of FLACs to measure")
    ap.add_argument("--root", default=DEFAULT_OUT, help="the corpus root")
    ap.add_argument("--from", dest="start", metavar="STAGE",
                    help="begin at this stage instead of the first")
    ap.add_argument("--only", metavar="STAGE", action="append",
                    help="run only these stages (repeatable)")
    ap.add_argument("--skip", metavar="STAGE", action="append", default=[],
                    help="skip these stages (repeatable)")
    ap.add_argument("--jobs", type=int, help="scan workers")
    ap.add_argument("--net-jobs", type=int, default=8,
                    help="enrichment and push workers; the per-host rate "
                         "limits are shared, so this buys latency not volume")
    ap.add_argument("--neighbours", type=int, default=5)
    ap.add_argument("--parent", help="Notion parent page id for a first push")
    ap.add_argument("--force", action="store_true",
                    help="re-do work that is already done, in every stage")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore the HTTP cache and re-fetch")
    ap.add_argument("--prune", action="store_true",
                    help="drop Notion select options nothing uses any more")
    ap.add_argument("--transcribe", action="store_true")
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="let audit warnings stop the run too")
    ap.add_argument("--no-notion-audit", action="store_true",
                    help="audit the corpus on disk only")
    ap.add_argument("--keep-going", action="store_true",
                    help="do not stop when a stage fails (audit still gates)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and run nothing")
    args = ap.parse_args()

    loaded = load_env(args.root)
    if loaded:
        log(f"read {len(loaded)} key(s) from {os.path.join(args.root, ENV_FILE)}: "
            + ", ".join(sorted(loaded)))

    all_stages = stages()
    names = [s.name for s in all_stages]
    for name in (args.only or []) + args.skip + ([args.start] if args.start else []):
        if name not in names:
            log(f"error: unknown stage {name!r}; known: {', '.join(names)}")
            return 2

    chosen = all_stages
    if args.only:
        chosen = [s for s in all_stages if s.name in args.only]
    elif args.start:
        chosen = all_stages[names.index(args.start):]
    chosen = [s for s in chosen if s.name not in args.skip]

    missing = sorted({key for s in chosen for key in NEEDS.get(s.name, ())
                      if not os.environ.get(key)})
    if missing:
        # Named now rather than discovered as a column of blanks in Notion.
        log(f"warning: {', '.join(missing)} not set -- the stages that need "
            f"them will run and quietly collect nothing.  Put them in "
            f"{os.path.join(args.root, ENV_FILE)}.")

    log(f"corpus {args.root}")
    log(f"stages {' -> '.join(s.name for s in chosen)}")

    failed: list[str] = []
    for stage in chosen:
        code = run(stage, args)
        if code == 0:
            continue
        failed.append(stage.name)
        if stage.gate:
            log("")
            log(f"STOP: {stage.name} found something wrong.  Nothing after this "
                f"stage ran.  Read {os.path.join(args.root, 'audit.json')}, fix "
                f"the cause, and run again -- publishing over it would put the "
                f"defect in the evidence base.")
            return 1
        if not args.keep_going:
            log(f"STOP: {stage.name} failed; pass --keep-going to continue anyway")
            return code
    if failed:
        log(f"finished with failures in: {', '.join(failed)}")
        return 1
    log("")
    log("done: every stage clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
