"""`mtx.env` beside the corpus, loaded by whichever tool needs it.

Only `pipeline.py` used to read this file, so every tool run on its own got
whatever happened to be in the shell. That is fine for `audit.py --notion`,
which stops with `no Notion token` and tells you what is missing. It is not
fine for `transcribe.py`, which reads `MTX_WHISPER_MODEL`, finds nothing,
falls back to `base`, transcribes 1,321 tracks with the small model and
reports success -- a corpus of worse lyrics with no mark anywhere saying so.

The corpus root is the natural home for it. The keys belong to the library,
not to the checkout: `mtx.env` sits with the music, never enters git, and is
found by every tool that is pointed at that music.

Names, never values. A log line that echoes a token has published it to every
terminal scrollback on the machine, and to whatever ships those logs onward.
"""

from __future__ import annotations

import os

ENV_FILE = "mtx.env"


def load_env(root: str) -> list[str]:
    """Set any key in `<root>/mtx.env` that is not already set. Returns names."""
    path = os.path.join(root or ".", ENV_FILE)
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
