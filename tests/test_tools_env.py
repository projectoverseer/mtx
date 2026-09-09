"""The keys live with the music, and every tool pointed at the music finds them.

`mtx.env` sits beside the corpus and never enters git. Only `pipeline.py` read
it, so a tool run on its own got whatever was in the shell -- which is a
different failure for each tool:

  * `audit.py --notion` stops with `no Notion token`. Loud, harmless.
  * `transcribe.py` reads `MTX_WHISPER_MODEL`, finds nothing, falls back to
    `base`, and transcribes the corpus with the small model. It reports
    success. Nothing on disk records which model wrote the words.

The second is the one this exists for, and it is the corpus's usual defect
shape: a run that completes, reports cleanly, and produces worse data than the
one before it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from env import ENV_FILE, load_env  # noqa: E402


def write(root: Path, text: str) -> str:
    (root / ENV_FILE).write_text(text, encoding="utf-8")
    return str(root)


def test_a_key_beside_the_corpus_reaches_the_process(tmp_path, monkeypatch):
    monkeypatch.delenv("MTX_TEST_KEY", raising=False)
    root = write(tmp_path, "MTX_TEST_KEY=abc123\n")

    assert load_env(root) == ["MTX_TEST_KEY"]

    import os
    assert os.environ["MTX_TEST_KEY"] == "abc123"


def test_it_returns_names_and_never_values(tmp_path, monkeypatch):
    """A log line that echoes a token has published it.

    `pipeline.py` prints what this returns, so anything but a name here ends up
    in every terminal scrollback on the machine and in whatever ships logs on.
    """
    monkeypatch.delenv("MTX_TEST_SECRET", raising=False)
    root = write(tmp_path, "MTX_TEST_SECRET=hunter2\n")

    got = load_env(root)

    assert got == ["MTX_TEST_SECRET"]
    assert "hunter2" not in repr(got)


def test_the_shell_wins_over_the_file(tmp_path, monkeypatch):
    """`set VAR=... && tool` is how you override for one run. It must work."""
    monkeypatch.setenv("MTX_TEST_KEY", "from-shell")
    root = write(tmp_path, "MTX_TEST_KEY=from-file\n")

    assert load_env(root) == []

    import os
    assert os.environ["MTX_TEST_KEY"] == "from-shell"


def test_quotes_and_comments_and_blanks(tmp_path, monkeypatch):
    for name in ("MTX_TEST_A", "MTX_TEST_B", "MTX_TEST_C"):
        monkeypatch.delenv(name, raising=False)
    root = write(tmp_path, "\n".join([
        "# a comment",
        "",
        'MTX_TEST_A="double"',
        "MTX_TEST_B='single'",
        "  MTX_TEST_C = spaced  ",
        "not a pair",
    ]) + "\n")

    assert set(load_env(root)) == {"MTX_TEST_A", "MTX_TEST_B", "MTX_TEST_C"}

    import os
    assert os.environ["MTX_TEST_A"] == "double"
    assert os.environ["MTX_TEST_B"] == "single"
    assert os.environ["MTX_TEST_C"] == "spaced", "a stray space is not the key"


def test_no_file_is_not_an_error(tmp_path):
    """Most corpora will not have one, and every tool calls this unconditionally."""
    assert load_env(str(tmp_path)) == []


def test_every_tool_that_reads_a_key_loads_the_file():
    """The regression guard: a new tool that reads a token and not the file.

    Grepping is crude, but the alternative is running seven CLIs, and the
    thing being asserted really is textual -- did whoever added the tool wire
    the loader in.
    """
    tools = Path(__file__).resolve().parents[1] / "tools"
    want = ["audit.py", "transcribe.py", "embed.py", "charts.py",
            "identity.py", "enrich_fast.py", "pipeline.py", "notion/push.py"]

    for rel in want:
        src = (tools / rel).read_text(encoding="utf-8")
        assert "load_env(" in src, f"{rel} never loads {ENV_FILE}"


def test_only_one_loader_exists():
    """Two copies drift, and the copy that drifts is the one nobody runs."""
    tools = Path(__file__).resolve().parents[1] / "tools"
    defs = [p for p in tools.rglob("*.py")
            if "def load_env" in p.read_text(encoding="utf-8")]

    assert [p.name for p in defs] == ["env.py"]
