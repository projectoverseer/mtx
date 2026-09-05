"""What the push state remembers, and what it used to forget.

The state file maps a track's sha256 to the Notion page it became, so a
resumed run costs no requests to work out where it stopped. That much worked.

What it did not record is *what was pushed*. The skip check asked only
whether a sha had ever been sent, so an amended analysis -- 1,306 new
transcripts, 66 repaired lyrics, a fresh set of cohort percentiles -- came
back `0 pushed, 1321 already present`, exit 0, `done: every stage clean`, and
25 columns that stayed empty in Notion while every stage reported success.

The push is the one tool here that writes to live tables, and it had no tests
at all. These cover the decision that governs every one of those writes.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "tools")
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(TOOLS, "notion"))

push = pytest.importorskip("push")


# --- the decision ------------------------------------------------------------

def test_a_track_never_pushed_is_pushed():
    assert push.needs_push("sha1", "stamp-a", {}, {}) is True


def test_an_unchanged_track_is_skipped():
    assert push.needs_push("sha1", "stamp-a", {"sha1": "page"},
                           {"sha1": "stamp-a"}) is False


def test_a_changed_track_is_pushed_again():
    """The regression: this returned False and the amendment never shipped."""
    assert push.needs_push("sha1", "stamp-b", {"sha1": "page"},
                           {"sha1": "stamp-a"}) is True


def test_a_track_pushed_before_stamps_existed_is_pushed():
    """No stamp recorded means no evidence of what is live: send it."""
    assert push.needs_push("sha1", "stamp-a", {"sha1": "page"}, {}) is True


def test_a_track_whose_page_is_unknown_is_pushed():
    """A stamp without a page is not a published row."""
    assert push.needs_push("sha1", "stamp-a", {}, {"sha1": "stamp-a"}) is True


def test_force_overrides_everything():
    assert push.needs_push("sha1", "stamp-a", {"sha1": "page"},
                           {"sha1": "stamp-a"}, force=True) is True


# --- the fingerprints --------------------------------------------------------

def folder(tmp_path, **files):
    d = str(tmp_path)
    for name, body in files.items():
        with open(os.path.join(d, name), "w", encoding="utf-8") as fh:
            fh.write(body)
    return d


def test_amending_an_analysis_changes_the_folder_stamp(tmp_path):
    """A transcription pass is an amendment, and must be visible as one."""
    d = folder(tmp_path, **{"analysis.json": '{"lyrics": {}}',
                            "online.json": "{}", "corpus_row.json": "{}",
                            "mtx_source.json": "{}"})
    before = push.folder_stamp(d)

    with open(os.path.join(d, "analysis.json"), "w", encoding="utf-8") as fh:
        fh.write('{"lyrics": {"text": "a transcript now"}}')

    assert push.folder_stamp(d) != before


def test_a_missing_file_does_not_raise(tmp_path):
    """A folder mid-write must not take the whole run down."""
    d = folder(tmp_path, **{"analysis.json": "{}"})

    assert isinstance(push.folder_stamp(d), str)
    assert push.folder_stamp(d) == push.folder_stamp(d), "and it is stable"


def test_the_shared_stamp_follows_content_not_mtime(tmp_path):
    """`cohort.json` is rewritten every run whether or not a number moved.

    Stamping it by modification time would force a full 1,321-page re-push
    daily for nothing -- which is its own kind of wrong answer, and the kind
    that trains people to stop reading the output.
    """
    root = folder(tmp_path, **{"cohort.json": '{"cohorts": 742}',
                               "outcome.json": "{}", "artists.json": "{}"})
    before = push.shared_stamp(root)

    # Rewritten, same bytes: a new mtime and nothing to republish.
    with open(os.path.join(root, "cohort.json"), "w", encoding="utf-8") as fh:
        fh.write('{"cohorts": 742}')
    assert push.shared_stamp(root) == before

    # Rewritten with a different number: every page reads this.
    with open(os.path.join(root, "cohort.json"), "w", encoding="utf-8") as fh:
        fh.write('{"cohorts": 750}')
    assert push.shared_stamp(root) != before


def test_the_state_file_survives_a_missing_stamps_key(tmp_path):
    """State written before stamps existed must load, not crash the push."""
    path = os.path.join(str(tmp_path), "state.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"tracks": {"sha1": "page"}, "databases": {}}, fh)

    state = push.State(path)

    assert state.data["tracks"] == {"sha1": "page"}
    assert state.data.setdefault("stamps", {}) == {}


# --- create or update --------------------------------------------------------

class _Api:
    """Records which Notion call a push would make."""

    def __init__(self):
        self.calls = []

    def update_page(self, page_id, props):
        self.calls.append(("update", page_id))

    def create_page(self, db_id, props, blocks=None):
        self.calls.append(("create", db_id))
        return {"id": "new-page"}

    def append_blocks(self, page_id, blocks):
        pass


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(push, "properties_for", lambda doc: {})
    monkeypatch.setattr(push, "body_blocks", lambda doc: [])
    return _Api()


def test_a_known_page_is_updated_not_duplicated(api):
    """A re-push of a changed track must edit its page, never add a second.

    `push_track` creates whenever it is handed no page id, and the call site
    used to pass one only under `--force`.  That was safe only while a track
    already in the state could never reach the worker -- which stopped being
    true the moment a changed stamp could put it there.  Every amended
    analysis would have published a duplicate page beside the original.
    """
    got = push.push_track(api, "db", {}, "existing-page", False)

    assert api.calls == [("update", "existing-page")]
    assert got == "existing-page"


def test_an_unknown_track_creates_a_page(api):
    got = push.push_track(api, "db", {}, None, False)

    assert api.calls == [("create", "db")]
    assert got == "new-page"


# --- which failures are worth waiting out ------------------------------------

client = pytest.importorskip("client")


def _http_error(code):
    import urllib.error
    import io as _io
    return urllib.error.HTTPError(
        "https://api.notion.com/v1/pages/x", code, "boom",
        {"Retry-After": "0"}, _io.BytesIO(b'{"code":"conflict_error"}'))


def run_with(monkeypatch, codes):
    """Drive the client through a scripted sequence of HTTP outcomes."""
    calls = {"n": 0}
    slept = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    def fake_open(req, timeout=None):
        i = calls["n"]
        calls["n"] += 1
        if i < len(codes):
            raise _http_error(codes[i])
        return _Resp()

    monkeypatch.setattr(client.urllib.request, "urlopen", fake_open)
    monkeypatch.setattr(client.time, "sleep", lambda s: slept.append(s))
    api = client.Notion("token", log=lambda _m: None)
    return api, calls, slept


def test_a_conflict_is_retried(monkeypatch):
    """409 is Notion refusing a concurrent write, not a bad request.

    13 pages were lost to it at eight workers and none at three, with
    identical payloads.  A lost page keeps yesterday's numbers while the run
    reports a failure count small enough to read as noise.
    """
    api, calls, slept = run_with(monkeypatch, [409, 409])

    got = api.request("PATCH", "/pages/x", {"properties": {}})

    assert got == {}
    assert calls["n"] == 3, "two conflicts, then the write that worked"
    assert all(s > 0 for s in slept), "and it waited between them"


def test_a_bad_request_is_not_retried(monkeypatch):
    """400 is our own bug; retrying only makes the same mistake more slowly."""
    api, calls, _slept = run_with(monkeypatch, [400, 400])

    with pytest.raises(client.NotionError):
        api.request("PATCH", "/pages/x", {"properties": {}})

    assert calls["n"] == 1, "no retry"


# --- the schema sync ---------------------------------------------------------

def test_only_missing_properties_are_sent():
    """An existing select must not appear in a schema update at all.

    `database_schema()` describes a select as `{"options": []}`, which is what
    *creating* one needs.  Sent at an existing database, Notion reads the empty
    list as "these are the options now" and deletes every one -- and deleting
    an option blanks it on every page holding it.  That ran on every push,
    wiping all 22 select columns, and hid behind the same run rewriting all
    1,321 pages and re-creating them on the way through.

    The moment a run wrote only what had changed, the restore covered 13
    pages: 46 artists gone and 1,005 rows blanked.
    """
    live = {"properties": {"Artist": {"type": "select"},
                           "LUFS-I": {"type": "number"}}}
    wanted = {"Artist": {"select": {"options": []}},
              "LUFS-I": {"number": {}},
              "Lyric language": {"select": {"options": []}}}

    assert push.new_properties(live, wanted) == {
        "Lyric language": {"select": {"options": []}}}


def test_a_database_that_matches_gets_no_update():
    live = {"properties": {"Artist": {"type": "select"}}}

    assert push.new_properties(live, {"Artist": {"select": {"options": []}}}) == {}


def test_a_database_with_no_properties_gets_everything():
    wanted = {"Artist": {"select": {"options": []}}}

    assert push.new_properties({}, wanted) == wanted


def test_the_sync_never_calls_update_when_nothing_is_missing(monkeypatch):
    """No request at all is the only way to be sure nothing was touched."""
    calls = []

    class _Api:
        def request(self, method, path, body=None):
            return {"properties": {"Artist": {"type": "select"}}}

        def update_database(self, db_id, payload):
            calls.append(payload)

    added = push.add_new_properties(_Api(), "db",
                                    {"Artist": {"select": {"options": []}}})

    assert added == []
    assert calls == [], "an existing schema must produce no write"
