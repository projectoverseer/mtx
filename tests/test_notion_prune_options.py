"""Removing a select option is the operation that once blanked 1,005 rows.

Notion replaces a select property's option list wholesale; there is no patch.
The schema sync sent `{"options": []}` on every run, which deleted every
option in the table and, with them, the value on every page holding one. It
stayed invisible because the same run re-created what it had just destroyed --
until a later change stopped re-creating them.

So the tool that removes options deliberately has to be more careful than the
one that did it by accident. Two properties matter: it never drops an option a
row is using, and it refuses to act on a read that came back empty -- because
an empty read and "nothing uses any option" are indistinguishable, and one of
them means deleting the entire vocabulary.
"""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "notion"))

from prune_options import orphans, retained, used_options  # noqa: E402


def schema(**columns):
    props = {}
    for name, (kind, options) in columns.items():
        props[name.replace("_", " ")] = {
            "type": kind,
            kind: {"options": [{"id": f"id-{o}", "name": o, "color": "default"}
                               for o in options]},
        }
    return {"properties": props}


def page(**props):
    out = {}
    for name, (kind, value) in props.items():
        out[name.replace("_", " ")] = {"type": kind, kind: value}
    return {"properties": out}


def test_an_option_no_row_holds_is_an_orphan():
    sch = schema(Genre=("select", ["house", "techno/house"]))
    pages = [page(Genre=("select", {"name": "house"}))]

    assert orphans(sch, used_options(pages)) == {"Genre": ["techno/house"]}


def test_an_option_a_row_holds_is_never_an_orphan():
    sch = schema(Genre=("select", ["house"]))
    pages = [page(Genre=("select", {"name": "house"}))]

    assert orphans(sch, used_options(pages)) == {}


def test_multi_select_counts_every_option_on_the_row():
    sch = schema(Cohort_genres=("multi_select", ["house", "disco", "skweee"]))
    pages = [page(Cohort_genres=("multi_select",
                                 [{"name": "house"}, {"name": "disco"}]))]

    assert orphans(sch, used_options(pages)) == {"Cohort genres": ["skweee"]}


def test_one_row_out_of_many_is_enough_to_keep_an_option():
    """The whole risk: a rarely used option is not an unused one."""
    sch = schema(Genre=("select", ["house", "berlin school"]))
    pages = [page(Genre=("select", {"name": "house"})) for _ in range(500)]
    pages.append(page(Genre=("select", {"name": "berlin school"})))

    assert orphans(sch, used_options(pages)) == {}


def test_a_column_that_is_not_a_select_is_left_alone():
    sch = schema(Notes=("rich_text", []), Genre=("select", ["dead"]))
    sch["properties"]["Notes"] = {"type": "rich_text", "rich_text": {}}

    assert set(orphans(sch, {})) == {"Genre"}


def test_the_retained_payload_keeps_every_option_it_is_not_dropping():
    sch = schema(Genre=("select", ["house", "disco", "dead"]))

    got = retained(sch, "Genre", {"dead"})

    names = [o["name"] for o in got["Genre"]["select"]["options"]]
    assert names == ["house", "disco"]


def test_the_retained_payload_carries_the_option_ids():
    """An option re-sent without its id is a new option.

    The rows pointing at the old one lose their value -- which is the 1,005-row
    failure, reached by a different route.
    """
    sch = schema(Genre=("select", ["house", "dead"]))

    got = retained(sch, "Genre", {"dead"})

    assert [o["id"] for o in got["Genre"]["select"]["options"]] == ["id-house"]


def test_the_payload_keeps_the_property_kind():
    """Sending `select` options to a `multi_select` would rewrite the column."""
    sch = schema(Cohort_genres=("multi_select", ["house", "dead"]))

    got = retained(sch, "Cohort genres", {"dead"})

    assert "multi_select" in got["Cohort genres"]
    assert "select" not in got["Cohort genres"]


def test_an_empty_read_makes_every_option_look_unused():
    """Documents why the tool refuses to act on it rather than trusting it.

    This is the shape of the accident: with no rows in hand, every option in
    the table is an orphan, and removing them all is exactly the bug.
    """
    sch = schema(Genre=("select", ["house", "disco"]))

    assert orphans(sch, used_options([])) == {"Genre": ["house", "disco"]}
