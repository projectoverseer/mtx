"""A guard against empty columns that reported every checkbox as empty.

`notion.dead_column` is the general form of this corpus's worst defect: a
column that is populated, typed, named and reading the wrong key. `Delivery`
sat blank on 1,321 rows for the life of the column because it read
`vocals.delivery.classification` and the value is under `inference`.

The counting loop skipped checkboxes, correctly reasoning that `false` is a
value rather than a blank -- and skipping the count left them at zero, which
is precisely what an empty column looks like. So all eleven checkbox columns
were reported dead on every run: sixteen findings of which five were real.

That is worse than not checking. A list two-thirds noise is a list nobody
reads, and this is the one guarding the defect that keeps recurring.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from audit import column_coverage  # noqa: E402


def page(**props):
    out = {}
    for name, (kind, value) in props.items():
        out[name.replace("_", " ")] = {"type": kind, kind: value}
    return {"properties": out}


def test_a_checkbox_is_never_counted_as_an_empty_column():
    """The regression: `Is single` is false on most rows and dead on none."""
    pages = [page(Is_single=("checkbox", False)),
             page(Is_single=("checkbox", True))]

    got = column_coverage(pages)

    assert got["filled"]["Is single"] == 2


def test_a_checkbox_false_on_every_row_is_still_counted():
    pages = [page(Keys_agree=("checkbox", False)) for _ in range(5)]

    got = column_coverage(pages)

    assert got["filled"]["Keys agree"] == 5, "not dead: every row has an answer"
    assert got["ticked"]["Keys agree"] == 0, "and the answer is always the same"


def test_how_often_a_checkbox_is_ticked_is_tracked_apart():
    pages = ([page(Explicit=("checkbox", True))] * 3
             + [page(Explicit=("checkbox", False))] * 7)

    got = column_coverage(pages)

    assert got["ticked"]["Explicit"] == 3
    assert got["filled"]["Explicit"] == 10


def test_an_empty_rich_text_is_a_blank_not_a_value():
    """Notion returns `[]` for an unset rich_text, which is falsy but present."""
    pages = [page(Mixing_engineer=("rich_text", []))]

    got = column_coverage(pages)

    assert not got["filled"].get("Mixing engineer")
    assert "Mixing engineer" in got["seen"], "seen, so it can be reported dead"


def test_a_filled_rich_text_counts():
    pages = [page(Mixing_engineer=("rich_text",
                                   [{"plain_text": "Serban Ghenea"}]))]

    assert column_coverage(pages)["filled"]["Mixing engineer"] == 1


def test_a_null_number_is_a_blank():
    pages = [page(Billboard_peak=("number", None)),
             page(Billboard_peak=("number", None))]

    got = column_coverage(pages)

    assert not got["filled"].get("Billboard peak")


def test_zero_is_a_measurement_not_a_blank():
    """`0` is falsy and is the answer on a track with no clipped samples."""
    pages = [page(New_overs=("number", 0))]

    assert column_coverage(pages)["filled"]["New overs"] == 1


def test_an_empty_multi_select_is_a_blank():
    pages = [page(Cohort_genres=("multi_select", []))]

    assert not column_coverage(pages)["filled"].get("Cohort genres")


def test_an_unset_select_is_a_blank():
    pages = [page(Certification=("select", None))]

    assert not column_coverage(pages)["filled"].get("Certification")


def test_only_checkboxes_are_listed_as_checkboxes():
    pages = [page(Is_single=("checkbox", True),
                  Cohort=("select", {"name": "house"}))]

    got = column_coverage(pages)

    assert got["checkboxes"] == {"Is single"}


def test_no_pages_reports_nothing_rather_than_everything():
    """An empty query must not read as "every column is dead"."""
    got = column_coverage([])

    assert got["seen"] == set() and not got["filled"]
