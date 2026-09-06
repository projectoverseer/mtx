"""Two columns empty on 1,321 rows, with the data one key away.

`online.credits` is keyed by the role names MusicBrainz and Discogs use, and
those are not the words a column heading uses. The block holds `mixing
engineer` on 1,059 of these tracks and `mastering engineer` on 499. The schema
asked for `mixer` and `mastering`. Neither key has ever existed.

Nothing raised. The columns were correctly typed, correctly named, present on
every row, and empty -- which reads as "nobody is credited with mixing these
records", about records that all have a named mixing engineer.
"""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "notion"))

from schema import credit  # noqa: E402


def doc(credits):
    return {"online": {"credits": credits}}


def test_the_role_name_the_providers_actually_write():
    got = credit("mixing engineer", "mixer", "mix")(
        doc({"mixing engineer": [{"name": "Serban Ghenea"}]}))

    assert got == "Serban Ghenea"


def test_mastering_too():
    got = credit("mastering engineer", "mastering")(
        doc({"mastering engineer": [{"name": "Bob Ludwig"}]}))

    assert got == "Bob Ludwig"


def test_the_old_spelling_still_resolves():
    """Aliases, so one provider changing its wording does not empty a column."""
    assert credit("mixing engineer", "mixer")(
        doc({"mixer": [{"name": "Tom Elmhirst"}]})) == "Tom Elmhirst"


def test_the_match_ignores_case_and_padding():
    assert credit("producer")(
        doc({"  Producer ": [{"name": "Nile Rodgers"}]})) == "Nile Rodgers"


def test_several_people_in_one_column():
    got = credit("engineer")(doc({"engineer": [{"name": "A"}, {"name": "B"}]}))

    assert got == "A, B"


def test_one_person_credited_by_two_providers_is_one_person():
    """MusicBrainz and Discogs both name them; the column should not say it twice."""
    got = credit("recording engineer", "engineer")(doc({
        "recording engineer": [{"name": "Bryce Bordone"}],
        "engineer": [{"name": "bryce bordone"}],
    }))

    assert got == "Bryce Bordone"


def test_a_role_nobody_filled_is_none_not_empty_string():
    """Notion writes an empty rich_text for `""`, which reads as a blank cell.

    Same rendering, different meaning: `None` says the credit is absent, and
    `""` says someone was credited with nothing.
    """
    assert credit("mixing engineer")(doc({"producer": [{"name": "X"}]})) is None
    assert credit("mixing engineer")(doc({})) is None
    assert credit("mixing engineer")({}) is None


def test_a_nameless_credit_entry_is_skipped():
    got = credit("producer")(doc({"producer": [{"name": ""}, None,
                                               {"name": "Real Name"}]}))

    assert got == "Real Name"


def test_the_schema_asks_for_role_names_this_corpus_contains():
    """The regression guard, and the reason the bug survived review.

    `credit("mixer")` is readable, plausible, and matches nothing. Only the
    data says which spelling is real, so this checks the wired-up columns
    against the vocabulary the providers actually emit.
    """
    from schema import PROPERTIES

    real_roles = {"main artist", "producer", "mixing engineer", "writer",
                  "engineer", "composer", "lyricist", "recording engineer",
                  "mastering engineer", "programming", "vocals"}
    wired = {p.name: p for p in PROPERTIES
             if p.name in ("Producer", "Mixing engineer", "Mastering engineer",
                           "Recording engineer")}
    assert len(wired) == 4, "a credit column was renamed or dropped"

    for name, prop in wired.items():
        found = [role for role in real_roles
                 if prop.source(doc({role: [{"name": "someone"}]}))]
        assert found, f"{name} matches no role this corpus contains"
