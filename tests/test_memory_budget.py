"""What a scan is allowed to hold, and how many workers are worth starting.

These exist because the arithmetic here decides whether an unattended run
finishes or dies.  Sized against the memory the machine *has* rather than the
memory that is *free*, `memory_budget` authorised 28 GB on a 34 GB machine
with 26 GB free and 8 GB already in use.  The admission gate then did exactly
as it was told, Windows killed a worker, and the resulting `BrokenProcessPool`
failed all 820 remaining tracks four minutes into a seven-hour run.
"""
import pytest

from mtx import parallel


@pytest.fixture
def machine(monkeypatch):
    """A machine with a stated total and free memory."""
    def set(total, free):
        monkeypatch.setattr(parallel, "total_memory_bytes", lambda: total)
        monkeypatch.setattr(parallel, "available_memory_bytes", lambda: free)
    return set


def test_the_budget_is_sized_against_free_memory_not_total(machine):
    """The bug, stated as a test: 8 GB in use has to come off the top."""
    machine(total=34_000_000_000, free=26_000_000_000)
    budget = parallel.memory_budget(streams=0, procs=0)
    assert budget <= 26_000_000_000 - parallel.MEMORY_RESERVE
    assert budget < 28_000_000_000, "must not authorise more than exists"


def test_an_empty_machine_is_still_bounded_by_its_own_reserve(machine):
    """With everything free, the ceiling is total minus the reserve."""
    machine(total=34_000_000_000, free=34_000_000_000)
    assert parallel.memory_budget() == 34_000_000_000 - parallel.MEMORY_RESERVE


def test_workers_are_charged_for_what_they_hold_before_decoding(machine):
    """824 MB of interpreter and numpy each is not free, and was counted as 0."""
    machine(total=34_000_000_000, free=26_000_000_000)
    alone = parallel.memory_budget(procs=0)
    six = parallel.memory_budget(procs=6)
    assert alone - six == 6 * parallel.WORKER_RESERVE


def test_separation_streams_are_charged_too(machine):
    machine(total=34_000_000_000, free=26_000_000_000)
    assert (parallel.memory_budget(streams=0) - parallel.memory_budget(streams=1)
            == parallel.SEPARATION_RESERVE)


def test_a_machine_whose_memory_cannot_be_read_disables_the_gate(monkeypatch):
    """Zero means "schedule by worker count", which is the old behaviour."""
    monkeypatch.setattr(parallel, "total_memory_bytes", lambda: 0)
    assert parallel.memory_budget(streams=1, procs=6) == 0


def test_a_starved_machine_narrows_instead_of_disabling_the_gate(machine):
    """A tiny budget must not read as zero: zero turns the gate off entirely.

    A machine this short of memory needs the admission gate more than a roomy
    one does, so the floor keeps it on and `scan.drive` degrades to one track
    at a time.
    """
    machine(total=34_000_000_000, free=3_500_000_000)
    budget = parallel.memory_budget(streams=1, procs=6)
    assert budget > 0, "the gate stays on"
    assert budget == parallel.SEPARATION_RESERVE


def test_no_more_workers_are_started_than_memory_can_run(machine):
    """Six lanes of a 2.8 GB track do not fit beside six workers in 26 GB."""
    machine(total=34_000_000_000, free=26_000_000_000)
    procs = parallel.workers_that_fit(6, typical=2_800_000_000, streams=1)
    assert 1 < procs < 6
    assert (procs * 2_800_000_000
            <= parallel.memory_budget(streams=1, procs=procs)), \
        "the answer fits inside the budget its own worker count produces"


def test_a_roomy_machine_keeps_every_worker_it_was_asked_for(machine):
    machine(total=128_000_000_000, free=120_000_000_000)
    assert parallel.workers_that_fit(6, typical=2_800_000_000, streams=1) == 6


def test_the_pool_is_sized_on_the_typical_track_not_the_largest(machine):
    """One 9.6 GB master must not collapse the pool for the other 820 tracks."""
    machine(total=34_000_000_000, free=26_000_000_000)
    on_typical = parallel.workers_that_fit(6, typical=2_800_000_000, streams=1)
    on_largest = parallel.workers_that_fit(6, typical=9_650_000_000, streams=1)
    assert on_largest == 1
    assert on_typical > on_largest, \
        "the gate handles the outlier; the pool is sized for the common case"


def test_unreadable_memory_leaves_the_worker_count_alone(monkeypatch):
    monkeypatch.setattr(parallel, "total_memory_bytes", lambda: 0)
    assert parallel.workers_that_fit(6, typical=2_800_000_000) == 6
