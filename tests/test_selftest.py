"""The section 7 suite, run under pytest so CI fails when an assertion does."""

from __future__ import annotations

from mtx.selftest import Suite, TESTS, run_selftest


def test_every_selftest_assertion_passes():
    assert run_selftest(verbose=False) == 0, "mtx selftest reported a failure"


def test_suite_reports_failures_as_failures():
    """The harness itself must not swallow a false assertion."""
    s = Suite(verbose=False)
    s.check("deliberately false", False, "measured", "expected")
    assert s.failed == 1 and s.passed == 0


def test_every_named_test_is_registered():
    names = [name for name, _ in TESTS]
    assert len(names) == len(set(names))
    required = ["clipping below full scale (trap #1)", "inter-sample peaks",
                "bass fundamental resolution", "brickwall cutoff detection",
                "effective bit depth", "tempo on a click track"]
    for r in required:
        assert r in names, f"the suite lost its {r!r} case"
