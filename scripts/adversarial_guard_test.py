"""Adversarial controls for the manual production audit."""

from __future__ import annotations

import argparse
import copy
import math

import production_guard as guard


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-production-build", action="store_true")
    parser.parse_args()
    reports = guard.parse_axiom_reports(
        "'x' depends on axioms: [propext, Classical.choice, Quot.sound]"
    )
    if reports.get("x") != guard.ALLOWED_AXIOMS:
        raise AssertionError(f"axiom parser mismatch: {reports}")
    bad = guard.parse_axiom_reports("'x' depends on axioms: [propext, sorryAx]")
    if "sorryAx" not in bad.get("x", set()):
        raise AssertionError("sorryAx escaped the axiom parser")
    duplicate = guard.parse_axiom_reports(
        "'x' depends on axioms: [propext, sorryAx]\n"
        "'x' does not depend on any axioms"
    )
    if "sorryAx" not in duplicate.get("x", set()):
        raise AssertionError("duplicate report erased an axiom dependency")
    payload = guard.load(guard.EVIDENCE_010)
    if guard.validate_dirac010_evidence(payload):
        raise AssertionError("passing DIRAC-010 evidence was rejected")
    broken = copy.deepcopy(payload)
    broken["status"] = "FAIL"
    if not guard.validate_dirac010_evidence(broken):
        raise AssertionError("failed evidence was accepted")
    nonfinite = copy.deepcopy(payload)
    nonfinite["max_error"] = math.nan
    if not guard.validate_dirac010_evidence(nonfinite):
        raise AssertionError("NaN evidence was accepted")
    missing = copy.deepcopy(payload)
    missing["errors"] = {}
    if not guard.validate_dirac010_evidence(missing):
        raise AssertionError("incomplete component evidence was accepted")
    payload_011 = guard.load(guard.EVIDENCE_011)
    if guard.validate_dirac011_evidence(payload_011):
        raise AssertionError("passing DIRAC-011 evidence was rejected")
    broken_011 = copy.deepcopy(payload_011)
    broken_011["max_error"] = broken_011["tolerance"] * 2
    if not guard.validate_dirac011_evidence(broken_011):
        raise AssertionError("out-of-tolerance DIRAC-011 evidence was accepted")
    nonfinite_011 = copy.deepcopy(payload_011)
    nonfinite_011["errors"][next(iter(nonfinite_011["errors"]))] = math.inf
    if not guard.validate_dirac011_evidence(nonfinite_011):
        raise AssertionError("non-finite DIRAC-011 component was accepted")
    print("ADVERSARIAL_GUARD_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
