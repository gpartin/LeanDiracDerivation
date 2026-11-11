"""Small negative controls for the manual production audit."""

from __future__ import annotations

import copy
import math

import production_guard as guard


def main() -> int:
    for token in (
        "by sorry",
        "by admit",
        "axiom injected : Prop",
        "private axiom injected : False",
        "@[irreducible] axiom injected : False",
    ):
        if not guard.forbidden_tokens(token):
            raise AssertionError(f"forbidden token escaped detection: {token}")
    if guard.forbidden_tokens("theorem clean : True := by trivial"):
        raise AssertionError("clean Lean source was rejected")
    reports = guard.parse_axiom_reports(
        "'x' depends on axioms: [propext, Classical.choice, Quot.sound]"
    )
    if reports.get("x") != guard.ALLOWED_AXIOMS:
        raise AssertionError(f"axiom parser mismatch: {reports}")
    bad = guard.parse_axiom_reports("'x' depends on axioms: [propext, sorryAx]")
    if "sorryAx" not in bad.get("x", set()):
        raise AssertionError("sorryAx escaped the axiom parser")
    payload = guard.load(guard.STABILITY)
    if guard.validate_stability(payload):
        raise AssertionError("passing stability evidence was rejected")
    broken = copy.deepcopy(payload)
    broken["verdict"] = "FAIL"
    if not guard.validate_stability(broken):
        raise AssertionError("failed stability evidence was accepted")
    incomplete = copy.deepcopy(payload)
    del incomplete["runs"][0]["maximums"]["track_pattern_defect"]
    if not guard.validate_stability(incomplete):
        raise AssertionError("incomplete defect metrics were accepted")
    nonfinite = copy.deepcopy(payload)
    nonfinite["runs"][0]["maximum_relative_energy_deviation"] = math.nan
    if not guard.validate_stability(nonfinite):
        raise AssertionError("NaN energy evidence was accepted")
    nested = guard.ROOT / "LeanDirac" / "Nested" / "Injected.lean"
    nested.parent.mkdir(parents=True, exist_ok=True)
    try:
        nested.write_text("private axiom injected : False\n", encoding="utf-8")
        if nested not in guard.production_sources():
            raise AssertionError("nested Lean source escaped recursive enumeration")
        if not guard.forbidden_tokens(nested.read_text(encoding="utf-8")):
            raise AssertionError("nested custom axiom escaped forbidden-token scan")
    finally:
        nested.unlink(missing_ok=True)
        nested.parent.rmdir()
    print("ADVERSARIAL_GUARD_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
