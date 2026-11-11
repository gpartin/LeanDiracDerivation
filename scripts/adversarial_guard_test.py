"""Small negative controls for the manual production audit."""

from __future__ import annotations

import copy
import math
import shutil
import tempfile
from pathlib import Path

import production_guard as guard


def main() -> int:
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
        raise AssertionError("duplicate empty report erased an axiom dependency")
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
    missing_dt = copy.deepcopy(payload)
    del missing_dt["runs"][0]["dt"]
    if not guard.validate_stability(missing_dt):
        raise AssertionError("missing timestep evidence was accepted")
    nonfinite_dt = copy.deepcopy(payload)
    nonfinite_dt["runs"][0]["dt"] = math.inf
    if not guard.validate_stability(nonfinite_dt):
        raise AssertionError("non-finite timestep evidence was accepted")
    fixture_root = Path(tempfile.mkdtemp(
        prefix="AuditFixture_", dir=guard.ROOT / "LeanDirac"
    ))
    nested = fixture_root / "Injected.lean"
    audit = guard.ROOT / f"{fixture_root.name}_Audit.lean"
    module = f"LeanDirac.{fixture_root.name}.Injected"
    try:
        audit_build = guard.run(["lake", "build", "LeanDirac.EnvironmentAudit"])
        if audit_build.returncode:
            raise AssertionError(
                "failed to build compiled-environment audit module:\n"
                f"{audit_build.stdout}{audit_build.stderr}"
            )
        nested.write_text("private axiom injected : False\n", encoding="utf-8")
        if nested not in guard.production_sources():
            raise AssertionError("nested Lean source escaped recursive enumeration")
        fixture_build = guard.run(["lake", "build", module])
        if fixture_build.returncode:
            raise AssertionError(
                "failed to compile custom-axiom fixture:\n"
                f"{fixture_build.stdout}{fixture_build.stderr}"
            )
        audit.write_text(
            "import LeanDirac.EnvironmentAudit\n"
            f"import {module}\n"
            "#audit_lean_dirac_environment\n",
            encoding="utf-8",
        )
        custom_axiom = guard.run(["lake", "env", "lean", audit.name])
        if custom_axiom.returncode == 0 or "custom axiom declaration" not in (
            custom_axiom.stdout + custom_axiom.stderr
        ):
            raise AssertionError(
                "compiled-environment audit accepted a custom axiom: "
                f"exit={custom_axiom.returncode}\n{custom_axiom.stdout}{custom_axiom.stderr}"
            )

        nested.write_text(
            'def interpolation : String := s!"{(by sorry : Nat)}"\n',
            encoding="utf-8",
        )
        fixture_build = guard.run(["lake", "build", module])
        if fixture_build.returncode:
            raise AssertionError(
                "failed to compile interpolated-sorry fixture:\n"
                f"{fixture_build.stdout}{fixture_build.stderr}"
            )
        hidden_sorry = guard.run(["lake", "env", "lean", audit.name])
        if hidden_sorry.returncode == 0 or "contains sorry" not in (
            hidden_sorry.stdout + hidden_sorry.stderr
        ):
            raise AssertionError("compiled-environment audit accepted an interpolated sorry")
    finally:
        audit.unlink(missing_ok=True)
        shutil.rmtree(fixture_root)
    print("ADVERSARIAL_GUARD_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
