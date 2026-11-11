"""Small negative controls for the manual production audit."""

from __future__ import annotations

import argparse
import copy
import math
import re
import shutil
import tempfile
from pathlib import Path

import production_guard as guard


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-production-build",
        action="store_true",
        help="reuse the complete production build performed by production_guard.py",
    )
    args = parser.parse_args()
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
    external = guard.ROOT / f"{fixture_root.name}_External.lean"
    external_olean = (
        guard.ROOT / ".lake" / "build" / "lib" / "lean"
        / f"{external.stem}.olean"
    )
    try:
        if not args.skip_production_build:
            audit_build = guard.run(["lake", "build", *guard.production_modules()])
            if audit_build.returncode:
                raise AssertionError(
                    "failed to build every production module before auditing:\n"
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

        escaped_identifier_fixtures = (
            'def «"» : Nat := 0\nprivate axiom injected : False\n',
            'def «/-» : Nat := 0\nprivate axiom injected : False\n',
            'def rawText : String := r#"""#\nprivate axiom injected : False\n',
        )
        for source in escaped_identifier_fixtures:
            nested.write_text(source, encoding="utf-8")
            fixture_build = guard.run(["lake", "build", module])
            if fixture_build.returncode:
                raise AssertionError(
                    "failed to compile delimiter fixture:\n"
                    f"{fixture_build.stdout}{fixture_build.stderr}"
                )
            hidden_axiom = guard.run(["lake", "env", "lean", audit.name])
            if hidden_axiom.returncode == 0 or "custom axiom declaration" not in (
                hidden_axiom.stdout + hidden_axiom.stderr
            ):
                raise AssertionError(
                    "compiled-environment audit accepted an axiom after a delimiter fixture"
                )

        external_module = external.stem
        external.write_text("axiom importedAxiom : False\n", encoding="utf-8")
        external_build = guard.run([
            "lake", "env", "lean", external.name,
            "-o", str(external_olean),
        ])
        if external_build.returncode:
            raise AssertionError(
                "failed to compile external axiom fixture:\n"
                f"{external_build.stdout}{external_build.stderr}"
            )
        nested.write_text(
            f"import {external_module}\n"
            "theorem imported_dependency : False := by exact importedAxiom\n",
            encoding="utf-8",
        )
        fixture_build = guard.run(["lake", "build", module])
        if fixture_build.returncode:
            raise AssertionError(
                "failed to compile imported-axiom fixture:\n"
                f"{fixture_build.stdout}{fixture_build.stderr}"
            )
        imported_axiom = guard.run(["lake", "env", "lean", audit.name])
        if imported_axiom.returncode == 0 or "depends on nonstandard axiom" not in (
            imported_axiom.stdout + imported_axiom.stderr
        ):
            raise AssertionError(
                "compiled-environment audit accepted an imported nonstandard axiom"
            )

        nested.write_text("def coverageSentinel : Nat := 1\n", encoding="utf-8")
        fixture_build = guard.run(["lake", "build", module])
        if fixture_build.returncode:
            raise AssertionError(
                "failed to rebuild clean coverage fixture:\n"
                f"{fixture_build.stdout}{fixture_build.stderr}"
            )
        coverage = guard.run_compiled_audit()
        if coverage.returncode:
            raise AssertionError(
                "generated all-module audit failed:\n"
                f"{coverage.stdout}{coverage.stderr}"
            )
        audited_modules = set(re.findall(
            r"LEAN_DIRAC_AUDITED_MODULE\s+([A-Za-z_][A-Za-z0-9_.]*)",
            coverage.stdout + coverage.stderr,
        ))
        expected_modules = set(guard.production_modules())
        if audited_modules != expected_modules:
            raise AssertionError(
                "generated audit did not cover every production module: "
                f"missing={sorted(expected_modules - audited_modules)}, "
                f"extra={sorted(audited_modules - expected_modules)}"
            )
    finally:
        audit.unlink(missing_ok=True)
        external.unlink(missing_ok=True)
        external_olean.unlink(missing_ok=True)
        shutil.rmtree(fixture_root)
    print("ADVERSARIAL_GUARD_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
