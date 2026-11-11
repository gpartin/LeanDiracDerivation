"""Manual clean-rebuild audit for the admitted DIRAC-001--009 chain.

This script is deliberately not a Git hook and never changes the index.  It
checks the repository tree that the caller invokes, runs Lean, audits axioms,
verifies the locked sources and checks the nondegenerate DIRAC-009 evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "project" / "action_lock.json"
STATUS = ROOT / "project" / "status.json"
REVIEW = ROOT / "project" / "no_bolton_review.json"
STABILITY = (
    ROOT
    / "experiments"
    / "dirac009_native_coupling"
    / "candidate_007_l4_nondegenerate_stability.json"
)
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def production_sources() -> list[Path]:
    return sorted((ROOT / "LeanDirac").rglob("*.lean")) + [
        ROOT / "LeanDirac.lean",
        ROOT / "LeanDiracAudit.lean",
    ]


def lean_module_name(path: Path) -> str:
    relative = path.relative_to(ROOT).with_suffix("")
    return ".".join(relative.parts)


def production_modules() -> list[str]:
    return sorted({lean_module_name(path) for path in production_sources()})


def compiled_audit_source() -> str:
    imports = "\n".join(f"import {module}" for module in production_modules())
    reports = "\n".join(
        line.strip()
        for line in (ROOT / "LeanDiracAudit.lean").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip().startswith("#print axioms ")
    )
    return f"{imports}\n\n#audit_lean_dirac_environment\n\n{reports}\n"


def run_compiled_audit() -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".lean",
        prefix="LeanDiracProductionAudit_",
        dir=ROOT,
        delete=False,
    ) as driver:
        driver.write(compiled_audit_source())
        driver_path = Path(driver.name)
    try:
        return run(["lake", "env", "lean", driver_path.name])
    finally:
        driver_path.unlink(missing_ok=True)


def parse_axiom_reports(output: str) -> dict[str, set[str]]:
    reports: dict[str, set[str]] = {}
    pattern = re.compile(
        r"'([A-Za-z_][A-Za-z0-9_.]*)' depends on axioms:\s*\[([^\]]*)\]",
        re.S,
    )
    for name, block in pattern.findall(output):
        dependencies = set(re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", block))
        reports.setdefault(name, set()).update(dependencies)
    for name in re.findall(
        r"'([A-Za-z_][A-Za-z0-9_.]*)' does not depend on any axioms", output
    ):
        reports.setdefault(name, set())
    return reports


def expected_axiom_reports() -> set[str]:
    source = (ROOT / "LeanDiracAudit.lean").read_text(encoding="utf-8")
    return set(re.findall(r"#print\s+axioms\s+([A-Za-z_][A-Za-z0-9_.]*)", source))


def validate_stability(payload: dict) -> list[str]:
    errors: list[str] = []
    if payload.get("verdict") != "PASS":
        errors.append("nondegenerate DIRAC-009 stability verdict is not PASS")
    if payload.get("lattice") != "4x4x4 periodic C19":
        errors.append("DIRAC-009 stability did not use the locked L=4 lattice")
    if payload.get("cycles") != 100:
        errors.append("DIRAC-009 stability did not run 100 cycles")
    runs = payload.get("runs", [])
    if [row.get("updates_per_cycle") for row in runs] != [64, 128, 256]:
        errors.append("DIRAC-009 timestep ensemble is incomplete")
    for row in runs:
        required_maxima = {
            "psi_pattern_defect",
            "chi_pattern_defect",
            "track_pattern_defect",
        }
        maxima = row.get("maximums")
        if not row.get("passes"):
            errors.append(f"stability row failed: {row.get('updates_per_cycle')}")
        finite_scalars = {
            "dt": row.get("dt"),
            "initial_incidence_energy": row.get("initial_incidence_energy"),
            "initial_incidence_residual_norm": row.get(
                "initial_incidence_residual_norm"),
            "maximum_relative_energy_deviation": row.get(
                "maximum_relative_energy_deviation"),
        }
        if any(not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in finite_scalars.values()):
            errors.append("stability row contains a missing or non-finite scalar")
            continue
        if not isinstance(maxima, dict) or set(maxima) != required_maxima:
            errors.append("stability row has incomplete defect metrics")
            continue
        if any(not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in maxima.values()):
            errors.append("stability row contains a non-finite defect metric")
            continue
        if finite_scalars["initial_incidence_energy"] <= 1.0e-14:
            errors.append("incidence energy was inactive in a stability row")
        if finite_scalars["initial_incidence_residual_norm"] <= 1.0e-10:
            errors.append("incidence residual was inactive in a stability row")
        if finite_scalars["maximum_relative_energy_deviation"] >= 1.0e-4:
            errors.append("maximum energy deviation exceeded 1e-4")
        if max(maxima.values()) >= 0.01:
            errors.append("pattern defect exceeded 1%")
    return errors


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


def validate(run_builds: bool = True) -> list[str]:
    errors: list[str] = []
    lock = load(LOCK)
    status = load(STATUS)
    review = load(REVIEW)

    if status.get("highest_passed_gate") != "DIRAC-009":
        errors.append("status does not stop at DIRAC-009")
    if status.get("next_gate") != "DIRAC-010":
        errors.append("DIRAC-010 is not the next gate")
    if status.get("completed") != 9 or status.get("total") != 15:
        errors.append("gate counts are incorrect")
    if review.get("status") != "PASS_ZERO_CRITICAL_HIGH":
        errors.append("critical/high red-team findings remain")
    if review.get("critical_findings") or review.get("high_findings"):
        errors.append("red-team finding lists are not empty")
    if lock.get("publication_theorem") != "LeanDirac.current_highest_gate_is_dirac009":
        errors.append("action lock names the wrong publication theorem")

    actual = {path.relative_to(ROOT).as_posix() for path in production_sources()}
    locked = set(lock.get("files", {}))
    if actual != locked:
        errors.append(
            f"locked source set differs: extra={sorted(actual - locked)}, "
            f"missing={sorted(locked - actual)}"
        )
    for relative, expected in lock.get("files", {}).items():
        path = ROOT / relative
        if not path.exists():
            errors.append(f"locked source missing: {relative}")
        elif sha256(path) != expected:
            errors.append(f"locked source changed: {relative}")

    errors.extend(validate_stability(load(STABILITY)))

    if run_builds:
        build = run(["lake", "build", *production_modules()])
        if build.returncode:
            errors.append(
                "complete production-module build failed\n"
                + build.stdout + build.stderr
            )
        audit = run_compiled_audit()
        if audit.returncode:
            errors.append("Lean axiom audit build failed\n" + audit.stdout + audit.stderr)
        else:
            audit_output = audit.stdout + audit.stderr
            if "LEAN_DIRAC_ENVIRONMENT_AUDIT_PASS" not in audit_output:
                errors.append("compiled-environment axiom audit marker is missing")
            audited_modules = set(re.findall(
                r"LEAN_DIRAC_AUDITED_MODULE\s+([A-Za-z_][A-Za-z0-9_.]*)",
                audit_output,
            ))
            expected_modules = set(production_modules())
            if audited_modules != expected_modules:
                errors.append(
                    "compiled-environment module coverage differs: "
                    f"extra={sorted(audited_modules - expected_modules)}, "
                    f"missing={sorted(expected_modules - audited_modules)}"
                )
            reports = parse_axiom_reports(audit_output)
            expected = expected_axiom_reports()
            missing = expected - set(reports)
            if missing:
                errors.append(f"missing axiom reports: {sorted(missing)}")
            publication = "LeanDirac.current_highest_gate_is_dirac009"
            if publication not in reports:
                errors.append("publication-theorem axiom report is missing")
            unexpected = set().union(*reports.values()) - ALLOWED_AXIOMS \
                if reports else set()
            if unexpected:
                errors.append(f"unexpected axioms: {sorted(unexpected)}")
    return errors


def main() -> int:
    errors = validate(run_builds=True)
    if errors:
        print("PRODUCTION_AUDIT_FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PRODUCTION_AUDIT_PASS")
    print("highest_passed_gate=DIRAC-009")
    print("next_gate=DIRAC-010 status=OPEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
