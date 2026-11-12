"""One-command release audit for the admitted DIRAC-001--013 chain.

This is a read-only verifier, not a Git hook.
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
FRAMEWORK_LOCK = ROOT / "project" / "framework_lock.json"
PROOF_MANIFEST = ROOT / "project" / "proof_manifest.json"
EVIDENCE_MANIFEST = ROOT / "project" / "evidence_manifest.json"
RELEASE_MANIFEST = ROOT / "project" / "release_manifest.json"
STATUS = ROOT / "project" / "status.json"
REVIEW = ROOT / "project" / "no_bolton_review.json"
EVIDENCE_010 = ROOT / "experiments" / "dirac010_standard_map" / "result.json"
EVIDENCE_011 = ROOT / "experiments" / "dirac011_frequency_mass" / "result.json"
EVIDENCE_012 = ROOT / "experiments" / "dirac012_continuum_evolution" / "result.json"
EVIDENCE_013 = ROOT / "experiments" / "dirac013_density_current" / "result.json"
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def validate_manifest(path: Path) -> tuple[dict, list[str]]:
    payload = load(path)
    errors: list[str] = []
    for relative, expected in payload.get("files", {}).items():
        source = ROOT / relative
        if not source.is_file():
            errors.append(f"{path.name}: missing file: {relative}")
        elif sha256(source) != expected:
            errors.append(f"{path.name}: changed file: {relative}")
    return payload, errors


def production_sources() -> list[Path]:
    return sorted((ROOT / "LeanDirac").rglob("*.lean")) + [
        ROOT / "LeanDirac.lean", ROOT / "LeanDiracAudit.lean"
    ]


def lean_module_name(path: Path) -> str:
    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


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


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=ROOT, text=True, encoding="utf-8", errors="replace",
        capture_output=True, check=False
    )


def run_compiled_audit() -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".lean",
        prefix="LeanDiracProductionAudit_", dir=ROOT, delete=False
    ) as driver:
        driver.write(compiled_audit_source())
        path = Path(driver.name)
    try:
        return run(["lake", "env", "lean", path.name])
    finally:
        path.unlink(missing_ok=True)


def parse_axiom_reports(output: str) -> dict[str, set[str]]:
    reports: dict[str, set[str]] = {}
    pattern = re.compile(
        r"'([A-Za-z_][A-Za-z0-9_.]*)' depends on axioms:\s*\[([^\]]*)\]",
        re.S,
    )
    for name, block in pattern.findall(output):
        reports.setdefault(name, set()).update(
            re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", block)
        )
    for name in re.findall(
        r"'([A-Za-z_][A-Za-z0-9_.]*)' does not depend on any axioms", output
    ):
        reports.setdefault(name, set())
    return reports


def expected_axiom_reports() -> set[str]:
    source = (ROOT / "LeanDiracAudit.lean").read_text(encoding="utf-8")
    return set(re.findall(r"#print\s+axioms\s+([A-Za-z_][A-Za-z0-9_.]*)", source))


def validate_dirac010_evidence(payload: dict) -> list[str]:
    errors: list[str] = []
    if payload.get("status") != "PASS":
        errors.append("DIRAC-010 independent matrix verdict is not PASS")
    value = payload.get("max_error")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        errors.append("DIRAC-010 maximum error is missing or non-finite")
    elif value > 1.0e-12:
        errors.append("DIRAC-010 maximum error exceeds 1e-12")
    errors_map = payload.get("errors")
    if not isinstance(errors_map, dict) or not errors_map:
        errors.append("DIRAC-010 component error table is missing")
    elif any(not isinstance(v, (int, float)) or not math.isfinite(v)
             for v in errors_map.values()):
        errors.append("DIRAC-010 component errors are incomplete or non-finite")
    return errors


def validate_dirac011_evidence(payload: dict) -> list[str]:
    errors: list[str] = []
    if payload.get("status") != "PASS":
        errors.append("DIRAC-011 independent verdict is not PASS")
    value = payload.get("max_error")
    tolerance = payload.get("tolerance")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        errors.append("DIRAC-011 maximum error is missing or non-finite")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance):
        errors.append("DIRAC-011 tolerance is missing or non-finite")
    elif tolerance > 1.0e-11 or tolerance <= 0:
        errors.append("DIRAC-011 tolerance is outside the admitted range")
    if isinstance(value, (int, float)) and math.isfinite(value) and \
            isinstance(tolerance, (int, float)) and math.isfinite(tolerance) and \
            value > tolerance:
        errors.append("DIRAC-011 maximum error exceeds its tolerance")
    errors_map = payload.get("errors")
    if not isinstance(errors_map, dict) or not errors_map:
        errors.append("DIRAC-011 component error table is missing")
    elif any(not isinstance(v, (int, float)) or not math.isfinite(v)
             for v in errors_map.values()):
        errors.append("DIRAC-011 component errors are incomplete or non-finite")
    cases = payload.get("cases")
    if not isinstance(cases, int) or cases < 1:
        errors.append("DIRAC-011 case count is missing")
    return errors


def validate_dirac012_evidence(payload: dict) -> list[str]:
    errors: list[str] = []
    if payload.get("status") != "PASS":
        errors.append("DIRAC-012 independent verdict is not PASS")
    value = payload.get("max_error")
    tolerance = payload.get("tolerance")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        errors.append("DIRAC-012 maximum error is missing or non-finite")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance):
        errors.append("DIRAC-012 tolerance is missing or non-finite")
    elif tolerance > 1.0e-11 or tolerance <= 0:
        errors.append("DIRAC-012 tolerance is outside the admitted range")
    if isinstance(value, (int, float)) and math.isfinite(value) and \
            isinstance(tolerance, (int, float)) and math.isfinite(tolerance) and \
            value > tolerance:
        errors.append("DIRAC-012 maximum error exceeds its tolerance")
    checks = payload.get("checks")
    required_checks = {
        "identity_error_within_tolerance",
        "every_proved_bound_holds",
        "residual_decreases_under_joint_refinement",
        "normalized_equation_error_decreases_under_joint_refinement",
        "c19_direction_count_is_18",
        "c19_weight_sum_is_24",
    }
    if not isinstance(checks, dict) or set(checks) != required_checks or \
            not all(value is True for value in checks.values()):
        errors.append("DIRAC-012 required checks are incomplete or false")
    errors_map = payload.get("errors")
    if not isinstance(errors_map, dict) or not errors_map:
        errors.append("DIRAC-012 component error table is missing")
    elif any(not isinstance(v, (int, float)) or not math.isfinite(v)
             for v in errors_map.values()):
        errors.append("DIRAC-012 component errors are incomplete or non-finite")
    for field in (
        "minimum_bound_margin",
        "minimum_normalized_bound_margin",
        "minimum_refinement_ratio",
        "minimum_normalized_refinement_ratio",
    ):
        metric = payload.get(field)
        if not isinstance(metric, (int, float)) or not math.isfinite(metric):
            errors.append(f"DIRAC-012 {field} is missing or non-finite")
    if isinstance(payload.get("minimum_normalized_bound_margin"), (int, float)) and \
            payload["minimum_normalized_bound_margin"] < -1.0e-12:
        errors.append("DIRAC-012 normalized bound is violated")
    cases = payload.get("cases")
    if not isinstance(cases, int) or cases < 32:
        errors.append("DIRAC-012 case count is incomplete")
    return errors


def validate_dirac013_evidence(payload: dict) -> list[str]:
    errors: list[str] = []
    if payload.get("status") != "PASS":
        errors.append("DIRAC-013 independent verdict is not PASS")
    value = payload.get("max_error")
    tolerance = payload.get("tolerance")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        errors.append("DIRAC-013 maximum error is missing or non-finite")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance):
        errors.append("DIRAC-013 tolerance is missing or non-finite")
    elif tolerance > 2.0e-9 or tolerance <= 0:
        errors.append("DIRAC-013 tolerance is outside the admitted range")
    if isinstance(value, (int, float)) and math.isfinite(value) and \
            isinstance(tolerance, (int, float)) and math.isfinite(tolerance) and \
            value > tolerance:
        errors.append("DIRAC-013 maximum error exceeds its tolerance")
    required_checks = {
        "action_phase_variation_reconstructed",
        "reverse_bond_law_reconstructed",
        "exact_local_continuity_reconstructed",
        "local_action_divergence_reconstructed",
        "positive_density_normalization_reconstructed",
        "finite_c19_flux_reconstructed",
        "velocity_from_hamiltonian_reconstructed",
        "dirac_current_identity_reconstructed",
        "both_cube_copy_density_scales_reconstructed",
        "c19_flux_slope_converges_quadratically",
        "c19_direction_count_is_18",
        "c19_weight_sum_is_24",
    }
    checks = payload.get("checks")
    if not isinstance(checks, dict) or set(checks) != required_checks or \
            not all(check is True for check in checks.values()):
        errors.append("DIRAC-013 required checks are incomplete or false")
    errors_map = payload.get("errors")
    if not isinstance(errors_map, dict) or not errors_map:
        errors.append("DIRAC-013 component error table is missing")
    elif any(not isinstance(v, (int, float)) or not math.isfinite(v)
             for v in errors_map.values()):
        errors.append("DIRAC-013 component errors are incomplete or non-finite")
    ratio = payload.get("minimum_flux_refinement_ratio")
    if not isinstance(ratio, (int, float)) or not math.isfinite(ratio):
        errors.append("DIRAC-013 flux refinement ratio is missing or non-finite")
    elif ratio <= 3.9:
        errors.append("DIRAC-013 C19 flux slope does not converge quadratically")
    action_fd = payload.get("max_action_divergence_fd_error")
    if not isinstance(action_fd, (int, float)) or not math.isfinite(action_fd):
        errors.append("DIRAC-013 action-divergence error is missing or non-finite")
    elif action_fd > 2.0e-7:
        errors.append("DIRAC-013 action-divergence reconstruction exceeds tolerance")
    cases = payload.get("cases")
    if not isinstance(cases, int) or cases < 32:
        errors.append("DIRAC-013 case count is incomplete")
    return errors


def validate(run_builds: bool = True) -> list[str]:
    errors: list[str] = []
    framework, e1 = validate_manifest(FRAMEWORK_LOCK)
    proof, e2 = validate_manifest(PROOF_MANIFEST)
    evidence, e3 = validate_manifest(EVIDENCE_MANIFEST)
    release, e4 = validate_manifest(RELEASE_MANIFEST)
    errors.extend(e1 + e2 + e3 + e4)
    status = load(STATUS)
    review = load(REVIEW)
    if status.get("highest_passed_gate") != "DIRAC-013":
        errors.append("status does not stop at DIRAC-013")
    if status.get("next_gate") != "DIRAC-014":
        errors.append("DIRAC-014 is not the next gate")
    if status.get("completed") != 13 or status.get("total") != 15:
        errors.append("gate counts are incorrect")
    if review.get("status") != "PASS_ZERO_CRITICAL_HIGH":
        errors.append("critical/high red-team findings remain")
    if review.get("critical_findings") or review.get("high_findings"):
        errors.append("red-team finding lists are not empty")
    publication = "LeanDirac.current_highest_gate_is_dirac013"
    if proof.get("publication_theorem") != publication:
        errors.append("proof manifest names the wrong publication theorem")
    action_ids = {m.get("action_id") for m in (framework, proof, evidence, release)}
    if len(action_ids) != 1 or None in action_ids:
        errors.append(f"role manifests disagree on action id: {sorted(map(str, action_ids))}")
    actual = {path.relative_to(ROOT).as_posix() for path in production_sources()}
    locked = set(proof.get("files", {}))
    if actual != locked:
        errors.append(
            f"proof source set differs: extra={sorted(actual - locked)}, "
            f"missing={sorted(locked - actual)}"
        )
    framework_files = set(framework.get("files", {}))
    if not framework_files or not framework_files < locked:
        errors.append("framework lock is not a nonempty strict subset of proof files")
    if evidence.get("numerical_evidence") != EVIDENCE_013.relative_to(ROOT).as_posix():
        errors.append("evidence manifest names the wrong result")
    cumulative = set(evidence.get("cumulative_numerical_evidence", []))
    expected_evidence = {
        EVIDENCE_010.relative_to(ROOT).as_posix(),
        EVIDENCE_011.relative_to(ROOT).as_posix(),
        EVIDENCE_012.relative_to(ROOT).as_posix(),
        EVIDENCE_013.relative_to(ROOT).as_posix(),
    }
    if cumulative != expected_evidence:
        errors.append("evidence manifest does not retain every admitted reconstruction")
    errors.extend(validate_dirac010_evidence(load(EVIDENCE_010)))
    errors.extend(validate_dirac011_evidence(load(EVIDENCE_011)))
    errors.extend(validate_dirac012_evidence(load(EVIDENCE_012)))
    errors.extend(validate_dirac013_evidence(load(EVIDENCE_013)))
    if run_builds:
        build = run(["lake", "build", *production_modules()])
        if build.returncode:
            errors.append("production build failed\n" + build.stdout + build.stderr)
        audit = run_compiled_audit()
        if audit.returncode:
            errors.append("Lean axiom audit failed\n" + audit.stdout + audit.stderr)
        else:
            output = audit.stdout + audit.stderr
            if "LEAN_DIRAC_ENVIRONMENT_AUDIT_PASS" not in output:
                errors.append("compiled-environment audit marker is missing")
            audited = set(re.findall(
                r"LEAN_DIRAC_AUDITED_MODULE\s+([A-Za-z_][A-Za-z0-9_.]*)", output
            ))
            expected_modules = set(production_modules())
            if audited != expected_modules:
                errors.append(
                    "compiled module coverage differs: "
                    f"extra={sorted(audited - expected_modules)}, "
                    f"missing={sorted(expected_modules - audited)}"
                )
            reports = parse_axiom_reports(output)
            missing = expected_axiom_reports() - set(reports)
            if missing:
                errors.append(f"missing axiom reports: {sorted(missing)}")
            if publication not in reports:
                errors.append("publication theorem axiom report is missing")
            unexpected = set().union(*reports.values()) - ALLOWED_AXIOMS if reports else set()
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
    print("highest_passed_gate=DIRAC-013")
    print("next_gate=DIRAC-014 status=OPEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
