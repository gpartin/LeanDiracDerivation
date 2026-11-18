"""One-command release audit for the completed DIRAC-001--018 chain.

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
AUDIT_014 = ROOT / "project" / "dirac014_final_audit_status.json"
REPRODUCTION_015 = ROOT / "project" / "dirac015_clean_reproduction_status.json"
STATUS_016 = ROOT / "project" / "dirac016_copy_translation_status.json"
STATUS_017 = ROOT / "project" / "dirac017_spin_half_representation_status.json"
STATUS_018 = ROOT / "project" / "dirac018_factorization_classification_status.json"
READINESS_018 = ROOT / "project" / "dirac018_release_readiness_status.json"
EVIDENCE_010 = ROOT / "experiments" / "dirac010_standard_map" / "result.json"
EVIDENCE_011 = ROOT / "experiments" / "dirac011_frequency_mass" / "result.json"
EVIDENCE_012 = ROOT / "experiments" / "dirac012_continuum_evolution" / "result.json"
EVIDENCE_013 = ROOT / "experiments" / "dirac013_density_current" / "result.json"
EVIDENCE_016 = ROOT / "experiments" / "dirac016_copy_translation" / "result.json"
EVIDENCE_017 = ROOT / "experiments" / "dirac017_spin_half" / "result.json"
EVIDENCE_018 = ROOT / "experiments" / "dirac018_factorization_classification" / "result.json"
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
        "recurrence_derived_first_order_branch_reproduces",
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


def validate_dirac016_evidence(payload: dict) -> list[str]:
    errors: list[str] = []
    tolerance = payload.get("tolerance")
    maximum = payload.get("max_error")
    if payload.get("status") != "PASS":
        errors.append("DIRAC-016 independent verdict is not PASS")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or \
            tolerance <= 0 or tolerance > 1.0e-12:
        errors.append("DIRAC-016 tolerance is missing or outside the admitted range")
    if not isinstance(maximum, (int, float)) or not math.isfinite(maximum):
        errors.append("DIRAC-016 maximum error is missing or non-finite")
    elif isinstance(tolerance, (int, float)) and maximum > tolerance:
        errors.append("DIRAC-016 maximum error exceeds its tolerance")
    required_checks = {
        "all_translations_are_unitary_involutions",
        "axis_2_exchanges_both_rank_four_copies",
        "c19_copy_class_weights_are_balanced",
        "coordinate_half_shift_is_involutive",
        "signed_incidence_preserves_copy",
    }
    checks = payload.get("checks")
    if not isinstance(checks, dict) or set(checks) != required_checks or \
            not all(value is True for value in checks.values()):
        errors.append("DIRAC-016 required checks are incomplete or false")
    errors_map = payload.get("errors")
    if not isinstance(errors_map, dict) or not errors_map or any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in errors_map.values()):
        errors.append("DIRAC-016 component errors are incomplete or non-finite")
    if payload.get("c19_exchange_weight") != 12 or \
            payload.get("c19_preserving_weight") != 12:
        errors.append("DIRAC-016 C19 copy-class weights are incorrect")
    return errors


def validate_dirac017_evidence(payload: dict) -> list[str]:
    errors: list[str] = []
    tolerance = payload.get("tolerance")
    maximum = payload.get("max_error")
    if payload.get("status") != "PASS":
        errors.append("DIRAC-017 independent verdict is not PASS")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or \
            tolerance <= 0 or tolerance > 2.0e-12:
        errors.append("DIRAC-017 tolerance is missing or outside the admitted range")
    if not isinstance(maximum, (int, float)) or not math.isfinite(maximum):
        errors.append("DIRAC-017 maximum error is missing or non-finite")
    elif isinstance(tolerance, (int, float)) and maximum > tolerance:
        errors.append("DIRAC-017 maximum error exceeds its tolerance")
    required_checks = {
        "continuous_hamiltonian_covariance_reconstructed",
        "continuous_rotations_are_unitary_with_two_pi_sign_and_four_pi_return",
        "native_spin_algebra_reconstructed",
        "quarter_turn_is_unitary_and_double_valued",
        "spin_half_spectrum_on_both_copies",
    }
    checks = payload.get("checks")
    if not isinstance(checks, dict) or set(checks) != required_checks or \
            not all(value is True for value in checks.values()):
        errors.append("DIRAC-017 required checks are incomplete or false")
    errors_map = payload.get("errors")
    if not isinstance(errors_map, dict) or not errors_map or any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in errors_map.values()):
        errors.append("DIRAC-017 component errors are incomplete or non-finite")
    return errors


def validate_dirac018_evidence(payload: dict) -> list[str]:
    errors: list[str] = []
    tolerance = payload.get("tolerance")
    maximum = payload.get("max_error")
    if payload.get("status") != "PASS":
        errors.append("DIRAC-018 independent verdict is not PASS")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or \
            tolerance <= 0 or tolerance > 3.0e-12:
        errors.append("DIRAC-018 tolerance is missing or outside the admitted range")
    if not isinstance(maximum, (int, float)) or not math.isfinite(maximum):
        errors.append("DIRAC-018 maximum error is missing or non-finite")
    elif isinstance(tolerance, (int, float)) and maximum > tolerance:
        errors.append("DIRAC-018 maximum error exceeds its tolerance")
    required_checks = {
        "canonical_root_squares_to_scalar",
        "cubic_commutant_basis_is_identity",
        "cubic_commutant_has_dimension_one",
        "flavor_has_exactly_two_rank_four_sectors",
        "negative_orientation_control_squares_but_is_rejected",
        "noncovariant_reflection_squares_but_is_rejected",
        "tested_periodic_brillouin_zones_have_only_origin_zero",
    }
    checks = payload.get("checks")
    if not isinstance(checks, dict) or set(checks) != required_checks or \
            not all(value is True for value in checks.values()):
        errors.append("DIRAC-018 required checks are incomplete or false")
    errors_map = payload.get("errors")
    if not isinstance(errors_map, dict) or not errors_map or any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in errors_map.values()):
        errors.append("DIRAC-018 component errors are incomplete or non-finite")
    if payload.get("commutant_rank") != 8 or payload.get("commutant_nullity") != 1:
        errors.append("DIRAC-018 cubic commutant census is incorrect")
    if payload.get("plus_sector_rank") != 4 or payload.get("minus_sector_rank") != 4:
        errors.append("DIRAC-018 flavor-sector ranks are incorrect")
    zero_counts = payload.get("periodic_zone_zero_counts")
    expected_sizes = {str(size) for size in range(2, 33)}
    if not isinstance(zero_counts, dict) or set(zero_counts) != expected_sizes or \
            any(value != 1 for value in zero_counts.values()):
        errors.append("DIRAC-018 periodic-zone zero census is incomplete or incorrect")
    excluded = set(payload.get("excluded_claims", []))
    required_exclusions = {
        "classification of arbitrary nonlocal square roots",
        "classification of flavor-mixing roots outside the native flavor-preserving class",
        "localized nonlinear electron selection or stability",
        "electromagnetic coupling",
    }
    if not required_exclusions <= excluded:
        errors.append("DIRAC-018 excluded-claim boundary is incomplete")
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
    audit_014 = load(AUDIT_014)
    reproduction_015 = load(REPRODUCTION_015)
    status_016 = load(STATUS_016)
    status_017 = load(STATUS_017)
    status_018 = load(STATUS_018)
    readiness_018 = load(READINESS_018)
    if status.get("highest_passed_gate") != "DIRAC-018":
        errors.append("status does not stop at DIRAC-018")
    if status.get("next_gate") != "EXACT_FINITE_LATTICE_FACTORIZATION_OR_SCOPED_OBSTRUCTION":
        errors.append("exact finite-lattice factorization is not the next boundary")
    if status.get("completed") != 18 or status.get("total") != 18:
        errors.append("gate counts are incorrect")
    expected_gate_017 = "PASS_CONDITIONAL_CUBICAL_SPIN_HALF_REPRESENTATION"
    if status.get("gates", {}).get("DIRAC-017") != expected_gate_017:
        errors.append("DIRAC-017 machine status does not preserve conditionality")
    if review.get("status") != "PASS_ZERO_CRITICAL_HIGH":
        errors.append("critical/high red-team findings remain")
    if review.get("critical_findings") or review.get("high_findings"):
        errors.append("red-team finding lists are not empty")
    if audit_014.get("status") != "PASS" or \
            audit_014.get("verdict") != "PASS_ZERO_CRITICAL_HIGH":
        errors.append("DIRAC-014 final adversarial audit is not PASS")
    if audit_014.get("critical_findings") or audit_014.get("high_findings"):
        errors.append("DIRAC-014 retains critical/high findings")
    if reproduction_015.get("status") != "PASS":
        errors.append("DIRAC-015 clean reproduction is not PASS")
    if reproduction_015.get("formal_theorem") != \
            "LeanDirac.lfm_free_dirac_continuum_limit":
        errors.append("DIRAC-015 names the wrong final theorem")
    if reproduction_015.get("historical") is not True:
        errors.append("DIRAC-015 package record is not marked historical")
    if readiness_018.get("status") != "PASS" or \
            readiness_018.get("highest_gate") != "DIRAC-018":
        errors.append("DIRAC-018 release readiness is not current")
    if readiness_018.get("package_artifacts") != [
            "LeanDiracDerivation-DIRAC-018.zip",
            "LeanDiracDerivation-DIRAC-018.zip.sha256"]:
        errors.append("DIRAC-018 release artifact names are stale")
    if status_016.get("status") != "PASS" or \
            status_016.get("formal_theorem") != \
            "LeanDirac.dirac016_exact_copy_translation_structure":
        errors.append("DIRAC-016 copy-translation status is not a formal PASS")
    red_team_016 = status_016.get("red_team", {})
    if red_team_016.get("critical_findings") or red_team_016.get("high_findings"):
        errors.append("DIRAC-016 retains critical/high red-team findings")
    if red_team_016.get("claim_boundary_explicit") is not True:
        errors.append("DIRAC-016 claim boundary is not explicit")
    if status_017.get("status") != \
            "PASS_CONDITIONAL_CUBICAL_SPIN_HALF_REPRESENTATION" or \
            status_017.get("formal_theorem") != \
            "LeanDirac.dirac017_closed_spin_half_representation":
        errors.append("DIRAC-017 scoped spin representation is not a formal PASS")
    red_team_017 = status_017.get("red_team", {})
    if red_team_017.get("critical_findings") or red_team_017.get("high_findings"):
        errors.append("DIRAC-017 retains critical/high red-team findings")
    if red_team_017.get("assumed_spin_matrix") is not False or \
            red_team_017.get("assumed_half_angle") is not False:
        errors.append("DIRAC-017 assumes the spin answer")
    for key in (
            "standard_dirac_spin_intertwiner_proved",
            "two_plus_two_minus_spectrum_proved",
            "positive_definite_inner_product_proved",
            "full_unitarity_proved",
            "continuous_rotation_group_proved",
            "continuous_angle_uses_derived_spin_scale",
            "quarter_turn_uniquely_characterized",
            "mass_grading_compatibility_proved",
            "quarter_turn_hamiltonian_covariance_proved",
            "continuous_hamiltonian_covariance_proved",
            "both_copies_retained",
            "localized_dynamics_claim_excluded"):
        if red_team_017.get(key) is not True:
            errors.append(f"DIRAC-017 closure check is absent: {key}")
    if red_team_017.get("claim_boundary_explicit") is not True:
        errors.append("DIRAC-017 claim boundary is not explicit")
    if status_018.get("status") != \
            "PASS_SCOPED_NATIVE_CUBICAL_FACTORIZATION_CLASSIFICATION" or \
            status_018.get("formal_theorem") != \
            "LeanDirac.dirac018_native_factorization_classification":
        errors.append("DIRAC-018 scoped factorization classification is not a formal PASS")
    red_team_018 = status_018.get("red_team", {})
    for key, expected in (
            ("new_action_term", False),
            ("new_native_field", False),
            ("assumed_dirac_matrices", False),
            ("global_copy_symmetry_assumed", False),
            ("sector_local_square_conditions", True),
            ("orientation_hypothesis_explicit", True),
            ("cubic_covariance_hypothesis_explicit", True),
            ("hypothesis_necessity_controls_proved", True),
            ("two_copy_exhaustion_proved", True),
            ("finite_lattice_overclaim_excluded", True),
            ("nielsen_ninomiya_overclaim_excluded", True),
            ("claim_boundary_explicit", True)):
        if red_team_018.get(key) is not expected:
            errors.append(f"DIRAC-018 red-team contract differs: {key}")
    if red_team_018.get("critical_findings") or red_team_018.get("high_findings"):
        errors.append("DIRAC-018 retains critical/high red-team findings")
    publication = "LeanDirac.current_highest_gate_is_dirac018"
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
    if evidence.get("numerical_evidence") != EVIDENCE_018.relative_to(ROOT).as_posix():
        errors.append("evidence manifest names the wrong result")
    cumulative = set(evidence.get("cumulative_numerical_evidence", []))
    expected_evidence = {
        EVIDENCE_010.relative_to(ROOT).as_posix(),
        EVIDENCE_011.relative_to(ROOT).as_posix(),
        EVIDENCE_012.relative_to(ROOT).as_posix(),
        EVIDENCE_013.relative_to(ROOT).as_posix(),
        EVIDENCE_016.relative_to(ROOT).as_posix(),
        EVIDENCE_017.relative_to(ROOT).as_posix(),
        EVIDENCE_018.relative_to(ROOT).as_posix(),
    }
    if cumulative != expected_evidence:
        errors.append("evidence manifest does not retain every admitted reconstruction")
    errors.extend(validate_dirac010_evidence(load(EVIDENCE_010)))
    errors.extend(validate_dirac011_evidence(load(EVIDENCE_011)))
    errors.extend(validate_dirac012_evidence(load(EVIDENCE_012)))
    errors.extend(validate_dirac013_evidence(load(EVIDENCE_013)))
    errors.extend(validate_dirac016_evidence(load(EVIDENCE_016)))
    errors.extend(validate_dirac017_evidence(load(EVIDENCE_017)))
    errors.extend(validate_dirac018_evidence(load(EVIDENCE_018)))
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
    print("highest_passed_gate=DIRAC-018")
    print("next_gate=EXACT_FINITE_LATTICE_FACTORIZATION_OR_SCOPED_OBSTRUCTION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
