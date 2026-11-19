"""One-command release audit for the completed DIRAC-001--025 chain.

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
STATUS_019 = ROOT / "project" / "dirac019_wavepacket_transport_status.json"
READINESS_019 = ROOT / "project" / "dirac019_release_readiness_status.json"
STATUS_020 = ROOT / "project" / "dirac020_coherent_spatial_transport_status.json"
READINESS_020 = ROOT / "project" / "dirac020_release_readiness_status.json"
STATUS_021 = ROOT / "project" / "dirac021_rest_clock_status.json"
READINESS_021 = ROOT / "project" / "dirac021_release_readiness_status.json"
STATUS_022 = ROOT / "project" / "dirac022_zitterbewegung_status.json"
READINESS_022 = ROOT / "project" / "dirac022_release_readiness_status.json"
STATUS_023 = ROOT / "project" / "dirac023_phase_harmony_status.json"
READINESS_023 = ROOT / "project" / "dirac023_release_readiness_status.json"
STATUS_024 = ROOT / "project" / "dirac024_projective_zitter_spin_status.json"
READINESS_024 = ROOT / "project" / "dirac024_release_readiness_status.json"
STATUS_025 = ROOT / "project" / "dirac025_velocity_clock_status.json"
READINESS_025 = ROOT / "project" / "dirac025_release_readiness_status.json"
EVIDENCE_010 = ROOT / "experiments" / "dirac010_standard_map" / "result.json"
EVIDENCE_011 = ROOT / "experiments" / "dirac011_frequency_mass" / "result.json"
EVIDENCE_012 = ROOT / "experiments" / "dirac012_continuum_evolution" / "result.json"
EVIDENCE_013 = ROOT / "experiments" / "dirac013_density_current" / "result.json"
EVIDENCE_016 = ROOT / "experiments" / "dirac016_copy_translation" / "result.json"
EVIDENCE_017 = ROOT / "experiments" / "dirac017_spin_half" / "result.json"
EVIDENCE_018 = ROOT / "experiments" / "dirac018_factorization_classification" / "result.json"
EVIDENCE_019 = ROOT / "experiments" / "dirac019_wavepacket_transport" / "result.json"
EVIDENCE_020 = ROOT / "experiments" / "dirac020_coherent_spatial_transport" / "result.json"
EVIDENCE_021 = ROOT / "experiments" / "dirac021_rest_clock" / "result.json"
EVIDENCE_022 = ROOT / "experiments" / "dirac022_zitterbewegung" / "result.json"
EVIDENCE_023 = ROOT / "experiments" / "dirac023_phase_harmony" / "result.json"
EVIDENCE_024 = ROOT / "experiments" / "dirac024_projective_zitter_spin" / "result.json"
EVIDENCE_025 = ROOT / "experiments" / "dirac025_velocity_clock" / "result.json"
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


def validate_dirac019_evidence(payload: dict) -> list[str]:
    errors: list[str] = []
    tolerance = payload.get("tolerance")
    maximum = payload.get("max_identity_error")
    if payload.get("status") != "PASS":
        errors.append("DIRAC-019 independent verdict is not PASS")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or \
            tolerance <= 0 or tolerance > 2.0e-11:
        errors.append("DIRAC-019 tolerance is missing or outside the admitted range")
    if not isinstance(maximum, (int, float)) or not math.isfinite(maximum):
        errors.append("DIRAC-019 maximum error is missing or non-finite")
    elif isinstance(tolerance, (int, float)) and maximum > tolerance:
        errors.append("DIRAC-019 maximum error exceeds its tolerance")
    required_checks = {
        "all_step_current_is_preserved",
        "all_massive_modes_strictly_subluminal",
        "explicit_matrix_identities_within_tolerance",
        "fourier_packet_density_is_preserved",
        "fourier_packet_velocity_converges_to_current_prediction",
        "generator_defect_converges_first_order_or_better",
        "negative_frequency_control_is_not_admitted_as_positive",
        "unstable_timestep_is_rejected",
    }
    checks = payload.get("checks")
    if not isinstance(checks, dict) or set(checks) != required_checks or \
            not all(value is True for value in checks.values()):
        errors.append("DIRAC-019 required checks are incomplete or false")
    for key in (
            "minimum_speed_margin",
            "minimum_generator_refinement_ratio",
            "negative_frequency_control_residual"):
        value = payload.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            errors.append(f"DIRAC-019 metric is missing or non-finite: {key}")
    controls = payload.get("real_space_packet_controls")
    if not isinstance(controls, list) or len(controls) != 3:
        errors.append("DIRAC-019 Fourier packet controls are incomplete")
    else:
        required = {"dt", "steps", "duration", "predicted_velocity",
                    "measured_velocity", "velocity_error", "density_drift"}
        if any(set(entry) != required or any(
                not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in entry.values()) for entry in controls):
            errors.append("DIRAC-019 Fourier packet controls are malformed")
    excluded = set(payload.get("excluded_claims", []))
    required_exclusions = {
        "nonlinear localized electron transport",
        "native selection of a positive-frequency packet",
        "uniform finite-time continuum error bound",
        "exact native finite-spacing packet evolution",
        "unrestricted live-chi evolution",
        "electromagnetic motion or response",
    }
    if not required_exclusions <= excluded:
        errors.append("DIRAC-019 excluded-claim boundary is incomplete")
    return errors


def validate_dirac020_evidence(payload: dict) -> list[str]:
    errors: list[str] = []
    tolerance = payload.get("tolerance")
    if payload.get("status") != "PASS":
        errors.append("DIRAC-020 independent verdict is not PASS")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or \
            tolerance <= 0 or tolerance > 3.0e-11:
        errors.append("DIRAC-020 tolerance is missing or outside the admitted range")
    admitted_tolerance = tolerance if isinstance(tolerance, (int, float)) and \
        math.isfinite(tolerance) and tolerance > 0 else 0.0
    required_checks = {
        "coherent_packet_density_is_preserved",
        "coherent_packet_velocity_converges",
        "coherent_spatial_parseval_identity",
        "explicit_fourier_frame_is_orthonormal",
        "finite_step_pointwise_error_bound_holds",
        "fourier_character_matches_mode_momentum",
        "integrated_spatial_current_equals_spectral_current",
        "spectral_current_equals_density_weighted_velocity",
        "unnormalized_frame_fails_parseval_control",
        "unstable_timestep_is_rejected",
        "zero_velocity_moment_gives_no_transport_control",
    }
    checks = payload.get("checks")
    if not isinstance(checks, dict) or set(checks) != required_checks or \
            not all(value is True for value in checks.values()):
        errors.append("DIRAC-020 required checks are incomplete or false")
    controls = payload.get("coherent_packet_controls")
    required_metrics = {
        "character_momentum_phase_error", "current_velocity_moment_error",
        "density_drift", "dt", "duration",
        "fourier_orthonormality_error", "integrated_current_error",
        "maximum_finite_step_bound_violation", "measured_velocity",
        "parseval_error", "predicted_velocity", "steps", "velocity_error",
    }
    if not isinstance(controls, list) or len(controls) != 3:
        errors.append("DIRAC-020 coherent packet controls are incomplete")
    elif any(set(entry) != required_metrics or any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in entry.values()) for entry in controls):
        errors.append("DIRAC-020 coherent packet controls are malformed")
    else:
        expected_grid = [(0.02, 300), (0.01, 600), (0.005, 1200)]
        if [(entry["dt"], entry["steps"]) for entry in controls] != expected_grid:
            errors.append("DIRAC-020 timestep refinement grid differs")
        if any(abs(entry["duration"] - 6.0) > admitted_tolerance
               for entry in controls):
            errors.append("DIRAC-020 controls do not share the fixed duration")
        for metric in (
                "character_momentum_phase_error", "current_velocity_moment_error",
                "density_drift",
                "fourier_orthonormality_error", "integrated_current_error",
                "maximum_finite_step_bound_violation", "parseval_error"):
            if max(abs(entry[metric]) for entry in controls) > admitted_tolerance:
                errors.append(f"DIRAC-020 {metric} exceeds tolerance")
        velocity_errors = [entry["velocity_error"] for entry in controls]
        if not (velocity_errors[2] < velocity_errors[1] < velocity_errors[0]):
            errors.append("DIRAC-020 packet velocity does not converge")
        if min(
                velocity_errors[0] / velocity_errors[1],
                velocity_errors[1] / velocity_errors[2]) <= 3.8:
            errors.append("DIRAC-020 packet velocity lacks second-order refinement")
    negative = payload.get("negative_controls")
    required_negative = {
        "raw_fourier_parseval_ratio", "symmetric_net_velocity",
        "unstable_timestep_rejected",
    }
    if not isinstance(negative, dict) or set(negative) != required_negative or \
            negative.get("unstable_timestep_rejected") is not True or any(
                not isinstance(negative.get(key), (int, float)) or
                not math.isfinite(negative[key])
                for key in ("raw_fourier_parseval_ratio", "symmetric_net_velocity")):
        errors.append("DIRAC-020 negative controls are malformed")
    elif abs(negative["raw_fourier_parseval_ratio"] - 32.0) > \
            admitted_tolerance or abs(negative["symmetric_net_velocity"]) > \
            admitted_tolerance:
        errors.append("DIRAC-020 negative controls differ from their exact values")
    excluded = set(payload.get("excluded_claims", []))
    required_exclusions = {
        "exact native finite-spacing Dirac evolution",
        "uniform fixed-physical-time analytic convergence rate",
        "nonlinear localized electron transport",
        "unrestricted live-chi evolution",
        "electron formation or identity",
        "electromagnetic response",
    }
    if not required_exclusions <= excluded:
        errors.append("DIRAC-020 excluded-claim boundary is incomplete")
    return errors


def validate_dirac021_evidence(payload: dict) -> list[str]:
    errors: list[str] = []
    tolerance = payload.get("tolerance")
    if payload.get("status") != "PASS":
        errors.append("DIRAC-021 independent verdict is not PASS")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or \
            tolerance <= 0 or tolerance > 2.0e-12:
        errors.append("DIRAC-021 tolerance is missing or outside the admitted range")
    admitted = tolerance if isinstance(tolerance, (int, float)) and \
        math.isfinite(tolerance) and tolerance > 0 else 0.0
    max_error = payload.get("max_error")
    if not isinstance(max_error, (int, float)) or not math.isfinite(max_error) or \
            max_error > admitted:
        errors.append("DIRAC-021 maximum error is non-finite or exceeds tolerance")
    required_values = {
        "rest_angular_frequency": 19.0,
        "branch_beat_angular_frequency": 38.0,
        "rest_clock_period": 2.0 * math.pi / 19.0,
        "branch_beat_period": math.pi / 19.0,
        "inverse_mass_scale": 1.0 / 19.0,
        "reduced_compton_length": math.sqrt(12.0) / 19.0,
        "c19_continuum_speed": math.sqrt(12.0),
    }
    values = payload.get("native_values")
    if not isinstance(values, dict) or set(values) != set(required_values) or any(
            not isinstance(values.get(key), (int, float)) or
            not math.isfinite(values[key]) or
            abs(values[key] - expected) > admitted
            for key, expected in required_values.items()):
        errors.append("DIRAC-021 native values are incomplete or incorrect")
    required_errors = {
        "positive_rest_energy", "negative_rest_energy",
        "positive_dirac_time_equation", "negative_dirac_time_equation",
        "relative_phase_doubles", "positive_clock_return",
        "negative_clock_return", "branch_beat_return", "rest_frequency",
        "branch_beat_frequency", "inverse_mass_scale",
        "native_reduced_compton_length",
    }
    component_errors = payload.get("errors")
    if not isinstance(component_errors, dict) or \
            set(component_errors) != required_errors or any(
                not isinstance(value, (int, float)) or not math.isfinite(value) or
                abs(value) > admitted for value in component_errors.values()):
        errors.append("DIRAC-021 component errors are incomplete or exceed tolerance")
    required_checks = {
        "zero_momentum_rest_frequency_is_chi0",
        "positive_and_negative_rest_energies_are_opposite",
        "both_rest_states_satisfy_the_dirac_time_equation",
        "opposite_branch_relative_phase_has_frequency_two_chi0",
        "single_branch_period_is_two_pi_over_chi0",
        "branch_coherence_period_is_pi_over_chi0",
        "pure_branch_density_has_no_clock_signal",
        "wrong_mass_control_is_rejected",
        "same_sign_control_has_no_doubled_phase",
    }
    checks = payload.get("checks")
    if not isinstance(checks, dict) or set(checks) != required_checks or \
            not all(value is True for value in checks.values()):
        errors.append("DIRAC-021 required checks are incomplete or false")
    negative = payload.get("negative_controls")
    required_negative = {
        "wrong_mass_max_phase_error", "same_sign_max_phase_error"
    }
    if not isinstance(negative, dict) or set(negative) != required_negative or any(
            not isinstance(negative.get(key), (int, float)) or
            not math.isfinite(negative[key]) or negative[key] <= 0.1
            for key in required_negative):
        errors.append("DIRAC-021 negative controls are malformed or ineffective")
    excluded = set(payload.get("excluded_claims", []))
    required_exclusions = {
        "an SI electron mass or SI Compton wavelength",
        "a directly observable clock in a pure energy eigenstate",
        "zitterbewegung without a specified cross-branch observable",
        "electromagnetic coupling or magnetic g-factor",
        "nonlinear localized-electron dynamics",
    }
    if not required_exclusions <= excluded:
        errors.append("DIRAC-021 excluded-claim boundary is incomplete")
    return errors


def validate_dirac022_evidence(payload: dict) -> list[str]:
    errors: list[str] = []
    tolerance = payload.get("tolerance")
    if payload.get("status") != "PASS":
        errors.append("DIRAC-022 independent verdict is not PASS")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or \
            tolerance <= 0 or tolerance > 2.0e-11:
        errors.append("DIRAC-022 tolerance is missing or outside the admitted range")
    admitted = tolerance if isinstance(tolerance, (int, float)) and \
        math.isfinite(tolerance) and tolerance > 0 else 0.0
    max_error = payload.get("max_error")
    if not isinstance(max_error, (int, float)) or not math.isfinite(max_error) or \
            max_error > admitted:
        errors.append("DIRAC-022 maximum error is non-finite or exceeds tolerance")
    required_values = {
        "action_selected_mass": 19.0,
        "zitter_angular_frequency": 38.0,
        "c19_continuum_speed": math.sqrt(12.0),
        "zitter_position_amplitude": math.sqrt(12.0) / 38.0,
        "zitter_peak_to_peak_excursion": math.sqrt(12.0) / 19.0,
        "native_reduced_compton_length": math.sqrt(12.0) / 19.0,
    }
    values = payload.get("native_values")
    if not isinstance(values, dict) or set(values) != set(required_values) or any(
            not isinstance(values.get(key), (int, float)) or
            not math.isfinite(values[key]) or
            abs(values[key] - expected) > admitted
            for key, expected in required_values.items()):
        errors.append("DIRAC-022 native values are incomplete or incorrect")
    required_errors = {
        "state_normalization",
        "velocity_from_hamiltonian_momentum_derivative",
        "rest_hamiltonian_velocity_anticommutator",
        "positive_branch_current", "negative_branch_current",
        "coherent_velocity_law", "position_derivative_law",
        "amplitude_identity", "peak_to_peak_identity", "position_bound",
        "same_sign_current", "incoherent_mixture_current",
    }
    component_errors = payload.get("errors")
    if not isinstance(component_errors, dict) or \
            set(component_errors) != required_errors or any(
                not isinstance(value, (int, float)) or not math.isfinite(value) or
                abs(value) > admitted for value in component_errors.values()):
        errors.append("DIRAC-022 component errors are incomplete or exceed tolerance")
    required_checks = {
        "all_evidence_is_finite", "normalized_cross_branch_state",
        "velocity_is_hamiltonian_momentum_response",
        "rest_velocity_anticommutes_with_rest_hamiltonian",
        "pure_rest_branches_have_zero_velocity",
        "coherent_velocity_frequency_is_two_chi0",
        "position_derivative_equals_velocity",
        "position_amplitude_is_half_reduced_compton_length",
        "same_sign_control_has_no_signal", "incoherent_mixture_has_no_signal",
        "wrong_mass_control_is_rejected",
    }
    checks = payload.get("checks")
    if not isinstance(checks, dict) or set(checks) != required_checks or \
            not all(value is True for value in checks.values()):
        errors.append("DIRAC-022 required checks are incomplete or false")
    negative = payload.get("negative_controls")
    required_negative = {
        "wrong_mass", "wrong_mass_max_signal_mismatch",
        "same_sign_max_current", "incoherent_mixture_current",
    }
    if not isinstance(negative, dict) or set(negative) != required_negative or \
            any(not isinstance(negative.get(key), (int, float)) or
                not math.isfinite(negative[key]) for key in required_negative):
        errors.append("DIRAC-022 negative controls are malformed")
    elif abs(negative["wrong_mass"] - 18.0) > admitted or \
            negative["wrong_mass_max_signal_mismatch"] <= 0.1 or \
            abs(negative["same_sign_max_current"]) > admitted or \
            abs(negative["incoherent_mixture_current"]) > admitted:
        errors.append("DIRAC-022 negative controls are ineffective")
    required_exclusions = {
        "a localized nonlinear native lattice electron",
        "measured trembling of a physical electron",
        "an SI frequency or SI length",
        "electromagnetic response or magnetic g-factor",
        "zitterbewegung in a pure branch or incoherent mixture",
    }
    if not required_exclusions <= set(payload.get("excluded_claims", [])):
        errors.append("DIRAC-022 excluded-claim boundary is incomplete")
    return errors


def validate_dirac023_evidence(payload: dict) -> list[str]:
    errors: list[str] = []
    tolerance = payload.get("tolerance")
    if payload.get("status") != "PASS":
        errors.append("DIRAC-023 independent verdict is not PASS")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or \
            tolerance <= 0 or tolerance > 2.0e-11:
        errors.append("DIRAC-023 tolerance is missing or outside the admitted range")
    admitted = tolerance if isinstance(tolerance, (int, float)) and \
        math.isfinite(tolerance) and tolerance > 0 else 0.0
    max_error = payload.get("max_error")
    if not isinstance(max_error, (int, float)) or not math.isfinite(max_error) or \
            max_error > admitted:
        errors.append("DIRAC-023 maximum error is non-finite or exceeds tolerance")
    if payload.get("sample_count") != 6:
        errors.append("DIRAC-023 sample census is incomplete")
    required_values = {
        "action_selected_rest_frequency": 19.0,
        "c19_continuum_speed_squared": 12.0,
        "c19_continuum_speed": math.sqrt(12.0),
        "zitter_angular_frequency": 38.0,
        "zitter_amplitude": math.sqrt(12.0) / 38.0,
        "zitter_amplitude_frequency_product": math.sqrt(12.0),
    }
    values = payload.get("native_values")
    if not isinstance(values, dict) or set(values) != set(required_values) or any(
            not isinstance(values.get(key), (int, float)) or
            not math.isfinite(values[key]) or
            abs(values[key] - expected) > admitted
            for key, expected in required_values.items()):
        errors.append("DIRAC-023 native values are incomplete or incorrect")
    required_errors = {
        "dispersion_identity", "packet_center_phase_rate",
        "energy_clock_reciprocity", "relativistic_clock_factor",
        "directional_group_phase_reciprocity", "rest_phase_rate",
        "zitter_amplitude_frequency_lock",
    }
    component_errors = payload.get("errors")
    if not isinstance(component_errors, dict) or \
            set(component_errors) != required_errors or any(
                not isinstance(value, (int, float)) or not math.isfinite(value) or
                abs(value) > admitted for value in component_errors.values()):
        errors.append("DIRAC-023 component errors are incomplete or exceed tolerance")
    required_checks = {
        "all_evidence_is_finite", "dispersion_is_inherited",
        "packet_center_phase_rate_is_mass_square_over_energy",
        "energy_and_clock_ratios_are_reciprocal",
        "clock_slowdown_matches_speed_factor",
        "directional_group_phase_product_is_c_eff_square",
        "rest_control_returns_action_clock", "all_sampled_modes_are_subluminal",
        "zitter_amplitude_times_frequency_is_c_eff",
        "wrong_transport_scale_is_rejected", "wrong_rest_frequency_is_rejected",
    }
    checks = payload.get("checks")
    if not isinstance(checks, dict) or set(checks) != required_checks or \
            not all(value is True for value in checks.values()):
        errors.append("DIRAC-023 required checks are incomplete or false")
    diagnostics = payload.get("diagnostics")
    margin = diagnostics.get(
        "minimum_sampled_c_eff_squared_minus_speed_squared") \
        if isinstance(diagnostics, dict) else None
    if not isinstance(margin, (int, float)) or not math.isfinite(margin) or \
            margin <= 0:
        errors.append("DIRAC-023 subluminal diagnostic is invalid")
    negative = payload.get("negative_controls")
    required_negative = {
        "wrong_transport_scale", "wrong_transport_max_reciprocity_defect",
        "wrong_rest_frequency", "wrong_rest_min_energy_clock_defect",
    }
    if not isinstance(negative, dict) or set(negative) != required_negative or any(
            not isinstance(negative.get(key), (int, float)) or
            not math.isfinite(negative[key]) for key in required_negative):
        errors.append("DIRAC-023 negative controls are malformed")
    elif abs(negative["wrong_transport_scale"] - 0.9) > admitted or \
            negative["wrong_transport_max_reciprocity_defect"] <= 0.1 or \
            abs(negative["wrong_rest_frequency"] - 18.0) > admitted or \
            negative["wrong_rest_min_energy_clock_defect"] <= 0.1:
        errors.append("DIRAC-023 negative controls are ineffective")
    required_exclusions = {
        "an exact finite-spacing phase-harmony identity",
        "an SI-calibrated electron clock",
        "a localized nonlinear native lattice electron",
        "unrestricted live-chi localized-electron motion",
        "electromagnetic response or magnetic g-factor",
        "zitterbewegung physically causes spin one-half",
    }
    if not required_exclusions <= set(payload.get("excluded_claims", [])):
        errors.append("DIRAC-023 excluded-claim boundary is incomplete")
    return errors


def validate_dirac024_evidence(payload: dict) -> list[str]:
    errors: list[str] = []
    tolerance = payload.get("tolerance")
    if payload.get("status") != "PASS":
        errors.append("DIRAC-024 independent verdict is not PASS")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or \
            tolerance <= 0 or tolerance > 2.0e-11:
        errors.append("DIRAC-024 tolerance is missing or outside the admitted range")
    admitted = tolerance if isinstance(tolerance, (int, float)) and \
        math.isfinite(tolerance) and tolerance > 0 else 0.0
    max_error = payload.get("max_error")
    if not isinstance(max_error, (int, float)) or not math.isfinite(max_error) or \
            max_error > admitted:
        errors.append("DIRAC-024 maximum error is non-finite or exceeds tolerance")
    if payload.get("sample_count") != 5:
        errors.append("DIRAC-024 sample census is incomplete")
    required_values = {
        "c19_continuum_speed": math.sqrt(12.0),
        "rest_angular_frequency": 19.0,
        "zitter_angular_frequency": 38.0,
        "zitter_period": math.pi / 19.0,
        "two_cycle_spinor_period": 2.0 * math.pi / 19.0,
    }
    values = payload.get("native_values")
    if not isinstance(values, dict) or set(values) != set(required_values) or any(
            not isinstance(values.get(key), (int, float)) or
            not math.isfinite(values[key]) or
            abs(values[key] - expected) > admitted
            for key, expected in required_values.items()):
        errors.append("DIRAC-024 native values are incomplete or incorrect")
    component_errors = payload.get("errors")
    required_base = {"period_equals_pi_over_chi0",
                     "half_period_velocity_reversal",
                     "time_generator_positive_branch",
                     "time_generator_negative_branch",
                     "spin_generator_same_on_positive_branch",
                     "spin_generator_same_on_negative_branch"}
    for sample in range(5):
        required_base.update({
            f"density_return_{sample}", f"velocity_return_{sample}",
            f"position_return_{sample}", f"state_antiperiod_{sample}",
            f"state_two_cycle_return_{sample}",
        })
    required_bridges = {
        f"{copy}_{axis}_{turn}_bridge_{sample}"
        for copy in ("plus", "minus") for axis in range(3)
        for turn in ("2pi", "4pi") for sample in range(5)
    }
    required_errors = required_base | required_bridges
    if not isinstance(component_errors, dict) or \
            set(component_errors) != required_errors or any(
                not isinstance(value, (int, float)) or not math.isfinite(value) or
                abs(value) > admitted for value in component_errors.values()):
        errors.append("DIRAC-024 component errors are incomplete or exceed tolerance")
    required_checks = {
        "density_velocity_and_position_return_after_one_cycle",
        "half_period_is_reversal_not_return",
        "one_zitter_cycle_negates_the_spinor",
        "time_and_spin_generators_remain_distinct",
        "two_pi_and_four_pi_bridges_hold_on_both_copies",
        "two_zitter_cycles_return_the_spinor",
        "wrong_period_is_rejected",
    }
    checks = payload.get("checks")
    if not isinstance(checks, dict) or set(checks) != required_checks or \
            not all(value is True for value in checks.values()):
        errors.append("DIRAC-024 required checks are incomplete or false")
    negative = payload.get("negative_controls")
    required_negative = {
        "half_period_return_defect", "wrong_period_min_antiperiod_defect",
        "wrong_period_scale",
    }
    if not isinstance(negative, dict) or set(negative) != required_negative or any(
            not isinstance(negative.get(key), (int, float)) or
            not math.isfinite(negative[key]) for key in required_negative):
        errors.append("DIRAC-024 negative controls are malformed")
    elif negative["half_period_return_defect"] <= 1.0 or \
            negative["wrong_period_min_antiperiod_defect"] <= 0.1 or \
            abs(negative["wrong_period_scale"] - 0.9) > admitted:
        errors.append("DIRAC-024 negative controls are ineffective")
    required_exclusions = {
        "time evolution is identical to spatial rotation",
        "zitterbewegung is the physical cause of spin one-half",
        "exact finite-spacing localized-electron motion",
        "electromagnetic coupling or a magnetic g-factor",
    }
    if not required_exclusions <= set(payload.get("excluded_claims", [])):
        errors.append("DIRAC-024 excluded-claim boundary is incomplete")
    return errors


def validate_dirac025_evidence(payload: dict) -> list[str]:
    errors: list[str] = []
    tolerance = payload.get("tolerance")
    if payload.get("status") != "PASS":
        errors.append("DIRAC-025 independent verdict is not PASS")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or \
            tolerance <= 0 or tolerance > 5.0e-11:
        errors.append("DIRAC-025 tolerance is missing or outside the admitted range")
    admitted = tolerance if isinstance(tolerance, (int, float)) and \
        math.isfinite(tolerance) and tolerance > 0 else 0.0
    max_error = payload.get("max_error")
    if not isinstance(max_error, (int, float)) or not math.isfinite(max_error) or \
            max_error > admitted:
        errors.append("DIRAC-025 maximum error is non-finite or exceeds tolerance")
    if payload.get("sample_count") != 5 or payload.get("spin_state_count") != 10:
        errors.append("DIRAC-025 momentum/spin census is incomplete")
    required_values = {
        "c19_continuum_speed": math.sqrt(12.0),
        "c19_continuum_speed_squared": 12.0,
        "rest_angular_frequency": 19.0,
    }
    values = payload.get("native_values")
    if not isinstance(values, dict) or set(values) != set(required_values) or any(
            not isinstance(values.get(key), (int, float)) or
            not math.isfinite(values[key]) or
            abs(values[key] - expected) > admitted
            for key, expected in required_values.items()):
        errors.append("DIRAC-025 native values are incomplete or incorrect")
    component_errors = payload.get("errors")
    required_errors = {
        f"sample_{sample}_{name}"
        for sample in range(5)
        for name in (
            "operator_square", "positive_energy_spectrum",
            "spin_0_mean", "spin_0_second_moment", "spin_0_variance",
            "spin_0_rms_clock", "spin_0_pythagorean", "spin_0_speed",
            "spin_1_mean", "spin_1_second_moment", "spin_1_variance",
            "spin_1_rms_clock", "spin_1_pythagorean", "spin_1_speed",
        )
    } | {
        "checkpoint_three_four_five", "checkpoint_four_three_five",
        "checkpoint_sqrt_three_half",
    }
    if not isinstance(component_errors, dict) or \
            set(component_errors) != required_errors or any(
                not isinstance(value, (int, float)) or not math.isfinite(value) or
                abs(value) > admitted for value in component_errors.values()):
        errors.append("DIRAC-025 component errors are incomplete or exceed tolerance")
    required_checks = {
        "arbitrary_three_direction_momenta_pass",
        "both_positive_energy_spin_states_pass",
        "known_relativistic_ratios_pass",
        "transverse_substitution_is_rejected",
        "unnormalized_direction_is_rejected",
        "wrong_rest_mass_is_rejected",
    }
    checks = payload.get("checks")
    if not isinstance(checks, dict) or set(checks) != required_checks or \
            not all(value is True for value in checks.values()):
        errors.append("DIRAC-025 required checks are incomplete or false")
    checkpoints = payload.get("known_checkpoints")
    expected_checkpoints = {
        "three_four_five": (3.0 / 5.0, 4.0 / 5.0),
        "four_three_five": (4.0 / 5.0, 3.0 / 5.0),
        "sqrt_three_half": (math.sqrt(3.0) / 2.0, 1.0 / 2.0),
    }
    if not isinstance(checkpoints, dict) or set(checkpoints) != set(expected_checkpoints):
        errors.append("DIRAC-025 known-ratio controls are incomplete")
    else:
        for key, (speed, clock) in expected_checkpoints.items():
            row = checkpoints.get(key)
            if not isinstance(row, dict) or set(row) != {
                    "speed_fraction", "expected_clock_ratio",
                    "observed_clock_ratio", "error"} or any(
                        not isinstance(value, (int, float)) or not math.isfinite(value)
                        for value in row.values()) or \
                    abs(row["speed_fraction"] - speed) > admitted or \
                    abs(row["expected_clock_ratio"] - clock) > admitted or \
                    abs(row["observed_clock_ratio"] - clock) > admitted or \
                    row["error"] > admitted:
                errors.append(f"DIRAC-025 known-ratio control is invalid: {key}")
    negative = payload.get("negative_controls")
    required_negative = {
        "unnormalized_direction_operator_square_defect",
        "transverse_direction_longitudinal_relation_defect",
        "wrong_rest_mass_clock_defect",
    }
    if not isinstance(negative, dict) or set(negative) != required_negative or any(
            not isinstance(negative.get(key), (int, float)) or
            not math.isfinite(negative[key]) for key in required_negative):
        errors.append("DIRAC-025 negative controls are malformed")
    elif negative["unnormalized_direction_operator_square_defect"] <= 1.0 or \
            negative["transverse_direction_longitudinal_relation_defect"] <= 0.1 or \
            negative["wrong_rest_mass_clock_defect"] <= 1.0e-3:
        errors.append("DIRAC-025 negative controls are ineffective")
    required_exclusions = {
        "a classical hidden path moving at c_eff",
        "an exact finite-spacing native velocity identity",
        "a localized nonlinear electron",
        "an SI calibration",
        "electromagnetic coupling or a magnetic g-factor",
    }
    if not required_exclusions <= set(payload.get("excluded_claims", [])):
        errors.append("DIRAC-025 excluded-claim boundary is incomplete")
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
    status_019 = load(STATUS_019)
    status_020 = load(STATUS_020)
    status_021 = load(STATUS_021)
    status_022 = load(STATUS_022)
    status_023 = load(STATUS_023)
    status_024 = load(STATUS_024)
    status_025 = load(STATUS_025)
    readiness_019 = load(READINESS_019)
    readiness_020 = load(READINESS_020)
    readiness_021 = load(READINESS_021)
    readiness_022 = load(READINESS_022)
    readiness_023 = load(READINESS_023)
    readiness_024 = load(READINESS_024)
    readiness_025 = load(READINESS_025)
    if status.get("highest_passed_gate") != "DIRAC-025":
        errors.append("status does not stop at DIRAC-025")
    if status.get("next_gate") != \
            "DIRAC_MAGNETIC_PROBE_AND_G_FACTOR":
        errors.append("magnetic probe is not the next boundary")
    if status.get("completed") != 25 or status.get("total") != 25:
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
    if readiness_019.get("status") != "PASS" or \
            readiness_019.get("highest_gate") != "DIRAC-019":
        errors.append("historical DIRAC-019 release readiness is invalid")
    if readiness_019.get("package_artifacts") != [
            "LeanDiracDerivation-DIRAC-019.zip",
            "LeanDiracDerivation-DIRAC-019.zip.sha256"]:
        errors.append("DIRAC-019 release artifact names are stale")
    if readiness_020.get("package_artifacts") != [
            "LeanDiracDerivation-DIRAC-020.zip",
            "LeanDiracDerivation-DIRAC-020.zip.sha256"]:
        errors.append("DIRAC-020 release artifact names are stale")
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
    if status_019.get("status") != \
            "PASS_SCOPED_FINITE_POSITIVE_FREQUENCY_MODE_TRANSPORT" or \
            status_019.get("formal_theorem") != \
            "LeanDirac.dirac019_finite_packet_transport":
        errors.append("DIRAC-019 scoped transport is not a formal PASS")
    red_team_019 = status_019.get("red_team", {})
    for key, expected in (
            ("new_action_term", False),
            ("new_native_field", False),
            ("runtime_translation_rule", False),
            ("positive_frequency_is_an_explicit_hypothesis", True),
            ("stable_timestep_is_an_explicit_hypothesis", True),
            ("mode_family_is_not_mislabeled_as_a_localized_electron", True),
            ("positive_frequency_sector_is_formally_nonvacuous", True),
            ("exact_native_finite_spacing_motion_is_not_claimed", True),
            ("all_update_counts_proved", True),
            ("all_update_current_persistence_proved", True),
            ("negative_frequency_control_rejected", True),
            ("unstable_timestep_control_rejected", True)):
        if red_team_019.get(key) is not expected:
            errors.append(f"DIRAC-019 red-team contract differs: {key}")
    if red_team_019.get("critical_findings") or red_team_019.get("high_findings"):
        errors.append("DIRAC-019 retains critical/high red-team findings")
    if status_020.get("status") != "PASS_SCOPED_COHERENT_SPATIAL_TRANSPORT" or \
            status_020.get("formal_theorem") != \
            "LeanDirac.dirac020_coherent_spatial_transport":
        errors.append("DIRAC-020 coherent spatial transport is not a formal PASS")
    red_team_020 = status_020.get("red_team", {})
    for key, expected in (
            ("new_action_term", False),
            ("new_native_field", False),
            ("runtime_translation_rule", False),
            ("fitted_coefficient", False),
            ("spatial_frame_is_explicit_and_orthonormality_is_proved", True),
            ("spatial_character_and_mode_momentum_are_proved_compatible", True),
            ("stable_timestep_is_an_explicit_hypothesis", True),
            ("nonzero_motion_requires_nonzero_derived_current", True),
            ("zero_current_control_rejected", True),
            ("unnormalized_fourier_control_rejected", True),
            ("exact_native_finite_spacing_motion_is_not_claimed", True),
            ("nonlinear_electron_motion_is_not_claimed", True)):
        if red_team_020.get(key) is not expected:
            errors.append(f"DIRAC-020 red-team contract differs: {key}")
    if red_team_020.get("critical_findings") or red_team_020.get("high_findings"):
        errors.append("DIRAC-020 retains critical/high red-team findings")
    if readiness_020.get("status") != "PASS" or \
            readiness_020.get("highest_gate") != "DIRAC-020":
        errors.append("DIRAC-020 release readiness is not PASS")
    if status_021.get("status") != "PASS_SCOPED_ACTION_DERIVED_REST_CLOCK" or \
            status_021.get("formal_theorem") != \
            "LeanDirac.dirac021_action_derived_rest_clock":
        errors.append("DIRAC-021 scoped rest-clock result is not a formal PASS")
    red_team_021 = status_021.get("red_team", {})
    for key, expected in (
            ("new_action_term", False),
            ("new_native_field", False),
            ("fitted_mass", False),
            ("rest_mass_is_inherited_from_action_derived_dirac011", True),
            ("opposite_rest_energies_are_proved", True),
            ("phase_derivatives_are_proved", True),
            ("relative_frequency_is_derived_from_branch_quotient", True),
            ("pure_global_phase_is_not_claimed_observable", True),
            ("inverse_mass_and_native_coordinate_length_are_distinguished", True),
            ("si_mass_is_not_claimed", True),
            ("zitterbewegung_is_not_claimed", True)):
        if red_team_021.get(key) is not expected:
            errors.append(f"DIRAC-021 red-team contract differs: {key}")
    if red_team_021.get("critical_findings") or red_team_021.get("high_findings"):
        errors.append("DIRAC-021 retains critical/high red-team findings")
    if readiness_021.get("status") != "PASS" or \
            readiness_021.get("highest_gate") != "DIRAC-021":
        errors.append("DIRAC-021 release readiness is not PASS")
    if readiness_021.get("package_artifacts") != [
            "LeanDiracDerivation-DIRAC-021.zip",
            "LeanDiracDerivation-DIRAC-021.zip.sha256"]:
        errors.append("DIRAC-021 release artifact names are stale")
    if status_022.get("status") != \
            "PASS_SCOPED_ACTION_DERIVED_FREE_ZITTERBEWEGUNG" or \
            status_022.get("formal_theorem") != \
            "LeanDirac.dirac022_action_derived_zitterbewegung":
        errors.append("DIRAC-022 scoped observable is not a formal PASS")
    red_team_022 = status_022.get("red_team", {})
    for key, expected in (
            ("new_action_term", False), ("new_native_field", False),
            ("fitted_mass_or_frequency", False),
            ("velocity_is_inherited_from_hamiltonian_response", True),
            ("coherence_and_normalization_are_explicit", True),
            ("factor_two_is_derived_from_opposite_rest_phases", True),
            ("position_is_integral_of_proved_velocity", True),
            ("pure_branch_control_passes", True),
            ("same_sign_control_passes", True),
            ("incoherent_mixture_control_passes", True),
            ("wrong_mass_control_passes", True),
            ("localized_native_electron_is_not_claimed", True),
            ("si_prediction_is_not_claimed", True)):
        if red_team_022.get(key) is not expected:
            errors.append(f"DIRAC-022 red-team contract differs: {key}")
    if red_team_022.get("critical_findings") or red_team_022.get("high_findings"):
        errors.append("DIRAC-022 retains critical/high red-team findings")
    if readiness_022.get("status") != "PASS" or \
            readiness_022.get("highest_gate") != "DIRAC-022" or \
            readiness_022.get("package_artifacts") != [
                "LeanDiracDerivation-DIRAC-022.zip",
                "LeanDiracDerivation-DIRAC-022.zip.sha256"]:
        errors.append("DIRAC-022 release readiness is invalid")
    if status_023.get("status") != \
            "PASS_SCOPED_CONTINUUM_PHASE_HARMONY_AND_TIME_DILATION" or \
            status_023.get("formal_theorem") != \
            "LeanDirac.dirac023_phase_harmony_and_time_dilation":
        errors.append("DIRAC-023 scoped phase-harmony result is not a formal PASS")
    red_team_023 = status_023.get("red_team", {})
    for key, expected in (
            ("new_action_term", False), ("new_native_field", False),
            ("fitted_coefficient", False), ("lorentz_factor_assumed", False),
            ("phase_velocity_operationally_verified", True),
            ("packet_phase_evaluated_on_independently_derived_current_trajectory", True),
            ("clock_slowdown_derived_from_phase_accumulation", True),
            ("energy_clock_reciprocity_proved", True),
            ("group_phase_velocity_reciprocity_proved", True),
            ("zitter_scale_lock_inherited_from_dirac022", True),
            ("exact_finite_spacing_claim_excluded", True),
            ("si_calibration_claim_excluded", True),
            ("localized_electron_claim_excluded", True)):
        if red_team_023.get(key) is not expected:
            errors.append(f"DIRAC-023 red-team contract differs: {key}")
    if red_team_023.get("critical_findings") or red_team_023.get("high_findings"):
        errors.append("DIRAC-023 retains critical/high red-team findings")
    if readiness_023.get("status") != "PASS" or \
            readiness_023.get("highest_gate") != "DIRAC-023" or \
            readiness_023.get("package_artifacts") != [
                "LeanDiracDerivation-DIRAC-023.zip",
                "LeanDiracDerivation-DIRAC-023.zip.sha256"]:
        errors.append("DIRAC-023 release readiness is invalid")
    if status_024.get("status") != \
            "PASS_SCOPED_PROJECTIVE_ZITTER_SPIN_CYCLE" or \
            status_024.get("formal_theorem") != \
            "LeanDirac.dirac024_projective_zitter_spin_cycle":
        errors.append("DIRAC-024 scoped projective-cycle result is not a formal PASS")
    red_team_024 = status_024.get("red_team", {})
    for key, expected in (
            ("new_action_term", False), ("new_native_field", False),
            ("fitted_coefficient", False), ("zitter_period_inserted", False),
            ("two_pi_sign_inserted", False),
            ("one_cycle_state_antiperiod_proved", True),
            ("two_cycle_state_return_proved", True),
            ("observables_return_after_one_cycle_proved", True),
            ("bridge_proved_on_both_native_copies", True),
            ("time_and_rotation_generators_kept_distinct", True),
            ("half_period_and_wrong_period_controls_rejected", True),
            ("causal_spin_origin_claim_excluded", True),
            ("localized_electron_claim_excluded", True)):
        if red_team_024.get(key) is not expected:
            errors.append(f"DIRAC-024 red-team contract differs: {key}")
    if red_team_024.get("critical_findings") or red_team_024.get("high_findings"):
        errors.append("DIRAC-024 retains critical/high red-team findings")
    if readiness_024.get("status") != "PASS" or \
            readiness_024.get("highest_gate") != "DIRAC-024" or \
            readiness_024.get("package_artifacts") != [
                "LeanDiracDerivation-DIRAC-024.zip",
                "LeanDiracDerivation-DIRAC-024.zip.sha256"]:
        errors.append("DIRAC-024 release readiness is invalid")
    if status_025.get("status") != \
            "PASS_SCOPED_VELOCITY_VARIANCE_CLOCK_IDENTITY" or \
            status_025.get("formal_theorem") != \
            "LeanDirac.dirac025_velocity_variance_clock_identity":
        errors.append("DIRAC-025 scoped velocity-clock result is not a formal PASS")
    red_team_025 = status_025.get("red_team", {})
    for key, expected in (
            ("new_action_term", False), ("new_native_field", False),
            ("fitted_coefficient", False), ("velocity_operator_inserted", False),
            ("variance_answer_inserted", False),
            ("same_state_and_inner_product_used", True),
            ("arbitrary_nonzero_three_direction_momentum_proved", True),
            ("operator_square_proved", True),
            ("current_expectation_bridge_proved", True),
            ("first_and_second_moments_proved", True),
            ("clock_identity_proved", True),
            ("known_relativistic_ratio_controls_pass", True),
            ("negative_controls_rejected", True),
            ("classical_hidden_path_claim_excluded", True),
            ("exact_finite_spacing_claim_excluded", True),
            ("localized_electron_claim_excluded", True)):
        if red_team_025.get(key) is not expected:
            errors.append(f"DIRAC-025 red-team contract differs: {key}")
    if red_team_025.get("critical_findings") or red_team_025.get("high_findings"):
        errors.append("DIRAC-025 retains critical/high red-team findings")
    if readiness_025.get("status") != "PASS" or \
            readiness_025.get("highest_gate") != "DIRAC-025" or \
            readiness_025.get("package_artifacts") != [
                "LeanDiracDerivation-DIRAC-025.zip",
                "LeanDiracDerivation-DIRAC-025.zip.sha256"]:
        errors.append("DIRAC-025 release readiness is invalid")
    publication = "LeanDirac.current_highest_gate_is_dirac025"
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
    if evidence.get("numerical_evidence") != EVIDENCE_025.relative_to(ROOT).as_posix():
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
        EVIDENCE_019.relative_to(ROOT).as_posix(),
        EVIDENCE_020.relative_to(ROOT).as_posix(),
        EVIDENCE_021.relative_to(ROOT).as_posix(),
        EVIDENCE_022.relative_to(ROOT).as_posix(),
        EVIDENCE_023.relative_to(ROOT).as_posix(),
        EVIDENCE_024.relative_to(ROOT).as_posix(),
        EVIDENCE_025.relative_to(ROOT).as_posix(),
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
    errors.extend(validate_dirac019_evidence(load(EVIDENCE_019)))
    errors.extend(validate_dirac020_evidence(load(EVIDENCE_020)))
    errors.extend(validate_dirac021_evidence(load(EVIDENCE_021)))
    errors.extend(validate_dirac022_evidence(load(EVIDENCE_022)))
    errors.extend(validate_dirac023_evidence(load(EVIDENCE_023)))
    errors.extend(validate_dirac024_evidence(load(EVIDENCE_024)))
    errors.extend(validate_dirac025_evidence(load(EVIDENCE_025)))
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
    print("highest_passed_gate=DIRAC-025")
    print("next_gate=DIRAC_MAGNETIC_PROBE_AND_G_FACTOR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
