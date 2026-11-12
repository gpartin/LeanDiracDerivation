"""Read-only audit of the completed DIRAC-001--015 derivation contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"


def load(name: str) -> dict:
    return json.loads((PROJECT / name).read_text(encoding="utf-8"))


def contains(path: str, text: str) -> bool:
    return text in (ROOT / path).read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    status = load("status.json")
    contract = load("derivation_contract.json")
    review = load("no_bolton_review.json")
    evidence_010 = json.loads((
        ROOT / "experiments" / "dirac010_standard_map" / "result.json"
    ).read_text(encoding="utf-8"))
    evidence_011 = json.loads((
        ROOT / "experiments" / "dirac011_frequency_mass" / "result.json"
    ).read_text(encoding="utf-8"))
    evidence_012 = json.loads((
        ROOT / "experiments" / "dirac012_continuum_evolution" / "result.json"
    ).read_text(encoding="utf-8"))
    evidence_013 = json.loads((
        ROOT / "experiments" / "dirac013_density_current" / "result.json"
    ).read_text(encoding="utf-8"))
    checks = {
        "highest_gate_is_015": status.get("highest_passed_gate") == "DIRAC-015",
        "all_fifteen_gates_complete": status.get("completed") == 15,
        "base_action_has_no_incidence_import": not contains(
            "LeanDirac/Dirac002.lean", "ChiIncidence"),
        "continuum_square_root_present": contains(
            "LeanDirac/Dirac009Geometry.lean",
            "theorem c19_continuum_dirac_square_is_native_quadratic_symbol"),
        "finite_lattice_remainder_present": contains(
            "LeanDirac/Dirac009Geometry.lean",
            "theorem c19_spatial_remainder_bound"),
        "standard_map_present": contains(
            "LeanDirac/Dirac010.lean",
            "theorem dirac010_native_operator_equals_standard_dirac"),
        "frequency_mass_map_present": contains(
            "LeanDirac/Dirac011.lean",
            "theorem dirac011_common_frequency_and_mass"),
        "continuum_evolution_present": contains(
            "LeanDirac/Dirac012.lean",
            "theorem dirac012_controlled_continuum_evolution"),
        "normalized_residual_present": contains(
            "LeanDirac/Dirac012.lean",
            "theorem normalized_native_dirac_mode_residual_bound"),
        "native_continuity_present": contains(
            "LeanDirac/Dirac013.lean",
            "theorem native_tangent_local_continuity"),
        "action_to_current_bridge_present": contains(
            "LeanDirac/Dirac013.lean",
            "theorem scalar_tangent_local_phase_variation_from_action"),
        "positive_density_bridge_present": contains(
            "LeanDirac/Dirac013.lean",
            "theorem native_cell4_positive_density_is_dirac_density"),
        "continuum_current_bridge_present": contains(
            "LeanDirac/Dirac013.lean",
            "theorem native_cell4_flux_continuum_limit_is_dirac_current"),
        "velocity_is_hamiltonian_response": contains(
            "LeanDirac/Dirac013.lean",
            "theorem standard_c19_velocity_is_hamiltonian_momentum_response"),
        "final_kg_dirac_square_present": contains(
            "LeanDirac/Dirac015.lean",
            "diracSquareIsContinuumKleinGordon"),
        "final_action_root_present": contains(
            "LeanDirac/Dirac015.lean",
            "kleinGordonTangentFromAction"),
        "publication_stops_at_015": contains(
            "LeanDirac/Publication.lean",
            "theorem current_highest_gate_is_dirac015"),
        "independent_matrix_check_passes": (
            evidence_010.get("status") == "PASS"
            and evidence_010.get("max_error", 1.0) <= 1.0e-12),
        "independent_frequency_mass_check_passes": (
            evidence_011.get("status") == "PASS"
            and evidence_011.get("max_error", 1.0) <= 1.0e-11),
        "independent_continuum_check_passes": (
            evidence_012.get("status") == "PASS"
            and evidence_012.get("max_error", 1.0) <= 1.0e-11
            and all(evidence_012.get("checks", {}).values())),
        "independent_density_current_check_passes": (
            evidence_013.get("status") == "PASS"
            and evidence_013.get("max_error", 1.0) <= 2.0e-9
            and evidence_013.get("minimum_flux_refinement_ratio", 0.0) > 3.9
            and all(evidence_013.get("checks", {}).values())),
        "zero_critical_high": (
            review.get("status") == "PASS_ZERO_CRITICAL_HIGH"
            and not review.get("critical_findings")
            and not review.get("high_findings")),
    }
    passed = all(checks.values())
    print(json.dumps({
        "action_id": contract["action_id"],
        "checks": checks,
        "verdict": "DIRAC015_READY" if passed else "DIRAC015_OPEN",
        "claim_boundary": status["claim_boundary"],
    }, indent=2))
    return 1 if args.strict and not passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
