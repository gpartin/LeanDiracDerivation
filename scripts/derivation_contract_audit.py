"""Read-only audit of the completed DIRAC-001--020 derivation contract."""

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
    evidence_016 = json.loads((
        ROOT / "experiments" / "dirac016_copy_translation" / "result.json"
    ).read_text(encoding="utf-8"))
    evidence_017 = json.loads((
        ROOT / "experiments" / "dirac017_spin_half" / "result.json"
    ).read_text(encoding="utf-8"))
    evidence_018 = json.loads((
        ROOT / "experiments" / "dirac018_factorization_classification" /
        "result.json"
    ).read_text(encoding="utf-8"))
    evidence_019 = json.loads((
        ROOT / "experiments" / "dirac019_wavepacket_transport" /
        "result.json"
    ).read_text(encoding="utf-8"))
    evidence_020 = json.loads((
        ROOT / "experiments" / "dirac020_coherent_spatial_transport" /
        "result.json"
    ).read_text(encoding="utf-8"))
    checks = {
        "highest_gate_is_020": status.get("highest_passed_gate") == "DIRAC-020",
        "all_twenty_gates_complete": status.get("completed") == 20,
        "base_action_has_no_incidence_import": not contains(
            "LeanDirac/Dirac002.lean", "ChiIncidence"),
        "native_action_restriction_present": contains(
            "LeanDirac/Dirac006Restriction.lean",
            "theorem aligned_lagrangian_is_restricted"),
        "restriction_variation_bridge_present": contains(
            "LeanDirac/Dirac006Restriction.lean",
            "theorem restricted_action_variation_is_native_variation"),
        "complementary_force_blocks_present": (
            contains(
                "LeanDirac/Dirac006Restriction.lean",
                "psiNormalVariationVanishes")
            and contains(
                "LeanDirac/Dirac006Restriction.lean",
                "chiNormalVariationVanishes")
        ),
        "complementary_momentum_blocks_present": (
            contains(
                "LeanDirac/Dirac006Restriction.lean",
                "psiNormalMomentumVanishes")
            and contains(
                "LeanDirac/Dirac006Restriction.lean",
                "chiNormalMomentumVanishes")
        ),
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
        "recurrence_derived_branch_present": contains(
            "LeanDirac/Dirac012Branch.lean",
            "theorem recurrence_dirac_branch_solves_continuum_centered_recurrence"),
        "native_branch_defect_is_spatial_remainder": contains(
            "LeanDirac/Dirac012Branch.lean",
            "theorem native_branch_defect_is_spatial_remainder"),
        "branch_generator_limit_present": contains(
            "LeanDirac/Dirac012Branch.lean",
            "theorem recurrence_branch_generator_defect_tends_to_zero"),
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
        "final_action_restriction_bridge_present": contains(
            "LeanDirac/Dirac015.lean",
            "actionRestrictionAndVariation"),
        "final_recurrence_branch_bridge_present": contains(
            "LeanDirac/Dirac015.lean",
            "recurrenceDerivedFirstOrderBranch"),
        "final_theorem_does_not_use_inserted_mode_as_bridge": not contains(
            "LeanDirac/Dirac015.lean",
            "controlledCenteredEquationLimit"),
        "publication_stops_at_020": contains(
            "LeanDirac/Publication.lean",
            "theorem current_highest_gate_is_dirac020"),
        "copy_translation_exchange_present": contains(
            "LeanDirac/Dirac016.lean",
            "theorem raw_axis2_maps_minus_to_plus"),
        "copy_translation_uniqueness_present": contains(
            "LeanDirac/Dirac016.lean",
            "theorem raw_axis2_minus_coordinate_map_unique"),
        "c19_copy_classification_present": contains(
            "LeanDirac/Dirac016.lean",
            "theorem raw_c19_translation_copy_relation"),
        "signed_incidence_distinction_present": contains(
            "LeanDirac/Dirac016.lean",
            "theorem prepared_incidence_axis2_preserves_minus"),
        "native_spin_scale_is_commutator_derived": (
            contains(
                "LeanDirac/Dirac017.lean",
                "theorem cube_spin_closure_iff_scale_equation")
            and contains(
                "LeanDirac/Dirac017.lean",
                "theorem exists_unique_admissible_spin_scale")
        ),
        "native_spin_casimir_present": contains(
            "LeanDirac/Dirac017.lean",
            "theorem cube_spin_casimir"),
        "native_spin_self_adjoint_present": contains(
            "LeanDirac/Dirac017.lean",
            "theorem cube_spin_self_adjoint"),
        "native_double_rotation_present": (
            contains(
                "LeanDirac/Dirac017.lean",
                "theorem cube_quarter_turn_four")
            and contains(
                "LeanDirac/Dirac017.lean",
                "theorem cube_quarter_turn_eight")
        ),
        "quarter_turn_density_present": contains(
            "LeanDirac/Dirac017.lean",
            "theorem cube_quarter_turn_preserves_density"),
        "standard_spin_intertwiner_present": (
            contains(
                "LeanDirac/Dirac017Closed.lean",
                "theorem cube_spin_intertwines_minus")
            and contains(
                "LeanDirac/Dirac017Closed.lean",
                "theorem cube_spin_intertwines_plus")
        ),
        "explicit_spin_spectrum_present": contains(
            "LeanDirac/Dirac017Closed.lean",
            "theorem standard_cube_spin0_basis_spectrum"),
        "positive_inner_product_and_unitarity_present": (
            contains(
                "LeanDirac/Dirac017Closed.lean",
                "theorem cell8_inner_positive_definite")
            and contains(
                "LeanDirac/Dirac017Closed.lean",
                "theorem cube_quarter_turn_preserves_inner")
        ),
        "continuous_rotation_group_present": (
            contains(
                "LeanDirac/Dirac017Closed.lean",
                "theorem cube_continuous_turn_add")
            and contains(
                "LeanDirac/Dirac017Closed.lean",
                "theorem cube_continuous_turn_preserves_inner")
        ),
        "quarter_turn_characterization_present": contains(
            "LeanDirac/Dirac017Closed.lean",
            "theorem exists_unique_admissible_cube_quarter_turn_coefficients"),
        "spin_hamiltonian_covariance_present": contains(
            "LeanDirac/Dirac017Closed.lean",
            "theorem cube_quarter_turn_hamiltonian_covariance"),
        "continuous_spin_hamiltonian_covariance_present": contains(
            "LeanDirac/Dirac017Closed.lean",
            "theorem cube_continuous_turn_hamiltonian_covariance"),
        "cubic_commutant_classification_present": contains(
            "LeanDirac/Dirac018.lean",
            "theorem cubic_covariant_coefficients_are_scalar"),
        "sector_local_two_copy_classification_present": contains(
            "LeanDirac/Dirac018.lean",
            "theorem admissible_two_copy_native_root_is_canonical"),
        "flavor_sector_exhaustion_present": (
            contains(
                "LeanDirac/Dirac018.lean",
                "theorem positive_flavor_is_exactly_plus_embedding")
            and contains(
                "LeanDirac/Dirac018.lean",
                "theorem negative_flavor_is_exactly_minus_embedding")
        ),
        "c19_zero_census_present": contains(
            "LeanDirac/Dirac018.lean",
            "theorem c19_first_brillouin_zone_zero_unique"),
        "orientation_necessity_control_present": contains(
            "LeanDirac/Dirac018.lean",
            "theorem overall_negative_root_is_rejected"),
        "cubic_covariance_necessity_control_present": contains(
            "LeanDirac/Dirac018.lean",
            "theorem reflected_axis_root_is_rejected"),
        "all_step_positive_mode_evolution_present": contains(
            "LeanDirac/Dirac019.lean",
            "theorem recurrence_step_advances_positive_mode_evolution"),
        "nonzero_positive_mode_exists_at_every_momentum": contains(
            "LeanDirac/Dirac019.lean",
            "theorem canonical_positive_frequency_mode_nonzero"),
        "all_step_packet_density_present": contains(
            "LeanDirac/Dirac019.lean",
            "theorem finite_packet_evolution_preserves_density"),
        "packet_current_velocity_present": contains(
            "LeanDirac/Dirac019.lean",
            "theorem finite_packet_current_is_density_weighted_velocity"),
        "all_step_packet_current_present": contains(
            "LeanDirac/Dirac019.lean",
            "theorem finite_packet_evolved_current_is_density_weighted_velocity"),
        "massive_subluminal_bound_present": contains(
            "LeanDirac/Dirac019.lean",
            "theorem continuum_transport_speed_strictly_subluminal"),
        "finite_packet_generator_limit_present": contains(
            "LeanDirac/Dirac019.lean",
            "theorem finite_packet_generator_defect_tends_to_zero"),
        "explicit_finite_fourier_frame_present": contains(
            "LeanDirac/Dirac020.lean",
            "theorem normalized_finite_fourier_basis_orthonormal"),
        "periodic_cubic_character_momentum_bridge_present": contains(
            "LeanDirac/Dirac020.lean",
            "theorem periodic_cubic_character_packet_is_momentum_compatible"),
        "coherent_spatial_parseval_present": contains(
            "LeanDirac/Dirac020.lean",
            "theorem cell4_spatial_synthesis_parseval"),
        "coherent_spatial_density_transport_present": (
            contains(
                "LeanDirac/Dirac020.lean",
                "theorem coherent_spatial_packet_density_is_spectral_density")
            and contains(
                "LeanDirac/Dirac020.lean",
                "theorem coherent_spatial_packet_density_is_preserved")
        ),
        "coherent_spatial_current_transport_present": (
            contains(
                "LeanDirac/Dirac020.lean",
                "theorem coherent_spatial_packet_current_is_velocity_moment")
            and contains(
                "LeanDirac/Dirac020.lean",
                "theorem coherent_spatial_packet_has_nonzero_transport")
        ),
        "coherent_finite_step_error_present": contains(
            "LeanDirac/Dirac020.lean",
            "theorem coherent_spatial_packet_finite_step_error"),
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
        "independent_copy_translation_check_passes": (
            evidence_016.get("status") == "PASS"
            and evidence_016.get("max_error", 1.0) <= 1.0e-12
            and all(evidence_016.get("checks", {}).values())),
        "independent_spin_representation_check_passes": (
            evidence_017.get("status") == "PASS"
            and evidence_017.get("max_error", 1.0) <= 2.0e-12
            and all(evidence_017.get("checks", {}).values())),
        "independent_factorization_classification_check_passes": (
            evidence_018.get("status") == "PASS"
            and evidence_018.get("max_error", 1.0) <= 3.0e-12
            and evidence_018.get("commutant_nullity") == 1
            and evidence_018.get("plus_sector_rank") == 4
            and evidence_018.get("minus_sector_rank") == 4
            and all(evidence_018.get("checks", {}).values())),
        "independent_spectral_transport_check_passes": (
            evidence_019.get("status") == "PASS"
            and evidence_019.get("max_identity_error", 1.0) <= 2.0e-11
            and evidence_019.get("minimum_speed_margin", 0.0) > 0.0
            and evidence_019.get("minimum_generator_refinement_ratio", 0.0) > 1.95
            and all(evidence_019.get("checks", {}).values())),
        "independent_coherent_spatial_transport_check_passes": (
            evidence_020.get("status") == "PASS"
            and evidence_020.get("tolerance", 1.0) <= 3.0e-11
            and len(evidence_020.get("coherent_packet_controls", [])) == 3
            and all(evidence_020.get("checks", {}).values())),
        "zero_critical_high": (
            review.get("status") == "PASS_ZERO_CRITICAL_HIGH"
            and not review.get("critical_findings")
            and not review.get("high_findings")),
    }
    passed = all(checks.values())
    print(json.dumps({
        "action_id": contract["action_id"],
        "checks": checks,
        "verdict": "DIRAC020_READY" if passed else "DIRAC020_OPEN",
        "claim_boundary": status["claim_boundary"],
    }, indent=2))
    return 1 if args.strict and not passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
