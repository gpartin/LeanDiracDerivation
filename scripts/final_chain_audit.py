"""Final read-only adversarial audit for DIRAC-001 through DIRAC-024."""

from __future__ import annotations

import json
from pathlib import Path

import production_guard as guard


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> int:
    errors = guard.validate(run_builds=False)
    status = load("project/status.json")
    review = load("project/no_bolton_review.json")
    audit = load("project/dirac014_final_audit_status.json")
    reproduction = load("project/dirac015_clean_reproduction_status.json")
    proof = load("project/proof_manifest.json")
    copy_status = load("project/dirac016_copy_translation_status.json")
    spin_status = load("project/dirac017_spin_half_representation_status.json")
    classification_status = load(
        "project/dirac018_factorization_classification_status.json"
    )
    transport_status = load("project/dirac019_wavepacket_transport_status.json")
    spatial_status = load(
        "project/dirac020_coherent_spatial_transport_status.json"
    )
    clock_status = load("project/dirac021_rest_clock_status.json")
    zitter_status = load("project/dirac022_zitterbewegung_status.json")
    phase_status = load("project/dirac023_phase_harmony_status.json")
    cycle_status = load("project/dirac024_projective_zitter_spin_status.json")
    readiness = load("project/dirac024_release_readiness_status.json")
    boundary = status.get("claim_boundary", "").lower()

    checks = {
        "all_twenty_four_gates_have_honest_passing_status": (
            status.get("completed") == 24
            and status.get("highest_passed_gate") == "DIRAC-024"
            and all(
                value == "PASS" for gate, value in status.get("gates", {}).items()
                if gate not in (
                    "DIRAC-017", "DIRAC-018", "DIRAC-019", "DIRAC-020",
                    "DIRAC-021", "DIRAC-022", "DIRAC-023", "DIRAC-024"
                )
            )
            and status.get("gates", {}).get("DIRAC-017") ==
                "PASS_CONDITIONAL_CUBICAL_SPIN_HALF_REPRESENTATION"
            and status.get("gates", {}).get("DIRAC-018") ==
                "PASS_SCOPED_NATIVE_CUBICAL_FACTORIZATION_CLASSIFICATION"
            and status.get("gates", {}).get("DIRAC-019") ==
                "PASS_SCOPED_FINITE_POSITIVE_FREQUENCY_MODE_TRANSPORT"
            and status.get("gates", {}).get("DIRAC-020") ==
                "PASS_SCOPED_COHERENT_SPATIAL_TRANSPORT"
            and status.get("gates", {}).get("DIRAC-021") ==
                "PASS_SCOPED_ACTION_DERIVED_REST_CLOCK"
            and status.get("gates", {}).get("DIRAC-022") ==
                "PASS_SCOPED_ACTION_DERIVED_FREE_ZITTERBEWEGUNG"
            and status.get("gates", {}).get("DIRAC-023") ==
                "PASS_SCOPED_CONTINUUM_PHASE_HARMONY_AND_TIME_DILATION"
            and status.get("gates", {}).get("DIRAC-024") ==
                "PASS_SCOPED_PROJECTIVE_ZITTER_SPIN_CYCLE"
        ),
        "zero_critical_high": (
            review.get("status") == "PASS_ZERO_CRITICAL_HIGH"
            and not review.get("critical_findings")
            and not review.get("high_findings")
        ),
        "audit_gate_passes": audit.get("status") == "PASS",
        "clean_reproduction_gate_passes": reproduction.get("status") == "PASS",
        "current_release_readiness_passes": readiness.get("status") == "PASS",
        "final_theorem_is_action_rooted": (
            proof.get("highest_gate") == "DIRAC-024"
            and "LeanDirac/Dirac015.lean" in proof.get("files", {})
            and "LeanDirac/Dirac016.lean" in proof.get("files", {})
            and "LeanDirac/Dirac017.lean" in proof.get("files", {})
            and "LeanDirac/Dirac017Closed.lean" in proof.get("files", {})
            and "LeanDirac/Dirac018.lean" in proof.get("files", {})
            and "LeanDirac/Dirac019.lean" in proof.get("files", {})
            and "LeanDirac/Dirac020.lean" in proof.get("files", {})
            and "LeanDirac/Dirac021.lean" in proof.get("files", {})
            and "LeanDirac/Dirac022.lean" in proof.get("files", {})
            and "LeanDirac/Dirac023.lean" in proof.get("files", {})
            and "LeanDirac/Dirac024.lean" in proof.get("files", {})
            and "LeanDirac/Publication.lean" in proof.get("files", {})
            and all(audit.get("checks", {}).get(name) is True for name in (
                "native_action_contains_no_dirac_matrix_spinor_or_projector",
                "native_action_restriction_and_variation_bridge_is_explicit",
                "finite_lattice_and_timestep_remainders_remain_explicit",
                "first_order_branch_is_derived_from_the_centered_recurrence",
                "current_is_derived_from_action_and_hamiltonian_response",
            ))
        ),
        "publication_names_final_theorem": (
            proof.get("publication_theorem") ==
                "LeanDirac.current_highest_gate_is_dirac024"
            and reproduction.get("formal_theorem") ==
                "LeanDirac.lfm_free_dirac_continuum_limit"
            and reproduction.get("publication_theorem") ==
                "LeanDirac.current_highest_gate_is_dirac015"
            and copy_status.get("publication_theorem") ==
                "LeanDirac.current_highest_gate_is_dirac016"
            and spin_status.get("publication_theorem") ==
                "LeanDirac.current_highest_gate_is_dirac017"
            and classification_status.get("publication_theorem") ==
                "LeanDirac.current_highest_gate_is_dirac018"
            and transport_status.get("publication_theorem") ==
                "LeanDirac.current_highest_gate_is_dirac019"
            and spatial_status.get("publication_theorem") ==
                "LeanDirac.current_highest_gate_is_dirac020"
            and clock_status.get("publication_theorem") ==
                "LeanDirac.current_highest_gate_is_dirac021"
            and zitter_status.get("publication_theorem") ==
                "LeanDirac.current_highest_gate_is_dirac022"
            and phase_status.get("publication_theorem") ==
                "LeanDirac.current_highest_gate_is_dirac023"
            and cycle_status.get("publication_theorem") ==
                proof.get("publication_theorem")
        ),
        "copy_translation_extension_is_scoped": (
            copy_status.get("status") == "PASS"
            and copy_status.get("action_changed") is False
            and copy_status.get("red_team", {}).get(
                "signed_incidence_distinguished_from_unsigned_translation") is True
            and not copy_status.get("red_team", {}).get("critical_findings")
            and not copy_status.get("red_team", {}).get("high_findings")
        ),
        "cubical_spin_representation_is_scoped": (
            spin_status.get("status") ==
                "PASS_CONDITIONAL_CUBICAL_SPIN_HALF_REPRESENTATION"
            and spin_status.get("action_changed") is False
            and spin_status.get("red_team", {}).get(
                "standard_dirac_spin_intertwiner_proved") is True
            and spin_status.get("red_team", {}).get(
                "assumed_spin_matrix") is False
            and spin_status.get("red_team", {}).get(
                "assumed_half_angle") is False
            and spin_status.get("red_team", {}).get(
                "continuous_hamiltonian_covariance_proved") is True
            and spin_status.get("red_team", {}).get(
                "quarter_turn_hamiltonian_covariance_proved") is True
            and spin_status.get("red_team", {}).get(
                "quarter_turn_uniquely_characterized") is True
            and spin_status.get("red_team", {}).get(
                "localized_dynamics_claim_excluded") is True
            and not spin_status.get("red_team", {}).get("critical_findings")
            and not spin_status.get("red_team", {}).get("high_findings")
        ),
        "native_factorization_classification_is_scoped": (
            classification_status.get("status") ==
                "PASS_SCOPED_NATIVE_CUBICAL_FACTORIZATION_CLASSIFICATION"
            and classification_status.get("action_changed") is False
            and classification_status.get("red_team", {}).get(
                "global_copy_symmetry_assumed") is False
            and classification_status.get("red_team", {}).get(
                "sector_local_square_conditions") is True
            and classification_status.get("red_team", {}).get(
                "hypothesis_necessity_controls_proved") is True
            and classification_status.get("red_team", {}).get(
                "two_copy_exhaustion_proved") is True
            and classification_status.get("red_team", {}).get(
                "finite_lattice_overclaim_excluded") is True
            and classification_status.get("red_team", {}).get(
                "nielsen_ninomiya_overclaim_excluded") is True
            and not classification_status.get("red_team", {}).get(
                "critical_findings")
            and not classification_status.get("red_team", {}).get(
                "high_findings")
        ),
        "finite_positive_frequency_transport_is_scoped": (
            transport_status.get("status") ==
                "PASS_SCOPED_FINITE_POSITIVE_FREQUENCY_MODE_TRANSPORT"
            and transport_status.get("action_changed") is False
            and transport_status.get("red_team", {}).get(
                "runtime_translation_rule") is False
            and transport_status.get("red_team", {}).get(
                "positive_frequency_is_an_explicit_hypothesis") is True
            and transport_status.get("red_team", {}).get(
                "stable_timestep_is_an_explicit_hypothesis") is True
            and transport_status.get("red_team", {}).get(
                "all_update_counts_proved") is True
            and transport_status.get("red_team", {}).get(
                "all_update_current_persistence_proved") is True
            and transport_status.get("red_team", {}).get(
                "positive_frequency_sector_is_formally_nonvacuous") is True
            and transport_status.get("red_team", {}).get(
                "exact_native_finite_spacing_motion_is_not_claimed") is True
            and transport_status.get("red_team", {}).get(
                "mode_family_is_not_mislabeled_as_a_localized_electron") is True
            and not transport_status.get("red_team", {}).get(
                "critical_findings")
            and not transport_status.get("red_team", {}).get(
                "high_findings")
        ),
        "coherent_spatial_transport_is_scoped": (
            spatial_status.get("status") ==
                "PASS_SCOPED_COHERENT_SPATIAL_TRANSPORT"
            and spatial_status.get("action_changed") is False
            and spatial_status.get("red_team", {}).get(
                "runtime_translation_rule") is False
            and spatial_status.get("red_team", {}).get(
                "fitted_coefficient") is False
            and spatial_status.get("red_team", {}).get(
                "spatial_frame_is_explicit_and_orthonormality_is_proved") is True
            and spatial_status.get("red_team", {}).get(
                "spatial_character_and_mode_momentum_are_proved_compatible") is True
            and spatial_status.get("red_team", {}).get(
                "stable_timestep_is_an_explicit_hypothesis") is True
            and spatial_status.get("red_team", {}).get(
                "nonzero_motion_requires_nonzero_derived_current") is True
            and spatial_status.get("red_team", {}).get(
                "zero_current_control_rejected") is True
            and spatial_status.get("red_team", {}).get(
                "unnormalized_fourier_control_rejected") is True
            and spatial_status.get("red_team", {}).get(
                "exact_native_finite_spacing_motion_is_not_claimed") is True
            and spatial_status.get("red_team", {}).get(
                "nonlinear_electron_motion_is_not_claimed") is True
            and not spatial_status.get("red_team", {}).get(
                "critical_findings")
            and not spatial_status.get("red_team", {}).get(
                "high_findings")
        ),
        "action_derived_rest_clock_is_scoped": (
            clock_status.get("status") ==
                "PASS_SCOPED_ACTION_DERIVED_REST_CLOCK"
            and clock_status.get("action_changed") is False
            and clock_status.get("red_team", {}).get("fitted_mass") is False
            and clock_status.get("red_team", {}).get(
                "rest_mass_is_inherited_from_action_derived_dirac011") is True
            and clock_status.get("red_team", {}).get(
                "opposite_rest_energies_are_proved") is True
            and clock_status.get("red_team", {}).get(
                "relative_frequency_is_derived_from_branch_quotient") is True
            and clock_status.get("red_team", {}).get(
                "pure_global_phase_is_not_claimed_observable") is True
            and clock_status.get("red_team", {}).get(
                "zitterbewegung_is_not_claimed") is True
            and not clock_status.get("red_team", {}).get("critical_findings")
            and not clock_status.get("red_team", {}).get("high_findings")
        ),
        "action_derived_zitterbewegung_is_scoped": (
            zitter_status.get("status") ==
                "PASS_SCOPED_ACTION_DERIVED_FREE_ZITTERBEWEGUNG"
            and zitter_status.get("action_changed") is False
            and zitter_status.get("native_field_changed") is False
            and zitter_status.get("red_team", {}).get(
                "velocity_is_inherited_from_hamiltonian_response") is True
            and zitter_status.get("red_team", {}).get(
                "factor_two_is_derived_from_opposite_rest_phases") is True
            and zitter_status.get("red_team", {}).get(
                "position_is_integral_of_proved_velocity") is True
            and zitter_status.get("red_team", {}).get(
                "localized_native_electron_is_not_claimed") is True
            and not zitter_status.get("red_team", {}).get("critical_findings")
            and not zitter_status.get("red_team", {}).get("high_findings")
        ),
        "phase_harmony_and_time_dilation_are_scoped": (
            phase_status.get("status") ==
                "PASS_SCOPED_CONTINUUM_PHASE_HARMONY_AND_TIME_DILATION"
            and phase_status.get("action_changed") is False
            and phase_status.get("native_field_changed") is False
            and phase_status.get("coefficient_changed") is False
            and phase_status.get("red_team", {}).get(
                "lorentz_factor_assumed") is False
            and phase_status.get("red_team", {}).get(
                "phase_velocity_operationally_verified") is True
            and phase_status.get("red_team", {}).get(
                "packet_phase_evaluated_on_independently_derived_current_trajectory") is True
            and phase_status.get("red_team", {}).get(
                "clock_slowdown_derived_from_phase_accumulation") is True
            and phase_status.get("red_team", {}).get(
                "exact_finite_spacing_claim_excluded") is True
            and not phase_status.get("red_team", {}).get("critical_findings")
            and not phase_status.get("red_team", {}).get("high_findings")
        ),
        "projective_zitter_spin_cycle_is_scoped": (
            cycle_status.get("status") ==
                "PASS_SCOPED_PROJECTIVE_ZITTER_SPIN_CYCLE"
            and cycle_status.get("action_changed") is False
            and cycle_status.get("native_field_changed") is False
            and cycle_status.get("coefficient_changed") is False
            and cycle_status.get("red_team", {}).get(
                "zitter_period_inserted") is False
            and cycle_status.get("red_team", {}).get(
                "two_pi_sign_inserted") is False
            and cycle_status.get("red_team", {}).get(
                "bridge_proved_on_both_native_copies") is True
            and cycle_status.get("red_team", {}).get(
                "time_and_rotation_generators_kept_distinct") is True
            and cycle_status.get("red_team", {}).get(
                "causal_spin_origin_claim_excluded") is True
            and not cycle_status.get("red_team", {}).get("critical_findings")
            and not cycle_status.get("red_team", {}).get("high_findings")
        ),
        "claim_boundary_is_retained": all(
            phrase in boundary for phrase in (
                "conditional free-dirac continuum derivation",
                "does not classify arbitrary nonlocal or flavor-mixing roots",
                "electromagnetic coupling",
                "localized electron transport",
            )
        ),
        "clean_package_builder_present": (
            ROOT.joinpath("scripts/build_publication_package.py").is_file()
        ),
    }
    errors.extend(name for name, passed in checks.items() if not passed)

    print(json.dumps({
        "checks": checks,
        "verdict": "DIRAC_001_024_FINAL_AUDIT_PASS" if not errors
            else "DIRAC_001_024_FINAL_AUDIT_FAIL",
        "errors": errors,
    }, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
