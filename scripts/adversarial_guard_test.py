"""Adversarial controls for the manual production audit."""

from __future__ import annotations

import argparse
import copy
import math

import production_guard as guard


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-production-build", action="store_true")
    parser.parse_args()
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
        raise AssertionError("duplicate report erased an axiom dependency")
    payload = guard.load(guard.EVIDENCE_010)
    if guard.validate_dirac010_evidence(payload):
        raise AssertionError("passing DIRAC-010 evidence was rejected")
    broken = copy.deepcopy(payload)
    broken["status"] = "FAIL"
    if not guard.validate_dirac010_evidence(broken):
        raise AssertionError("failed evidence was accepted")
    nonfinite = copy.deepcopy(payload)
    nonfinite["max_error"] = math.nan
    if not guard.validate_dirac010_evidence(nonfinite):
        raise AssertionError("NaN evidence was accepted")
    missing = copy.deepcopy(payload)
    missing["errors"] = {}
    if not guard.validate_dirac010_evidence(missing):
        raise AssertionError("incomplete component evidence was accepted")
    payload_011 = guard.load(guard.EVIDENCE_011)
    if guard.validate_dirac011_evidence(payload_011):
        raise AssertionError("passing DIRAC-011 evidence was rejected")
    broken_011 = copy.deepcopy(payload_011)
    broken_011["max_error"] = broken_011["tolerance"] * 2
    if not guard.validate_dirac011_evidence(broken_011):
        raise AssertionError("out-of-tolerance DIRAC-011 evidence was accepted")
    nonfinite_011 = copy.deepcopy(payload_011)
    nonfinite_011["errors"][next(iter(nonfinite_011["errors"]))] = math.inf
    if not guard.validate_dirac011_evidence(nonfinite_011):
        raise AssertionError("non-finite DIRAC-011 component was accepted")
    payload_012 = guard.load(guard.EVIDENCE_012)
    if guard.validate_dirac012_evidence(payload_012):
        raise AssertionError("passing DIRAC-012 evidence was rejected")
    broken_012 = copy.deepcopy(payload_012)
    broken_012["checks"]["every_proved_bound_holds"] = False
    if not guard.validate_dirac012_evidence(broken_012):
        raise AssertionError("violated DIRAC-012 bound was accepted")
    broken_branch_012 = copy.deepcopy(payload_012)
    broken_branch_012["checks"][
        "recurrence_derived_first_order_branch_reproduces"
    ] = False
    if not guard.validate_dirac012_evidence(broken_branch_012):
        raise AssertionError("failed recurrence-derived branch was accepted")
    nonfinite_012 = copy.deepcopy(payload_012)
    nonfinite_012["minimum_normalized_bound_margin"] = math.nan
    if not guard.validate_dirac012_evidence(nonfinite_012):
        raise AssertionError("non-finite DIRAC-012 metric was accepted")
    incomplete_012 = copy.deepcopy(payload_012)
    incomplete_012["checks"].pop(
        "normalized_equation_error_decreases_under_joint_refinement"
    )
    if not guard.validate_dirac012_evidence(incomplete_012):
        raise AssertionError("incomplete DIRAC-012 checks were accepted")
    payload_013 = guard.load(guard.EVIDENCE_013)
    if guard.validate_dirac013_evidence(payload_013):
        raise AssertionError("passing DIRAC-013 evidence was rejected")
    broken_013 = copy.deepcopy(payload_013)
    broken_013["checks"]["exact_local_continuity_reconstructed"] = False
    if not guard.validate_dirac013_evidence(broken_013):
        raise AssertionError("failed DIRAC-013 continuity was accepted")
    nonfinite_013 = copy.deepcopy(payload_013)
    nonfinite_013["minimum_flux_refinement_ratio"] = math.nan
    if not guard.validate_dirac013_evidence(nonfinite_013):
        raise AssertionError("non-finite DIRAC-013 refinement was accepted")
    incomplete_013 = copy.deepcopy(payload_013)
    incomplete_013["checks"].pop("dirac_current_identity_reconstructed")
    if not guard.validate_dirac013_evidence(incomplete_013):
        raise AssertionError("incomplete DIRAC-013 checks were accepted")
    payload_016 = guard.load(guard.EVIDENCE_016)
    if guard.validate_dirac016_evidence(payload_016):
        raise AssertionError("passing DIRAC-016 evidence was rejected")
    broken_016 = copy.deepcopy(payload_016)
    broken_016["checks"]["signed_incidence_preserves_copy"] = False
    if not guard.validate_dirac016_evidence(broken_016):
        raise AssertionError("failed DIRAC-016 copy check was accepted")
    nonfinite_016 = copy.deepcopy(payload_016)
    nonfinite_016["errors"]["minus_to_plus"] = math.nan
    if not guard.validate_dirac016_evidence(nonfinite_016):
        raise AssertionError("non-finite DIRAC-016 evidence was accepted")
    payload_017 = guard.load(guard.EVIDENCE_017)
    if guard.validate_dirac017_evidence(payload_017):
        raise AssertionError("passing DIRAC-017 evidence was rejected")
    broken_017 = copy.deepcopy(payload_017)
    broken_017["checks"]["continuous_hamiltonian_covariance_reconstructed"] = False
    if not guard.validate_dirac017_evidence(broken_017):
        raise AssertionError("failed DIRAC-017 covariance was accepted")
    nonfinite_017 = copy.deepcopy(payload_017)
    nonfinite_017["max_error"] = math.inf
    if not guard.validate_dirac017_evidence(nonfinite_017):
        raise AssertionError("non-finite DIRAC-017 evidence was accepted")
    payload_018 = guard.load(guard.EVIDENCE_018)
    if guard.validate_dirac018_evidence(payload_018):
        raise AssertionError("passing DIRAC-018 evidence was rejected")
    broken_018 = copy.deepcopy(payload_018)
    broken_018["checks"]["cubic_commutant_has_dimension_one"] = False
    if not guard.validate_dirac018_evidence(broken_018):
        raise AssertionError("failed DIRAC-018 commutant check was accepted")
    nonfinite_018 = copy.deepcopy(payload_018)
    nonfinite_018["max_error"] = math.nan
    if not guard.validate_dirac018_evidence(nonfinite_018):
        raise AssertionError("non-finite DIRAC-018 evidence was accepted")
    wrong_rank_018 = copy.deepcopy(payload_018)
    wrong_rank_018["plus_sector_rank"] = 3
    if not guard.validate_dirac018_evidence(wrong_rank_018):
        raise AssertionError("incorrect DIRAC-018 flavor rank was accepted")
    incomplete_zeros_018 = copy.deepcopy(payload_018)
    incomplete_zeros_018["periodic_zone_zero_counts"].pop("32")
    if not guard.validate_dirac018_evidence(incomplete_zeros_018):
        raise AssertionError("incomplete DIRAC-018 zero census was accepted")
    inflated_claim_018 = copy.deepcopy(payload_018)
    inflated_claim_018["excluded_claims"].remove(
        "classification of arbitrary nonlocal square roots"
    )
    if not guard.validate_dirac018_evidence(inflated_claim_018):
        raise AssertionError("inflated DIRAC-018 claim boundary was accepted")
    payload_019 = guard.load(guard.EVIDENCE_019)
    if guard.validate_dirac019_evidence(payload_019):
        raise AssertionError("passing DIRAC-019 evidence was rejected")
    broken_019 = copy.deepcopy(payload_019)
    broken_019["checks"]["fourier_packet_velocity_converges_to_current_prediction"] = False
    if not guard.validate_dirac019_evidence(broken_019):
        raise AssertionError("failed DIRAC-019 movement check was accepted")
    broken_current_019 = copy.deepcopy(payload_019)
    broken_current_019["checks"]["all_step_current_is_preserved"] = False
    if not guard.validate_dirac019_evidence(broken_current_019):
        raise AssertionError("failed DIRAC-019 sustained current was accepted")
    unstable_019 = copy.deepcopy(payload_019)
    unstable_019["checks"]["unstable_timestep_is_rejected"] = False
    if not guard.validate_dirac019_evidence(unstable_019):
        raise AssertionError("unstable DIRAC-019 timestep was accepted")
    nonfinite_019 = copy.deepcopy(payload_019)
    nonfinite_019["real_space_packet_controls"][0]["velocity_error"] = math.nan
    if not guard.validate_dirac019_evidence(nonfinite_019):
        raise AssertionError("non-finite DIRAC-019 movement evidence was accepted")
    inflated_claim_019 = copy.deepcopy(payload_019)
    inflated_claim_019["excluded_claims"].remove(
        "nonlinear localized electron transport"
    )
    if not guard.validate_dirac019_evidence(inflated_claim_019):
        raise AssertionError("inflated DIRAC-019 claim boundary was accepted")
    payload_020 = guard.load(guard.EVIDENCE_020)
    if guard.validate_dirac020_evidence(payload_020):
        raise AssertionError("passing DIRAC-020 evidence was rejected")
    broken_frame_020 = copy.deepcopy(payload_020)
    broken_frame_020["checks"]["explicit_fourier_frame_is_orthonormal"] = False
    if not guard.validate_dirac020_evidence(broken_frame_020):
        raise AssertionError("failed DIRAC-020 Fourier frame was accepted")
    broken_momentum_020 = copy.deepcopy(payload_020)
    broken_momentum_020["checks"][
        "fourier_character_matches_mode_momentum"
    ] = False
    if not guard.validate_dirac020_evidence(broken_momentum_020):
        raise AssertionError("failed DIRAC-020 momentum bridge was accepted")
    broken_current_020 = copy.deepcopy(payload_020)
    broken_current_020["checks"][
        "integrated_spatial_current_equals_spectral_current"
    ] = False
    if not guard.validate_dirac020_evidence(broken_current_020):
        raise AssertionError("failed DIRAC-020 spatial current was accepted")
    broken_bound_020 = copy.deepcopy(payload_020)
    broken_bound_020["checks"]["finite_step_pointwise_error_bound_holds"] = False
    if not guard.validate_dirac020_evidence(broken_bound_020):
        raise AssertionError("failed DIRAC-020 finite-step bound was accepted")
    nonfinite_020 = copy.deepcopy(payload_020)
    nonfinite_020["coherent_packet_controls"][0]["density_drift"] = math.nan
    if not guard.validate_dirac020_evidence(nonfinite_020):
        raise AssertionError("non-finite DIRAC-020 evidence was accepted")
    inflated_claim_020 = copy.deepcopy(payload_020)
    inflated_claim_020["excluded_claims"].remove(
        "nonlinear localized electron transport"
    )
    if not guard.validate_dirac020_evidence(inflated_claim_020):
        raise AssertionError("inflated DIRAC-020 claim boundary was accepted")
    payload_021 = guard.load(guard.EVIDENCE_021)
    if guard.validate_dirac021_evidence(payload_021):
        raise AssertionError("passing DIRAC-021 evidence was rejected")
    broken_frequency_021 = copy.deepcopy(payload_021)
    broken_frequency_021["native_values"]["branch_beat_angular_frequency"] = 19.0
    if not guard.validate_dirac021_evidence(broken_frequency_021):
        raise AssertionError("failed DIRAC-021 branch doubling was accepted")
    broken_equation_021 = copy.deepcopy(payload_021)
    broken_equation_021["checks"][
        "both_rest_states_satisfy_the_dirac_time_equation"
    ] = False
    if not guard.validate_dirac021_evidence(broken_equation_021):
        raise AssertionError("failed DIRAC-021 time equation was accepted")
    nonfinite_021 = copy.deepcopy(payload_021)
    nonfinite_021["errors"]["relative_phase_doubles"] = math.nan
    if not guard.validate_dirac021_evidence(nonfinite_021):
        raise AssertionError("non-finite DIRAC-021 evidence was accepted")
    pure_density_overclaim_021 = copy.deepcopy(payload_021)
    pure_density_overclaim_021["checks"][
        "pure_branch_density_has_no_clock_signal"
    ] = False
    if not guard.validate_dirac021_evidence(pure_density_overclaim_021):
        raise AssertionError("observable pure-state clock claim was accepted")
    inflated_claim_021 = copy.deepcopy(payload_021)
    inflated_claim_021["excluded_claims"].remove(
        "zitterbewegung without a specified cross-branch observable"
    )
    if not guard.validate_dirac021_evidence(inflated_claim_021):
        raise AssertionError("inflated DIRAC-021 claim boundary was accepted")
    payload_022 = guard.load(guard.EVIDENCE_022)
    if guard.validate_dirac022_evidence(payload_022):
        raise AssertionError("passing DIRAC-022 evidence was rejected")
    inserted_frequency_022 = copy.deepcopy(payload_022)
    inserted_frequency_022["native_values"]["zitter_angular_frequency"] = 19.0
    if not guard.validate_dirac022_evidence(inserted_frequency_022):
        raise AssertionError("failed DIRAC-022 doubled frequency was accepted")
    broken_velocity_origin_022 = copy.deepcopy(payload_022)
    broken_velocity_origin_022["checks"][
        "velocity_is_hamiltonian_momentum_response"
    ] = False
    if not guard.validate_dirac022_evidence(broken_velocity_origin_022):
        raise AssertionError("inserted DIRAC-022 velocity was accepted")
    nonfinite_022 = copy.deepcopy(payload_022)
    nonfinite_022["errors"]["coherent_velocity_law"] = math.nan
    if not guard.validate_dirac022_evidence(nonfinite_022):
        raise AssertionError("non-finite DIRAC-022 evidence was accepted")
    broken_incoherent_control_022 = copy.deepcopy(payload_022)
    broken_incoherent_control_022["negative_controls"][
        "incoherent_mixture_current"
    ] = 1.0
    if not guard.validate_dirac022_evidence(broken_incoherent_control_022):
        raise AssertionError("failed DIRAC-022 incoherent control was accepted")
    inflated_claim_022 = copy.deepcopy(payload_022)
    inflated_claim_022["excluded_claims"].remove(
        "a localized nonlinear native lattice electron"
    )
    if not guard.validate_dirac022_evidence(inflated_claim_022):
        raise AssertionError("inflated DIRAC-022 claim boundary was accepted")
    payload_023 = guard.load(guard.EVIDENCE_023)
    if guard.validate_dirac023_evidence(payload_023):
        raise AssertionError("passing DIRAC-023 evidence was rejected")
    broken_clock_023 = copy.deepcopy(payload_023)
    broken_clock_023["checks"]["clock_slowdown_matches_speed_factor"] = False
    if not guard.validate_dirac023_evidence(broken_clock_023):
        raise AssertionError("failed DIRAC-023 clock factor was accepted")
    broken_phase_023 = copy.deepcopy(payload_023)
    broken_phase_023["errors"]["packet_center_phase_rate"] = 1.0
    if not guard.validate_dirac023_evidence(broken_phase_023):
        raise AssertionError("failed DIRAC-023 phase rate was accepted")
    nonfinite_023 = copy.deepcopy(payload_023)
    nonfinite_023["diagnostics"][
        "minimum_sampled_c_eff_squared_minus_speed_squared"
    ] = math.nan
    if not guard.validate_dirac023_evidence(nonfinite_023):
        raise AssertionError("non-finite DIRAC-023 diagnostic was accepted")
    ineffective_control_023 = copy.deepcopy(payload_023)
    ineffective_control_023["negative_controls"][
        "wrong_transport_max_reciprocity_defect"
    ] = 0.0
    if not guard.validate_dirac023_evidence(ineffective_control_023):
        raise AssertionError("ineffective DIRAC-023 control was accepted")
    inflated_claim_023 = copy.deepcopy(payload_023)
    inflated_claim_023["excluded_claims"].remove(
        "an exact finite-spacing phase-harmony identity"
    )
    if not guard.validate_dirac023_evidence(inflated_claim_023):
        raise AssertionError("inflated DIRAC-023 claim boundary was accepted")
    payload_024 = guard.load(guard.EVIDENCE_024)
    if guard.validate_dirac024_evidence(payload_024):
        raise AssertionError("passing DIRAC-024 evidence was rejected")
    broken_antiperiod_024 = copy.deepcopy(payload_024)
    broken_antiperiod_024["checks"][
        "one_zitter_cycle_negates_the_spinor"
    ] = False
    if not guard.validate_dirac024_evidence(broken_antiperiod_024):
        raise AssertionError("failed DIRAC-024 antiperiod was accepted")
    nonfinite_024 = copy.deepcopy(payload_024)
    nonfinite_024["errors"]["state_antiperiod_0"] = math.nan
    if not guard.validate_dirac024_evidence(nonfinite_024):
        raise AssertionError("non-finite DIRAC-024 evidence was accepted")
    ineffective_control_024 = copy.deepcopy(payload_024)
    ineffective_control_024["negative_controls"][
        "wrong_period_min_antiperiod_defect"
    ] = 0.0
    if not guard.validate_dirac024_evidence(ineffective_control_024):
        raise AssertionError("ineffective DIRAC-024 control was accepted")
    inflated_claim_024 = copy.deepcopy(payload_024)
    inflated_claim_024["excluded_claims"].remove(
        "time evolution is identical to spatial rotation"
    )
    if not guard.validate_dirac024_evidence(inflated_claim_024):
        raise AssertionError("inflated DIRAC-024 claim boundary was accepted")
    payload_025 = guard.load(guard.EVIDENCE_025)
    if guard.validate_dirac025_evidence(payload_025):
        raise AssertionError("passing DIRAC-025 evidence was rejected")
    broken_identity_025 = copy.deepcopy(payload_025)
    broken_identity_025["checks"][
        "known_relativistic_ratios_pass"
    ] = False
    if not guard.validate_dirac025_evidence(broken_identity_025):
        raise AssertionError("failed DIRAC-025 ratio identity was accepted")
    nonfinite_025 = copy.deepcopy(payload_025)
    nonfinite_025["errors"]["sample_0_operator_square"] = math.nan
    if not guard.validate_dirac025_evidence(nonfinite_025):
        raise AssertionError("non-finite DIRAC-025 evidence was accepted")
    ineffective_control_025 = copy.deepcopy(payload_025)
    ineffective_control_025["negative_controls"][
        "wrong_rest_mass_clock_defect"
    ] = 0.0
    if not guard.validate_dirac025_evidence(ineffective_control_025):
        raise AssertionError("ineffective DIRAC-025 control was accepted")
    inflated_claim_025 = copy.deepcopy(payload_025)
    inflated_claim_025["excluded_claims"].remove(
        "a classical hidden path moving at c_eff"
    )
    if not guard.validate_dirac025_evidence(inflated_claim_025):
        raise AssertionError("inflated DIRAC-025 claim boundary was accepted")
    print("ADVERSARIAL_GUARD_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
