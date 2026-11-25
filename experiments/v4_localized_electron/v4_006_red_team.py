"""Adversarial completion audit for the V4-006 movement claim."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE/name).read_text(encoding="utf-8"))


def main():
    tangent = load("cube_incidence_extended_packet_refinement.json")
    width = load("cube_incidence_extended_packet_width.json")
    current = load("cube_incidence_extended_packet_dirac_current.json")
    raw_charge = load("cube_incidence_extended_packet_current.json")
    nonlinear = load("cube_incidence_extended_packet_nonlinear.json")
    generic_action = load("cube_incidence_generic_action_gate.json")
    generic_box = load("cube_incidence_generic_box_packet.json")
    long_run = load("cube_incidence_extended_packet_long_tangent.json")
    long_dt8 = load("cube_incidence_extended_packet_long_tangent_dt8_10cycle.json")
    long_dt16 = load("cube_incidence_extended_packet_long_tangent_dt16_10cycle.json")
    coefficient_screen = load("cube_incidence_long_transport_coefficient_screen.json")
    repository_basis = load("cube_incidence_repository_action_basis_screen.json")
    complete_frame = load("cube_incidence_complete_frame_inverse.json")
    hodge_rest = load("cube_incidence_hodge_rest_gate.json")
    hodge_branch = load("cube_incidence_hodge_branch_gate.json")
    hodge_response = load("cube_incidence_hodge_branch_response.json")
    hodge_unactivated = load(
        "cube_incidence_hodge_unactivated_branch_response.json")
    hodge_readout = load("cube_incidence_hodge_readout_bridge.json")
    hodge_phase_space = load("cube_incidence_hodge_phase_space_bridge.json")
    hodge_beta = load("cube_incidence_hodge_beta_grading_gate.json")
    complete_cochains = load("cube_incidence_full_hodge_clifford.json")
    complete_generator = load(
        "cube_incidence_complete_cochain_generator_audit.json")
    rigid_orbit = load(
        "cube_incidence_complete_cochain_rigid_orbit_screen.json")
    fractional_orbit = load(
        "cube_incidence_complete_cochain_fractional_orbit_screen.json")
    orbit_rca = load("cube_incidence_fractional_orbit_action_rca.json")
    moving_substrate = load("cube_incidence_moving_substrate_gate.json")
    rca = load("cube_incidence_long_tangent_rca.json")
    checks = [
        {"check": "timestep-refined tangent bridge",
         "passed": tangent["verdict"] == "EXTENDED_PACKET_CONTINUUM_TANGENT_SURVIVES"},
        {"check": "packet width and momentum robustness",
         "passed": width["verdict"] == "EXTENDED_PACKET_WIDTH_ROBUST"},
        {"check": "positive-frequency density/current bridge",
         "passed": current["verdict"] == "EXTENDED_PACKET_DIRAC_CURRENT_PASS"},
        {"check": "short unrestricted nonlinear run",
         "passed": nonlinear["verdict"] == "EXTENDED_PACKET_NONLINEAR_SHORT_RUN_PASS"},
        {"check": "raw nonlinear phase charge equals Dirac probability",
         "passed": False,
         "evidence": raw_charge["verdict"]},
        {"check": "arbitrary-box action identity and rest transfer",
         "passed": generic_action["verdict"] == "GENERIC_ACTION_IDENTITY_AND_REST_PASS"},
        {"check": "arbitrary-box tangent residual convergence",
         "passed": generic_box["verdict"] == "GENERIC_BOX_PACKET_RESIDUAL_CONVERGES",
         "evidence": generic_box["per_side"]},
        {"check": "long tangent packet remains in the Dirac carrier",
         "passed": False,
         "evidence": {
             "verdict": long_run["verdict"],
             "maximum_100_cycle_shape_error": long_run[
                 "maximum_phase_aligned_shape_error"],
             "maximum_100_cycle_norm_drift": long_run["maximum_norm_drift"],
             "ten_cycle_dt_over_8_shape_error": long_dt8[
                 "maximum_phase_aligned_shape_error"],
             "ten_cycle_dt_over_16_shape_error": long_dt16[
                 "maximum_phase_aligned_shape_error"],
         }},
        {"check": "existing connection-term span can close the long residual",
         "passed": False,
         "evidence": {
             "verdict": coefficient_screen["verdict"],
             "best_inverse_diagnostic": coefficient_screen["best_diagnostic_row"],
             "defect_channel": rca["defect_power_fractions"],
         }},
        {"check": "already-derived reciprocal action vocabulary spans the residual",
         "passed": False,
         "evidence": {
             "verdict": repository_basis["verdict"],
             "response_rank": repository_basis["action_response_rank"],
             "best_linearized_relative_error": repository_basis[
                 "best_linearized_repository_basis_relative_error"],
         }},
        {"check": "complete native overlap-area frame spans the residual",
         "passed": False,
         "evidence": {
             "verdict": complete_frame["verdict"],
             "response_rank": complete_frame["action_response_rank"],
             "best_linearized_relative_error": complete_frame[
                 "best_complete_frame_plus_repository_linearized_relative_error"],
             "relative_improvement": complete_frame["relative_improvement"],
         }},
        {"check": "complete odd C19 Hodge connection has a native rest branch",
         "passed": (
             hodge_rest["square_identity"]["scalar_relative_error"] < 1e-12
             and hodge_rest["square_identity"]["vector_relative_error"] < 1e-12
             and hodge_branch["verdict"]
             == "HODGE_DEFORMED_REST_BRANCH_EXISTS_TO_FIRST_ORDER"),
         "evidence": {
             "square_identity": hodge_rest["square_identity"],
             "compatible_branches": hodge_branch["compatible_branches"],
         }},
        {"check": "complete odd C19 Hodge branch closes motion",
         "passed": False,
         "evidence": {
             "response_rank": hodge_response["small_union"]["response_rank"],
             "best_linearized_relative_error": hodge_response[
                 "small_union"]["diagnostic_relative_error"],
             "coefficient_admitted": hodge_response[
                 "research_action_admitted"],
             "unactivated_best_linearized_relative_error": (
                 hodge_unactivated["small_union"][
                     "diagnostic_relative_error"]),
         }},
        {"check": "Hodge phase-space readout is lossless and invariant",
         "passed": False,
         "evidence": {
             "configuration_variants": [{
                 "variant": variant["variant"],
                 "ranks": [row["mapped_real_rank"]
                           for row in variant["rows"]],
                 "cross_blocks": [row["hodge_relative_cross_block"]
                                  for row in variant["rows"]],
             } for variant in hodge_readout["variants"]],
             "phase_space_rows": hodge_phase_space["rows"],
             "verdict": hodge_phase_space["verdict"],
         }},
        {"check": "configuration-image projector supplies native beta",
         "passed": False,
         "evidence": {
             "verdict": hodge_beta["verdict"],
             "hodge_anticommutator_relative": hodge_beta[
                 "beta_checks"]["hodge_anticommutator_relative"],
             "same_grade_spatial_fractions": [
                 hodge_beta["beta_checks"][
                     "positive_same_grade_spatial_fraction"],
                 hodge_beta["beta_checks"][
                     "negative_same_grade_spatial_fraction"],
             ],
         }},
        {"check": "complete cube cochains supply beta and two closed copies",
         "passed": (complete_cochains["verdict"] ==
                    "COMPLETE_CUBICAL_COCHAINS_SUPPLY_BETA_AND_TWO_DIRAC_COPIES"),
         "evidence": {
             "verdict": complete_cochains["verdict"],
             "reduced_beta_anticommutant_dimension": complete_cochains[
                 "reduced_scalar_vector_space"][
                     "beta_anticommutant_dimension"],
             "copies": complete_cochains["complete_cubical_space"][
                 "four_component_copies"],
         }},
        {"check": "fixed-center Fourier lift is an uncontaminated motion comparator",
         "passed": False,
         "evidence": {
             "verdict": complete_generator["verdict"],
             "worst_one_step_error": complete_generator[
                 "worst_native_vs_discrete_dirac_map_error"],
             "worst_transport_increment_error": complete_generator[
                 "worst_transport_increment_error_after_rest_subtraction"],
             "reason": "nonzero rest mismatch and momentum mixing about a localized fixed center",
         }},
        {"check": "simple moving-substrate drift stabilizes the carrier",
         "passed": False,
         "evidence": {
             "verdict": moving_substrate["verdict"],
             "minimum_translation_DEL_norm": moving_substrate[
                 "minimum_simple_translation_DEL_norm"],
             "maximum_linear_background_effect": moving_substrate[
                 "maximum_linear_background_effect_on_packet"],
         }},
        {"check": "two full translations both directions",
         "passed": False,
         "evidence": {
             "rigid_orbit_verdict": rigid_orbit["verdict"],
             "rigid_best_DEL_norm_over_state_norm": rigid_orbit[
                 "best_row"]["DEL_norm_over_state_norm"],
             "fractional_endpoint_defect": fractional_orbit[
                 "endpoint_translation_maximum_relative_defect"],
             "fractional_orbit_verdict": fractional_orbit["verdict"],
             "fractional_best_maximum_DEL_norm_over_state_norm": (
                 fractional_orbit["best_row"][
                     "maximum_DEL_norm_over_state_norm"]),
             "action_RCA": orbit_rca["speed_summary"],
         }},
        {"check": "100-cycle moving-carrier perturbation gate",
         "passed": False,
         "evidence": "the linearized packet already leaves the sector before a nonlinear endurance run is admissible"},
        {"check": "same-carrier localized motion theorem in Lean",
         "passed": False,
         "evidence": "not formalized"},
    ]
    result = {
        "status": "V4_006_RED_TEAM",
        "checks": checks,
        "passed_checks": sum(row["passed"] for row in checks),
        "total_checks": len(checks),
        "critical_open_findings": [row["check"] for row in checks
                                   if not row["passed"]],
        "verdict": "V4_006_REMAINS_OPEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "claim_boundary": (
            "The arbitrary-box and short-time continuum bridges pass, but "
            "the packet does not remain in the same native carrier over one "
            "cycle. The failure persists under timestep refinement and is "
            "not closed by the full tested reciprocal action vocabulary, "
            "including the complete overlap-area frame. The exact odd C19 "
            "Hodge connection adds compatible rest branches and a tenth "
            "response direction but does not close the endpoint. The reduced "
            "tau/W chart provably has no beta, while complete cube cochains "
            "supply exact beta and two closed rank-four copies. The existing "
            "fixed-center Fourier lift is contaminated at zero momentum, so "
            "V4-006 now requires a finite relative-periodic translated orbit "
            "measured with that complete cochain grading."),
    }
    (HERE/"v4_006_red_team.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
