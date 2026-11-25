"""Assemble the evidence boundary for the six-face V4-006 candidate."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name: str):
    return json.loads((HERE/name).read_text(encoding="utf-8"))


def main() -> None:
    geometry = load("live_face_transport_geometry_gate.json")
    rest = load(
        "live_clifford_bond_rest_gate_joint_face_wedge_laplacian_smooth.json")
    reflection = load(
        "live_link_reflection_covariance_gate_joint_face_wedge_laplacian_smooth.json")
    receiver = load(
        "live_link_receiver_pattern_gate_joint_face_wedge_laplacian_smooth.json")
    short = load(
        "live_clifford_nonlinear_gate_joint_face_laplacian_smooth_all8_priorchi.json")
    nine = load(
        "live_clifford_nonlinear_gate_joint_face_laplacian_smooth_plus_xminus_9updates.json")
    scale_limit = load(
        "live_clifford_joint_face_laplacian_scale_scan_limit_0p05cycle.json")
    uniform = load(
        "live_clifford_nonlinear_gate_joint_face_laplacian_smooth_uniform_k0_9updates.json")
    nine_samples = nine["rows"][0]["samples"]
    first_over_one_percent = next(
        (row["step"] for row in nine_samples
         if row["native_vs_dirac_relative_error"] > 0.01), None)
    result = {
        "status": "V4_006_FACE_TRANSPORT_PARTIAL_PASS",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "matter_transport_hypothesis": (
            "C19 remains the native field geometry; the added matter-current "
            "metric uses only the six incident face bonds."),
        "derived_omega_m": geometry["omega_m_geometric_ratio"],
        "action_gates": {
            "globally_smooth": True,
            "strictly_positive": geometry["strictly_positive"],
            "exact_rest_DEL_RMS": rest["rest_DEL_rms"],
            "reflection_defect": reflection[
                "maximum_interval_relative_defect"],
        },
        "receiver_gates": {
            "Psi_target_overlap": receiver["maximum_target_overlap"],
            "Psi_minimum_orthogonal_defect": receiver[
                "minimum_active_orthogonal_defect"],
            "best_common_launch_chi_target_overlap": max(
                row["chi_target_overlap"] for row in receiver["rows"]
                if row["launch_mode"] == "common"
                and row["chi_response_norm"] > 1e-10),
        },
        "short_nonlinear_gates": {
            "cases": 8,
            "updates": short["requested_updates"],
            "all_converged": short["all_solvers_converged"],
            "maximum_Dirac_error": short[
                "maximum_native_vs_dirac_relative_error"],
            "maximum_four_component_leakage": short[
                "maximum_outside_four_component_fraction"],
        },
        "first_accumulated_departure": {
            "first_update_above_one_percent": first_over_one_percent,
            "nine_update_Dirac_error": nine_samples[-1][
                "native_vs_dirac_relative_error"],
            "nine_update_leakage": nine_samples[-1][
                "outside_four_component_fraction"],
            "nine_update_norm_ratio": nine_samples[-1]["native_norm_ratio"],
        },
        "localization_controls": {
            "uniform_zero_momentum_nine_update_error": uniform["rows"][0][
                "samples"][-1]["native_vs_dirac_relative_error"],
            "compact_zero_momentum_nine_update_error": load(
                "live_clifford_nonlinear_gate_joint_face_laplacian_smooth_k0_9updates.json"
            )["rows"][0]["samples"][-1]["native_vs_dirac_relative_error"],
            "compact_moving_nine_update_error": nine_samples[-1][
                "native_vs_dirac_relative_error"],
            "interpretation": (
                "The uniform rest/time control passes. Most accumulated "
                "error enters when the four-component envelope is localized; "
                "momentum adds a smaller further defect."
            ),
        },
        "action_class_limit_control": {
            "largest_diagnostic_scale": scale_limit["rows"][-1]["scale"],
            "limiting_maximum_shape_error": scale_limit["rows"][-1][
                "maximum_shape_error"],
            "limiting_maximum_cross_block": scale_limit["rows"][-1][
                "maximum_psi_cross_block"],
            "interpretation": (
                "The face Laplacian removes almost all sector leakage, but "
                "a finite within-sector time/mass/current mismatch remains "
                "even in the diagnostic infinite-stiffness limit."),
        },
        "passed": [
            "six-face 6/19 coordination identity",
            "global smoothness and kinetic positivity",
            "exact retained rest orbit",
            "cubic reflections",
            "exact first Psi receiver face",
            "substantially aligned first chi receiver response",
            "both copies, both signs, axis and diagonal for four updates",
        ],
        "open": [
            "derive the lossless compact-gradient/readout bridge",
            "remove the smaller residual within-sector transport/current defect",
            "complete one full Dirac-time cell handoff",
            "repeat two handoffs in both signs of all three axes",
            "derive Noether charge/current and verify energy conservation",
            "100-cycle stability and perturbation gates",
            "dt and box refinement",
            "Lean action, variation, update, and same-carrier proof",
        ],
        "verdict": "V4_006_REMAINS_OPEN",
        "next_falsifiable_target": (
            "Compress the face-Laplacian DEL map through the frozen native "
            "four-component readout and isolate the residual beta/time/current "
            "block. A new action term is admissible only if it is a native, "
            "reciprocal invariant that removes that block while leaving the "
            "six-face transfer and rest nullspace unchanged."),
    }
    (HERE/"live_face_transport_verdict.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
