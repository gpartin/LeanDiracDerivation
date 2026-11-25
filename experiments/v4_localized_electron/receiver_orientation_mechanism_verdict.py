"""Assemble the ten receiver-orientation mechanism gates.

Each row points to a concrete action or preparation test.  A scoped candidate
failure is not promoted to a universal no-go for its broad literature class.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as native_box
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_extended_packet_gate as packet_gate
import cube_incidence_nonlinear_action as native_action


HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def main() -> None:
    connection = load("mechanism01_phase_space_connection_gate.json")
    spin = load("cube_incidence_spin_diagonal_packet_gate.json")
    exceptional = load("cube_incidence_exceptional_discretization_screen.json")
    translation = load("cube_incidence_dirac_translation_requirement.json")
    tail = load("canonical_deformed_relative_periodic_orbit.json")
    bistable = load("c19_bistable_chi_orientation_gate.json")
    rotor = load("canonical_rotor_wake_audit.json")
    wake = load("canonical_transfer_wake_branch.json")
    phase = load("joint_phase_lock_action_gate.json")
    current = load("cube_incidence_noether_launch.json")
    copy = load("cube_incidence_copy_exchange_packet_gate.json")

    bare, _matrices = native_action.expected_blocks()
    theta = float(np.arccos(
        1 - native_time.DT**2 * native_time.CHI0**2 / 2))
    rotation = native_spectrum.rotation_matrix(bare["rest"].size, theta)
    psi_now, _chi_now = native_box.unpack(
        torch.tensor(bare["rest"], dtype=torch.float64), 5)
    psi_next, _chi_next = native_box.unpack(
        torch.tensor(rotation @ bare["rest"], dtype=torch.float64), 5)
    rest_cycle_shift = packet_gate.wrapped_shift(
        packet_gate.circular_center(psi_now.detach().numpy()),
        packet_gate.circular_center(psi_next.detach().numpy())).tolist()

    rows = [
        {
            "id": 1,
            "mechanism": "native non-Abelian parallel transport",
            "implementation": "derived cube-incidence reciprocal symplectic connection on the spin-diagonal extended packet",
            "evidence": {
                "best_one_cycle_error": connection["best_row"]["best_case"]["one_cycle_phase_aligned_shape_error"],
                "held_direction_error": connection["best_row"]["selected_frame_worst_case_error"],
            },
            "verdict": "TESTED_IMPLEMENTATION_FAIL",
            "winner": False,
        },
        {
            "id": 2,
            "mechanism": "self-generated coin-and-shift",
            "implementation": "geometry spin rotation followed by unchanged native C19 evolution",
            "evidence": {
                "first_step_error": spin["best_case"]["one_step_native_vs_dirac_relative_error"],
                "one_cycle_error": spin["best_case"]["one_cycle_phase_aligned_shape_error"],
                "native_action_derivation": False,
            },
            "verdict": "DIAGNOSTIC_ROTATION_HELPS_BUT_NO_NATIVE_COIN_SHIFT",
            "winner": False,
        },
        {
            "id": 3,
            "mechanism": "Ablowitz-Ladik-like nonlinear coherent hopping",
            "implementation": "32 reciprocal C19 position-action redistributions and coherent-hopping invariants",
            "evidence": {
                "basis_size": len(exceptional["candidate_actions"]),
                "rest_preserving_nullity": exceptional["rest_force_nullity"],
                "best_rest_preserving_relative_residual": exceptional["best_rest_preserving_relative_residual"],
            },
            "verdict": "POSITION_ONLY_ACTION_CLASS_FAIL",
            "winner": False,
        },
        {
            "id": 4,
            "mechanism": "translation-zero-mode restoration",
            "implementation": "native localized center-displacement tangent",
            "evidence": {
                "translation_curvatures": [row["generalized_translation_curvature"] for row in translation["continuous_curvature_rows"]],
                "existing_action_best_relative_residual": translation["existing_action_inverse_screen"]["best_relative_residual"],
            },
            "verdict": "CURRENT_ACTION_PINNED",
            "winner": False,
        },
        {
            "id": 5,
            "mechanism": "self-consistent precursor or traveling tail",
            "implementation": "32-update charge-constrained freely deformed relative-periodic shooting",
            "evidence": {
                "boundary_residual": tail["row"]["final_boundary_residual"],
                "second_cell_residual": tail["row"]["unreset_second_cell_residual"],
                "psi_fidelity": tail["row"]["carrier_metrics"]["eight_site_psi_fidelity"],
                "containment": tail["row"]["carrier_metrics"]["psi_core_containment"],
            },
            "verdict": "TESTED_SOLVE_NO_SAME_CARRIER_ORBIT",
            "winner": False,
        },
        {
            "id": 6,
            "mechanism": "excitable or autocatalytic chi front",
            "implementation": "positive reciprocal C19 bistable chi-orientation action",
            "evidence": {
                "rest_preserved": bistable["rest_preserved"],
                "spectrum_improved": bistable["spectrum_improved"],
                "unit_strength_growth_rate": bistable["rows"][1]["growth_rate"],
            },
            "verdict": "TESTED_BISTABLE_ACTION_FAIL",
            "winner": False,
        },
        {
            "id": 7,
            "mechanism": "adiabatic make-before-break transfer",
            "implementation": "native rest-rotation receiver wake plus transfer-wake continuation",
            "evidence": {
                "maximum_rest_wake_receiver_force": rotor["maximum_receiving_psi_force_norm"],
                "maximum_rest_wake_phase_current": rotor["maximum_receiving_phase_current"],
                "continued_wake_residual_rms": wake["final_residual_rms"],
            },
            "verdict": "NO_AUTONOMOUS_DARK_HANDOFF_FOUND",
            "winner": False,
        },
        {
            "id": 8,
            "mechanism": "complete-frame phase entrainment",
            "implementation": "positive reciprocal joint Psi-chi phase-lock action at geometric scales",
            "evidence": {
                "rest_force_norm": phase["rest_force_norm"],
                "minimum_tested_growth_rate": min(row["maximum_growth_rate"] for row in phase["coefficient_scan"]),
            },
            "verdict": "TESTED_PHASE_LOCK_ACTION_FAIL",
            "winner": False,
        },
        {
            "id": 9,
            "mechanism": "autonomous topological pump",
            "implementation": "one complete internal rest cycle and native action-derived launch-current control",
            "evidence": {
                "rest_step_center_shift": rest_cycle_shift,
                "global_flux_controls": current["global_flux_controls"],
                "directed_launch_mode": current["selected_native_launch_mode"],
            },
            "verdict": "SIMPLE_INTERNAL_CYCLE_HAS_NO_AUTONOMOUS_PUMP",
            "winner": False,
        },
        {
            "id": 10,
            "mechanism": "two-copy relay or staggered handshake",
            "implementation": "exact Hermitian copy-exchange involution separating face and edge transport",
            "evidence": {
                "best_one_cycle_error": copy["best_case"]["one_cycle_phase_aligned_shape_error"],
            },
            "verdict": "SIMPLE_TWO_COPY_RELAY_FAIL",
            "winner": False,
        },
    ]
    winners = [row for row in rows if row["winner"]]
    result = {
        "status": "V4_RECEIVER_ORIENTATION_TEN_MECHANISM_VERDICT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "winner_definition": "same carrier, freely evolving, frozen discrete-Dirac comparator, rest preserved, no preload/projection/reset/fitted runtime support",
        "rows": rows,
        "winner_count": len(winners),
        "verdict": "WINNER_FOUND_STOP_FOR_ASSESSMENT" if winners else "NO_TESTED_MECHANISM_IS_A_WINNER",
        "claim_boundary": "Each verdict is scoped to the concrete tested implementation. Broad mechanism classes are not universal no-go theorems.",
    }
    (HERE / "receiver_orientation_mechanism_verdict.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
