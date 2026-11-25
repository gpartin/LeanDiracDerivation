"""Test whether the two cubical copies steer a fixed-charge edge snap.

The literal translated profile plus global field phases preserves shape but
does not transport the Psi Noether center.  The two rank-four cubical copies
provide two additional native phase directions: relative-copy phase in Psi
and in the carrier-versus-vacuum chi contrast.  Their four coefficients are
fixed without Dirac data by preserving both native charges and cancelling the
two transverse center responses.  The longitudinal response is then a native
prediction.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve

import localized_carrier_contrast_fixed_charge_launch as fixed
import localized_carrier_contrast_full_inverse_launch as full
import localized_carrier_contrast_inverse_launch as inverse
import localized_carrier_contrast_noether_motion as noether


HERE = Path(__file__).resolve().parent


def relative_copy_phase_tangents(
        following: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    psi_profile, chi_profile = inverse.contrast.carrier_contrast()
    _, _, _, theta = inverse.rest_layers()
    psi_profile = np.exp(1j*theta)*psi_profile
    flavors = inverse.local.flavor_bases()
    ones = np.ones(8, dtype=complex)
    plus = flavors["plus"]@np.linalg.lstsq(
        flavors["plus"], ones, rcond=None)[0]
    minus = flavors["minus"]@np.linalg.lstsq(
        flavors["minus"], ones, rcond=None)[0]
    relative = plus-minus
    dpsi = np.zeros((inverse.SIDE,)*3+(3,), dtype=complex)
    dchi = np.zeros_like(dpsi)
    for corner, bits in enumerate(itertools.product((0, 1), repeat=3)):
        site = tuple(inverse.ORIGIN+bit for bit in bits)
        dpsi[site] = 1j*relative[corner]*psi_profile[corner]
        dchi[site] = 1j*relative[corner]*chi_profile[corner]
    return (inverse.rank.pack(dpsi, np.zeros_like(dchi)),
            inverse.rank.pack(np.zeros_like(dpsi), dchi))


def response(previous: np.ndarray, current: np.ndarray,
             following: np.ndarray, launch: np.ndarray,
             target: np.ndarray, axis: int) -> np.ndarray:
    charges = fixed.charge_derivatives(
        previous, current, following, launch, target)
    current_center = fixed.center_derivative(
        previous, current, launch, None, "Psi")
    following_center = fixed.center_derivative(
        current, following, None, target, "Psi")
    displacement = following_center-current_center
    transverse = np.delete(displacement, axis)
    return np.concatenate((charges, transverse))


def main() -> None:
    inverse.rank.lock_inertia()
    inverse.rest.SIDE = inverse.SIDE
    inverse.rest.DT = inverse.DT
    inverse.current_audit.SIDE = inverse.SIDE
    inverse.current_audit.DT = inverse.DT
    previous, current, following, _ = inverse.rest_layers()
    base = tuple(inverse.torch.tensor(value, dtype=inverse.torch.float64)
                 for value in (previous, current, following))
    dimension = previous.size
    previous_block = inverse.batched_jvp(
        "previous", base, np.eye(dimension))
    factor = lu_factor(previous_block)
    psi_phase, chi_phase = fixed.local_phase_tangents(following)
    copy_psi_phase, copy_chi_phase = relative_copy_phase_tangents(following)
    steering_targets = np.column_stack((
        psi_phase, chi_phase, copy_psi_phase, copy_chi_phase))
    steering_images = inverse.batched_jvp(
        "following", base, steering_targets)
    steering_launches = lu_solve(factor, -steering_images)
    rows = []
    for axis in range(3):
        for sign in (-1, 1):
            shift_target = inverse.shifted_difference(following, axis, sign)
            shift_image = inverse.batched_jvp(
                "following", base, shift_target[:, None])[:, 0]
            shift_launch = lu_solve(factor, -shift_image)
            base_response = response(
                previous, current, following, shift_launch,
                shift_target, axis)
            response_matrix = np.column_stack([
                response(previous, current, following,
                         steering_launches[:, column],
                         steering_targets[:, column], axis)
                for column in range(4)
            ])
            coefficients, _, solve_rank, singular = np.linalg.lstsq(
                response_matrix, -base_response, rcond=1e-10)
            target = shift_target+steering_targets@coefficients
            launch = shift_launch+steering_launches@coefficients
            corrected = response(
                previous, current, following, launch, target, axis)
            current_center = fixed.center_derivative(
                previous, current, launch, None, "Psi")
            following_center = fixed.center_derivative(
                current, following, None, target, "Psi")
            displacement = following_center-current_center
            residual = previous_block@launch + inverse.batched_jvp(
                "following", base, target[:, None])[:, 0]
            condition = (None if singular[-1] <= 1e-14*singular[0] else
                         float(singular[0]/singular[-1]))
            rows.append({
                "axis": axis,
                "sign": sign,
                "steering_coefficients": {
                    "Psi_common_phase": float(coefficients[0]),
                    "chi_contrast_common_phase": float(coefficients[1]),
                    "Psi_relative_copy_phase": float(coefficients[2]),
                    "chi_relative_copy_phase": float(coefficients[3]),
                },
                "constraint_matrix_rank": int(solve_rank),
                "constraint_matrix_condition": condition,
                "constraint_matrix_singular_values": [
                    float(value) for value in singular],
                "uncorrected_native_constraints": base_response.tolist(),
                "corrected_native_constraints": corrected.tolist(),
                "DEL_relative_residual": float(
                    np.linalg.norm(residual)
                    / max(np.linalg.norm(shift_image), 1e-30)),
                "Psi_center_displacement_response": [float(value)
                                                     for value in displacement],
                "signed_requested_axis_response": float(sign*displacement[axis]),
                "moves_requested_way": bool(sign*displacement[axis] > 1e-8),
                "transverse_center_response_norm": float(np.linalg.norm(
                    np.delete(displacement, axis))),
                "launch_support_power_fractions": full.support_fractions(
                    launch),
            })
    max_constraint = max(max(abs(value) for value in row[
        "corrected_native_constraints"]) for row in rows)
    all_move = all(row["moves_requested_way"] for row in rows)
    result = {
        "status": "V4_CARRIER_CONTRAST_COPY_STEERED_LAUNCH_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "coefficient_provenance": (
            "four internal coefficients are solved from two native Noether "
            "charges and two transverse-center constraints; no longitudinal "
            "speed, Dirac current, or preferred magnitude is fitted"),
        "rows": rows,
        "max_native_constraint_residual": max_constraint,
        "all_six_longitudinal_responses_move_requested_way": all_move,
        "verdict": (
            "TWO_COPY_FIXED_CHARGE_EDGE_TANGENT_PASS" if
            max_constraint < 1e-3 and all_move else
            "TWO_COPY_FIXED_CHARGE_EDGE_TANGENT_FAIL"),
        "claim_boundary": (
            "This tests one linear interval. It does not prove a finite "
            "nonlinear hop, repeated transport, Dirac-current magnitude, or "
            "rest stability."),
        "corrective_action": (
            "On pass, continue the tangent to a nonlinear relative-periodic "
            "branch. On failure, the current action does not turn native "
            "copy phase into longitudinal charge motion at this order; "
            "decompose the missing response before admitting an action term."),
    }
    (HERE / "localized_carrier_contrast_copy_steered_launch.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
