"""Derive the fixed-charge internal rotation of the native edge launch.

A literal shape shift changes the two native phase charges at first order and
therefore is not a physical boost of the same state.  This audit augments the
translated endpoint only by its local Psi phase tangent and its local
carrier-versus-vacuum chi phase tangent.  Their coefficients are fixed by
preserving the independently derived Psi and chi Noether charges.  The full
previous layer is then solved from the unchanged native DEL equation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve

import localized_carrier_contrast_full_inverse_launch as full
import localized_carrier_contrast_inverse_launch as inverse
import localized_carrier_contrast_noether_motion as noether


HERE = Path(__file__).resolve().parent
STEP = 2e-6


def local_phase_tangents(following: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    psi, chi = inverse.rank.native.unpack(following, inverse.SIDE)
    vacuum = np.zeros_like(chi)
    vacuum[..., 2] = inverse.rank.R
    psi_phase = inverse.rank.pack(1j*psi, np.zeros_like(chi))
    chi_phase = inverse.rank.pack(np.zeros_like(psi), 1j*(chi-vacuum))
    return psi_phase, chi_phase


def charge_derivatives(previous: np.ndarray, current: np.ndarray,
                       following: np.ndarray, launch: np.ndarray,
                       target: np.ndarray) -> np.ndarray:
    values = []
    for mode in ("Psi", "chi"):
        current_derivative = (
            np.sum(noether.phase_charge_density(
                previous+STEP*launch, current, mode))
            - np.sum(noether.phase_charge_density(
                previous-STEP*launch, current, mode))) / (2*STEP)
        following_derivative = (
            np.sum(noether.phase_charge_density(
                current, following+STEP*target, mode))
            - np.sum(noether.phase_charge_density(
                current, following-STEP*target, mode))) / (2*STEP)
        values.append((current_derivative+following_derivative)/2)
    return np.asarray(values)


def center_derivative(left: np.ndarray, right: np.ndarray,
                      left_direction: np.ndarray | None,
                      right_direction: np.ndarray | None,
                      mode: str = "Psi") -> np.ndarray:
    def value(epsilon: float) -> np.ndarray:
        l = left if left_direction is None else left+epsilon*left_direction
        r = right if right_direction is None else right+epsilon*right_direction
        return noether.center(noether.phase_charge_density(l, r, mode))
    return (value(STEP)-value(-STEP))/(2*STEP)


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
    identity = np.eye(dimension)
    previous_block = inverse.batched_jvp("previous", base, identity)
    factor = lu_factor(previous_block)
    psi_phase, chi_phase = local_phase_tangents(following)
    phase_targets = np.column_stack((psi_phase, chi_phase))
    phase_images = inverse.batched_jvp("following", base, phase_targets)
    phase_launches = lu_solve(factor, -phase_images)

    rows = []
    for axis in range(3):
        for sign in (-1, 1):
            shift_target = inverse.shifted_difference(following, axis, sign)
            shift_image = inverse.batched_jvp(
                "following", base, shift_target[:, None])[:, 0]
            shift_launch = lu_solve(factor, -shift_image)
            base_charge = charge_derivatives(
                previous, current, following, shift_launch, shift_target)
            response = np.column_stack([
                charge_derivatives(
                    previous, current, following,
                    phase_launches[:, column], phase_targets[:, column])
                for column in range(2)
            ])
            coefficients = np.linalg.solve(response, -base_charge)
            target = shift_target+phase_targets@coefficients
            launch = shift_launch+phase_launches@coefficients
            charge_after = charge_derivatives(
                previous, current, following, launch, target)
            residual = previous_block@launch + inverse.batched_jvp(
                "following", base, target[:, None])[:, 0]
            current_center = center_derivative(
                previous, current, launch, None, "Psi")
            following_center = center_derivative(
                current, following, None, target, "Psi")
            displacement = following_center-current_center
            rows.append({
                "axis": axis,
                "sign": sign,
                "native_fixed_charge_Psi_phase_coefficient": float(
                    coefficients[0]),
                "native_fixed_charge_chi_contrast_phase_coefficient": float(
                    coefficients[1]),
                "charge_response_matrix": response.tolist(),
                "uncorrected_charge_derivatives": base_charge.tolist(),
                "corrected_charge_derivatives": charge_after.tolist(),
                "DEL_relative_residual": float(
                    np.linalg.norm(residual)
                    / max(np.linalg.norm(shift_image), 1e-30)),
                "launch_support_power_fractions": full.support_fractions(
                    launch),
                "Psi_center_displacement_response": [float(value)
                                                     for value in displacement],
                "signed_requested_axis_response": float(sign*displacement[axis]),
                "moves_requested_way": bool(sign*displacement[axis] > 0),
                "transverse_center_response_norm": float(np.linalg.norm(
                    np.delete(displacement, axis))),
            })
    max_charge = max(max(abs(value) for value in row[
        "corrected_charge_derivatives"]) for row in rows)
    all_move = all(row["moves_requested_way"] for row in rows)
    result = {
        "status": "V4_CARRIER_CONTRAST_FIXED_CHARGE_LAUNCH_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "coefficient_provenance": (
            "the two internal phase coefficients are uniquely solved from "
            "preservation of the native independent Psi and chi Noether "
            "charges; no Dirac velocity or current is fitted"),
        "rows": rows,
        "max_corrected_charge_derivative": max_charge,
        "all_six_Psi_charge_centers_move_requested_way": all_move,
        "verdict": (
            "FIXED_CHARGE_NATIVE_EDGE_TANGENT_PASS" if
            max_charge < 1e-4 and all_move else
            "FIXED_CHARGE_NATIVE_EDGE_TANGENT_OPEN"),
        "claim_boundary": (
            "This is a one-interval linear fixed-charge tangent. It is not a "
            "finite nonlinear or repeated translation and is not yet matched "
            "to the discrete Dirac current."),
        "corrective_action": (
            "On pass, continue this tangent into a nonlinear relative-periodic "
            "branch. On failure, add the independently native relative-copy "
            "phase directions and solve the same fixed-charge constraints."),
    }
    (HERE / "localized_carrier_contrast_fixed_charge_launch.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
