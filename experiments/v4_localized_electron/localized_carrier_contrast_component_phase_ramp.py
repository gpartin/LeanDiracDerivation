"""Resolve all-direction transport from the eight cubical component ramps.

The rest-state copy amplitudes yield strong x/y phase-gradient transport but
no z response.  A boost also changes the internal four-component amplitudes.
This audit therefore builds the neighboring-center odd phase ramp for every
component of both cubical copies, derives its previous layer from the native
DEL equation, and asks whether fixed native charges leave a three-directional
Noether-center response.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve

import localized_carrier_contrast_fixed_charge_launch as fixed
import localized_carrier_contrast_full_inverse_launch as full
import localized_carrier_contrast_inverse_launch as inverse
import localized_carrier_contrast_noether_motion as noether
import localized_carrier_contrast_phase_ramp as ramp


HERE = Path(__file__).resolve().parent


def component_targets(axis: int, psi_phase: complex) -> tuple[list[str], np.ndarray]:
    flavors = inverse.local.flavor_bases()
    offset_plus = tuple(1 if index == axis else 0 for index in range(3))
    offset_minus = tuple(-1 if index == axis else 0 for index in range(3))
    labels = []
    columns = []
    for flavor in ("plus", "minus"):
        for component in range(4):
            amplitudes = flavors[flavor][:, component]
            odd = (ramp.placed_copy(amplitudes, offset_plus, psi_phase)
                   - ramp.placed_copy(amplitudes, offset_minus, psi_phase))/2
            psi, chi = inverse.rank.native.unpack(odd, inverse.SIDE)
            columns.append(inverse.rank.pack(1j*psi, 1j*chi))
            labels.append(f"{flavor}_{component}")
    return labels, np.column_stack(columns)


def response(previous: np.ndarray, current: np.ndarray,
             following: np.ndarray, launch: np.ndarray,
             target: np.ndarray) -> np.ndarray:
    current_density = noether.phase_charge_density(
        previous, current, "Psi")
    following_density = noether.phase_charge_density(
        current, following, "Psi")
    current_psi_derivative = noether.phase_charge_density_derivative(
        previous, current, launch, None, "Psi")
    following_psi_derivative = noether.phase_charge_density_derivative(
        current, following, None, target, "Psi")
    current_chi_derivative = noether.phase_charge_density_derivative(
        previous, current, launch, None, "chi")
    following_chi_derivative = noether.phase_charge_density_derivative(
        current, following, None, target, "chi")
    charges = np.asarray((
        (np.sum(current_psi_derivative)+np.sum(following_psi_derivative))/2,
        (np.sum(current_chi_derivative)+np.sum(following_chi_derivative))/2,
    ))
    displacement = (
        noether.center_derivative_exact(
            following_density, following_psi_derivative)
        - noether.center_derivative_exact(
            current_density, current_psi_derivative))
    return np.concatenate((charges, displacement))


def main() -> None:
    inverse.rank.lock_inertia()
    inverse.rest.SIDE = inverse.SIDE
    inverse.rest.DT = inverse.DT
    previous, current, following, theta = inverse.rest_layers()
    base = tuple(inverse.torch.tensor(value, dtype=inverse.torch.float64)
                 for value in (previous, current, following))
    dimension = previous.size
    previous_block = inverse.batched_jvp(
        "previous", base, np.eye(dimension))
    factor = lu_factor(previous_block)
    axis_rows = []
    for axis in range(3):
        labels, targets = component_targets(axis, np.exp(1j*theta))
        images = inverse.batched_jvp("following", base, targets)
        launches = lu_solve(factor, -images)
        responses = np.column_stack([
            response(previous, current, following,
                     launches[:, column], targets[:, column])
            for column in range(targets.shape[1])
        ])
        center_matrix = responses[2:]
        fixed_charge_null = np.linalg.svd(responses[:2], full_matrices=True)[2][
            np.linalg.matrix_rank(responses[:2], tol=1e-8):].T
        fixed_charge_center = center_matrix@fixed_charge_null
        singular = np.linalg.svd(fixed_charge_center, compute_uv=False)
        response_rank = int(np.sum(singular > 1e-10*max(singular[0], 1.0)))

        center_scale = max(float(np.linalg.norm(center_matrix, axis=0).max()),
                           1e-30)
        row_scales = np.array([
            max(float(np.linalg.norm(responses[index])), 1e-30)
            for index in range(5)])
        normalized = responses/row_scales[:, None]
        target = np.zeros(5)
        target[2+axis] = center_scale
        normalized_target = target/row_scales
        coefficients, _, solve_rank, solve_singular = np.linalg.lstsq(
            normalized, normalized_target, rcond=1e-11)
        achieved = responses@coefficients
        combined_launch = launches@coefficients
        combined_target = targets@coefficients
        del_residual = (previous_block@combined_launch
                        + images@coefficients)
        axis_rows.append({
            "phase_ramp_axis": axis,
            "component_labels": labels,
            "response_matrix_rows": [
                "Psi_charge", "chi_charge", "center_x", "center_y", "center_z"],
            "response_matrix": responses.tolist(),
            "fixed_charge_center_rank": response_rank,
            "fixed_charge_center_singular_values": [float(value)
                                                     for value in singular],
            "normalized_constraint_rank": int(solve_rank),
            "normalized_constraint_singular_values": [
                float(value) for value in solve_singular],
            "native_geometry_target_center_scale": center_scale,
            "solved_component_coefficients": [float(value)
                                              for value in coefficients],
            "achieved_native_constraints": achieved.tolist(),
            "relative_constraint_defect": float(
                np.linalg.norm((achieved-target)/row_scales)
                / max(np.linalg.norm(normalized_target), 1e-30)),
            "DEL_relative_residual": float(
                np.linalg.norm(del_residual)
                / max(np.linalg.norm(images@coefficients), 1e-30)),
            "launch_support_power_fractions": full.support_fractions(
                combined_launch),
        })
    pass_all = all(row["fixed_charge_center_rank"] == 3
                   and row["relative_constraint_defect"] < 1e-6
                   for row in axis_rows)
    result = {
        "status": "V4_COMPONENT_PHASE_RAMP_RESPONSE_RANK_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "construction": (
            "odd neighboring-center phase gradients of all four components "
            "in both cubical copies"),
        "axis_rows": axis_rows,
        "all_axes_have_rank_three_fixed_charge_transport": all(
            row["fixed_charge_center_rank"] == 3 for row in axis_rows),
        "verdict": (
            "NATIVE_COMPONENT_RAMPS_SPAN_ALL_TRANSPORT" if pass_all else
            "NATIVE_COMPONENT_RAMP_TRANSPORT_INCOMPLETE"),
        "claim_boundary": (
            "This is an existence/rank result for linear native momentum "
            "directions. It does not select a nonlinear branch, determine "
            "Dirac velocity, or prove repeated movement."),
        "corrective_action": (
            "On pass, derive the native positive-frequency graph that selects "
            "the component coefficients. On rank failure, identify the missing "
            "axis representation and screen an admissible action response."),
    }
    (HERE / "localized_carrier_contrast_component_phase_ramp.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
