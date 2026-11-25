"""Exhaust the full native C3 launch space before changing the action.

For every cubical component in both flavor copies, this audit independently
ramps the real and imaginary Psi-ray, Psi-companion, Psi-normal, chi-ray,
chi-companion, and chi-normal channels across neighboring carrier centers.
The unchanged discrete Euler--Lagrange equation selects the previous layer.
The exact discrete Noether densities then determine whether fixed native
charges leave three independent carrier-center responses.

If this complete native preparation space remains rank two, neither a tied
Psi--chi launch nor omission of the C3-normal direction explains the missing
transport generator.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve
from torch.func import jvp, vmap

import localized_carrier_contrast_full_inverse_launch as full
import localized_carrier_contrast_inverse_launch as inverse
import localized_carrier_contrast_noether_motion as noether
import localized_embedding_independent_fields_tangent as independent


HERE = Path(__file__).resolve().parent


def targets(axis: int) -> tuple[list[str], np.ndarray]:
    _, _, psi_frame, chi_frame = inverse.local.carrier_frames()
    flavors = inverse.local.flavor_bases()
    plus = tuple(1 if index == axis else 0 for index in range(3))
    minus = tuple(-1 if index == axis else 0 for index in range(3))
    labels: list[str] = []
    columns: list[np.ndarray] = []
    channel_names = (
        "psi_ray_real", "psi_ray_imag",
        "chi_companion_real", "chi_companion_imag",
        "chi_normal_real", "chi_normal_imag",
        "psi_companion_real", "psi_companion_imag",
        "psi_normal_real", "psi_normal_imag",
        "chi_ray_real", "chi_ray_imag",
    )
    for flavor in ("plus", "minus"):
        basis = flavors[flavor]
        plus_columns = independent.independent_local_columns(
            basis, psi_frame, chi_frame, plus, full_native_frame=True)
        minus_columns = independent.independent_local_columns(
            basis, psi_frame, chi_frame, minus, full_native_frame=True)
        for index, (positive, negative) in enumerate(
                zip(plus_columns, minus_columns, strict=True)):
            component = index//len(channel_names)
            channel = channel_names[index % len(channel_names)]
            labels.append(f"{flavor}_{component}_{channel}")
            columns.append((positive-negative)/2)
    return labels, np.column_stack(columns)


def batched_density_derivative(
        left: np.ndarray, right: np.ndarray,
        left_directions: np.ndarray, right_directions: np.ndarray,
        mode: str) -> tuple[np.ndarray, np.ndarray]:
    left_tensor = torch.tensor(left, dtype=torch.float64)
    right_tensor = torch.tensor(right, dtype=torch.float64)
    dl = torch.tensor(left_directions.T, dtype=torch.float64)
    dr = torch.tensor(right_directions.T, dtype=torch.float64)
    function = lambda a, b: noether.phase_charge_density_tensor(a, b, mode)
    baseline = function(left_tensor, right_tensor).detach().numpy()
    derivatives = vmap(
        lambda va, vb: jvp(
            function, (left_tensor, right_tensor), (va, vb))[1],
        chunk_size=8)(dl, dr).detach().numpy()
    return baseline, derivatives


def center_derivatives(density: np.ndarray,
                       derivatives: np.ndarray) -> np.ndarray:
    total = float(np.sum(density))
    moments = np.asarray([
        np.sum(np.indices(density.shape)[axis]*density)
        for axis in range(3)])
    dtotal = derivatives.reshape(derivatives.shape[0], -1).sum(axis=1)
    result = np.zeros((derivatives.shape[0], 3))
    coordinates = np.indices(density.shape)
    for axis in range(3):
        dmoment = np.sum(
            derivatives*coordinates[axis][None, ...], axis=(1, 2, 3))
        result[:, axis] = (dmoment*total-moments[axis]*dtotal)/(total**2)
    return result


def main() -> None:
    inverse.rank.lock_inertia()
    inverse.rest.SIDE = inverse.SIDE
    inverse.rest.DT = inverse.DT
    previous, current, following, _ = inverse.rest_layers()
    base = tuple(torch.tensor(value, dtype=torch.float64)
                 for value in (previous, current, following))
    dimension = previous.size
    previous_block = inverse.batched_jvp("previous", base, np.eye(dimension))
    factor = lu_factor(previous_block)
    rows = []
    for axis in range(3):
        labels, target_matrix = targets(axis)
        images = inverse.batched_jvp("following", base, target_matrix)
        launches = lu_solve(factor, -images)
        zero = np.zeros_like(target_matrix)

        current_psi_density, current_psi_derivative = batched_density_derivative(
            previous, current, launches, zero, "Psi")
        following_psi_density, following_psi_derivative = batched_density_derivative(
            current, following, zero, target_matrix, "Psi")
        current_chi_density, current_chi_derivative = batched_density_derivative(
            previous, current, launches, zero, "chi")
        following_chi_density, following_chi_derivative = batched_density_derivative(
            current, following, zero, target_matrix, "chi")

        psi_charge = (current_psi_derivative.reshape(
            target_matrix.shape[1], -1).sum(axis=1)
            + following_psi_derivative.reshape(
                target_matrix.shape[1], -1).sum(axis=1))/2
        chi_charge = (current_chi_derivative.reshape(
            target_matrix.shape[1], -1).sum(axis=1)
            + following_chi_derivative.reshape(
                target_matrix.shape[1], -1).sum(axis=1))/2
        density_modes = {
            "Psi": (current_psi_density, following_psi_density,
                    current_psi_derivative, following_psi_derivative),
            "chi": (current_chi_density, following_chi_density,
                    current_chi_derivative, following_chi_derivative),
            "common": (current_psi_density+current_chi_density,
                       following_psi_density+following_chi_density,
                       current_psi_derivative+current_chi_derivative,
                       following_psi_derivative+following_chi_derivative),
        }
        charge_rows = np.vstack((psi_charge, chi_charge))
        charge_rank = np.linalg.matrix_rank(charge_rows, tol=1e-8)
        null = np.linalg.svd(
            charge_rows, full_matrices=True)[2][charge_rank:].T
        center_readouts = {}
        response_by_mode = {}
        for mode, (current_density, following_density,
                   current_derivative, following_derivative) in density_modes.items():
            displacement = (
                center_derivatives(following_density, following_derivative)
                - center_derivatives(current_density, current_derivative))
            responses_mode = np.vstack((charge_rows, displacement.T))
            fixed_center = displacement.T@null
            left_mode, singular_mode, _ = np.linalg.svd(
                fixed_center, full_matrices=True)
            threshold_mode = 1e-8*max(singular_mode[0], 1e-30)
            center_readouts[mode] = {
                "fixed_charge_center_rank": int(np.sum(
                    singular_mode > threshold_mode)),
                "fixed_charge_center_singular_values": [float(value)
                                                         for value in singular_mode],
                "relative_third_singular_value": float(
                    singular_mode[2]/max(singular_mode[0], 1e-30)),
                "missing_physical_direction": [float(value)
                                                for value in left_mode[:, -1]],
            }
            response_by_mode[mode] = responses_mode

        # Keep the historical Psi-center solve as a control.  A common-center
        # pass is reported separately and is not promoted without deriving
        # the physical Dirac density combination from the embedding.
        responses = response_by_mode["Psi"]
        singular = np.asarray(center_readouts["Psi"][
            "fixed_charge_center_singular_values"])
        response_rank = center_readouts["Psi"]["fixed_charge_center_rank"]

        row_scales = np.asarray([
            max(float(np.linalg.norm(responses[index])), 1e-30)
            for index in range(5)])
        normalized = responses/row_scales[:, None]
        requested = np.zeros(5)
        requested[2+axis] = 1.0
        normalized_target = requested/row_scales
        coefficients, _, solve_rank, solve_singular = np.linalg.lstsq(
            normalized, normalized_target, rcond=1e-10)
        achieved = responses@coefficients
        combined_launch = launches@coefficients
        combined_target = target_matrix@coefficients
        del_residual = previous_block@combined_launch+images@coefficients
        rows.append({
            "phase_ramp_axis": axis,
            "target_count": int(target_matrix.shape[1]),
            "center_readouts": center_readouts,
            "fixed_charge_center_rank": response_rank,
            "fixed_charge_center_singular_values": [float(value)
                                                     for value in singular],
            "relative_third_singular_value": float(
                singular[2]/max(singular[0], 1e-30)),
            "missing_physical_direction": center_readouts["Psi"][
                "missing_physical_direction"],
            "normalized_constraint_rank": int(solve_rank),
            "normalized_constraint_singular_values": [float(value)
                                                        for value in solve_singular],
            "coefficient_norm": float(np.linalg.norm(coefficients)),
            "achieved_constraints": achieved.tolist(),
            "relative_constraint_defect": float(np.linalg.norm(
                (achieved-requested)/row_scales)
                / max(np.linalg.norm(normalized_target), 1e-30)),
            "DEL_relative_residual": float(np.linalg.norm(del_residual)
                / max(np.linalg.norm(images@coefficients), 1e-30)),
            "launch_support_power_fractions": full.support_fractions(
                combined_launch),
            "target_support_power_fractions": full.support_fractions(
                combined_target),
            "largest_coefficient_channels": sorted((
                {"label": labels[index], "coefficient": float(value)}
                for index, value in enumerate(coefficients)),
                key=lambda row: abs(row["coefficient"]), reverse=True)[:12],
        })

    robust = all(row["fixed_charge_center_rank"] == 3
                 and row["relative_third_singular_value"] > 1e-6
                 and row["relative_constraint_defect"] < 1e-6
                 and row["DEL_relative_residual"] < 1e-9
                 for row in rows)
    common_robust = all(
        row["center_readouts"]["common"]["fixed_charge_center_rank"] == 3
        and row["center_readouts"]["common"][
            "relative_third_singular_value"] > 1e-6
        for row in rows)
    result = {
        "status": "V4_FULL_NATIVE_FIXED_CHARGE_TRANSPORT_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action": "unchanged incumbent discrete two-field interval action",
        "construction": (
            "odd neighboring-center ramps of every independent real/imaginary "
            "Psi and chi C3 frame channel in both cubical copies"),
        "axis_rows": rows,
        "all_axes_robust_rank_three": robust,
        "all_axes_common_charge_rank_three": common_robust,
        "verdict": (
            "UNCHANGED_ACTION_FULL_NATIVE_SPACE_HAS_THREE_AXIS_TRANSPORT"
            if robust else
            "UNCHANGED_ACTION_FULL_NATIVE_SPACE_REMAINS_TRANSPORT_DEFICIENT"),
        "claim_boundary": (
            "This is an exhaustive linearized preparation and readout screen "
            "within the declared local carrier-frame family. A pass is not "
            "native selection, nonlinear hopping, stability, or Dirac closure. "
            "A common-charge result is not a physical-current mapping until "
            "the same four-component embedding derives that density."),
    }
    (HERE / "localized_carrier_contrast_full_native_ramp.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
