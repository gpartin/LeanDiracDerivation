"""Test the native chi-normal channel omitted by the phase-ramp audit.

The full configuration tangent audit already showed that the same carrier's
three translation tangents lie in the source-plus-faces native span only when
the independent C3-normal chi response is retained.  The later fixed-charge
phase-ramp test inadvertently returned to the planar joint Psi/chi graph and
found rank two.  This audit adds both real and imaginary odd neighboring-cube
ramps of that *native chi component*, derives the previous layer from the
unchanged discrete action, and repeats the exact Noether-charge screen.

This is a preparation/readout audit.  It adds no action term, background,
Dirac matrix, copy selector, or runtime support.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve

import localized_carrier_contrast_component_phase_ramp as component
import localized_carrier_contrast_full_inverse_launch as full
import localized_carrier_contrast_inverse_launch as inverse
import localized_carrier_contrast_noether_motion as noether
import localized_carrier_contrast_phase_ramp as ramp


HERE = Path(__file__).resolve().parent


def placed_chi_normal(amplitudes: np.ndarray,
                      offset: tuple[int, int, int]) -> np.ndarray:
    dpsi = np.zeros((inverse.SIDE,)*3+(3,), dtype=complex)
    dchi = np.zeros_like(dpsi)
    for corner, bits in enumerate(itertools.product((0, 1), repeat=3)):
        site = tuple(inverse.ORIGIN+offset[axis]+bits[axis]
                     for axis in range(3))
        dchi[site][2] += amplitudes[corner]
    return inverse.rank.pack(dpsi, dchi)


def targets(axis: int, psi_phase: complex) -> tuple[list[str], np.ndarray]:
    labels, planar = component.component_targets(axis, psi_phase)
    flavors = inverse.local.flavor_bases()
    offset_plus = tuple(1 if index == axis else 0 for index in range(3))
    offset_minus = tuple(-1 if index == axis else 0 for index in range(3))
    columns = [planar[:, index] for index in range(planar.shape[1])]
    for flavor in ("plus", "minus"):
        for index in range(4):
            amplitudes = flavors[flavor][:, index]
            odd = (placed_chi_normal(amplitudes, offset_plus)
                   - placed_chi_normal(amplitudes, offset_minus))/2
            columns.extend((odd, inverse.rank.pack(*(
                lambda values: (np.zeros_like(values[1]), 1j*values[1]))(
                    inverse.rank.native.unpack(odd, inverse.SIDE)))))
            labels.extend((f"{flavor}_{index}_chi_normal_real",
                           f"{flavor}_{index}_chi_normal_imag"))
    return labels, np.column_stack(columns)


def response(previous: np.ndarray, current: np.ndarray,
             following: np.ndarray, launch: np.ndarray,
             target: np.ndarray) -> np.ndarray:
    current_density = noether.phase_charge_density(previous, current, "Psi")
    following_density = noether.phase_charge_density(current, following, "Psi")
    current_psi = noether.phase_charge_density_derivative(
        previous, current, launch, None, "Psi")
    following_psi = noether.phase_charge_density_derivative(
        current, following, None, target, "Psi")
    current_chi = noether.phase_charge_density_derivative(
        previous, current, launch, None, "chi")
    following_chi = noether.phase_charge_density_derivative(
        current, following, None, target, "chi")
    charges = np.asarray((
        (np.sum(current_psi)+np.sum(following_psi))/2,
        (np.sum(current_chi)+np.sum(following_chi))/2,
    ))
    displacement = (
        noether.center_derivative_exact(following_density, following_psi)
        - noether.center_derivative_exact(current_density, current_psi))
    return np.concatenate((charges, displacement))


def main() -> None:
    inverse.rank.lock_inertia()
    inverse.rest.SIDE = inverse.SIDE
    inverse.rest.DT = inverse.DT
    previous, current, following, theta = inverse.rest_layers()
    base = tuple(inverse.torch.tensor(value, dtype=inverse.torch.float64)
                 for value in (previous, current, following))
    dimension = previous.size
    previous_block = inverse.batched_jvp("previous", base, np.eye(dimension))
    factor = lu_factor(previous_block)
    rows = []
    for axis in range(3):
        labels, target_matrix = targets(axis, np.exp(1j*theta))
        images = inverse.batched_jvp("following", base, target_matrix)
        launches = lu_solve(factor, -images)
        responses = np.column_stack([
            response(previous, current, following,
                     launches[:, column], target_matrix[:, column])
            for column in range(target_matrix.shape[1])
        ])
        charge_rank = np.linalg.matrix_rank(responses[:2], tol=1e-8)
        null = np.linalg.svd(
            responses[:2], full_matrices=True)[2][charge_rank:].T
        fixed_center = responses[2:]@null
        left, singular, _ = np.linalg.svd(fixed_center, full_matrices=True)
        threshold = 1e-9*max(singular[0], 1e-30)
        response_rank = int(np.sum(singular > threshold))

        # Solve for a unit response along the requested physical axis while
        # exactly preserving both native phase charges.
        row_scales = np.asarray([
            max(float(np.linalg.norm(responses[index])), 1e-30)
            for index in range(5)])
        normalized = responses/row_scales[:, None]
        target = np.zeros(5)
        target[2+axis] = 1.0
        normalized_target = target/row_scales
        coefficients, _, solve_rank, solve_singular = np.linalg.lstsq(
            normalized, normalized_target, rcond=1e-11)
        achieved = responses@coefficients
        combined_launch = launches@coefficients
        combined_target = target_matrix@coefficients
        del_residual = previous_block@combined_launch+images@coefficients
        rows.append({
            "phase_ramp_axis": axis,
            "target_count": int(target_matrix.shape[1]),
            "fixed_charge_center_rank": response_rank,
            "fixed_charge_center_singular_values": [float(value)
                                                     for value in singular],
            "missing_physical_direction": [float(value)
                                            for value in left[:, -1]],
            "normalized_constraint_rank": int(solve_rank),
            "normalized_constraint_singular_values": [float(value)
                                                        for value in solve_singular],
            "achieved_constraints": achieved.tolist(),
            "relative_constraint_defect": float(np.linalg.norm(
                (achieved-target)/row_scales)
                / max(np.linalg.norm(normalized_target), 1e-30)),
            "DEL_relative_residual": float(np.linalg.norm(del_residual)
                / max(np.linalg.norm(images@coefficients), 1e-30)),
            "coefficient_norm": float(np.linalg.norm(coefficients)),
            "chi_normal_coefficient_fraction": float(np.linalg.norm(
                coefficients[8:])/max(np.linalg.norm(coefficients), 1e-30)),
            "launch_support_power_fractions": full.support_fractions(
                combined_launch),
            "target_support_power_fractions": full.support_fractions(
                combined_target),
            "component_labels": labels,
            "response_matrix": responses.tolist(),
        })

    rank_three = all(row["fixed_charge_center_rank"] == 3 for row in rows)
    solved = all(row["relative_constraint_defect"] < 1e-6
                 and row["DEL_relative_residual"] < 1e-9 for row in rows)
    result = {
        "status": "V4_CHI_NORMAL_FIXED_CHARGE_TRANSPORT_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action": "unchanged incumbent discrete two-field interval action",
        "construction": (
            "all planar four-component ramps in both copies plus independent "
            "real/imaginary C3-normal chi ramps on neighboring carrier centers"),
        "prior_evidence": (
            "localized_embedding_independent_fields_tangent.json proved that "
            "source-plus-face chi-normal columns span all three configuration "
            "translation tangents to about 1e-15"),
        "axis_rows": rows,
        "all_axes_rank_three": rank_three,
        "all_axes_constraints_solved": solved,
        "verdict": (
            "UNCHANGED_ACTION_HAS_THREE_AXIS_CHARGE_TRANSPORT_TANGENT"
            if rank_three and solved else
            "CHI_NORMAL_CHANNEL_DOES_NOT_CLOSE_CHARGE_TRANSPORT"),
        "claim_boundary": (
            "A pass proves existence of an exact linearized fixed-charge "
            "native tangent, not that unrestricted evolution selects it, not "
            "a nonlinear repeated edge hop, and not a Dirac-current match."),
        "corrective_action": (
            "On pass, derive the native positive-frequency graph selecting "
            "the chi-normal coefficient and run a nonlinear two-handoff test. "
            "On failure, decompose the remaining missing representation before "
            "testing a multi-site action."),
    }
    (HERE / "localized_carrier_contrast_chi_normal_ramp.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
