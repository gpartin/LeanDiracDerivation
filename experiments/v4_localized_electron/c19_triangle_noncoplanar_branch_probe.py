"""Lyapunov--Schmidt probe of the strongest noncoplanar null direction.

The full Hessian audit finds physical carrier-W directions in the true orbit
nullspace.  A linear zero is not yet a branch.  This script selects the
strongest carrier-W quotient direction, solves the complete Hessian-range
equations at several amplitudes, and measures the remaining nullspace
solvability residual.  No action coefficient or Dirac target is introduced.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import c19_triangle_gyro_candidate_equivalence as triangle
import c19_triangle_noncoplanar_full_audit as audit
import c27_body_gyro_basis_probe as direct
import c27_body_gyro_rest_spectrum as spectrum
import relative_kinetic_probe as relative


HERE = Path(__file__).resolve().parent
FRACTIONS = (
    1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4,
    1e-3, 3e-3, 1e-2, 3e-2,
)


def main() -> None:
    spectrum.configure()
    direct.base.inverse.rank.lock_inertia()
    _previous, current, _following, theta = direct.base.inverse.rest_layers()
    values = triangle.coefficients(theta)
    rotation = torch.tensor(relative.raw_rotation_matrix(
        spectrum.SIDE, theta, 0.0, 1), dtype=torch.float64)
    inverse_rotation = rotation.T

    def orbit_action(raw: torch.Tensor) -> torch.Tensor:
        return (triangle.triangle_interval(inverse_rotation @ raw, raw, values)
                + triangle.triangle_interval(raw, rotation @ raw, values))

    def residual(raw: np.ndarray) -> np.ndarray:
        variable = torch.tensor(raw, dtype=torch.float64, requires_grad=True)
        return torch.autograd.grad(
            orbit_action(variable), variable)[0].detach().numpy()

    base_tensor = torch.tensor(current, dtype=torch.float64,
                               requires_grad=True)
    hessian = torch.autograd.functional.hessian(
        orbit_action, base_tensor, vectorize=True).detach().numpy()
    hessian = (hessian + hessian.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    scale = max(float(np.max(abs(eigenvalues))), 1e-300)
    threshold = 1e-8 * scale
    null_mask = abs(eigenvalues) < threshold
    null_basis = eigenvectors[:, null_mask]
    range_basis = eigenvectors[:, ~null_mask]
    range_values = eigenvalues[~null_mask]

    psi, _chi = direct.base.inverse.rank.native.unpack(
        current, spectrum.SIDE)
    occupied = np.sum(abs(psi)**2, axis=-1) > 1e-12
    w_from_null = np.column_stack([
        audit.carrier_w_tangent(current, null_basis[:, index], occupied)
        for index in range(null_basis.shape[1])
    ])
    symmetry = audit.global_tangents(current)
    symmetry_w = np.column_stack([
        audit.carrier_w_tangent(current, symmetry[:, index], occupied)
        for index in range(symmetry.shape[1])
    ])
    symmetry_w_basis = audit.orth(symmetry_w)
    quotient = w_from_null - symmetry_w_basis @ (
        symmetry_w_basis.T @ w_from_null)
    _left, singular, vh = np.linalg.svd(quotient, full_matrices=False)
    direction = null_basis @ vh[0]
    direction /= np.linalg.norm(direction)
    base_norm = np.linalg.norm(current)
    base_residual = residual(current)
    rows = []
    solved_states = []

    for fraction in FRACTIONS:
        epsilon = fraction * base_norm
        state = current + epsilon * direction
        traces = []
        for _iteration in range(8):
            value = residual(state)
            range_coordinates = range_basis.T @ value
            null_coordinates = null_basis.T @ value
            traces.append({
                "total_rms": float(np.linalg.norm(value)
                                   / np.sqrt(value.size)),
                "range_norm": float(np.linalg.norm(range_coordinates)),
                "null_norm": float(np.linalg.norm(null_coordinates)),
            })
            correction = -range_basis @ (range_coordinates / range_values)
            accepted = False
            old_range = np.linalg.norm(range_coordinates)
            for power in range(10):
                trial = state + 0.5**power * correction
                trial_value = residual(trial)
                if np.linalg.norm(range_basis.T @ trial_value) < old_range:
                    state = trial
                    accepted = True
                    break
            if not accepted or old_range < 1e-11:
                break
        final = residual(state)
        range_final = range_basis.T @ final
        null_final = null_basis.T @ final
        displacement = state - current
        amplitude = float(direction @ displacement)
        transverse = displacement - amplitude * direction
        rows.append({
            "fraction_of_rest_norm": fraction,
            "requested_amplitude": epsilon,
            "retained_direction_amplitude": amplitude,
            "range_correction_norm": float(np.linalg.norm(transverse)),
            "final_total_rms": float(np.linalg.norm(final)
                                     / np.sqrt(final.size)),
            "final_range_norm": float(np.linalg.norm(range_final)),
            "final_null_solvability_norm": float(np.linalg.norm(null_final)),
            "null_norm_over_amplitude_squared": float(
                np.linalg.norm(null_final) / max(amplitude**2, 1e-300)),
            "trace": traces,
        })
        solved_states.append(state)

    ratios = [row["final_null_solvability_norm"]
              / max(row["requested_amplitude"]**2, 1e-300)
              for row in rows]
    nonlinear_obstruction = (
        np.median(ratios[-3:]) > 1e-6
        and all(row["final_null_solvability_norm"] > 1e-10
                for row in rows[-3:]))
    solved = any(row["final_total_rms"] < 1e-9
                 and row["fraction_of_rest_norm"] >= 1e-4
                 for row in rows)
    result = {
        "status": "V4_C19_TRIANGLE_NONCOPLANAR_BRANCH_PROBE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "full_nullity": int(null_basis.shape[1]),
        "strongest_carrier_W_quotient_singular_value": float(singular[0]),
        "base_rest_residual_rms": float(
            np.linalg.norm(base_residual) / np.sqrt(base_residual.size)),
        "rows": rows,
        "verdict": (
            "NONCOPLANAR_NEARBY_REST_BRANCH_NUMERIC_CANDIDATE" if solved else
            "STRONGEST_NONCOPLANAR_DIRECTION_QUADRATICALLY_OBSTRUCTED"
            if nonlinear_obstruction else
            "NONCOPLANAR_BRANCH_PROBE_INCONCLUSIVE"
        ),
        "claim_boundary": (
            "Range-corrected Lyapunov--Schmidt probe of one strongest null "
            "direction. A pass still requires a full augmented nonlinear "
            "solve and stability/transport tests; an obstruction here does "
            "not exclude combinations of the 17 physical null directions."
        ),
    }
    output = HERE / "c19_triangle_noncoplanar_branch_probe.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        HERE / "c19_triangle_noncoplanar_branch_probe.npz",
        rest=current,
        direction=direction,
        fractions=np.asarray(FRACTIONS),
        states=np.asarray(solved_states),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
