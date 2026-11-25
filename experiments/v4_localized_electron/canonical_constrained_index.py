"""Continuous energy--momentum index of the unscaffolded V40 rest branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig, null_space

import constrained_energy_index as reduced
import localized_stability as canonical


HERE = Path(__file__).resolve().parent


def rotation_generator(side, omega_psi, omega_chi):
    size = 12 * side**3
    result = np.empty((size, size))
    basis = np.zeros(size)
    pf = np.asarray((omega_psi, omega_psi, omega_psi - omega_chi))
    cf = np.asarray((omega_chi, omega_chi, 0.0))
    for column in range(size):
        basis[column] = 1
        psi, chi = canonical.unpack(basis, side)
        result[:, column] = canonical.pack(1j * pf * psi, 1j * cf * chi)
        basis[column] = 0
    return result


def symmetry_directions(psi, chi):
    zero_psi, zero_chi = np.zeros_like(psi), np.zeros_like(chi)
    columns = [canonical.pack(1j * psi, zero_chi),
               canonical.pack(zero_psi, 1j * chi)]
    names = ["psi_phase", "chi_phase"]
    for name, matrix in reduced.hermitian_basis():
        columns.append(canonical.pack(
            1j * np.einsum("ab,...b->...a", matrix, psi),
            1j * np.einsum("ab,...b->...a", matrix, chi)))
        names.append(name)
    return names, np.column_stack(columns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=3)
    parser.add_argument("--output", default="canonical_constrained_index.json")
    args = parser.parse_args()
    side = args.side
    canonical.B = 1197.0
    gap = canonical.R**2
    psi_frequency_sq = canonical.B * gap / (canonical.B - 1)
    chi_frequency_sq = (psi_frequency_sq - gap) / canonical.B
    omega_psi = np.sqrt(psi_frequency_sq)
    omega_chi = np.sqrt(chi_frequency_sq)
    amplitude_sq = psi_frequency_sq / (2 + 54 / canonical.R**2)
    psi, chi = canonical.carrier_fields(side, amplitude_sq)
    matrices = canonical.tangent_matrices(
        psi, chi, 0.0, 0.0, 1.0, 1.0)
    force_jacobian = matrices[1] + matrices[2]
    gamma = rotation_generator(side, omega_psi, omega_chi)
    size = force_jacobian.shape[0]
    mass = np.eye(size)
    gyro = 2 * gamma
    stiffness = gamma @ gamma - force_jacobian
    generator = np.block([
        [np.zeros_like(mass), np.eye(size)],
        [-stiffness, -gyro],
    ])
    omega_form = np.block([
        [gyro, mass],
        [-mass, np.zeros_like(mass)],
    ])
    hessian = np.block([
        [stiffness, np.zeros_like(mass)],
        [np.zeros_like(mass), mass],
    ])
    names, q_tangents = symmetry_directions(psi, chi)
    tangents = np.vstack((q_tangents, np.zeros_like(q_tangents)))
    tangents, tangent_singular, tangent_rank = reduced.independent_columns(
        tangents)
    momentum_gradients = omega_form @ tangents
    constraints, constraint_singular, constraint_rank = (
        reduced.independent_columns(np.column_stack((
            tangents, momentum_gradients))))
    basis = null_space(constraints.T, rcond=1e-10)
    reduced_hessian = basis.T @ hessian @ basis
    _, full_counts = reduced.index_counts(hessian)
    _, reduced_counts = reduced.index_counts(reduced_hessian)
    values = eig(generator, right=False, check_finite=False)
    result = {
        "status": "V40_CANONICAL_CONSTRAINED_ENERGY_INDEX",
        "canonical_action_changed": False,
        "side": side,
        "B": canonical.B,
        "amplitude_sq": float(amplitude_sq),
        "omega_psi": float(omega_psi),
        "omega_chi": float(omega_chi),
        "hamiltonian_structure_relative_residual": float(np.linalg.norm(
            generator.T @ omega_form + omega_form @ generator)
            / (np.linalg.norm(generator) * np.linalg.norm(omega_form))),
        "maximum_growth": float(np.max(values.real)),
        "growth_count_above_1e_2": int(np.sum(values.real > 1e-2)),
        "growth_count_above_1e_4": int(np.sum(values.real > 1e-4)),
        "independent_group_orbit_rank": tangent_rank,
        "independent_constraint_and_orbit_rank": constraint_rank,
        "full_hessian_index": full_counts,
        "constrained_hessian_index": reduced_counts,
        "claim_boundary": (
            "Index of the prepared canonical eight-site relative equilibrium; "
            "not a no-go for other charge/frequency branches."),
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
