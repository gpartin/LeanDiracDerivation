"""Energy--momentum index of the localized relative equilibrium.

The augmented quadratic Hamiltonian is reduced by every continuous native
field symmetry available to the current research action: simultaneous U(3)
internal rotations and independent global Psi/chi phases.  Both the momentum
constraints and their group-orbit directions are removed before the Morse
index is counted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import null_space

import action_response_rank as rank
import field_berry_probe as field


HERE = Path(__file__).resolve().parent


def hermitian_basis():
    result = []
    for a in range(3):
        value = np.zeros((3, 3), dtype=complex)
        value[a, a] = 1
        result.append((f"u3_diag_{a}", value))
    for a in range(3):
        for b in range(a + 1, 3):
            symmetric = np.zeros((3, 3), dtype=complex)
            symmetric[a, b] = symmetric[b, a] = 1
            result.append((f"u3_real_{a}{b}", symmetric))
            antisymmetric = np.zeros((3, 3), dtype=complex)
            antisymmetric[a, b] = -1j
            antisymmetric[b, a] = 1j
            result.append((f"u3_imag_{a}{b}", antisymmetric))
    return result


def generator_directions(raw, side):
    psi, chi = field.prior.bond.local.common.unpack_raw(raw, side)
    result = []
    result.append(("psi_phase", field.prior.bond.local.common.pack_raw(
        1j * psi, np.zeros_like(chi))))
    result.append(("chi_phase", field.prior.bond.local.common.pack_raw(
        np.zeros_like(psi), 1j * chi)))
    for name, matrix in hermitian_basis():
        dpsi = 1j * np.einsum("ab,...b->...a", matrix, psi)
        dchi = 1j * np.einsum("ab,...b->...a", matrix, chi)
        result.append((name, field.prior.bond.local.common.pack_raw(
            dpsi, dchi)))
    return result


def independent_columns(matrix, tolerance=1e-10):
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    rank_value = int(np.sum(singular > singular[0] * tolerance))
    return u[:, :rank_value], singular, rank_value


def index_counts(matrix, tolerance_scale=1e-8):
    values = np.linalg.eigvalsh((matrix + matrix.T) / 2)
    tolerance = max(np.max(abs(values)) * tolerance_scale, 1e-9)
    return values, {
        "negative": int(np.sum(values < -tolerance)),
        "zero": int(np.sum(abs(values) <= tolerance)),
        "positive": int(np.sum(values > tolerance)),
        "tolerance": float(tolerance),
        "minimum": float(values[0]),
        "maximum": float(values[-1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=int, default=1024)
    parser.add_argument("--output", default="constrained_energy_index.json")
    args = parser.parse_args()
    field.set_chi_inertia(229.0)
    (side, dt, coefficients, solution, layers, residual, rotation, mass,
     gyro, stiffness, generator) = rank.exact_rest(args.rate)
    size = mass.shape[0]
    omega = np.block([
        [gyro, mass],
        [-mass, np.zeros_like(mass)],
    ])
    hessian = np.block([
        [stiffness, np.zeros_like(mass)],
        [np.zeros_like(mass), mass],
    ])
    directions = generator_directions(layers[1], side)
    tangents = np.column_stack([
        np.concatenate((direction, np.zeros(size)))
        for _, direction in directions])
    tangents, tangent_singular, tangent_rank = independent_columns(tangents)
    momentum_gradients = omega @ tangents
    constraints = np.column_stack((tangents, momentum_gradients))
    constraints, constraint_singular, constraint_rank = independent_columns(
        constraints)
    reduced_basis = null_space(constraints.T, rcond=1e-10)
    reduced_hessian = reduced_basis.T @ hessian @ reduced_basis
    full_values, full_counts = index_counts(hessian)
    reduced_values, reduced_counts = index_counts(reduced_hessian)
    result = {
        "status": "V4_CONSTRAINED_ENERGY_INDEX",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rest_del_rms_residual": residual,
        "declared_symmetry_generators": [name for name, _ in directions],
        "independent_group_orbit_rank": tangent_rank,
        "independent_constraint_and_orbit_rank": constraint_rank,
        "full_phase_space_dimension": int(hessian.shape[0]),
        "reduced_dimension": int(reduced_hessian.shape[0]),
        "full_hessian_index": full_counts,
        "constrained_hessian_index": reduced_counts,
        "interpretation": (
            "A constrained energy minimum requires no negative directions "
            "after all native momentum constraints and group-orbit directions "
            "are removed. Remaining negative directions prove that this "
            "relative equilibrium is a saddle for the tested action."),
        "claim_boundary": (
            "This is an energy--momentum index of the current scaffolded "
            "research orbit, not a no-go for every state or action."),
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
