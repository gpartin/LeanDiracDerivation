"""Screen cross-bond chi^dagger Psi gyros against the nonlinear obstruction.

The scalar s_ij = chi_i^dagger Psi_j carries the relative native phase, while
Im(conj(s_ij) sdot_ij) is invariant under independent global Psi and chi
phase rotations.  Unlike a W_i-only term, it can orient a receiving chi site
from neighboring Psi before Psi occupies that site.  We test local, C19 and
body-diagonal common/relative derivatives, restrict to the exact-rest
nullspace, and measure response to both signed finite-clock residuals.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.func import grad

import c27_body_gyro_basis_probe as candidate
import c27_body_gyro_rest_spectrum as spectrum


HERE = Path(__file__).resolve().parent
BODY_OFFSETS = tuple((offset, 1.0) for offset in
                     itertools.product((-1, 1), repeat=3))
C19_OFFSETS = tuple(candidate.base.inverse.rank.native.DIRECTIONS)
BASIS_NAMES = (
    "local_cross_common",
    "local_cross_relative",
    "c19_cross_common",
    "c19_cross_relative",
    "body_cross_common",
    "body_cross_relative",
)


def midpoint_fields(left: torch.Tensor, right: torch.Tensor):
    left_psi, left_chi = candidate.fields(left)
    right_psi, right_chi = candidate.fields(right)
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    psi_velocity = (right_psi - left_psi) / candidate.base.inverse.DT
    chi_velocity = (right_chi - left_chi) / candidate.base.inverse.DT
    return psi, chi, psi_velocity, chi_velocity


def cross_pair(psi, chi, psi_velocity, chi_velocity,
               offsets, relative: bool) -> torch.Tensor:
    result = torch.zeros((), dtype=torch.float64)
    for offset, weight in offsets:
        neighbor_psi = torch.roll(psi, shifts=offset, dims=(0, 1, 2))
        neighbor_velocity = torch.roll(
            psi_velocity, shifts=offset, dims=(0, 1, 2))
        scalar = torch.sum(torch.conj(chi) * neighbor_psi, dim=-1)
        chi_part = torch.sum(
            torch.conj(chi_velocity) * neighbor_psi, dim=-1)
        psi_part = torch.sum(
            torch.conj(chi) * neighbor_velocity, dim=-1)
        scalar_velocity = (psi_part - chi_part if relative
                           else psi_part + chi_part)
        result = result + weight * torch.sum(torch.imag(
            torch.conj(scalar) * scalar_velocity))
    return candidate.base.inverse.DT * result / (
        2 * candidate.base.inverse.rank.R**2)


def gyro_basis(left: torch.Tensor, right: torch.Tensor):
    psi, chi, psi_velocity, chi_velocity = midpoint_fields(left, right)
    local = (((0, 0, 0), 1.0),)
    return (
        cross_pair(psi, chi, psi_velocity, chi_velocity, local, False),
        cross_pair(psi, chi, psi_velocity, chi_velocity, local, True),
        cross_pair(psi, chi, psi_velocity, chi_velocity,
                   C19_OFFSETS, False),
        cross_pair(psi, chi, psi_velocity, chi_velocity,
                   C19_OFFSETS, True),
        cross_pair(psi, chi, psi_velocity, chi_velocity,
                   BODY_OFFSETS, False),
        cross_pair(psi, chi, psi_velocity, chi_velocity,
                   BODY_OFFSETS, True),
    )


def native_coefficients() -> torch.Tensor:
    theta0 = candidate.base.inverse.rest_layers()[3]
    a = 2 * spectrum.DT / np.sin(theta0)
    return torch.tensor((
        a + candidate.base.inverse.rank.B / 4,
        a - candidate.base.inverse.rank.B / 4,
        -a / 2,
        0.0,
    ), dtype=torch.float64)


def interval_action(left: torch.Tensor, right: torch.Tensor,
                    extra: torch.Tensor) -> torch.Tensor:
    value = candidate.interval_action(
        left, right, 1.0, native_coefficients())
    return value + sum(coefficient * term for coefficient, term in zip(
        extra, gyro_basis(left, right)))


def make_del(extra: np.ndarray):
    coefficients = torch.tensor(extra, dtype=torch.float64)

    def action(previous, current, following):
        return (interval_action(previous, current, coefficients)
                + interval_action(current, following, coefficients))

    return grad(action, argnums=1)


def nullspace(matrix: np.ndarray, relative_tolerance: float = 1e-10):
    _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    threshold = relative_tolerance * max(singular[0], 1e-300)
    rank = int(np.sum(singular > threshold))
    return vh[rank:].T, singular, rank


def main() -> None:
    spectrum.configure()
    candidate.base.inverse.rank.lock_inertia()
    previous, current, following, _ = candidate.base.inverse.rest_layers()
    rest_layers = tuple(torch.tensor(value, dtype=torch.float64)
                        for value in (previous, current, following))
    zero = np.zeros(len(BASIS_NAMES))
    baseline_del = make_del(zero)
    rest_baseline = baseline_del(*rest_layers).detach().numpy()
    rest_columns = []
    unit_dels = []
    for index in range(len(BASIS_NAMES)):
        unit = zero.copy()
        unit[index] = 1.0
        unit_del = make_del(unit)
        unit_dels.append(unit_del)
        rest_columns.append(
            unit_del(*rest_layers).detach().numpy() - rest_baseline)
    rest_matrix = np.column_stack(rest_columns)
    rest_null, rest_singular, rest_rank = nullspace(rest_matrix)

    data = np.load(HERE / "c27_body_gyro_linked_clock_continuation.npz")
    branch_matrices = []
    targets = []
    for prefix in ("minus", "plus"):
        layers = tuple(torch.tensor(data[f"{prefix}_{name}"],
                                    dtype=torch.float64)
                       for name in ("previous", "state", "following"))
        baseline = baseline_del(*layers).detach().numpy()
        targets.append(-baseline)
        branch_matrices.append(np.column_stack([
            unit_del(*layers).detach().numpy() - baseline
            for unit_del in unit_dels]))
    response = np.vstack([matrix @ rest_null for matrix in branch_matrices])
    target = np.concatenate(targets)
    reduced, _, response_rank, response_singular = np.linalg.lstsq(
        response, target, rcond=1e-10)
    coefficients = rest_null @ reduced
    predicted = response @ reduced
    combined = float(np.linalg.norm(predicted - target) / max(
        np.linalg.norm(target), 1e-300))
    per_branch = []
    cursor = 0
    for prefix, branch_target in zip(("minus", "plus"), targets):
        size = branch_target.size
        branch_prediction = predicted[cursor:cursor + size]
        per_branch.append({
            "branch": prefix,
            "relative_remainder": float(np.linalg.norm(
                branch_prediction - branch_target) / max(
                    np.linalg.norm(branch_target), 1e-300)),
        })
        cursor += size
    result = {
        "status": "V4_C27_CROSS_BOND_GYRO_OBSTRUCTION_RANK",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "basis_names": list(BASIS_NAMES),
        "rest_force_rank": rest_rank,
        "rest_nullity": int(rest_null.shape[1]),
        "rest_singular_values": [float(value) for value in rest_singular],
        "admissible_response_rank": int(response_rank),
        "admissible_response_singular_values": [
            float(value) for value in response_singular],
        "diagnostic_coefficients": {
            name: float(value) for name, value in zip(BASIS_NAMES, coefficients)},
        "combined_relative_remainder": combined,
        "branches": per_branch,
        "verdict": (
            "CROSS_BOND_GYRO_BASIS_SPANS_NONLINEAR_OBSTRUCTION"
            if combined < 1e-6 else
            "CROSS_BOND_GYRO_BASIS_PARTIAL_NONLINEAR_RESPONSE"
            if combined < 0.99 else
            "CROSS_BOND_GYRO_BASIS_MISSES_NONLINEAR_OBSTRUCTION"),
        "claim_boundary": (
            "Exact-rest-nullspace response-rank screen only. Diagnostic "
            "coefficients are not admitted or fitted into the action."),
    }
    output = HERE / "c27_cross_bond_gyro_obstruction_rank.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
