"""Can native chi bond gyros span the first nonlinear clock obstruction?

The existing C27 joint-area potential plus reciprocal W gyros closes the
linear transport rank and preserves the exact rest orbit.  Finite linked-clock
continuation leaves a predominantly transverse exterior-chi residual.  This
probe screens the lowest-degree cubic-symmetric direct-chi one-forms without
installing any coefficient:

* local chi phase velocity;
* C19 and body-diagonal inner-product common/relative velocities;
* C19 and body-diagonal chi-area common/relative velocities.

It first restricts coefficients to the exact-rest nullspace, then asks whether
that admissible span contains both signed finite-branch residuals.  This is a
response-rank diagnostic, not coefficient admission.
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
    "local_chi_phase_velocity",
    "c19_chi_inner_common",
    "c19_chi_inner_relative",
    "body_chi_inner_common",
    "body_chi_inner_relative",
    "c19_chi_area_common",
    "c19_chi_area_relative",
    "body_chi_area_common",
    "body_chi_area_relative",
)


def chi_kinematics(left: torch.Tensor, right: torch.Tensor):
    _, left_chi = candidate.fields(left)
    _, right_chi = candidate.fields(right)
    chi = (left_chi + right_chi) / 2
    velocity = (right_chi - left_chi) / candidate.base.inverse.DT
    return chi, velocity


def local_phase(chi: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    return candidate.base.inverse.DT * torch.sum(torch.imag(
        torch.sum(torch.conj(chi) * velocity, dim=-1)))


def inner_pair(chi: torch.Tensor, velocity: torch.Tensor,
               offsets, relative: bool) -> torch.Tensor:
    result = torch.zeros((), dtype=torch.float64)
    for offset, weight in offsets:
        neighbor_chi = torch.roll(chi, shifts=offset, dims=(0, 1, 2))
        neighbor_velocity = torch.roll(
            velocity, shifts=offset, dims=(0, 1, 2))
        first = torch.sum(torch.conj(chi) * neighbor_velocity, dim=-1)
        second = torch.sum(
            torch.conj(neighbor_chi) * velocity, dim=-1)
        value = first - second if relative else first + second
        result = result + weight * torch.sum(torch.imag(value))
    return candidate.base.inverse.DT * result / (
        4 * candidate.base.inverse.rank.R**2)


def area_pair(chi: torch.Tensor, velocity: torch.Tensor,
              offsets, relative: bool) -> torch.Tensor:
    result = torch.zeros((), dtype=torch.float64)
    scale = candidate.base.inverse.rank.R**2
    for offset, weight in offsets:
        neighbor_chi = torch.roll(chi, shifts=offset, dims=(0, 1, 2))
        neighbor_velocity = torch.roll(
            velocity, shifts=offset, dims=(0, 1, 2))
        area = torch.linalg.cross(chi, neighbor_chi, dim=-1) / scale
        left_velocity = torch.linalg.cross(
            velocity, neighbor_chi, dim=-1) / scale
        right_velocity = torch.linalg.cross(
            chi, neighbor_velocity, dim=-1) / scale
        area_velocity = (left_velocity - right_velocity if relative
                         else left_velocity + right_velocity)
        result = result + weight * torch.sum(torch.imag(torch.sum(
            torch.conj(area) * area_velocity, dim=-1)))
    return candidate.base.inverse.DT * result / 4


def gyro_basis(left: torch.Tensor, right: torch.Tensor):
    chi, velocity = chi_kinematics(left, right)
    return (
        local_phase(chi, velocity),
        inner_pair(chi, velocity, C19_OFFSETS, False),
        inner_pair(chi, velocity, C19_OFFSETS, True),
        inner_pair(chi, velocity, BODY_OFFSETS, False),
        inner_pair(chi, velocity, BODY_OFFSETS, True),
        area_pair(chi, velocity, C19_OFFSETS, False),
        area_pair(chi, velocity, C19_OFFSETS, True),
        area_pair(chi, velocity, BODY_OFFSETS, False),
        area_pair(chi, velocity, BODY_OFFSETS, True),
    )


def interval_action(left: torch.Tensor, right: torch.Tensor,
                    extra: torch.Tensor) -> torch.Tensor:
    theta0 = candidate.base.inverse.rest_layers()[3]
    a = 2 * spectrum.DT / np.sin(theta0)
    native = torch.tensor((
        a + candidate.base.inverse.rank.B / 4,
        a - candidate.base.inverse.rank.B / 4,
        -a / 2,
        0.0,
    ), dtype=torch.float64)
    value = candidate.interval_action(left, right, 1.0, native)
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
        columns = [unit_del(*layers).detach().numpy() - baseline
                   for unit_del in unit_dels]
        branch_matrices.append(np.column_stack(columns))
    response = np.vstack([matrix @ rest_null for matrix in branch_matrices])
    target = np.concatenate(targets)
    reduced, _, response_rank, response_singular = np.linalg.lstsq(
        response, target, rcond=1e-10)
    coefficients = rest_null @ reduced
    predicted = response @ reduced
    relative_remainder = float(
        np.linalg.norm(predicted - target) / max(np.linalg.norm(target), 1e-300))
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
        "status": "V4_C27_CHI_BOND_GYRO_OBSTRUCTION_RANK",
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
        "combined_relative_remainder": relative_remainder,
        "branches": per_branch,
        "verdict": (
            "CHI_BOND_GYRO_BASIS_SPANS_NONLINEAR_OBSTRUCTION"
            if relative_remainder < 1e-6 else
            "CHI_BOND_GYRO_BASIS_INCOMPLETE_FOR_NONLINEAR_OBSTRUCTION"),
        "claim_boundary": (
            "Exact-rest-nullspace response-rank screen only. Diagnostic "
            "coefficients are not admitted or fitted into the action."),
    }
    output = HERE / "c27_chi_bond_gyro_obstruction_rank.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
