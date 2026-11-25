"""Screen collective C19 Psi-neighbor orientation potentials for chi.

At each site the C19 neighbors define a coherent vector S and covariance C.
Real cubic scalars built from chi, S, C and C^2 can orient a receiving chi site
using several incoming Psi signals at once.  Variation automatically returns
the reciprocal force to every contributing Psi neighbor.  The individual
scalars are screened only as a complete response basis: exact-rest nullspace
first, then the two signed nonlinear linked-clock obstructions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.func import grad

import c27_body_gyro_basis_probe as candidate
import c27_body_gyro_rest_spectrum as spectrum


HERE = Path(__file__).resolve().parent
DIRECTIONS = tuple(candidate.base.inverse.rank.native.DIRECTIONS)
WEIGHT_SUM = float(sum(weight for _, weight in DIRECTIONS))
BASIS_NAMES = (
    "rhochi_trace_covariance",
    "chi_covariance_chi",
    "rhochi_trace_covariance_squared",
    "chi_covariance_squared_chi",
    "covariance_variance",
    "rhochi_coherent_norm",
    "chi_coherent_overlap",
)


def real_scalar(value: torch.Tensor) -> torch.Tensor:
    return torch.real(value)


def orientation_basis(raw: torch.Tensor):
    psi, chi = candidate.fields(raw)
    coherent = torch.zeros_like(psi)
    covariance = torch.zeros(
        psi.shape[:-1] + (3, 3), dtype=torch.complex128)
    for offset, weight in DIRECTIONS:
        neighbor = torch.roll(psi, shifts=offset, dims=(0, 1, 2))
        coherent = coherent + weight * neighbor
        covariance = covariance + weight * (
            neighbor[..., :, None] * torch.conj(neighbor[..., None, :]))
    coherent = coherent / WEIGHT_SUM
    covariance = covariance / WEIGHT_SUM
    covariance_squared = covariance @ covariance
    rhochi = torch.sum(torch.conj(chi) * chi, dim=-1).real
    trace_covariance = torch.diagonal(
        covariance, dim1=-2, dim2=-1).sum(-1).real
    trace_covariance_squared = torch.diagonal(
        covariance_squared, dim1=-2, dim2=-1).sum(-1).real
    c_chi = torch.einsum("...ab,...b->...a", covariance, chi)
    c2_chi = torch.einsum("...ab,...b->...a", covariance_squared, chi)
    chi_c_chi = real_scalar(torch.sum(torch.conj(chi) * c_chi, dim=-1))
    chi_c2_chi = real_scalar(torch.sum(torch.conj(chi) * c2_chi, dim=-1))
    coherent_norm = torch.sum(
        torch.conj(coherent) * coherent, dim=-1).real
    coherent_overlap = torch.sum(torch.conj(chi) * coherent, dim=-1)
    return (
        torch.sum(rhochi * trace_covariance),
        torch.sum(chi_c_chi),
        torch.sum(rhochi * trace_covariance_squared),
        torch.sum(chi_c2_chi),
        torch.sum(rhochi * chi_c2_chi - chi_c_chi**2),
        torch.sum(rhochi * coherent_norm),
        torch.sum(torch.real(torch.conj(coherent_overlap)
                             * coherent_overlap)),
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
    left_terms = orientation_basis(left)
    right_terms = orientation_basis(right)
    potential = sum(coefficient * (left_term + right_term) / 2
                    for coefficient, left_term, right_term in zip(
                        extra, left_terms, right_terms))
    return value - candidate.base.inverse.DT * potential


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
        "status": "V4_C19_CHI_NEIGHBOR_ORIENTATION_OBSTRUCTION_RANK",
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
            "C19_NEIGHBOR_ORIENTATION_BASIS_SPANS_NONLINEAR_OBSTRUCTION"
            if combined < 1e-6 else
            "C19_NEIGHBOR_ORIENTATION_BASIS_PARTIAL_NONLINEAR_RESPONSE"
            if combined < 0.99 else
            "C19_NEIGHBOR_ORIENTATION_BASIS_MISSES_NONLINEAR_OBSTRUCTION"),
        "claim_boundary": (
            "Exact-rest-nullspace response-rank screen only. Diagnostic "
            "coefficients are not admitted or fitted into the action."),
    }
    output = HERE / "c19_chi_neighbor_orientation_obstruction_rank.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
