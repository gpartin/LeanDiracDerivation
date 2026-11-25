"""Test a native vacuum-chi alignment to neighboring Psi--chi area normals.

The localized rest orbit has rhoChi=rhoPsi on its eight carrier sites, while
ordinary vacuum has rhoPsi=0 and chi along the third internal direction.  The
carrier area W=chi x Psi points along that same third direction.  Therefore

  ((rhoChi-rhoPsi)/chi0^2)^2 |chi_i x W_j / chi0|^2

is positive, phase invariant and cubic symmetric; it and its first variation
vanish on the exact rest orbit.  It lets a quiet receiving chi site acquire
orientation from a neighboring occupied site without a prepared ray, corner
label, or external background.  C19 and body-diagonal versions are screened
against both signed nonlinear clock residuals.  Coefficients remain diagnostic.
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
C19_OFFSETS = tuple(candidate.base.inverse.rank.native.DIRECTIONS)
BODY_OFFSETS = tuple((offset, 1.0) for offset in
                     itertools.product((-1, 1), repeat=3))
BASIS_NAMES = ("c19_vacuum_mold_alignment", "body_vacuum_mold_alignment")


def squared_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.conj(value) * value)


def alignment_potential(raw: torch.Tensor, offsets) -> torch.Tensor:
    psi, chi = candidate.fields(raw)
    radius = candidate.base.inverse.rank.R
    rho_psi = torch.sum(squared_norm(psi), dim=-1)
    rho_chi = torch.sum(squared_norm(chi), dim=-1)
    activation = ((rho_chi - rho_psi) / radius**2) ** 2
    result = torch.zeros((), dtype=torch.float64)
    for offset, weight in offsets:
        neighbor_psi = torch.roll(psi, shifts=offset, dims=(0, 1, 2))
        neighbor_chi = torch.roll(chi, shifts=offset, dims=(0, 1, 2))
        neighbor_area = torch.linalg.cross(
            neighbor_chi, neighbor_psi, dim=-1) / radius
        mismatch = torch.linalg.cross(
            chi, neighbor_area, dim=-1) / radius
        result = result + weight * torch.sum(
            activation * torch.sum(squared_norm(mismatch), dim=-1))
    # Directed offsets count each undirected interface twice.
    return result / 2


def potential_basis(raw: torch.Tensor):
    return (
        alignment_potential(raw, C19_OFFSETS),
        alignment_potential(raw, BODY_OFFSETS),
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
    left_terms = potential_basis(left)
    right_terms = potential_basis(right)
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


def main() -> None:
    spectrum.configure()
    candidate.base.inverse.rank.lock_inertia()
    previous, current, following, _ = candidate.base.inverse.rest_layers()
    rest_layers = tuple(torch.tensor(value, dtype=torch.float64)
                        for value in (previous, current, following))
    zero = np.zeros(len(BASIS_NAMES))
    baseline_del = make_del(zero)
    rest_baseline = baseline_del(*rest_layers).detach().numpy()
    unit_dels = []
    rest_columns = []
    rest_energies = []
    current_tensor = rest_layers[1]
    for index in range(len(BASIS_NAMES)):
        unit = zero.copy()
        unit[index] = 1.0
        unit_del = make_del(unit)
        unit_dels.append(unit_del)
        rest_columns.append(
            unit_del(*rest_layers).detach().numpy() - rest_baseline)
        rest_energies.append(float(
            potential_basis(current_tensor)[index].detach()))
    rest_matrix = np.column_stack(rest_columns)

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
    response = np.vstack(branch_matrices)
    target = np.concatenate(targets)
    coefficients, _, response_rank, singular = np.linalg.lstsq(
        response, target, rcond=1e-10)
    predicted = response @ coefficients
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
        "status": "V4_C19_VACUUM_MOLD_ALIGNMENT_OBSTRUCTION",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "basis_names": list(BASIS_NAMES),
        "rest_potential_values": rest_energies,
        "rest_force_column_norms": [float(np.linalg.norm(column))
                                    for column in rest_columns],
        "response_rank": int(response_rank),
        "response_singular_values": [float(value) for value in singular],
        "diagnostic_coefficients": {
            name: float(value) for name, value in zip(BASIS_NAMES, coefficients)},
        "coefficients_nonnegative": bool(np.all(coefficients >= 0)),
        "combined_relative_remainder": combined,
        "branches": per_branch,
        "verdict": (
            "VACUUM_MOLD_ALIGNMENT_SPANS_NONLINEAR_OBSTRUCTION"
            if combined < 1e-6 else
            "VACUUM_MOLD_ALIGNMENT_PARTIAL_NONLINEAR_RESPONSE"
            if combined < 0.99 else
            "VACUUM_MOLD_ALIGNMENT_MISSES_NONLINEAR_OBSTRUCTION"),
        "claim_boundary": (
            "Positive-potential response screen only. Diagnostic coefficients "
            "are not admitted or fitted into the action."),
    }
    output = HERE / "c19_vacuum_mold_alignment_obstruction.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
