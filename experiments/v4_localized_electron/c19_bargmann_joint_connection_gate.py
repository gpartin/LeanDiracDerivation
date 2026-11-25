"""Rest-solvability gate for a symmetry-safe joint Bargmann connection.

For a loop B, the ordinary Berry one-form Im(conj(B) dB) reverses sign when
the two sender vertices are exchanged.  Selecting one triangle orientation
would therefore bake an orientation into the action.  Multiplication by
Im(B), which also reverses sign, gives the orientation-independent one-form

    Im(B) Im(conj(B) dB).

Its curvature is nonzero, so it is a genuine joint Psi--chi kinetic
connection rather than a boundary term.  This script asks whether this
connection, together with the two positive reversal-invariant loop squares,
has any nonzero coefficient combination whose discrete Euler--Lagrange force
vanishes on the exact V4 rest orbit.  Failure at this gate retires the basis
before a transport or simulation claim is attempted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.func import grad

import plaquette_factorization_completion_probe as base
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent
CHI0 = 19.0
BASIS = ("re_loop_square", "im_loop_square", "im_weighted_connection")


def squared_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.conj(value) * value)


def canonical_potential(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = base.inverse.current_audit.mold.unpack_tensor(
        raw, base.inverse.SIDE)
    rho_psi = torch.real(torch.sum(torch.conj(psi) * psi, dim=-1))
    rho_chi = torch.real(torch.sum(torch.conj(chi) * chi, dim=-1))
    local = torch.sum(
        0.25 * (rho_chi - rank.R**2)**2
        + 0.5 * rho_chi * rho_psi + 0.25 * rho_psi**2)
    area = torch.linalg.cross(chi, psi, dim=-1)
    bond = torch.zeros((), dtype=torch.float64)
    for offset, weight in rank.native.DIRECTIONS:
        difference = area - torch.roll(
            area, shifts=offset, dims=(0, 1, 2))
        bond = bond + weight * torch.sum(squared_norm(difference))
    return local + bond / (2 * rank.R**2)


def canonical_interval_action(left: torch.Tensor,
                              right: torch.Tensor) -> torch.Tensor:
    left_psi, left_chi = base.inverse.current_audit.mold.unpack_tensor(
        left, base.inverse.SIDE)
    right_psi, right_chi = base.inverse.current_audit.mold.unpack_tensor(
        right, base.inverse.SIDE)
    psi_dot = (right_psi - left_psi) / base.inverse.DT
    chi_dot = (right_chi - left_chi) / base.inverse.DT
    kinetic = 0.5 * torch.sum(squared_norm(psi_dot))
    kinetic = kinetic + rank.B / 2 * torch.sum(squared_norm(chi_dot))
    return base.inverse.DT * (
        kinetic - (canonical_potential(left) + canonical_potential(right)) / 2)


def inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.conj(left) * right, dim=-1)


def shift(field: torch.Tensor, axis: int, amount: int) -> torch.Tensor:
    offsets = tuple(amount if index == axis else 0 for index in range(3))
    return torch.roll(field, shifts=offsets, dims=(0, 1, 2))


def loops(raw: torch.Tensor) -> list[torch.Tensor]:
    psi, chi = base.inverse.current_audit.mold.unpack_tensor(
        raw, base.inverse.SIDE)
    values = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        for first_sign in (-1, 1):
            for second_sign in (-1, 1):
                psi_j = shift(psi, first, first_sign)
                psi_k = shift(psi_j, second, second_sign)
                values.append(inner(chi, psi_j) * inner(psi_j, psi_k)
                              * inner(psi_k, chi))
    return values


def potential_terms(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    re_square = torch.zeros((), dtype=torch.float64)
    im_square = torch.zeros((), dtype=torch.float64)
    for loop in loops(raw):
        re_square = re_square + torch.sum(torch.real(loop) ** 2)
        im_square = im_square + torch.sum(torch.imag(loop) ** 2)
    return re_square / CHI0**8, im_square / CHI0**8


def connection_term(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    result = torch.zeros((), dtype=torch.float64)
    for left_loop, right_loop in zip(loops(left), loops(right)):
        midpoint = (left_loop + right_loop) / 2
        increment = right_loop - left_loop
        one_form = torch.imag(torch.conj(midpoint) * increment)
        result = result + torch.sum(torch.imag(midpoint) * one_form)
    return result / CHI0**12


def basis_interval(left: torch.Tensor, right: torch.Tensor,
                   index: int) -> torch.Tensor:
    if index < 2:
        left_terms = potential_terms(left)
        right_terms = potential_terms(right)
        return -base.inverse.DT * (left_terms[index] + right_terms[index]) / 2
    return connection_term(left, right)


def basis_three(previous: torch.Tensor, current: torch.Tensor,
                following: torch.Tensor, index: int) -> torch.Tensor:
    return (basis_interval(previous, current, index)
            + basis_interval(current, following, index))


def reversal_error(left: torch.Tensor, right: torch.Tensor) -> float:
    # Complex conjugating every loop models exchange of its two senders.
    # Evaluate the scalar identity directly on deterministic loop tensors.
    errors = []
    for left_loop, right_loop in zip(loops(left), loops(right)):
        midpoint = (left_loop + right_loop) / 2
        increment = right_loop - left_loop
        original = torch.imag(midpoint) * torch.imag(
            torch.conj(midpoint) * increment)
        reversed_value = torch.imag(torch.conj(midpoint)) * torch.imag(
            midpoint * torch.conj(increment))
        errors.append(float(torch.max(torch.abs(
            original - reversed_value)).detach()))
    return max(errors)


def main() -> None:
    base.inverse.rank.lock_inertia()
    base.inverse.rest.SIDE = base.inverse.SIDE
    base.inverse.rest.DT = base.inverse.DT
    previous, current, following, _theta = base.inverse.rest_layers()
    tensors = tuple(torch.tensor(value, dtype=torch.float64)
                    for value in (previous, current, following))

    columns = []
    column_norms = []
    for index in range(len(BASIS)):
        derivative = grad(
            lambda p, c, f: basis_three(p, c, f, index), argnums=1)
        column = derivative(*tensors).detach().numpy()
        columns.append(column)
        column_norms.append(float(np.linalg.norm(column)))
    matrix = np.column_stack(columns)
    raw_scales = np.linalg.norm(matrix, axis=0)
    # The exact rest orbit is evaluated in double precision.  A column below
    # 1e-10 is a numerical zero at this gate and must not be normalized into
    # an artificial unit response.
    active = raw_scales > 1e-10
    scales = np.where(active, raw_scales, 1.0)
    normalized = matrix[:, active] / scales[active]
    _u, singular, vh = np.linalg.svd(normalized, full_matrices=True)
    tolerance = max(normalized.shape) * max(singular[0], 1.0) * 1e-12
    active_rank = int(np.sum(singular > tolerance))
    active_null = vh[active_rank:].T
    null_columns = []
    active_indices = np.flatnonzero(active)
    for column in range(active_null.shape[1]):
        vector = np.zeros(len(BASIS))
        vector[active_indices] = active_null[:, column]
        null_columns.append(vector)
    for index in np.flatnonzero(~active):
        vector = np.zeros(len(BASIS))
        vector[index] = 1.0
        null_columns.append(vector)
    nullspace = (np.column_stack(null_columns) if null_columns
                 else np.zeros((len(BASIS), 0)))
    rank = active_rank
    normalized_full = matrix / scales
    null_residuals = [float(np.linalg.norm(
        normalized_full @ nullspace[:, index]))
                      for index in range(nullspace.shape[1])]

    rng = np.random.default_rng(4197)
    random_left = torch.tensor(
        rng.normal(size=previous.size), dtype=torch.float64)
    random_right = torch.tensor(
        rng.normal(size=previous.size), dtype=torch.float64)
    reversal = reversal_error(random_left, random_right)

    potential_positive = all(float(value.detach()) >= -1e-14
                             for value in potential_terms(random_left))
    if nullspace.shape[1] == 0:
        verdict = "JOINT_BARGMANN_BASIS_HAS_NO_REST_PRESERVING_DIRECTION"
    else:
        verdict = "JOINT_BARGMANN_BASIS_HAS_REST_PRESERVING_DIRECTION"
    result = {
        "status": "V4_C19_BARGMANN_JOINT_CONNECTION_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "basis": list(BASIS),
        "basis_DEL_norms": column_norms,
        "numerically_active_rest_columns": [
            name for name, keep in zip(BASIS, active) if keep],
        "normalized_rest_response_rank": rank,
        "rest_preserving_dimension": int(nullspace.shape[1]),
        "normalized_singular_values": [float(value) for value in singular],
        "normalized_null_residuals": null_residuals,
        "nullspace_scaled_coordinates": nullspace.tolist(),
        "sender_reversal_max_error": reversal,
        "positive_square_terms_nonnegative": potential_positive,
        "verdict": verdict,
        "claim_boundary": (
            "Exact discrete rest-response rank for this three-element action "
            "basis only. A nonzero null direction would still require "
            "transport rank, symplectic nondegeneracy, stability, coefficient "
            "provenance, and Dirac-current closure."),
    }
    output = HERE / "c19_bargmann_joint_connection_gate.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
