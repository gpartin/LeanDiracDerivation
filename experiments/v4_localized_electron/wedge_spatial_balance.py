"""Derive the C19 spatial/local balance of the induced oriented-area wave."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import cube_incidence_nonlinear_action as prior_action
import edge_snap_action_rank as rank
import mixed_prong_dirac_long_tangent as gate
import wedge_augmented_action as action


HERE = Path(__file__).resolve().parent


def area_norm(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = action.local.unpack_tensor(raw, action.SIDE)
    w = action.oriented_area(psi, chi)
    return torch.sum(action.common.squared_norm(w))


def area_bond(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = action.local.unpack_tensor(raw, action.SIDE)
    w = action.oriented_area(psi, chi)
    result = torch.zeros((), dtype=torch.float64)
    for offset, weight in rank.native.DIRECTIONS:
        difference = w-torch.roll(w, shifts=offset, dims=(0, 1, 2))
        result += weight*torch.sum(action.common.squared_norm(difference))
    return result/2


def del_of_interval_term(term, previous, current, following):
    def summed(left, middle, right):
        return term(left, middle)+term(middle, right)
    return torch.func.grad(summed, argnums=1)(previous, current, following)


def main() -> None:
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    theta = float(np.arccos(1-gate.DT**2*rank.R**2/2))
    rminus = gate.phase_rotation_matrix(rest.size, -theta)
    rplus = gate.phase_rotation_matrix(rest.size, theta)
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in (rminus@rest, rest, rplus@rest))

    def local_unit(left, right):
        return -gate.DT*(area_norm(left)+area_norm(right))/4

    def bond_unit(left, right):
        return -gate.DT*(area_bond(left)+area_bond(right))/2

    local_response = del_of_interval_term(local_unit, *layers).detach().numpy()
    bond_response = del_of_interval_term(bond_unit, *layers).detach().numpy()
    delta = -float(np.dot(local_response, bond_response)
                   / np.dot(local_response, local_response))
    remainder = bond_response+delta*local_response
    result = {
        "status": "V4_WEDGE_SPATIAL_BALANCE",
        "derived_local_frequency_shift": delta,
        "derived_local_frequency_squared": rank.R**2+delta,
        "cube_corner_eigenvalue": 27.0,
        "directed_action_counting_factor": 2.0,
        "expected_action_frequency_shift": -54.0,
        "balance_relative_residual": float(
            np.linalg.norm(remainder)/np.linalg.norm(bond_response)),
        "local_response_norm": float(np.linalg.norm(local_response)),
        "bond_response_norm": float(np.linalg.norm(bond_response)),
        "verdict": (
            "CUBE_27_DIRECTED_54_BALANCE_DERIVED" if abs(delta+54) < 1e-8
            and np.linalg.norm(remainder)/np.linalg.norm(bond_response) < 1e-8
            else "WEDGE_SPATIAL_BALANCE_OPEN"),
    }
    path = HERE/"wedge_spatial_balance.json"
    path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
