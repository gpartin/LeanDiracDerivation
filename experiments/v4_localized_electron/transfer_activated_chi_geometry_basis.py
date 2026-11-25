"""Resolve transfer-time chi response into radial and orientation bonds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
import torch

import density_mold_connection_probe as local
import edge_snap_action_rank as rank
import edge_snap_residual_action_projection as projection


HERE = Path(__file__).resolve().parent
SIDE = 5
DT = 0.1
INPUT = "transfer_activated_chi_bond_four_update_orbit"
NAMES = ("full_difference", "radial_magnitude", "orientation_area")


def interval_action(left_raw: torch.Tensor, right_raw: torch.Tensor,
                    channel: str) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left_raw, SIDE)
    right_psi, right_chi = local.unpack_tensor(right_raw, SIDE)
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    psi_dot = (right_psi - left_psi) / DT
    rho_dot = 2 * torch.real(torch.sum(
        torch.conj(psi) * psi_dot, dim=-1))
    rho_psi = torch.sum(local.squared_norm(psi), dim=-1)
    rho_chi = torch.sum(local.squared_norm(chi), dim=-1)
    transfer = rho_dot / (
        2 * rank.R * (rho_psi + rank.KAPPA * rank.R**2))
    total = torch.zeros((), dtype=torch.float64)
    for offset, weight in rank.native.DIRECTIONS:
        chi_j = torch.roll(chi, shifts=offset, dims=(0, 1, 2))
        rho_j = torch.roll(rho_chi, shifts=offset, dims=(0, 1, 2))
        transfer_j = torch.roll(transfer, shifts=offset, dims=(0, 1, 2))
        activation = (transfer**2 + transfer_j**2) / 2
        denominator = rho_chi + rho_j + rank.KAPPA * rank.R**2
        if channel == "full_difference":
            bond = torch.sum(local.squared_norm(chi_j - chi), dim=-1)
        elif channel == "radial_magnitude":
            bond = (rho_j - rho_chi)**2 / denominator
        elif channel == "orientation_area":
            area = torch.linalg.cross(chi, chi_j, dim=-1)
            bond = torch.sum(local.squared_norm(area), dim=-1) / denominator
        else:
            raise ValueError(channel)
        total = total + weight * torch.sum(activation * bond)
    return DT * rank.B * total / (2 * rank.C19_WEIGHT_SUM)


def action_delta(triple, channel: str) -> np.ndarray:
    raws = [torch.tensor(rank.pack(*layer), dtype=torch.float64,
                        requires_grad=True) for layer in triple]
    action = interval_action(raws[0], raws[1], channel)
    action = action + interval_action(raws[1], raws[2], channel)
    return torch.autograd.grad(action, raws[1])[0].detach().numpy()


def main() -> None:
    rank.lock_inertia()
    triples, data = projection.layers_from_artifact(INPUT)
    target = -data["residuals"].reshape(-1)
    columns = [np.concatenate([action_delta(triple, name)
                               for triple in triples]) for name in NAMES]
    matrix = np.column_stack(columns)
    target_norm = np.linalg.norm(target)
    norms = np.linalg.norm(matrix, axis=0)
    normalized = matrix / norms
    coefficients_normalized, residual_norm = nnls(
        normalized, target / target_norm, maxiter=10000)
    coefficients = coefficients_normalized * target_norm / norms
    single = {}
    for name, column, norm in zip(NAMES, columns, norms):
        correlation = float(np.dot(column, target) / (norm * target_norm))
        optimum = float(np.dot(column, target) / np.dot(column, column))
        positive = max(optimum, 0.0)
        residual = np.linalg.norm(positive * column - target) / target_norm
        single[name] = {
            "correlation": correlation,
            "positive_optimal_coefficient": positive,
            "positive_explained_squared_fraction": float(1 - residual**2),
        }
    result = {
        "status": "V4_TRANSFER_ACTIVATED_CHI_GEOMETRY_BASIS",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "input_orbit": INPUT,
        "basis": list(NAMES),
        "all_terms_real_positive_reciprocal_cubic": True,
        "contains_corner_copy_or_Dirac_label": False,
        "single_channels": single,
        "positive_joint_fit": {
            "response_rank": int(np.linalg.matrix_rank(normalized, tol=1e-11)),
            "coefficients": {
                name: float(value) for name, value in zip(NAMES, coefficients)
                if value > 1e-12
            },
            "relative_residual": float(residual_norm),
            "explained_squared_fraction": float(1 - residual_norm**2),
        },
        "verdict": (
            "ADVANCE_SPLIT_CHI_GEOMETRY_BASIS"
            if residual_norm < 0.99 else
            "RETIRE_SPLIT_CHI_GEOMETRY_BASIS"),
        "claim_boundary": (
            "Inverse response coefficients are diagnostics only. A survivor "
            "still needs native coefficient provenance, spectrum, charge, "
            "and free-orbit gates."),
    }
    (HERE / "transfer_activated_chi_geometry_basis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
