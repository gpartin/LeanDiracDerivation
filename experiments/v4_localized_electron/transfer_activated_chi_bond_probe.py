"""Test a positive transfer-activated native chi bond.

The action term is a positive kinetic contribution.  It uses only the local
Psi density rate and the C19 chi difference, so it vanishes on the exact rest
orbit but can stiffen the receiving substrate during a handoff.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import density_mold_connection_probe as local
import edge_snap_action_rank as rank
import edge_snap_residual_action_projection as projection


HERE = Path(__file__).resolve().parent
DT = 0.1
SIDE = 5
INPUTS = (
    "edge_snap_transfer_charge_four_update_orbit",
    "direct_chi_bond_four_update_orbit",
)


def interval_extra_action(left_raw: torch.Tensor, right_raw: torch.Tensor,
                          dt: float) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left_raw, SIDE)
    right_psi, right_chi = local.unpack_tensor(right_raw, SIDE)
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    psi_dot = (right_psi - left_psi) / dt
    rho_dot = 2 * torch.real(torch.sum(
        torch.conj(psi) * psi_dot, dim=-1))
    rho_psi = torch.sum(local.squared_norm(psi), dim=-1)
    transfer = rho_dot / (
        2 * rank.R * (rho_psi + rank.KAPPA * rank.R**2))
    bond = torch.zeros((), dtype=torch.float64)
    for offset, weight in rank.native.DIRECTIONS:
        chi_neighbor = torch.roll(chi, shifts=offset, dims=(0, 1, 2))
        transfer_neighbor = torch.roll(
            transfer, shifts=offset, dims=(0, 1, 2))
        activation = (transfer**2 + transfer_neighbor**2) / 2
        difference_sq = torch.sum(
            local.squared_norm(chi_neighbor - chi), dim=-1)
        bond = bond + weight * torch.sum(activation * difference_sq)
    # Directed C19 bonds are normalized by their weighted degree.  The
    # coefficient multiplying this unit column remains an inverse diagnostic.
    return dt * rank.B * bond / (2 * rank.C19_WEIGHT_SUM)


def action_delta(triple) -> np.ndarray:
    raws = [torch.tensor(rank.pack(*layer), dtype=torch.float64,
                        requires_grad=True) for layer in triple]
    action = interval_extra_action(raws[0], raws[1], DT)
    action = action + interval_extra_action(raws[1], raws[2], DT)
    return torch.autograd.grad(action, raws[1])[0].detach().numpy()


def response(input_stem: str) -> dict:
    triples, data = projection.layers_from_artifact(input_stem)
    target = -data["residuals"].reshape(-1)
    column = np.concatenate([action_delta(triple) for triple in triples])
    column_norm = np.linalg.norm(column)
    target_norm = np.linalg.norm(target)
    correlation = float(np.dot(column, target) / (column_norm * target_norm))
    optimum = float(np.dot(column, target) / np.dot(column, column))
    positive = max(optimum, 0.0)
    residual = np.linalg.norm(positive * column - target) / target_norm
    half_residual = np.linalg.norm(0.5 * column - target) / target_norm
    unit_residual = np.linalg.norm(column - target) / target_norm
    return {
        "input_orbit": input_stem,
        "column_norm": float(column_norm),
        "correlation_with_required_response": correlation,
        "unconstrained_optimal_coefficient": optimum,
        "positive_optimal_coefficient": positive,
        "positive_relative_residual": float(residual),
        "positive_explained_squared_fraction": float(1 - residual**2),
        "undirected_half_coefficient_relative_residual": float(half_residual),
        "undirected_half_coefficient_improvement": float(1 - half_residual),
        "unit_coefficient_relative_residual": float(unit_residual),
    }


def rest_control() -> dict:
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    theta = float(np.arccos(1 - DT**2 * rank.R**2 / 2))
    layers = (
        (np.exp(-1j * theta) * psi, chi),
        (psi, chi),
        (np.exp(1j * theta) * psi, chi),
    )
    column = action_delta(layers)
    return {
        "action_value_left_interval": float(interval_extra_action(
            torch.tensor(rank.pack(*layers[0]), dtype=torch.float64),
            torch.tensor(rank.pack(*layers[1]), dtype=torch.float64),
            DT).detach()),
        "rest_DEL_column_norm": float(np.linalg.norm(column)),
    }


def main() -> None:
    rank.lock_inertia()
    controls = rest_control()
    rows = [response(stem) for stem in INPUTS]
    improves = any(row["positive_explained_squared_fraction"] > 1e-6
                   for row in rows)
    result = {
        "status": "V4_TRANSFER_ACTIVATED_CHI_BOND_RESPONSE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action_delta": (
            "Delta T=(B/(2*24)) sum_C19 w * "
            "[(a_i^2+a_j^2)/2] |chi_j-chi_i|^2, "
            "a=rhoPsiDot/[2 chi0(rhoPsi+kappa chi0^2)]"),
        "real_positive_reciprocal": True,
        "cubic_and_C19_symmetric": True,
        "contains_corner_copy_or_Dirac_label": False,
        "rest_control": controls,
        "movement_response": rows,
        "verdict": (
            "ADVANCE_TRANSFER_ACTIVATED_BOND" if improves else
            "RETIRE_TRANSFER_ACTIVATED_BOND"),
        "claim_boundary": (
            "This is an action-response screen. A favorable positive column "
            "does not establish coefficient provenance, rest stability, an "
            "exact moving orbit, or a Dirac pass."),
    }
    (HERE / "transfer_activated_chi_bond_probe.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
