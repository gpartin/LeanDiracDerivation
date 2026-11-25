"""Rest-spectrum gate for the transfer-activated native chi bond."""

from __future__ import annotations

import json
from pathlib import Path

import torch

import chi_transfer_softening_rest_spectrum as softening
import density_mold_connection_probe as local
import density_mold_rest_spectrum as rest
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent


def extra_action(left_raw: torch.Tensor, right_raw: torch.Tensor) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left_raw, rest.SIDE)
    right_psi, right_chi = local.unpack_tensor(right_raw, rest.SIDE)
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    psi_dot = (right_psi - left_psi) / rest.DT
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
    return rest.DT * rank.B * bond / (2 * rank.C19_WEIGHT_SUM)


def interval_action(left: torch.Tensor, right: torch.Tensor,
                    coefficient: float) -> torch.Tensor:
    return (softening.interval_action(left, right, 19.0)
            + coefficient * extra_action(left, right))


def classify(coefficient: float):
    original = rest.interval_action
    rest.interval_action = interval_action
    try:
        return rest.classify(coefficient)
    finally:
        rest.interval_action = original


def main() -> None:
    rank.lock_inertia()
    coefficients = (0.0, 0.5, 1.0)
    rows = []
    for coefficient in coefficients:
        row = classify(coefficient)
        row["coefficient"] = coefficient
        rows.append(row)
    incumbent = rows[0]
    candidate = rows[1]
    passed = (candidate["growth_rate_per_unit_time"]
              <= incumbent["growth_rate_per_unit_time"] + 1e-9)
    result = {
        "status": (
            "V4_TRANSFER_ACTIVATED_CHI_BOND_REST_GATE_PASS" if passed else
            "V4_TRANSFER_ACTIVATED_CHI_BOND_REST_GATE_FAIL"),
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": rest.SIDE,
        "dt": rest.DT,
        "rows": rows,
        "half_coefficient_growth_delta": float(
            candidate["growth_rate_per_unit_time"]
            - incumbent["growth_rate_per_unit_time"]),
        "verdict": (
            "ADVANCE_TO_FREE_ORBIT_GATE" if passed else
            "RETIRE_OR_REDERIVE_TRANSFER_ACTIVATED_BOND"),
        "claim_boundary": (
            "This is the complete side-three two-layer linear rest spectrum. "
            "It is not a nonlinear stability or movement pass."),
    }
    (HERE / "transfer_activated_chi_bond_rest_spectrum.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
