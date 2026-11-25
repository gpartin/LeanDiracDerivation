"""Rest-spectrum control for nonlinear χ transfer softening."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import density_mold_connection_probe as local
import density_mold_rest_spectrum as rest
import density_mold_tensor_probe as tensor
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent


def interval_action(left: torch.Tensor, right: torch.Tensor,
                    geometric_factor: float) -> torch.Tensor:
    psi_left, chi_left = local.unpack_tensor(left, rest.SIDE)
    psi_right, chi_right = local.unpack_tensor(right, rest.SIDE)
    psi = (psi_left + psi_right) / 2
    chi = (chi_left + chi_right) / 2
    psi_dot = (psi_right - psi_left) / rest.DT
    chi_dot = (chi_right - chi_left) / rest.DT
    rho_dot = 2 * torch.real(torch.sum(
        torch.conj(psi) * psi_dot, dim=-1))
    rho_psi = torch.sum(local.squared_norm(psi), dim=-1)
    curvature = tensor.density_curvature(rho_dot)
    trace = torch.diagonal(curvature, dim1=-2, dim2=-1).sum(-1)
    identity = torch.eye(3, dtype=torch.float64)
    traceless = (curvature - trace[..., None, None] * identity / 3).to(
        torch.complex128)
    response = (rho_dot[..., None] * chi
                + torch.einsum("...ab,...b->...a", traceless, chi))
    rho_chi = torch.sum(local.squared_norm(chi), dim=-1)
    chi_denominator = rho_chi + rank.KAPPA * rank.R**2
    material = chi_dot + local.ALPHA * response / (
        2 * chi_denominator[..., None])
    overlap = torch.sum(torch.conj(chi) * material, dim=-1)
    longitudinal = chi * overlap[..., None] / chi_denominator[..., None]
    transverse = material - longitudinal
    transfer = rho_dot / (
        2 * rank.R * (rho_psi + rank.KAPPA * rank.R**2))
    scale = torch.rsqrt(1 + geometric_factor * transfer**2)
    framed = longitudinal + scale[..., None] * transverse
    averaged = local.adjacency_average(framed)
    kinetic = 0.5 * torch.sum(local.squared_norm(psi_dot))
    kinetic = kinetic + rank.B / 2 * (
        rank.KAPPA * torch.sum(local.squared_norm(chi_dot))
        + torch.sum(local.squared_norm(averaged)))
    return rest.DT * (
        kinetic - (rest.potential(left) + rest.potential(right)) / 2)


def classify(factor: float):
    original = rest.interval_action
    rest.interval_action = interval_action
    try:
        return rest.classify(factor)
    finally:
        rest.interval_action = original


def main() -> None:
    rank.lock_inertia()
    factors = {
        "incumbent_zero": 0.0,
        "face_transport_channels": 6.0,
        "c19_count": 19.0,
        "radial_balance": 54.0,
    }
    rows = []
    for name, factor in factors.items():
        row = classify(factor)
        row["name"] = name
        row["geometric_factor"] = factor
        rows.append(row)
    incumbent = rows[0]
    max_growth_defect = max(abs(
        row["growth_rate_per_unit_time"]
        - incumbent["growth_rate_per_unit_time"])
        for row in rows[1:])
    max_hessian_defect = max(abs(
        row["spectral_radius"] - incumbent["spectral_radius"])
        for row in rows[1:])
    passed = max_growth_defect < 1e-7 and max_hessian_defect < 1e-10
    result = {
        "status": (
            "V4_CHI_TRANSFER_SOFTENING_REST_GATE_PASS" if passed else
            "V4_CHI_TRANSFER_SOFTENING_REST_GATE_FAIL"),
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": rest.SIDE,
        "dt": rest.DT,
        "rows": rows,
        "max_growth_rate_defect": float(max_growth_defect),
        "max_spectral_radius_defect": float(max_hessian_defect),
        "analytic_reason": (
            "transfer=0 on the rest orbit and scale=1+O(transfer^2), "
            "so the action change begins above quadratic order"),
        "verdict": (
            "ADVANCE_TO_FREE_ORBIT_GATE" if passed else
            "REST_HESSIAN_CHANGED_UNEXPECTEDLY"),
        "claim_boundary": (
            "This verifies the side-three two-layer linear rest spectrum; "
            "it is not nonlinear stability or movement."),
    }
    (HERE / "chi_transfer_softening_rest_spectrum.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
