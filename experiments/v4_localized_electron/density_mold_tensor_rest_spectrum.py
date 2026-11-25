"""Rest-spectrum gate for the traceless C19 density-curvature completion."""

from __future__ import annotations

import json
from pathlib import Path

import torch

import density_mold_connection_probe as local
import density_mold_rest_spectrum as rest
import density_mold_tensor_probe as tensor
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent


def tensor_interval_action(left: torch.Tensor, right: torch.Tensor,
                           coefficient: float) -> torch.Tensor:
    psi_left, chi_left = local.unpack_tensor(left, rest.SIDE)
    psi_right, chi_right = local.unpack_tensor(right, rest.SIDE)
    psi = (psi_left + psi_right) / 2
    chi = (chi_left + chi_right) / 2
    psi_dot = (psi_right - psi_left) / rest.DT
    chi_dot = (chi_right - chi_left) / rest.DT
    rho_dot = 2 * torch.real(torch.sum(
        torch.conj(psi) * psi_dot, dim=-1))
    curvature = tensor.density_curvature(rho_dot)
    trace = torch.diagonal(curvature, dim1=-2, dim2=-1).sum(-1)
    identity = torch.eye(3, dtype=torch.float64)
    traceless = (curvature - trace[..., None, None] * identity / 3).to(
        torch.complex128)
    response = (rho_dot[..., None] * chi
                + torch.einsum("...ab,...b->...a", traceless, chi))
    rho_chi = torch.sum(local.squared_norm(chi), dim=-1)
    denominator = rho_chi + rank.KAPPA * rank.R**2
    velocity = (chi_dot + local.ALPHA * response
                / (2 * denominator[..., None]))
    averaged = local.adjacency_average(velocity)
    kinetic = 0.5 * torch.sum(local.squared_norm(psi_dot))
    kinetic = kinetic + rank.B / 2 * (
        rank.KAPPA * torch.sum(local.squared_norm(chi_dot))
        + torch.sum(local.squared_norm(averaged)))
    return rest.DT * (
        kinetic - (rest.potential(left) + rest.potential(right)) / 2)


def main() -> None:
    rank.lock_inertia()
    local_result = json.loads(
        (HERE / "density_mold_rest_spectrum.json").read_text(
            encoding="utf-8"))
    local_row = local_result["rows"][1]
    original = rest.interval_action
    rest.interval_action = tensor_interval_action
    try:
        row = rest.classify(local.ALPHA)
    finally:
        rest.interval_action = original
    relative = ((local_row["growth_rate_per_unit_time"]
                 - row["growth_rate_per_unit_time"])
                / local_row["growth_rate_per_unit_time"])
    preserved = abs(relative) < 1e-8
    result = {
        "status": ("V4_DENSITY_MOLD_TENSOR_REST_GATE_PASS" if preserved or
                   relative > 0 else "V4_DENSITY_MOLD_TENSOR_REST_GATE_FAIL"),
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "mode": "local_plus_traceless",
        "local_incumbent": local_row,
        "tensor_candidate": row,
        "relative_growth_improvement": float(relative),
        "rest_growth_preserved": preserved,
        "verdict": (
            "ADVANCE_TO_FOUR_UPDATE_ORBIT" if preserved or relative > 0 else
            "RETIRE_TENSOR_COMPLETION"),
        "claim_boundary": (
            "This is a side-three implicit linear rest spectrum, not a "
            "nonlinear stability or movement pass."),
    }
    (HERE / "density_mold_tensor_rest_spectrum.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
