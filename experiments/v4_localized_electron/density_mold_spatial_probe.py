"""Screen where the Psi-density rate enters the positive chi mold metric.

All candidates use the same real positive reciprocal action and the same
geometry-derived coefficient.  They differ only in whether chi sees the
local density rate, its weighted C19 neighbor average, or the C19
Dirichlet difference.  No direction, carrier site, or Dirac data enters.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import density_mold_connection_probe as local
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent
SIDE = 5
MODES = ("local", "neighbor_average", "dirichlet", "symmetric_average")


def scalar_adjacency_average(value: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(value)
    for offset, weight in rank.native.DIRECTIONS:
        result = result + weight * torch.roll(
            value, shifts=offset, dims=(0, 1, 2))
    return result / rank.C19_WEIGHT_SUM


def interval_extra_action(left_raw: torch.Tensor, right_raw: torch.Tensor,
                          dt: float, mode: str) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left_raw, SIDE)
    right_psi, right_chi = local.unpack_tensor(right_raw, SIDE)
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    psi_dot = (right_psi - left_psi) / dt
    chi_dot = (right_chi - left_chi) / dt
    rho_dot = 2 * torch.real(torch.sum(
        torch.conj(psi) * psi_dot, dim=-1))
    neighbor = scalar_adjacency_average(rho_dot)
    if mode == "local":
        response = rho_dot
    elif mode == "neighbor_average":
        response = neighbor
    elif mode == "dirichlet":
        response = rho_dot - neighbor
    elif mode == "symmetric_average":
        response = (rho_dot + neighbor) / 2
    else:
        raise ValueError(mode)
    rho_chi = torch.sum(local.squared_norm(chi), dim=-1)
    denominator = rho_chi + rank.KAPPA * rank.R**2
    velocity = (chi_dot + local.ALPHA * response[..., None] * chi
                / (2 * denominator[..., None]))
    averaged = local.adjacency_average(velocity)
    candidate = rank.KAPPA * torch.sum(local.squared_norm(chi_dot))
    candidate = candidate + torch.sum(local.squared_norm(averaged))
    canonical = torch.sum(local.squared_norm(chi_dot))
    return dt * rank.B * (candidate - canonical) / 2


def action_delta(previous, current, following, dt: float,
                 mode: str) -> np.ndarray:
    raws = [torch.tensor(rank.pack(*layer), dtype=torch.float64,
                          requires_grad=True)
            for layer in (previous, current, following)]
    action = interval_extra_action(raws[0], raws[1], dt, mode)
    action = action + interval_extra_action(raws[1], raws[2], dt, mode)
    return torch.autograd.grad(action, raws[1])[0].detach().numpy()


def classify(mode: str):
    evidence = json.loads(
        (HERE / "edge_snap_verlet_gate.json").read_text(encoding="utf-8"))
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    force_psi, force_chi = rank.native.force(psi, chi)
    numerator = 0.0
    denominator = 0.0
    rows = []
    for row in evidence["rows"]:
        dt = row["best_dt"]
        layers = rank.translated_layers(row, psi, chi)
        baseline = rank.canonical_residual(
            *layers, dt, force_psi, force_chi)
        residual = baseline + action_delta(*layers, dt, mode)
        base_sq = float(np.vdot(baseline, baseline).real)
        residual_sq = float(np.vdot(residual, residual).real)
        numerator += residual_sq
        denominator += base_sq
        rows.append({
            "axis": row["axis"],
            "sign": row["sign"],
            "relative_residual": float(np.sqrt(residual_sq / base_sq)),
        })
    return {
        "mode": mode,
        "combined_relative_residual": float(np.sqrt(numerator / denominator)),
        "rows": rows,
    }


def main() -> None:
    rank.lock_inertia()
    rows = [classify(mode) for mode in MODES]
    best = min(rows, key=lambda row: row["combined_relative_residual"])
    incumbent = next(row for row in rows if row["mode"] == "local")
    result = {
        "status": "V4_DENSITY_MOLD_SPATIAL_RESPONSE_SCREEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "common_action": (
            "T_chi=(B/2)[kappa|chiDot|^2+|A19 Vchi/24|^2], "
            "Vchi=chiDot+alpha response(rhoPsiDot) chi/"
            "[2(rhoChi+kappa chi0^2)]"),
        "alpha": local.ALPHA,
        "modes": rows,
        "best_mode": best["mode"],
        "best_relative_residual": best["combined_relative_residual"],
        "relative_improvement_over_local": float(
            (incumbent["combined_relative_residual"]
             - best["combined_relative_residual"])
            / incumbent["combined_relative_residual"]),
        "verdict": (
            "SPATIAL_DENSITY_RESPONSE_IMPROVES_HANDOFF" if
            best["mode"] != "local" else
            "LOCAL_DENSITY_RESPONSE_REMAINS_BEST"),
        "claim_boundary": (
            "This is a fixed-endpoint inverse response screen, not an action "
            "admission, freely moving orbit, or Dirac pass."),
    }
    (HERE / "density_mold_spatial_probe.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
