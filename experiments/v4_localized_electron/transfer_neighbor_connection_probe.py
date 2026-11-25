"""Screen a transfer-driven C19 neighbour connection for chi.

The incumbent positive kinetic metric reacts to local Psi density transfer,
but the freely optimized handoff still fails mostly in chi.  This probe puts
the smallest C19 neighbour curvature inside the same positive kinetic square:

  K_i = chi_i - A19(chi)_i / 24
  a_i = rhoPsiDot_i / [2 chi0 (rhoPsi_i + kappa chi0^2)]
  V_i(c) = V_i(0) + c chi0 a_i K_i.

The connection is zero on the exact rest orbit because a_i=0.  It uses no
corner, copy, axis, shift, or Dirac object.  Full, chi-radial, and chi-normal
parts of K are screened independently.  Fixed coefficients in the scan are
native normalization/counting values only; the diagnostic optimum is not an
admitted coefficient.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import chi_transfer_softening_probe as incumbent
import density_mold_connection_probe as local
import density_mold_tensor_probe as tensor
import edge_snap_action_rank as rank
import edge_snap_residual_action_projection as projection


HERE = Path(__file__).resolve().parent
SIDE = 5
DT = 0.1
INPUT_STEM = "transfer_activated_chi_bond_four_update_orbit"
CHANNELS = ("full", "radial", "normal", "frame_reflection")
COEFFICIENTS = (
    -1.0,
    -0.5,
    -1.0 / 6.0,
    -1.0 / 19.0,
    -1.0 / 38.0,
    0.0,
    1.0 / 38.0,
    1.0 / 19.0,
    1.0 / 6.0,
    0.5,
    1.0,
)


def frame_velocity(chi: torch.Tensor, velocity: torch.Tensor,
                   transfer: torch.Tensor) -> torch.Tensor:
    """Apply the incumbent g=19 longitudinal/transverse chi frame."""
    rho_chi = torch.sum(local.squared_norm(chi), dim=-1)
    denominator = rho_chi + rank.KAPPA * rank.R**2
    overlap = torch.sum(torch.conj(chi) * velocity, dim=-1)
    radial = chi * overlap[..., None] / denominator[..., None]
    normal = velocity - radial
    scale = torch.rsqrt(1 + 19.0 * transfer**2)
    return radial + scale[..., None] * normal


def connection_channel(chi: torch.Tensor, channel: str) -> torch.Tensor:
    curvature = chi - local.adjacency_average(chi)
    rho_chi = torch.sum(local.squared_norm(chi), dim=-1)
    denominator = rho_chi + rank.KAPPA * rank.R**2
    overlap = torch.sum(torch.conj(chi) * curvature, dim=-1)
    radial = chi * overlap[..., None] / denominator[..., None]
    if channel == "full":
        return curvature
    if channel == "radial":
        return radial
    if channel == "normal":
        return curvature - radial
    if channel == "frame_reflection":
        return 2 * radial - curvature
    raise ValueError(channel)


def interval_action_delta(left_raw: torch.Tensor, right_raw: torch.Tensor,
                          dt: float, coefficient: float,
                          channel: str) -> torch.Tensor:
    side = round((left_raw.numel() / 12) ** (1 / 3))
    left_psi, left_chi = local.unpack_tensor(left_raw, side)
    right_psi, right_chi = local.unpack_tensor(right_raw, side)
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    psi_dot = (right_psi - left_psi) / dt
    chi_dot = (right_chi - left_chi) / dt

    rho_dot = 2 * torch.real(torch.sum(
        torch.conj(psi) * psi_dot, dim=-1))
    rho_psi = torch.sum(local.squared_norm(psi), dim=-1)
    transfer = rho_dot / (
        2 * rank.R * (rho_psi + rank.KAPPA * rank.R**2))

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

    added = (coefficient * rank.R * transfer[..., None]
             * connection_channel(chi, channel))
    old_average = local.adjacency_average(
        frame_velocity(chi, material, transfer))
    new_average = local.adjacency_average(
        frame_velocity(chi, material + added, transfer))
    delta = (torch.sum(local.squared_norm(new_average))
             - torch.sum(local.squared_norm(old_average)))
    return dt * rank.B * delta / 2


def action_delta(triple, coefficient: float, channel: str) -> np.ndarray:
    raws = [torch.tensor(rank.pack(*layer), dtype=torch.float64,
                        requires_grad=True) for layer in triple]
    action = interval_action_delta(
        raws[0], raws[1], DT, coefficient, channel)
    action = action + interval_action_delta(
        raws[1], raws[2], DT, coefficient, channel)
    return torch.autograd.grad(action, raws[1])[0].detach().numpy()


def movement_scan() -> tuple[list[dict], dict, list[dict]]:
    triples, data = projection.layers_from_artifact(INPUT_STEM)
    target = -data["residuals"].reshape(-1)
    target_norm = np.linalg.norm(target)
    rows = []
    tangent_rows = []
    for channel in CHANNELS:
        for coefficient in COEFFICIENTS:
            column = np.concatenate([
                action_delta(triple, coefficient, channel)
                for triple in triples
            ])
            corrected = target - column
            relative = float(np.linalg.norm(corrected) / target_norm)
            rows.append({
                "channel": channel,
                "coefficient": coefficient,
                "column_norm": float(np.linalg.norm(column)),
                "relative_residual": relative,
                "relative_improvement": float(1 - relative),
            })
        epsilon = 1e-5
        tangent = np.concatenate([
            action_delta(triple, epsilon, channel) / epsilon
            for triple in triples
        ])
        optimum = float(np.dot(tangent, target) / np.dot(tangent, tangent))
        tangent_relative = float(
            np.linalg.norm(optimum * tangent - target) / target_norm)
        tangent_rows.append({
            "channel": channel,
            "linearized_optimal_coefficient_diagnostic_only": optimum,
            "correlation_with_required_response": float(
                np.dot(tangent, target)
                / (np.linalg.norm(tangent) * target_norm)),
            "relative_residual": tangent_relative,
            "explained_squared_fraction": float(1 - tangent_relative**2),
        })
    best = min(rows, key=lambda row: row["relative_residual"])
    return rows, best, tangent_rows


def rest_control() -> dict:
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    theta = float(np.arccos(1 - DT**2 * rank.R**2 / 2))
    triple = (
        (np.exp(-1j * theta) * psi, chi),
        (psi, chi),
        (np.exp(1j * theta) * psi, chi),
    )
    rows = []
    for channel in CHANNELS:
        column = action_delta(triple, 1.0, channel)
        rows.append({
            "channel": channel,
            "unit_coefficient_DEL_norm": float(np.linalg.norm(column)),
        })
    return {
        "analytic_reason": "rhoPsiDot=0 on the exact rest orbit, hence a=0",
        "rows": rows,
    }


def main() -> None:
    rank.lock_inertia()
    scan, best, tangent = movement_scan()
    rest = rest_control()
    improves = best["relative_improvement"] > 1e-6
    result = {
        "status": "V4_TRANSFER_NEIGHBOR_CONNECTION_RESPONSE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "input_orbit": INPUT_STEM,
        "action_candidate": (
            "replace Vchi by Vchi+c*chi0*a*Kchi inside the positive incumbent "
            "C19-average kinetic square; Kchi=chi-A19(chi)/24, with full, "
            "chi-radial, and chi-normal channels screened separately"
        ),
        "real_positive_invertible_kinetic_square": True,
        "reciprocal_by_single_discrete_action": True,
        "cubic_C19_and_phase_covariant": True,
        "contains_corner_copy_axis_shift_or_Dirac_label": False,
        "rest_control": rest,
        "movement_scan": scan,
        "linearized_response_diagnostic": tangent,
        "best_fixed_native_coefficient": best,
        "verdict": (
            "ADVANCE_TRANSFER_NEIGHBOR_CONNECTION" if improves else
            "RETIRE_TRANSFER_NEIGHBOR_CONNECTION"
        ),
        "claim_boundary": (
            "This is a response screen on one freely optimized handoff. "
            "A favorable fixed coefficient must still pass the exact-rest "
            "spectrum, charge, free-orbit, held-out direction, and Dirac gates."
        ),
    }
    (HERE / "transfer_neighbor_connection_probe.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
