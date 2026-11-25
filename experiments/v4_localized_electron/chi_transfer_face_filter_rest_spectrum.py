"""Rest-spectrum gate for the six-face transfer-activated C19 filter.

The candidate is the positive kinetic square

  F = P_chi V + (1 + 19 a^2)^(-1/2) (I-P_chi) V
  K = A19(F)/24 + h(a) D19(F)/24
  h(a) = (a^2/6)/(1 + a^2/6).

The coefficient 1/6 is the equal-share normalization of the six face
transport channels.  This file checks rather than assumes that the candidate
leaves the complete two-layer rest Hessian unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import density_mold_connection_probe as local
import density_mold_rest_spectrum as rest
import density_mold_tensor_probe as tensor
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent
SOFTENING_FACTOR = 19.0


def c19_dirichlet(value: torch.Tensor) -> torch.Tensor:
    return value * rank.C19_WEIGHT_SUM - sum(
        weight * torch.roll(value, shifts=offset, dims=(0, 1, 2))
        for offset, weight in rank.native.DIRECTIONS)


def interval_action(left: torch.Tensor, right: torch.Tensor,
                    filter_factor: float) -> torch.Tensor:
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
    denominator = rho_chi + rank.KAPPA * rank.R**2
    material = chi_dot + local.ALPHA * response / (
        2 * denominator[..., None])
    overlap = torch.sum(torch.conj(chi) * material, dim=-1)
    longitudinal = chi * overlap[..., None] / denominator[..., None]
    transverse = material - longitudinal
    transfer = rho_dot / (
        2 * rank.R * (rho_psi + rank.KAPPA * rank.R**2))
    scale = torch.rsqrt(1 + SOFTENING_FACTOR * transfer**2)
    framed = longitudinal + scale[..., None] * transverse
    average = local.adjacency_average(framed)
    dirichlet = c19_dirichlet(framed) / rank.C19_WEIGHT_SUM
    activation = (filter_factor * transfer**2
                  / (1 + filter_factor * transfer**2))
    filtered = average + activation[..., None] * dirichlet
    kinetic = 0.5 * torch.sum(local.squared_norm(psi_dot))
    kinetic = kinetic + rank.B / 2 * (
        rank.KAPPA * torch.sum(local.squared_norm(chi_dot))
        + torch.sum(local.squared_norm(filtered)))
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
    rows = []
    for name, factor in (("unfiltered", 0.0),
                         ("six_face_equal_share", 1/6)):
        row = classify(factor)
        row["name"] = name
        row["filter_factor"] = factor
        rows.append(row)
    incumbent, candidate = rows
    growth_defect = abs(
        candidate["growth_rate_per_unit_time"]
        - incumbent["growth_rate_per_unit_time"])
    radius_defect = abs(
        candidate["spectral_radius"] - incumbent["spectral_radius"])
    passed = growth_defect < 1e-7 and radius_defect < 1e-10
    result = {
        "status": (
            "V4_CHI_FACE_FILTER_REST_GATE_PASS" if passed else
            "V4_CHI_FACE_FILTER_REST_GATE_FAIL"),
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": rest.SIDE,
        "dt": rest.DT,
        "rows": rows,
        "growth_rate_defect": float(growth_defect),
        "spectral_radius_defect": float(radius_defect),
        "coefficient_provenance": (
            "1/6 is the equal-share normalization of the six C19 face "
            "transport channels"),
        "verdict": (
            "ADVANCE_TO_FREE_ORBIT_GATE" if passed else
            "RETIRE_FILTER_AT_REST_GATE"),
        "claim_boundary": (
            "This verifies the side-three complete two-layer linear rest "
            "spectrum only; it is not nonlinear stability or movement."),
    }
    (HERE / "chi_transfer_face_filter_rest_spectrum.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
