"""Rest-spectrum gate for the native three-channel Psi--chi frame metric."""

from __future__ import annotations

import json
from pathlib import Path

import torch

import density_mold_connection_probe as local
import density_mold_rest_spectrum as rest
import density_mold_tensor_probe as tensor
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent
PLANE_FACTOR = 19.0


def interval_action(left: torch.Tensor, right: torch.Tensor,
                    normal_factor: float) -> torch.Tensor:
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
    regulator = rank.KAPPA * rank.R**2
    chi_denominator = rho_chi + regulator
    material = chi_dot + local.ALPHA * response / (
        2 * chi_denominator[..., None])
    chi_overlap = torch.sum(torch.conj(chi) * material, dim=-1)
    longitudinal = chi * chi_overlap[..., None] / chi_denominator[..., None]
    transverse = material - longitudinal
    chi_psi_overlap = torch.sum(torch.conj(chi) * psi, dim=-1)
    psi_orthogonal = (psi - chi * chi_psi_overlap[..., None]
                      / chi_denominator[..., None])
    rho_plane = torch.sum(local.squared_norm(psi_orthogonal), dim=-1)
    plane_denominator = rho_plane + regulator
    plane_overlap = torch.sum(
        torch.conj(psi_orthogonal) * transverse, dim=-1)
    plane = (psi_orthogonal * plane_overlap[..., None]
             / plane_denominator[..., None])
    normal = transverse - plane
    transfer = rho_dot / (2 * rank.R * (rho_psi + regulator))
    plane_scale = torch.rsqrt(1 + PLANE_FACTOR * transfer**2)
    normal_scale = torch.rsqrt(1 + normal_factor * transfer**2)
    framed = (longitudinal + plane_scale[..., None] * plane
              + normal_scale[..., None] * normal)
    average = local.adjacency_average(framed)
    kinetic = 0.5 * torch.sum(local.squared_norm(psi_dot))
    kinetic = kinetic + rank.B / 2 * (
        rank.KAPPA * torch.sum(local.squared_norm(chi_dot))
        + torch.sum(local.squared_norm(average)))
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
    for name, factor in (("equal_transverse", 19.0),
                         ("six_face_normal", 6.0)):
        row = classify(factor)
        row["name"] = name
        row["normal_factor"] = factor
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
            "V4_CHI_NATIVE_FRAME_REST_GATE_PASS" if passed else
            "V4_CHI_NATIVE_FRAME_REST_GATE_FAIL"),
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": rest.SIDE,
        "dt": rest.DT,
        "rows": rows,
        "growth_rate_defect": float(growth_defect),
        "spectral_radius_defect": float(radius_defect),
        "coefficient_provenance": (
            "normal factor 6 from the six C19 face transport directions"),
        "verdict": (
            "ADVANCE_TO_FREE_ORBIT_GATE" if passed else
            "RETIRE_NATIVE_FRAME_AT_REST_GATE"),
        "claim_boundary": (
            "Complete side-three two-layer rest-spectrum gate only; not "
            "nonlinear stability or movement."),
    }
    (HERE / "chi_transfer_native_frame_rest_spectrum.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
