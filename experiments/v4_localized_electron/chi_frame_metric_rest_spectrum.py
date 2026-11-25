"""Full two-layer rest spectrum of the χ-frame kinetic metric."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import chi_frame_metric_probe as frame
import density_mold_connection_probe as local
import density_mold_rest_spectrum as rest
import density_mold_tensor_probe as tensor
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent


def frame_interval_action(left: torch.Tensor, right: torch.Tensor,
                          transverse_scale: float) -> torch.Tensor:
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
    material = chi_dot + local.ALPHA * response / (
        2 * denominator[..., None])
    overlap = torch.sum(torch.conj(chi) * material, dim=-1)
    longitudinal = chi * overlap[..., None] / denominator[..., None]
    transverse = material - longitudinal
    framed = longitudinal + transverse_scale * transverse
    averaged = local.adjacency_average(framed)

    kinetic = 0.5 * torch.sum(local.squared_norm(psi_dot))
    kinetic = kinetic + rank.B / 2 * (
        rank.KAPPA * torch.sum(local.squared_norm(chi_dot))
        + torch.sum(local.squared_norm(averaged)))
    return rest.DT * (
        kinetic - (rest.potential(left) + rest.potential(right)) / 2)


def classify(transverse_scale: float):
    original = rest.interval_action
    rest.interval_action = frame_interval_action
    try:
        # The classify argument is passed through as the frame scale.
        return rest.classify(transverse_scale)
    finally:
        rest.interval_action = original


def main() -> None:
    rank.lock_inertia()
    scales = {
        "zero": 0.0,
        "inverse_sqrt_63": np.sqrt(rank.KAPPA),
        "inverse_sqrt_54": 1 / np.sqrt(54.0),
        "unit_tensor_incumbent": 1.0,
    }
    rows = []
    for name, scale in scales.items():
        row = classify(scale)
        row["name"] = name
        row["transverse_scale"] = float(scale)
        rows.append(row)
    incumbent = next(
        row for row in rows if row["name"] == "unit_tensor_incumbent")
    candidate = next(row for row in rows if row["name"] == "inverse_sqrt_54")
    well_posed = all(row["growth_rate_per_unit_time"] is not None
                     for row in rows)
    improvement = (float(
        (incumbent["growth_rate_per_unit_time"]
         - candidate["growth_rate_per_unit_time"])
        / incumbent["growth_rate_per_unit_time"])
        if well_posed else None)
    result = {
        "status": (
            "V4_CHI_FRAME_METRIC_REST_GATE_PASS"
            if well_posed and improvement >= 0 else
            "V4_CHI_FRAME_METRIC_REST_GATE_FAIL"),
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": rest.SIDE,
        "dt": rest.DT,
        "action": (
            "T_chi=(B/2)[kappa|chiDot|^2+"
            "|A19(Pchi Vchi+s(I-Pchi)Vchi)/24|^2]"),
        "coefficient_provenance_tested": "s=1/sqrt(54)",
        "rows": rows,
        "relative_growth_improvement_over_s1": improvement,
        "verdict": (
            "ADVANCE_TO_FREE_ORBIT_TEST"
            if well_posed and improvement >= 0 else
            "RETIRE_OR_REPAIR_FRAME_METRIC"),
        "claim_boundary": (
            "This is the complete side-three two-layer linear spectrum. It "
            "does not establish nonlinear stability, movement, or Dirac."),
    }
    (HERE / "chi_frame_metric_rest_spectrum.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
