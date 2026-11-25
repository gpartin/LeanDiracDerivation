"""Screen transfer softening in the complete native Psi--chi frame.

The incumbent metric splits chi velocity only into chi-longitudinal and
chi-transverse pieces.  The covariant free-orbit RCA shows that the residual
distinguishes the Psi-in-plane and remaining orientation-normal directions.
This probe derives that split from the live local fields themselves:

  e_chi  ~ chi,
  e_plane ~ Psi - P_chi Psi,
  normal  = I - P_chi - P_plane.

All denominators use the existing kappa chi0^2 regularizer.  These are native
field-frame operators, not prepared corner or spinor projectors.  The action
is a sum of positive kinetic squares and all scales become one at rest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
import torch

import density_mold_connection_probe as local
import density_mold_tensor_probe as tensor
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent
SIDE = 5
PLANE_FACTOR = 19.0


def interval_extra_action(left_raw: torch.Tensor, right_raw: torch.Tensor,
                          dt: float, normal_factor: float) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left_raw, SIDE)
    right_psi, right_chi = local.unpack_tensor(right_raw, SIDE)
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    psi_dot = (right_psi - left_psi) / dt
    chi_dot = (right_chi - left_chi) / dt
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

    transfer = rho_dot / (
        2 * rank.R * (rho_psi + regulator))
    plane_scale = torch.rsqrt(1 + PLANE_FACTOR * transfer**2)
    normal_scale = torch.rsqrt(1 + normal_factor * transfer**2)
    framed = (longitudinal + plane_scale[..., None] * plane
              + normal_scale[..., None] * normal)
    averaged = local.adjacency_average(framed)
    candidate = rank.KAPPA * torch.sum(local.squared_norm(chi_dot))
    candidate = candidate + torch.sum(local.squared_norm(averaged))
    canonical = torch.sum(local.squared_norm(chi_dot))
    return dt * rank.B * (candidate - canonical) / 2


def action_delta(previous, current, following, dt: float,
                 normal_factor: float) -> np.ndarray:
    raws = [torch.tensor(rank.pack(*layer), dtype=torch.float64,
                        requires_grad=True)
            for layer in (previous, current, following)]
    action = interval_extra_action(raws[0], raws[1], dt, normal_factor)
    action = action + interval_extra_action(raws[1], raws[2], dt,
                                            normal_factor)
    return torch.autograd.grad(action, raws[1])[0].detach().numpy()


def evaluate(factor: float, rows, psi, chi, force_psi, force_chi):
    numerator = denominator = 0.0
    direction_rows = []
    for row in rows:
        dt = row["best_dt"]
        layers = rank.translated_layers(row, psi, chi)
        baseline = rank.canonical_residual(
            *layers, dt, force_psi, force_chi)
        residual = baseline + action_delta(*layers, dt, factor)
        base_sq = float(np.vdot(baseline, baseline).real)
        residual_sq = float(np.vdot(residual, residual).real)
        numerator += residual_sq
        denominator += base_sq
        direction_rows.append({
            "axis": row["axis"], "sign": row["sign"],
            "relative_residual": float(np.sqrt(residual_sq / base_sq)),
        })
    return float(np.sqrt(numerator / denominator)), direction_rows


def main() -> None:
    rank.lock_inertia()
    rows = json.loads((HERE / "edge_snap_verlet_gate.json").read_text(
        encoding="utf-8"))["rows"]
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    force_psi, force_chi = rank.native.force(psi, chi)
    factors = (0.0, 1/6, 1.0, 6.0, 19.0, 63.0, 1197.0)
    scan = []
    for factor in factors:
        residual, direction_rows = evaluate(
            factor, rows, psi, chi, force_psi, force_chi)
        scan.append({"normal_factor": factor,
                     "combined_relative_residual": residual,
                     "rows": direction_rows})
    optimum = minimize_scalar(
        lambda factor: evaluate(
            factor, rows, psi, chi, force_psi, force_chi)[0],
        bounds=(0.0, 2000.0), method="bounded",
        options={"xatol": 1e-8})
    incumbent = next(row for row in scan if row["normal_factor"] == 19.0)
    best = min((row for row in scan if row is not incumbent),
               key=lambda row: row["combined_relative_residual"])
    improvement = ((incumbent["combined_relative_residual"]
                    - best["combined_relative_residual"])
                   / incumbent["combined_relative_residual"])
    result = {
        "status": "V4_CHI_NATIVE_FRAME_TRANSFER_SCREEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "strictly_positive_metric": True,
        "phase_covariant": True,
        "cubic_covariant": True,
        "native_frame_only": True,
        "plane_factor": PLANE_FACTOR,
        "scan": scan,
        "incumbent_equal_transverse_factor": incumbent,
        "best_native": best,
        "diagnostic_optimum": {
            "normal_factor": float(optimum.x),
            "combined_relative_residual": float(optimum.fun),
            "admitted": False},
        "relative_improvement_over_equal_transverse": float(improvement),
        "verdict": (
            "ADVANCE_NATIVE_FRAME_SPLIT" if improvement > 0 else
            "RETIRE_NATIVE_FRAME_SPLIT"),
        "claim_boundary": (
            "Frozen six-direction response only; not an admitted action, "
            "rest-spectrum pass, or free moving orbit."),
    }
    (HERE / "chi_transfer_native_frame_probe.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
