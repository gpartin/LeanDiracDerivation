"""Screen a transfer-activated native bond-frame transporter for chi.

The edge-snap hypothesis requires a receiving cube to interpret the incoming
substrate velocity in its own live Psi--chi frame.  Raw C19 averaging compares
global components instead.  For each bond i<-j this probe constructs the
phase- and cubic-covariant native frame map

  M_ij v = chi_i (chi_j^dag v)/Nchi_ij
         + Psi_i (Psi_j^dag v)/Npsi_ij
         + W_i   (W_j^dag v)/Nw_ij,   W=chi x Psi.

The kinetic square blends raw and transported neighbor velocities only while
Psi density is crossing the site.  At rest the blend is exactly raw C19
averaging.  The action remains a positive square with the native kappa floor;
no carrier, corner, direction, copy, or Dirac object occurs in it.
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
TRANSVERSE_FACTOR = 19.0


def squared_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.sum(value.real**2 + value.imag**2, dim=-1)


def neighbor(value: torch.Tensor, offset) -> torch.Tensor:
    return torch.roll(value, shifts=offset, dims=(0, 1, 2))


def transported_average(psi: torch.Tensor, chi: torch.Tensor,
                        velocity: torch.Tensor) -> torch.Tensor:
    regulator = rank.KAPPA * rank.R**2
    rho_psi = squared_norm(psi)
    rho_chi = squared_norm(chi)
    area = torch.linalg.cross(chi, psi, dim=-1)
    rho_area = squared_norm(area)
    result = torch.zeros_like(velocity)
    for offset, weight in rank.native.DIRECTIONS:
        psi_j = neighbor(psi, offset)
        chi_j = neighbor(chi, offset)
        area_j = neighbor(area, offset)
        velocity_j = neighbor(velocity, offset)
        dchi = torch.sqrt((rho_chi + regulator)
                          * (neighbor(rho_chi, offset) + regulator))
        dpsi = torch.sqrt((rho_psi + regulator)
                          * (neighbor(rho_psi, offset) + regulator))
        darea = torch.sqrt((rho_area + regulator**2)
                           * (neighbor(rho_area, offset) + regulator**2))
        chi_part = chi * torch.sum(
            torch.conj(chi_j) * velocity_j, dim=-1)[..., None] / dchi[..., None]
        psi_part = psi * torch.sum(
            torch.conj(psi_j) * velocity_j, dim=-1)[..., None] / dpsi[..., None]
        area_part = area * torch.sum(
            torch.conj(area_j) * velocity_j, dim=-1)[..., None] / darea[..., None]
        result = result + weight * (chi_part + psi_part + area_part)
    return result / rank.C19_WEIGHT_SUM


def interval_extra_action(left_raw: torch.Tensor, right_raw: torch.Tensor,
                          dt: float, factor: float) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left_raw, SIDE)
    right_psi, right_chi = local.unpack_tensor(right_raw, SIDE)
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    psi_dot = (right_psi - left_psi) / dt
    chi_dot = (right_chi - left_chi) / dt
    rho_dot = 2 * torch.real(torch.sum(
        torch.conj(psi) * psi_dot, dim=-1))
    rho_psi = squared_norm(psi)
    curvature = tensor.density_curvature(rho_dot)
    trace = torch.diagonal(curvature, dim1=-2, dim2=-1).sum(-1)
    identity = torch.eye(3, dtype=torch.float64)
    traceless = (curvature - trace[..., None, None] * identity / 3).to(
        torch.complex128)
    response = (rho_dot[..., None] * chi
                + torch.einsum("...ab,...b->...a", traceless, chi))
    rho_chi = squared_norm(chi)
    regulator = rank.KAPPA * rank.R**2
    denominator = rho_chi + regulator
    material = chi_dot + local.ALPHA * response / (
        2 * denominator[..., None])
    overlap = torch.sum(torch.conj(chi) * material, dim=-1)
    longitudinal = chi * overlap[..., None] / denominator[..., None]
    transverse = material - longitudinal
    transfer = rho_dot / (2 * rank.R * (rho_psi + regulator))
    transverse_scale = torch.rsqrt(
        1 + TRANSVERSE_FACTOR * transfer**2)
    framed = longitudinal + transverse_scale[..., None] * transverse
    raw_average = local.adjacency_average(framed)
    frame_average = transported_average(psi, chi, framed)
    activation = factor * transfer**2 / (1 + factor * transfer**2)
    blended = ((1 - activation)[..., None] * raw_average
               + activation[..., None] * frame_average)
    candidate = rank.KAPPA * torch.sum(squared_norm(chi_dot))
    candidate = candidate + torch.sum(squared_norm(blended))
    canonical = torch.sum(squared_norm(chi_dot))
    return dt * rank.B * (candidate - canonical) / 2


def action_delta(previous, current, following, dt: float,
                 factor: float) -> np.ndarray:
    raws = [torch.tensor(rank.pack(*layer), dtype=torch.float64,
                        requires_grad=True)
            for layer in (previous, current, following)]
    action = interval_extra_action(raws[0], raws[1], dt, factor)
    action = action + interval_extra_action(raws[1], raws[2], dt, factor)
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
        scan.append({"factor": factor,
                     "combined_relative_residual": residual,
                     "rows": direction_rows})
    optimum = minimize_scalar(
        lambda factor: evaluate(
            factor, rows, psi, chi, force_psi, force_chi)[0],
        bounds=(0.0, 2000.0), method="bounded",
        options={"xatol": 1e-8})
    incumbent = scan[0]
    best = min(scan[1:], key=lambda row: row["combined_relative_residual"])
    improvement = ((incumbent["combined_relative_residual"]
                    - best["combined_relative_residual"])
                   / incumbent["combined_relative_residual"])
    result = {
        "status": "V4_CHI_BOND_FRAME_TRANSPORT_SCREEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "strictly_positive_metric": True,
        "phase_covariant": True,
        "cubic_covariant": True,
        "native_bond_frame_only": True,
        "scan": scan,
        "best_native": best,
        "diagnostic_optimum": {
            "factor": float(optimum.x),
            "combined_relative_residual": float(optimum.fun),
            "admitted": False},
        "relative_improvement_over_incumbent": float(improvement),
        "verdict": (
            "ADVANCE_BOND_FRAME_TRANSPORT" if improvement > 0 else
            "RETIRE_BOND_FRAME_TRANSPORT"),
        "claim_boundary": (
            "Frozen six-direction response only; not an admitted action, "
            "rest-spectrum pass, or free moving orbit."),
    }
    (HERE / "chi_bond_frame_transport_probe.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
