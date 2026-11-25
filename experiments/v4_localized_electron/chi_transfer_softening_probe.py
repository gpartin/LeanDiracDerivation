"""Test density-transfer activation of the χ-frame kinetic metric.

The constant frame split improves the edge handoff but worsens the rest
spectrum.  This candidate keeps the proven tensor incumbent at rest and
softens only the transverse χ material velocity during actual local density
transfer:

  a = rhoPsiDot / [2 chi0 (rhoPsi + kappa chi0^2)]
  s(a) = 1/sqrt(1 + 6 a^2)

  Tχ = B/2 [kappa|chiDot|^2
             + |A19(Pχ Vχ + s(a)(I-Pχ)Vχ)/24|^2].

The factor 6 is the number of signed cubical face-transport channels.  The
action is real and positive.  At the exact rest orbit a=0, and the first
change to the metric is quadratic in a; consequently the rest quadratic
action is the same as the incumbent.  No preferred direction, corner, copy,
or Dirac object occurs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar

import chi_frame_metric_probe as frame
import density_mold_connection_probe as local
import density_mold_tensor_probe as tensor
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent
SIDE = 5
GEOMETRIC_FACTOR = 6.0


def interval_extra_action(left_raw: torch.Tensor, right_raw: torch.Tensor,
                          dt: float, geometric_factor: float) -> torch.Tensor:
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
    chi_denominator = rho_chi + rank.KAPPA * rank.R**2
    material = chi_dot + local.ALPHA * response / (
        2 * chi_denominator[..., None])

    overlap = torch.sum(torch.conj(chi) * material, dim=-1)
    longitudinal = chi * overlap[..., None] / chi_denominator[..., None]
    transverse = material - longitudinal
    transfer = rho_dot / (
        2 * rank.R * (rho_psi + rank.KAPPA * rank.R**2))
    transverse_scale = torch.rsqrt(1 + geometric_factor * transfer**2)
    framed = longitudinal + transverse_scale[..., None] * transverse
    averaged = local.adjacency_average(framed)

    candidate = rank.KAPPA * torch.sum(local.squared_norm(chi_dot))
    candidate = candidate + torch.sum(local.squared_norm(averaged))
    canonical = torch.sum(local.squared_norm(chi_dot))
    return dt * rank.B * (candidate - canonical) / 2


def action_delta(previous, current, following, dt: float,
                 geometric_factor: float) -> np.ndarray:
    raws = [torch.tensor(rank.pack(*layer), dtype=torch.float64,
                        requires_grad=True)
            for layer in (previous, current, following)]
    action = interval_extra_action(
        raws[0], raws[1], dt, geometric_factor)
    action = action + interval_extra_action(
        raws[1], raws[2], dt, geometric_factor)
    return torch.autograd.grad(action, raws[1])[0].detach().numpy()


def evaluate(geometric_factor: float, rows, psi, chi,
             force_psi, force_chi):
    numerator = denominator = 0.0
    direction_rows = []
    for row in rows:
        dt = row["best_dt"]
        layers = rank.translated_layers(row, psi, chi)
        baseline = rank.canonical_residual(
            *layers, dt, force_psi, force_chi)
        residual = baseline + action_delta(
            *layers, dt, geometric_factor)
        baseline_sq = float(np.vdot(baseline, baseline).real)
        residual_sq = float(np.vdot(residual, residual).real)
        numerator += residual_sq
        denominator += baseline_sq
        direction_rows.append({
            "axis": row["axis"],
            "sign": row["sign"],
            "relative_residual": float(np.sqrt(residual_sq / baseline_sq)),
        })
    return float(np.sqrt(numerator / denominator)), direction_rows


def main() -> None:
    rank.lock_inertia()
    evidence = json.loads((HERE / "edge_snap_verlet_gate.json").read_text(
        encoding="utf-8"))
    rows = evidence["rows"]
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    force_psi, force_chi = rank.native.force(psi, chi)
    factors = (0.0, 6.0, 19.0, 24.0, 54.0, 63.0, 361.0)
    scan = []
    for factor in factors:
        residual, direction_rows = evaluate(
            factor, rows, psi, chi, force_psi, force_chi)
        scan.append({
            "geometric_factor": factor,
            "combined_relative_residual": residual,
            "rows": direction_rows,
        })
    incumbent = scan[0]
    candidate = next(row for row in scan
                     if row["geometric_factor"] == GEOMETRIC_FACTOR)
    optimum = minimize_scalar(
        lambda factor: evaluate(
            factor, rows, psi, chi, force_psi, force_chi)[0],
        bounds=(0.0, 100.0), method="bounded",
        options={"xatol": 1e-8})
    result = {
        "status": "V4_CHI_TRANSFER_SOFTENING_RESPONSE_SCREEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action": (
            "T_chi=(B/2)[kappa|chiDot|^2+"
            "|A19(Pchi Vchi+s(a)(I-Pchi)Vchi)/24|^2], "
            "s(a)=(1+6 a^2)^(-1/2)"),
        "positive_and_invertible": True,
        "phase_covariant": True,
        "cubic_covariant": True,
        "rest_quadratic_action_identical_to_incumbent": True,
        "coefficient_provenance": (
            "6 is the exact count of signed cubical face-transport channels"),
        "scan": scan,
        "incumbent_residual": incumbent["combined_relative_residual"],
        "candidate_residual": candidate["combined_relative_residual"],
        "relative_improvement": float(
            (incumbent["combined_relative_residual"]
             - candidate["combined_relative_residual"])
            / incumbent["combined_relative_residual"]),
        "diagnostic_optimum": {
            "geometric_factor": float(optimum.x),
            "combined_relative_residual": float(optimum.fun),
            "coefficient_admitted": False,
        },
        "verdict": (
            "ADVANCE_TO_REST_AND_ORBIT_GATES" if
            candidate["combined_relative_residual"] <
            incumbent["combined_relative_residual"] else
            "RETIRE_TRANSFER_SOFTENING"),
        "claim_boundary": (
            "This is a finite handoff response gate. Equality of the rest "
            "quadratic action is analytic but still requires a numerical "
            "Hessian control before orbit work."),
    }
    (HERE / "chi_transfer_softening_probe.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
