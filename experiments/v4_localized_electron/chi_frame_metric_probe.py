"""Screen a positive χ-frame kinetic metric for edge transport.

The incumbent density-mold metric transports every component of the material
χ velocity through the same C19 average.  The saved multi-layer RCA shows that
the remaining defect is overwhelmingly χ and is dominated by its third
component.  The smallest field-native refinement is therefore to let the
local χ direction split its material velocity into longitudinal and
transverse pieces before the same C19 average:

  Tχ = B/2 [κ|χdot|² + |A19(Pχ Vχ + s (I-Pχ)Vχ)/24|²].

Pχ is the smooth Hermitian frame tensor χχ†/(ρχ+κχ0²).  The term is real,
positive, phase covariant and cubic covariant.  It contains no corner, copy,
direction, projector from Dirac algebra, or target force.  ``s=1`` is exactly
the incumbent tensor density-mold metric.  This script is an inverse response
screen only; a non-unit value is not admitted without native provenance and a
full rest-spectrum test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar

import density_mold_connection_probe as local
import density_mold_tensor_probe as tensor
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent
SIDE = 5


def interval_extra_action(left_raw: torch.Tensor, right_raw: torch.Tensor,
                          dt: float, transverse_scale: float) -> torch.Tensor:
    left_psi, left_chi = local.unpack_tensor(left_raw, SIDE)
    right_psi, right_chi = local.unpack_tensor(right_raw, SIDE)
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    psi_dot = (right_psi - left_psi) / dt
    chi_dot = (right_chi - left_chi) / dt

    rho_dot = 2 * torch.real(torch.sum(
        torch.conj(psi) * psi_dot, dim=-1))
    curvature = tensor.density_curvature(rho_dot)
    trace = torch.diagonal(curvature, dim1=-2, dim2=-1).sum(-1)
    identity = torch.eye(3, dtype=torch.float64)
    traceless = curvature - trace[..., None, None] * identity / 3
    response = (rho_dot[..., None] * chi
                + torch.einsum(
                    "...ab,...b->...a", traceless.to(torch.complex128), chi))
    rho_chi = torch.sum(local.squared_norm(chi), dim=-1)
    denominator = rho_chi + rank.KAPPA * rank.R**2
    material = chi_dot + local.ALPHA * response / (
        2 * denominator[..., None])

    # Smooth χ-native longitudinal tensor.  It is not a Dirac projector.
    overlap = torch.sum(torch.conj(chi) * material, dim=-1)
    longitudinal = chi * overlap[..., None] / denominator[..., None]
    transverse = material - longitudinal
    framed = longitudinal + transverse_scale * transverse
    averaged = local.adjacency_average(framed)

    candidate = rank.KAPPA * torch.sum(local.squared_norm(chi_dot))
    candidate = candidate + torch.sum(local.squared_norm(averaged))
    canonical = torch.sum(local.squared_norm(chi_dot))
    return dt * rank.B * (candidate - canonical) / 2


def action_delta(previous, current, following, dt: float,
                 transverse_scale: float) -> np.ndarray:
    raws = [torch.tensor(rank.pack(*layer), dtype=torch.float64,
                        requires_grad=True)
            for layer in (previous, current, following)]
    action = interval_extra_action(
        raws[0], raws[1], dt, transverse_scale)
    action = action + interval_extra_action(
        raws[1], raws[2], dt, transverse_scale)
    return torch.autograd.grad(action, raws[1])[0].detach().numpy()


def evaluate(transverse_scale: float, rows, psi, chi,
             force_psi, force_chi):
    numerator = 0.0
    denominator = 0.0
    direction_rows = []
    for row in rows:
        dt = row["best_dt"]
        layers = rank.translated_layers(row, psi, chi)
        baseline = rank.canonical_residual(
            *layers, dt, force_psi, force_chi)
        residual = baseline + action_delta(
            *layers, dt, transverse_scale)
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


def derivative_control(row, psi, chi, transverse_scale: float):
    dt = row["best_dt"]
    layers = rank.translated_layers(row, psi, chi)
    raws = [torch.tensor(rank.pack(*layer), dtype=torch.float64)
            for layer in layers]
    current = raws[1].clone().requires_grad_(True)

    def total(value):
        return (interval_extra_action(raws[0], value, dt, transverse_scale)
                + interval_extra_action(
                    value, raws[2], dt, transverse_scale))

    gradient = torch.autograd.grad(total(current), current)[0].detach().numpy()
    rng = np.random.default_rng(40402)
    direction = rng.normal(size=gradient.size)
    direction /= np.linalg.norm(direction)
    epsilon = 1e-5
    plus = current.detach() + epsilon * torch.tensor(
        direction, dtype=torch.float64)
    minus = current.detach() - epsilon * torch.tensor(
        direction, dtype=torch.float64)
    finite = float((total(plus) - total(minus)) / (2 * epsilon))
    analytic = float(np.dot(gradient, direction))
    return {
        "epsilon": epsilon,
        "analytic_directional_derivative": analytic,
        "finite_directional_derivative": finite,
        "relative_defect": float(
            abs(analytic-finite)/max(abs(analytic), abs(finite), 1.0)),
    }


def main() -> None:
    rank.lock_inertia()
    evidence = json.loads((HERE / "edge_snap_verlet_gate.json").read_text(
        encoding="utf-8"))
    rows = evidence["rows"]
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    force_psi, force_chi = rank.native.force(psi, chi)

    scan = []
    geometric_scales = {
        "zero": 0.0,
        "inverse_chi0": 1 / rank.R,
        "inverse_sqrt_54": 1 / np.sqrt(54.0),
        "inverse_sqrt_63": np.sqrt(rank.KAPPA),
        "unit": 1.0,
    }
    for scale in (-2.0, -1.0, 0.0, 1 / rank.R,
                  np.sqrt(rank.KAPPA), 1 / np.sqrt(54.0),
                  0.5, 1.0, 2.0, 3.0):
        residual, direction_rows = evaluate(
            scale, rows, psi, chi, force_psi, force_chi)
        scan.append({
            "transverse_scale": scale,
            "combined_relative_residual": residual,
            "rows": direction_rows,
        })
    optimum = minimize_scalar(
        lambda scale: evaluate(
            scale, rows, psi, chi, force_psi, force_chi)[0],
        bounds=(-4.0, 4.0), method="bounded",
        options={"xatol": 1e-10})
    optimum_rows = evaluate(
        float(optimum.x), rows, psi, chi, force_psi, force_chi)[1]
    incumbent = next(row for row in scan if row["transverse_scale"] == 1.0)
    result = {
        "status": "V4_CHI_FRAME_METRIC_RESPONSE_SCREEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action": (
            "T_chi=(B/2)[kappa|chiDot|^2+"
            "|A19(Pchi Vchi+s(I-Pchi)Vchi)/24|^2]"),
        "positive_and_invertible": True,
        "phase_covariant": True,
        "cubic_covariant": True,
        "rest_orbit_action_delta": 0.0,
        "scan": scan,
        "geometric_scales": {
            name: {
                "transverse_scale": float(scale),
                "combined_relative_residual": float(evaluate(
                    scale, rows, psi, chi, force_psi, force_chi)[0]),
            }
            for name, scale in geometric_scales.items()
        },
        "incumbent_s1_residual": incumbent["combined_relative_residual"],
        "diagnostic_optimum": {
            "transverse_scale": float(optimum.x),
            "combined_relative_residual": float(optimum.fun),
            "relative_improvement_over_s1": float(
                (incumbent["combined_relative_residual"] - optimum.fun)
                / incumbent["combined_relative_residual"]),
            "rows": optimum_rows,
            "coefficient_admitted": False,
        },
        "derivative_control": derivative_control(
            rows[0], psi, chi, float(optimum.x)),
        "verdict": (
            "FRAME_SPLIT_RESPONSE_IMPROVES" if optimum.fun <
            0.99 * incumbent["combined_relative_residual"] else
            "FRAME_SPLIT_RESPONSE_RETIRED"),
        "claim_boundary": (
            "The optimum is an inverse diagnostic, not coefficient "
            "provenance or action admission. A survivor still requires an "
            "exact rest spectrum, charge/current derivation and free orbit."),
    }
    (HERE / "chi_frame_metric_probe.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
