"""Screen a positive density-driven C19 chi kinetic connection.

The literal edge-snap residual is almost entirely in chi and is dominated by
the substrate component normal to the prepared Psi plane.  The smallest
native way for Psi to tell chi that density is arriving or leaving is the
time derivative of the local density.  This probe therefore tests

  T_chi = B/2 [kappa |chi_dot|^2 + |A19 V_chi / 24|^2],

  V_chi = chi_dot
          + s rhoPsi_dot chi / (2 (rhoChi + kappa chi0^2)).

The square is real and nonnegative, uses only native fields and C19
neighbors, and generates reciprocal Psi and chi terms by variation.  It is
evaluated here through automatic differentiation of the discrete action;
the edge-snap target never enters the action.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
import torch

import c19_kinetic_metric_probe as metric_probe
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent
SIDE = 5
ALPHA = 1.0 + 54.0 / rank.R**2


def squared_norm(value: torch.Tensor) -> torch.Tensor:
    """Smooth real quadratic norm, including at an exactly zero component."""
    return torch.real(torch.conj(value) * value)


def unpack_tensor(raw: torch.Tensor, side: int):
    data = raw.reshape(side, side, side, 4, 3)
    psi = torch.complex(data[..., 0, :], data[..., 1, :])
    chi = torch.complex(data[..., 2, :], data[..., 3, :])
    return psi, chi


def adjacency_average(value: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(value)
    for offset, weight in rank.native.DIRECTIONS:
        result = result + weight * torch.roll(
            value, shifts=offset, dims=(0, 1, 2))
    return result / rank.C19_WEIGHT_SUM


def interval_extra_action(left_raw: torch.Tensor, right_raw: torch.Tensor,
                          dt: float, coefficient: float) -> torch.Tensor:
    """Discrete kinetic-action delta relative to canonical local chi inertia."""
    left_psi, left_chi = unpack_tensor(left_raw, SIDE)
    right_psi, right_chi = unpack_tensor(right_raw, SIDE)
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    psi_dot = (right_psi - left_psi) / dt
    chi_dot = (right_chi - left_chi) / dt
    rho_psi_dot = 2 * torch.real(torch.sum(
        torch.conj(psi) * psi_dot, dim=-1))
    rho_chi = torch.real(torch.sum(torch.conj(chi) * chi, dim=-1))
    denominator = rho_chi + rank.KAPPA * rank.R**2
    mold_velocity = (
        chi_dot
        + coefficient * rho_psi_dot[..., None] * chi
        / (2 * denominator[..., None]))
    averaged = adjacency_average(mold_velocity)
    candidate = rank.KAPPA * torch.sum(squared_norm(chi_dot))
    candidate = candidate + torch.sum(squared_norm(averaged))
    canonical = torch.sum(squared_norm(chi_dot))
    return dt * rank.B * (candidate - canonical) / 2


def action_delta(previous, current, following, dt: float,
                 coefficient: float) -> np.ndarray:
    raws = [torch.tensor(rank.pack(*layer), dtype=torch.float64,
                          requires_grad=True)
            for layer in (previous, current, following)]
    action = interval_extra_action(raws[0], raws[1], dt, coefficient)
    action = action + interval_extra_action(
        raws[1], raws[2], dt, coefficient)
    gradient = torch.autograd.grad(action, raws[1])[0]
    return gradient.detach().numpy()


def directional_residuals(coefficient: float):
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
        residual = baseline + action_delta(
            *layers, dt, coefficient)
        base_norm_sq = float(np.vdot(baseline, baseline).real)
        residual_norm_sq = float(np.vdot(residual, residual).real)
        numerator += residual_norm_sq
        denominator += base_norm_sq
        rows.append({
            "axis": row["axis"],
            "sign": row["sign"],
            "relative_residual": float(
                np.sqrt(residual_norm_sq / base_norm_sq)),
        })
    return float(np.sqrt(numerator / denominator)), rows


def derivative_control() -> float:
    """Compare autograd with a centered directional derivative."""
    evidence = json.loads(
        (HERE / "edge_snap_verlet_gate.json").read_text(encoding="utf-8"))
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    layers = rank.translated_layers(evidence["rows"][0], psi, chi)
    dt = evidence["rows"][0]["best_dt"]
    previous = torch.tensor(rank.pack(*layers[0]), dtype=torch.float64)
    current = torch.tensor(rank.pack(*layers[1]), dtype=torch.float64,
                           requires_grad=True)
    following = torch.tensor(rank.pack(*layers[2]), dtype=torch.float64)
    action = interval_extra_action(previous, current, dt, ALPHA)
    action = action + interval_extra_action(current, following, dt, ALPHA)
    gradient = torch.autograd.grad(action, current)[0].detach().numpy()
    rng = np.random.default_rng(402)
    direction = rng.standard_normal(gradient.size)
    direction /= np.linalg.norm(direction)
    epsilon = 1e-6

    def value(raw):
        raw_tensor = torch.tensor(raw, dtype=torch.float64)
        return float((interval_extra_action(previous, raw_tensor, dt, ALPHA)
                      + interval_extra_action(raw_tensor, following, dt,
                                              ALPHA)).detach())

    center = current.detach().numpy()
    finite = (value(center + epsilon * direction)
              - value(center - epsilon * direction)) / (2 * epsilon)
    exact = float(np.dot(gradient, direction))
    return float(abs(finite - exact) / max(abs(exact), 1e-12))


def main() -> None:
    rank.lock_inertia()
    controls = []
    for coefficient in (0.0, 1.0, ALPHA, 2.0):
        residual, rows = directional_residuals(coefficient)
        controls.append({
            "coefficient": coefficient,
            "combined_relative_residual": residual,
            "rows": rows,
        })

    objective = lambda value: directional_residuals(float(value))[0]
    optimum = minimize_scalar(
        objective, bounds=(-1.0, 4.0), method="bounded",
        options={"xatol": 1e-7})
    baseline = controls[0]["combined_relative_residual"]
    natural = controls[2]["combined_relative_residual"]
    result = {
        "status": "V4_DENSITY_MOLD_CONNECTION_PARTIAL_IMPROVEMENT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action": (
            "T_chi=(B/2)[kappa|chiDot|^2+|A19 Vchi/24|^2], "
            "Vchi=chiDot+s rhoPsiDot chi/[2(rhoChi+kappa chi0^2)]"),
        "B": rank.B,
        "kappa": rank.KAPPA,
        "chi0": rank.R,
        "native_coefficient": ALPHA,
        "native_coefficient_provenance": (
            "s=1+54/chi0^2 is the differentiated carrier radial-balance "
            "factor rhoChi+s rhoPsi=chi0^2; 54/chi0^2 is the exact C19 "
            "area-stiffness correction of the localized rest orbit"),
        "positive_kinetic_square": True,
        "rest_orbit_action_delta_zero": True,
        "controls": controls,
        "natural_coefficient_relative_improvement": float(
            (baseline - natural) / baseline),
        "diagnostic_optimum": {
            "coefficient": float(optimum.x),
            "relative_residual": float(optimum.fun),
            "admitted": False,
            "reason": "movement-fitted coefficients are diagnostic only",
        },
        "autodiff_directional_derivative_relative_defect": derivative_control(),
        "verdict": "PARTIAL_IMPROVEMENT_REQUIRES_REST_SPECTRUM",
        "root_cause_update": (
            "The dominant handoff mismatch is chi radial mold motion rather "
            "than an internal Psi-frame rotation.  Psi density departure and "
            "arrival must respectively heal and open the neighboring chi "
            "mold inside the kinetic geometry."),
        "next_action": (
            "Derive the complete implicit discrete Euler-Lagrange tangent "
            "operator of this field-dependent metric.  Retain it only if the "
            "native coefficient improves the full rest spectrum as well as "
            "the edge-snap response; then solve a multi-update relative-"
            "periodic one-edge orbit rather than equating one integrator step "
            "with one physical hop."),
        "claim_boundary": (
            "This is an inverse edge-snap response test. It is not a freely "
            "moving orbit, rest-stability pass, Dirac pass, or admitted action."),
    }
    (HERE / "density_mold_connection_probe.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
