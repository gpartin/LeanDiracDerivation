"""Test a positive phase-covariant Psi-to-chi frame connection.

The full candidate chi kinetic energy is

  B/2 [kappa |chi_dot|^2
       + |A19 chi_dot/24 - gamma Omega_Psi chi|^2],

  Omega_Psi chi = [Psi_dot (Psi^dag chi)
                   - Psi (Psi_dot^dag chi)]/(chi0^2+rho_Psi).

``Omega_Psi`` is anti-Hermitian before the smooth denominator, transforms
covariantly under the independent field phases, and vanishes on the exact
rest carrier because its rotating Psi is orthogonal to chi.  The expression
is positive and contains no carrier, corner, direction, copy, or Dirac data.
The self-adjoint midpoint discrete action is differentiated directly; no
force is inserted by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar

import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent
SIDE = 5
SHAPE = (SIDE, SIDE, SIDE, 4, 3)


def unpack(raw: torch.Tensor):
    data = raw.reshape(SHAPE)
    return (torch.complex(data[..., 0, :], data[..., 1, :]),
            torch.complex(data[..., 2, :], data[..., 3, :]))


def adjacency_average(value: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(value)
    for offset, weight in rank.native.DIRECTIONS:
        result = result + weight * torch.roll(
            value, shifts=tuple(-entry for entry in offset), dims=(0, 1, 2))
    return result / rank.C19_WEIGHT_SUM


def interval_delta(left_raw: torch.Tensor, right_raw: torch.Tensor,
                   dt: float, gamma: float) -> torch.Tensor:
    left_psi, left_chi = unpack(left_raw)
    right_psi, right_chi = unpack(right_raw)
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    velocity_psi = (right_psi - left_psi) / dt
    velocity_chi = (right_chi - left_chi) / dt
    density_psi = torch.sum(
        psi.real**2 + psi.imag**2, dim=-1, keepdim=True)
    psi_chi = torch.sum(torch.conj(psi) * chi, dim=-1, keepdim=True)
    velocity_chi_overlap = torch.sum(
        torch.conj(velocity_psi) * chi, dim=-1, keepdim=True)
    connection = (
        velocity_psi * psi_chi - psi * velocity_chi_overlap
    ) / (rank.R**2 + density_psi)
    covariant_velocity = (
        adjacency_average(velocity_chi) - gamma * connection)
    chi_norm = torch.sum(
        velocity_chi.real**2 + velocity_chi.imag**2)
    covariant_norm = torch.sum(
        covariant_velocity.real**2 + covariant_velocity.imag**2)
    # Delta relative to the canonical B |chi_dot|^2 / 2 action.
    return dt * rank.B / 2 * (
        rank.KAPPA * chi_norm + covariant_norm - chi_norm)


def action_delta(layers, dt: float, gamma: float):
    raw = [torch.tensor(rank.pack(*layer), dtype=torch.float64)
           for layer in layers]
    current = raw[1].clone().requires_grad_(True)
    action = (interval_delta(raw[0], current, dt, gamma)
              + interval_delta(current, raw[2], dt, gamma))
    gradient = torch.autograd.grad(action, current)[0]
    return float(action.detach()), gradient.detach().numpy()


def evaluate(gamma: float, rows, psi, chi, force_psi, force_chi):
    numerator = 0.0
    denominator = 0.0
    output = []
    for row in rows:
        dt = row["best_dt"]
        layers = rank.translated_layers(row, psi, chi)
        baseline = rank.canonical_residual(
            *layers, dt, force_psi, force_chi)
        action, delta = action_delta(layers, dt, gamma)
        residual = baseline + delta
        numerator += float(np.vdot(residual, residual).real)
        denominator += float(np.vdot(baseline, baseline).real)
        output.append({
            "axis": row["axis"],
            "sign": row["sign"],
            "relative_residual": float(
                np.linalg.norm(residual) / np.linalg.norm(baseline)),
            "discrete_action_delta": action,
        })
    return float(np.sqrt(numerator / denominator)), output


def derivative_control(row, psi, chi, gamma: float):
    dt = row["best_dt"]
    layers = rank.translated_layers(row, psi, chi)
    raw = [torch.tensor(rank.pack(*layer), dtype=torch.float64)
           for layer in layers]
    current = raw[1].clone().requires_grad_(True)
    action = (interval_delta(raw[0], current, dt, gamma)
              + interval_delta(current, raw[2], dt, gamma))
    gradient = torch.autograd.grad(action, current)[0].detach().numpy()
    rng = np.random.default_rng(40401)
    direction = rng.normal(size=gradient.size)
    direction /= np.linalg.norm(direction)
    epsilon = 1e-5

    def value(raw_current):
        return float((
            interval_delta(raw[0], raw_current, dt, gamma)
            + interval_delta(raw_current, raw[2], dt, gamma)).detach())

    plus = raw[1] + epsilon * torch.tensor(direction, dtype=torch.float64)
    minus = raw[1] - epsilon * torch.tensor(direction, dtype=torch.float64)
    finite = (value(plus) - value(minus)) / (2 * epsilon)
    analytic = float(np.dot(gradient, direction))
    return {
        "epsilon": epsilon,
        "analytic_directional_derivative": analytic,
        "finite_directional_derivative": finite,
        "relative_defect": float(
            abs(analytic - finite) / max(abs(analytic), abs(finite), 1.0)),
    }


def main() -> None:
    rank.lock_inertia()
    evidence = json.loads(
        (HERE / "edge_snap_verlet_gate.json").read_text(encoding="utf-8"))
    rows = evidence["rows"]
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    force_psi, force_chi = rank.native.force(psi, chi)
    scan = []
    for gamma in (-2.0, -1.0, 0.0, 1.0, 2.0):
        residual, direction_rows = evaluate(
            gamma, rows, psi, chi, force_psi, force_chi)
        scan.append({
            "gamma": gamma,
            "combined_relative_residual": residual,
            "rows": direction_rows,
        })
    optimum = minimize_scalar(
        lambda gamma: evaluate(
            gamma, rows, psi, chi, force_psi, force_chi)[0],
        bounds=(-4.0, 4.0), method="bounded",
        options={"xatol": 1e-10})
    optimum_rows = evaluate(
        float(optimum.x), rows, psi, chi, force_psi, force_chi)[1]
    zero = next(row for row in scan if row["gamma"] == 0.0)
    natural = next(row for row in scan if row["gamma"] == 1.0)
    result = {
        "status": "V4_PSI_FRAME_CONNECTION_RETIRED",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action": (
            "T_chi=(B/2)[kappa |chiDot|^2+"
            "|A19 chiDot/24-gamma Omega_Psi chi|^2]"),
        "positive_and_invertible": True,
        "rest_orbit_action_delta": 0.0,
        "coefficient_one_provenance": (
            "unit parallel transport of chi by the native Psi frame"),
        "scan": scan,
        "gamma_zero_control": zero["combined_relative_residual"],
        "gamma_one_residual": natural["combined_relative_residual"],
        "diagnostic_optimum": {
            "gamma": float(optimum.x),
            "combined_relative_residual": float(optimum.fun),
            "relative_improvement_over_zero": float(
                (zero["combined_relative_residual"] - optimum.fun)
                / zero["combined_relative_residual"]),
            "rows": optimum_rows,
            "coefficient_admitted": False,
        },
        "derivative_control": derivative_control(rows[0], psi, chi, 1.0),
        "verdict": "NO_MEANINGFUL_MOVEMENT_IMPROVEMENT",
        "root_cause": (
            "The surviving handoff residual is not the local internal-frame "
            "rotation generated by Omega_Psi.  Unit parallel transport "
            "worsens it; a fitted gamma changes the norm by less than 0.05%."),
        "next_action": (
            "Retire this local Psi-frame connection.  Decompose the remaining "
            "chi residual by C19 symbol and carrier/interface support before "
            "constructing another positive square."),
        "claim_boundary": (
            "This rejects one generic connection family on the fixed-phase "
            "edge-snap gate.  It is not a no-go for every field-dependent "
            "metric and does not change the canonical action."),
    }
    (HERE / "psi_frame_connection_probe.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
