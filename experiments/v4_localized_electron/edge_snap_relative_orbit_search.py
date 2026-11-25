"""Two-update relative-periodic edge-snap feasibility search.

The endpoint constraint is native translation, not a runtime force:

    q[n+2] = exp(i phase) tau_edge q[n].

The intermediate full Psi/chi lattice layer is solved from the discrete
Euler--Lagrange equations.  This tests the refined snap hypothesis: the
carrier may be localized at recurring endpoint times while its fields occupy
a nonlocalized bond-current configuration between endpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import density_mold_connection_probe as mold
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent
SIDE = 5
AXIS = 0
SIGN = 1


def potential(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = mold.unpack_tensor(raw, SIDE)
    rho_psi = torch.sum(mold.squared_norm(psi), dim=-1)
    rho_chi = torch.sum(mold.squared_norm(chi), dim=-1)
    local = torch.sum(
        0.25 * (rho_chi - rank.R**2)**2
        + 0.5 * rho_chi * rho_psi
        + 0.25 * rho_psi**2)
    area = torch.linalg.cross(chi, psi, dim=-1)
    bond = torch.zeros((), dtype=torch.float64)
    for offset, weight in rank.native.DIRECTIONS:
        difference = area - torch.roll(
            area, shifts=offset, dims=(0, 1, 2))
        bond = bond + weight * torch.sum(mold.squared_norm(difference))
    return local + bond / (2 * rank.R**2)


def interval_action(left: torch.Tensor, right: torch.Tensor, dt: float,
                    coefficient: float) -> torch.Tensor:
    left_psi, left_chi = mold.unpack_tensor(left, SIDE)
    right_psi, right_chi = mold.unpack_tensor(right, SIDE)
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    psi_dot = (right_psi - left_psi) / dt
    chi_dot = (right_chi - left_chi) / dt
    rho_psi_dot = 2 * torch.real(torch.sum(
        torch.conj(psi) * psi_dot, dim=-1))
    rho_chi = torch.sum(mold.squared_norm(chi), dim=-1)
    denominator = rho_chi + rank.KAPPA * rank.R**2
    velocity = (
        chi_dot
        + coefficient * rho_psi_dot[..., None] * chi
        / (2 * denominator[..., None]))
    averaged = mold.adjacency_average(velocity)
    kinetic = 0.5 * torch.sum(mold.squared_norm(psi_dot))
    kinetic = kinetic + rank.B / 2 * (
        rank.KAPPA * torch.sum(mold.squared_norm(chi_dot))
        + torch.sum(mold.squared_norm(averaged)))
    return dt * (kinetic - (potential(left) + potential(right)) / 2)


def translate(raw: torch.Tensor, amount: int, psi_phase: float,
              chi_phase: float) -> torch.Tensor:
    psi, chi = mold.unpack_tensor(raw, SIDE)
    psi = torch.roll(psi, shifts=amount, dims=AXIS)
    chi = torch.roll(chi, shifts=amount, dims=AXIS)
    def phase_factor(value):
        value = torch.as_tensor(value, dtype=torch.float64)
        return torch.complex(torch.cos(value), torch.sin(value))

    psi = psi * phase_factor(psi_phase)
    chi = chi * phase_factor(chi_phase)
    return torch.stack(
        (psi.real, psi.imag, chi.real, chi.imag), dim=-2).reshape(-1)


def del_residuals(q0: torch.Tensor, q1: torch.Tensor, dt: float,
                  coefficient: float, psi_phase: float, chi_phase: float):
    qm1 = translate(q1, -SIGN, -psi_phase, -chi_phase)
    q2 = translate(q0, SIGN, psi_phase, chi_phase)
    q0_work = q0.detach().clone().requires_grad_(True)
    q1_work = q1
    action0 = (interval_action(qm1, q0_work, dt, coefficient)
               + interval_action(q0_work, q1_work, dt, coefficient))
    residual0 = torch.autograd.grad(
        action0, q0_work, create_graph=True)[0]
    action1 = (interval_action(q0, q1_work, dt, coefficient)
               + interval_action(q1_work, q2, dt, coefficient))
    residual1 = torch.autograd.grad(
        action1, q1_work, create_graph=True)[0]
    return residual0, residual1, q2


def initial_midpoint(q0: torch.Tensor, psi_phase: float,
                     chi_phase: float) -> torch.Tensor:
    q2 = translate(q0, SIGN, psi_phase, chi_phase)
    midpoint = q0 + q2
    # The sum can suppress an opposite-phase Psi endpoint.  Add the tangent
    # direction in that case; this is an initial guess only, never a force.
    if torch.linalg.vector_norm(midpoint) < 0.1 * torch.linalg.vector_norm(q0):
        midpoint = q0 + 0.5 * (q2 - q0)
    return midpoint / 2


def search(coefficient: float, iterations: int = 400):
    rank.lock_inertia()
    evidence = json.loads(
        (HERE / "edge_snap_verlet_gate.json").read_text(encoding="utf-8"))
    endpoint = next(row for row in evidence["rows"]
                    if row["axis"] == AXIS and row["sign"] == SIGN)
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    q0 = torch.tensor(rank.pack(psi, chi), dtype=torch.float64)
    psi_phase = float(endpoint["best_phase_psi"])
    chi_phase = float(endpoint["best_phase_chi"])
    dt = float(endpoint["best_dt"] / 2)
    q1 = initial_midpoint(q0, psi_phase / 2, chi_phase / 2)
    q1 = q1.detach().clone().requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [q1], lr=0.5, max_iter=iterations, max_eval=iterations * 2,
        tolerance_grad=1e-10, tolerance_change=1e-14,
        history_size=40, line_search_fn="strong_wolfe")
    trace = []

    def closure():
        optimizer.zero_grad()
        r0, r1, _ = del_residuals(
            q0, q1, dt, coefficient, psi_phase, chi_phase)
        scale = rank.B * rank.R / max(dt, 1e-12)
        loss = (torch.mean((r0 / scale)**2)
                + torch.mean((r1 / scale)**2))
        loss.backward()
        trace.append(float(torch.sqrt(loss.detach())))
        return loss

    initial = float(torch.sqrt(closure().detach()))
    optimizer.step(closure)
    r0, r1, q2 = del_residuals(
        q0, q1, dt, coefficient, psi_phase, chi_phase)
    scale = rank.B * rank.R / max(dt, 1e-12)
    final = float(torch.sqrt(
        torch.mean((r0 / scale)**2) + torch.mean((r1 / scale)**2)).detach())
    psi1, chi1 = mold.unpack_tensor(q1.detach(), SIDE)
    rho1 = torch.sum(mold.squared_norm(psi1), dim=-1)
    coordinates = torch.tensor(
        np.indices((SIDE, SIDE, SIDE)).reshape(3, -1).T,
        dtype=torch.float64)
    center = torch.sum(
        coordinates * rho1.reshape(-1, 1), dim=0) / torch.sum(rho1)
    charge0 = float(np.sum(abs(psi)**2))
    charge1 = float(torch.sum(rho1))
    endpoint_defect = float(torch.linalg.vector_norm(
        q2 - translate(q0, SIGN, psi_phase, chi_phase)))
    return {
        "coefficient": coefficient,
        "dt": dt,
        "translation_phase_psi": psi_phase,
        "translation_phase_chi": chi_phase,
        "initial_scaled_del_rms": initial,
        "final_scaled_del_rms": final,
        "relative_residual_reduction": float(1 - final / initial),
        "optimizer_evaluations": len(trace),
        "minimum_trace_residual": min(trace),
        "intermediate_psi_charge": charge1,
        "endpoint_psi_charge": charge0,
        "charge_relative_change": charge1 / charge0 - 1,
        "intermediate_center_of_density": center.detach().numpy().tolist(),
        "endpoint_constraint_defect": endpoint_defect,
    }


def main() -> None:
    rows = [search(0.0), search(mold.ALPHA)]
    candidate_better = (
        rows[1]["final_scaled_del_rms"] < rows[0]["final_scaled_del_rms"])
    converged = rows[1]["final_scaled_del_rms"] < 1e-8
    result = {
        "status": ("V4_TWO_UPDATE_EDGE_ORBIT_PASS" if converged else
                   "V4_TWO_UPDATE_EDGE_ORBIT_INCONCLUSIVE"),
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "relative_period_updates": 2,
        "axis": AXIS,
        "sign": SIGN,
        "rows": rows,
        "density_mold_candidate_better_than_metric_incumbent": candidate_better,
        "verdict": (
            "EXACT_RELATIVE_PERIODIC_EDGE_ORBIT_FOUND" if converged else
            "NO_CONVERGED_TWO_UPDATE_ORBIT_FROM_THIS_INITIALIZATION"),
        "claim_boundary": (
            "A failed nonlinear solve is not a no-go theorem. A pass would "
            "still require independent free evolution, all directions, rest "
            "stability, conservation, and Dirac current checks."),
    }
    (HERE / "edge_snap_relative_orbit_search.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
