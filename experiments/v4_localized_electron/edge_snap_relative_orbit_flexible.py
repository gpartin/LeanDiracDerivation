"""Solve the two-update edge orbit with native period and phases free.

The fixed-period search inherited the one-step inverse recurrence timing.
This follow-up treats the relative period and both native U(1) phases as
unknown boundary-orbit data.  No field is preloaded or reset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import density_mold_connection_probe as mold
import edge_snap_action_rank as rank
import edge_snap_relative_orbit_search as orbit


HERE = Path(__file__).resolve().parent
DT_MIN = 0.005
DT_MAX = 0.2


def dt_from_raw(raw: torch.Tensor) -> torch.Tensor:
    return DT_MIN + (DT_MAX - DT_MIN) * torch.sigmoid(raw)


def raw_from_dt(value: float) -> float:
    fraction = (value - DT_MIN) / (DT_MAX - DT_MIN)
    fraction = min(max(fraction, 1e-8), 1 - 1e-8)
    return float(np.log(fraction / (1 - fraction)))


def discrete_charge(left: torch.Tensor, right: torch.Tensor, dt,
                    coefficient: float, field: str) -> float:
    right_work = right.detach().clone().requires_grad_(True)
    action = orbit.interval_action(left.detach(), right_work, dt, coefficient)
    momentum = torch.autograd.grad(action, right_work)[0]
    psi, chi = mold.unpack_tensor(right_work, orbit.SIDE)
    zero_psi = torch.zeros_like(psi)
    zero_chi = torch.zeros_like(chi)
    generator = (orbit.rank.pack(
        (1j * psi.detach().numpy()) if field == "psi" else
        zero_psi.detach().numpy(),
        (1j * chi.detach().numpy()) if field == "chi" else
        zero_chi.detach().numpy()))
    return float(torch.dot(
        momentum.detach(), torch.tensor(generator, dtype=torch.float64)))


def search(iterations: int = 800):
    rank.lock_inertia()
    evidence = json.loads(
        (HERE / "edge_snap_verlet_gate.json").read_text(encoding="utf-8"))
    endpoint = next(row for row in evidence["rows"]
                    if row["axis"] == orbit.AXIS
                    and row["sign"] == orbit.SIGN)
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(orbit.SIDE, amplitude_sq)
    q0 = torch.tensor(rank.pack(psi, chi), dtype=torch.float64)
    phase_psi = torch.tensor(
        float(endpoint["best_phase_psi"]), dtype=torch.float64,
        requires_grad=True)
    phase_chi = torch.tensor(
        float(endpoint["best_phase_chi"]), dtype=torch.float64,
        requires_grad=True)
    raw_dt = torch.tensor(
        raw_from_dt(float(endpoint["best_dt"] / 2)),
        dtype=torch.float64, requires_grad=True)
    q1 = orbit.initial_midpoint(
        q0, phase_psi.detach() / 2, phase_chi.detach() / 2)
    q1 = q1.detach().clone().requires_grad_(True)
    parameters = [q1, raw_dt, phase_psi, phase_chi]
    optimizer = torch.optim.LBFGS(
        parameters, lr=0.35, max_iter=iterations, max_eval=iterations * 2,
        tolerance_grad=1e-11, tolerance_change=1e-15,
        history_size=60, line_search_fn="strong_wolfe")
    dt_reference = float(endpoint["best_dt"] / 2)
    scale = rank.B * rank.R / dt_reference
    trace = []

    def closure():
        optimizer.zero_grad()
        dt = dt_from_raw(raw_dt)
        r0, r1, _ = orbit.del_residuals(
            q0, q1, dt, mold.ALPHA, phase_psi, phase_chi)
        loss = (torch.mean((r0 / scale)**2)
                + torch.mean((r1 / scale)**2))
        loss.backward()
        trace.append({
            "residual": float(torch.sqrt(loss.detach())),
            "dt": float(dt.detach()),
            "phase_psi": float(phase_psi.detach()),
            "phase_chi": float(phase_chi.detach()),
        })
        return loss

    initial = float(torch.sqrt(closure().detach()))
    optimizer.step(closure)
    dt = dt_from_raw(raw_dt)
    r0, r1, q2 = orbit.del_residuals(
        q0, q1, dt, mold.ALPHA, phase_psi, phase_chi)
    final = float(torch.sqrt(
        torch.mean((r0 / scale)**2) + torch.mean((r1 / scale)**2)).detach())
    q_minus1 = orbit.translate(
        q1.detach(), -orbit.SIGN, -phase_psi.detach(), -phase_chi.detach())
    charges = {}
    for field in ("psi", "chi"):
        first = discrete_charge(
            q_minus1, q0, dt.detach(), mold.ALPHA, field)
        second = discrete_charge(
            q0, q1.detach(), dt.detach(), mold.ALPHA, field)
        charges[field] = {
            "incoming": first,
            "outgoing": second,
            "relative_defect": float(
                abs(second - first) / max(abs(first), abs(second), 1e-30)),
        }
    psi1, _ = mold.unpack_tensor(q1.detach(), orbit.SIDE)
    density = torch.sum(mold.squared_norm(psi1), dim=-1)
    result = {
        "status": ("V4_FLEXIBLE_EDGE_ORBIT_PASS" if final < 1e-8 else
                   "V4_FLEXIBLE_EDGE_ORBIT_INCONCLUSIVE"),
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "updates_per_translation": 2,
        "axis": orbit.AXIS,
        "sign": orbit.SIGN,
        "initial_scaled_del_rms": initial,
        "final_scaled_del_rms": final,
        "relative_residual_reduction": float(1 - final / initial),
        "optimizer_evaluations": len(trace),
        "solved_dt": float(dt.detach()),
        "solved_translation_time": float(2 * dt.detach()),
        "solved_phase_psi": float(phase_psi.detach()),
        "solved_phase_chi": float(phase_chi.detach()),
        "intermediate_psi_density_sum": float(torch.sum(density)),
        "discrete_noether_charge_controls": charges,
        "endpoint_constraint_defect": float(torch.linalg.vector_norm(
            q2 - orbit.translate(q0, orbit.SIGN, phase_psi, phase_chi))),
        "trace_tail": trace[-10:],
        "verdict": (
            "EXACT_RELATIVE_PERIODIC_EDGE_ORBIT_FOUND" if final < 1e-8 else
            "FREE_PERIOD_AND_PHASES_DO_NOT_YET_CLOSE_THE_ORBIT"),
        "claim_boundary": (
            "Nonconvergence from one initial path is inconclusive. The solved "
            "parameters are not action coefficients and do not alter the "
            "native law."),
    }
    (HERE / "edge_snap_relative_orbit_flexible.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    search()
