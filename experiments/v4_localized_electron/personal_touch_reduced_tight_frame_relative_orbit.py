"""Free one-cell relative-periodic shooting for the reduced tight-frame action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as small
import personal_touch_unit_geometric_nonlinear as action


HERE = Path(__file__).resolve().parent
SIDE = 5
DT_MIN = 1/4096
DT_MAX = 0.08


def phase(value: torch.Tensor) -> torch.Tensor:
    return torch.complex(torch.cos(value), torch.sin(value))


def translate(raw: torch.Tensor, axis: int, sign: int,
              psi_phase: torch.Tensor, chi_phase: torch.Tensor) -> torch.Tensor:
    psi, chi = action.unpack(raw)
    return action.pack(
        torch.roll(psi, sign, axis)*phase(psi_phase),
        torch.roll(chi, sign, axis)*phase(chi_phase))


def dt_from_raw(raw: torch.Tensor) -> torch.Tensor:
    return DT_MIN+(DT_MAX-DT_MIN)*torch.sigmoid(raw)


def raw_from_dt(value: float) -> float:
    ratio = (value-DT_MIN)/(DT_MAX-DT_MIN)
    ratio = min(max(ratio, 1e-8), 1-1e-8)
    return float(np.log(ratio/(1-ratio)))


def components(raw: torch.Tensor):
    psi, chi = action.unpack(raw)
    vacuum = torch.zeros_like(chi)
    vacuum[..., 2] = small.CHI0
    return psi, chi-vacuum


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    ap, ac = components(actual)
    ep, ec = components(expected)
    return (torch.sum(torch.abs(ap-ep)**2)/torch.sum(torch.abs(ep)**2)
            + torch.sum(torch.abs(ac-ec)**2)/torch.sum(torch.abs(ec)**2))


def shoot(updates: int, axis: int, sign: int, iterations: int,
          device: torch.device) -> dict:
    action.ACTION_KIND = "reduced_tight_frame"
    raw = np.load(
        HERE/"canonical_noncoplanar_large_box_solvability.npz")["rest"]
    q0 = torch.tensor(raw, dtype=torch.float64, device=device)
    mass = action.masses(device)
    dt_guess = min(0.04, 1/(2*updates))
    total_phase_guess = small.CHI0*updates*dt_guess
    shifted = translate(
        q0, axis, sign, torch.tensor(total_phase_guess, device=device),
        torch.tensor(0.0, device=device))
    qminus = (q0-(shifted-q0)/updates).detach().requires_grad_(True)
    raw_dt = torch.tensor(raw_from_dt(dt_guess), dtype=torch.float64,
                          device=device, requires_grad=True)
    psi_phase = torch.tensor(total_phase_guess, dtype=torch.float64,
                             device=device, requires_grad=True)
    chi_phase = torch.tensor(0.0, dtype=torch.float64,
                             device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [qminus, raw_dt, psi_phase, chi_phase], lr=0.35,
        max_iter=iterations, max_eval=iterations*2,
        tolerance_grad=1e-10, tolerance_change=1e-14,
        history_size=50, line_search_fn="strong_wolfe")
    trace: list[float] = []

    def update(previous: torch.Tensor, current: torch.Tensor,
               dt: torch.Tensor) -> torch.Tensor:
        gradient = torch.func.grad(action.total_potential)(current)
        return 2*current-previous-dt**2*gradient/mass

    def evolve():
        dt = dt_from_raw(raw_dt)
        states = [qminus, q0]
        for _ in range(updates):
            states.append(update(states[-2], states[-1], dt))
        return dt, states

    def objective():
        optimizer.zero_grad()
        dt, states = evolve()
        endpoint = translate(q0, axis, sign, psi_phase, chi_phase)
        before = translate(qminus, axis, sign, psi_phase, chi_phase)
        loss = relative_error(states[-1], endpoint)+relative_error(
            states[-2], before)
        fraction = (dt-DT_MIN)/(DT_MAX-DT_MIN)
        loss = loss+1e-10*(1/fraction+1/(1-fraction))
        loss.backward()
        trace.append(float(torch.sqrt(loss.detach()).cpu()))
        return loss

    initial = float(torch.sqrt(objective().detach()).cpu())
    optimizer.step(objective)
    dt, states = evolve()
    endpoint = translate(q0, axis, sign, psi_phase, chi_phase)
    before = translate(qminus, axis, sign, psi_phase, chi_phase)
    end_error = float(torch.sqrt(relative_error(
        states[-1], endpoint)).detach().cpu())
    layer_error = float(torch.sqrt(relative_error(
        states[-2], before)).detach().cpu())
    second = list(states)
    for _ in range(updates):
        second.append(update(second[-2], second[-1], dt))
    endpoint2 = translate(endpoint, axis, sign, psi_phase, chi_phase)
    second_error = float(torch.sqrt(relative_error(
        second[-1], endpoint2)).detach().cpu())
    combined = float(np.hypot(end_error, layer_error))
    return {
        "axis": axis, "sign": sign, "updates_per_cell": updates,
        "initial_relative_boundary_residual": initial,
        "final_relative_boundary_residual": combined,
        "endpoint_relative_residual": end_error,
        "preceding_layer_relative_residual": layer_error,
        "unreset_second_cell_endpoint_residual": second_error,
        "solved_dt": float(dt.detach().cpu()),
        "solved_translation_time": float((updates*dt).detach().cpu()),
        "solved_psi_phase": float(psi_phase.detach().cpu()),
        "solved_chi_phase": float(chi_phase.detach().cpu()),
        "optimizer_evaluations": len(trace),
        "minimum_objective_trace": min(trace),
        "converged": combined < 1e-8 and second_error < 1e-7,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", nargs="+", type=int, default=(2, 4))
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--axis", type=int, default=0)
    parser.add_argument("--sign", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", default="personal_touch_reduced_tight_frame_relative_orbit.json")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = [shoot(value, args.axis, args.sign, args.iterations, device)
            for value in args.updates]
    passed = any(row["converged"] for row in rows)
    result = {
        "status": "V4_REDUCED_TIGHT_FRAME_RELATIVE_ORBIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "device": str(device),
        "rows": rows,
        "verdict": (
            "EXACT_ONE_CELL_RELATIVE_PERIODIC_ORBIT_FOUND" if passed
            else "NO_ORBIT_CONVERGED_FROM_TESTED_SEEDS"),
        "claim_boundary": (
            "Free one-cell shooting with fixed exact carrier at one layer. "
            "Nonconvergence is not a no-go for a deformed carrier or other "
            "periods. V4-006 passes only with two handoffs in both signs of "
            "all axes and the same-state Dirac-current gate."),
    }
    (HERE/args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
