"""Shoot for a one-cell relative-periodic orbit of canonical V40.

The canonical explicit centered update is evolved freely from two layers.  We
hold the exact localized rest carrier at layer zero and solve only its previous
layer, the timestep, and the two global endpoint phases.  After ``N`` updates
the state and its preceding layer must equal the initial two-layer state
translated by one cell.  No receiving field, projection, or Dirac force is
applied during evolution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent
SIDE = 5
AXIS = 0
SIGN = 1
DT_MIN = 1.0 / 4096.0
DT_MAX = 0.08


def unpack(raw: torch.Tensor):
    data = raw.reshape(SIDE, SIDE, SIDE, 4, 3)
    return (torch.complex(data[..., 0, :], data[..., 1, :]),
            torch.complex(data[..., 2, :], data[..., 3, :]))


def pack(psi: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    return torch.stack((psi.real, psi.imag, chi.real, chi.imag), dim=-2).reshape(-1)


def norm2(value: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.conj(value) * value).sum(dim=-1)


def canonical_force(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = unpack(raw)
    p = norm2(psi)[..., None]
    c = norm2(chi)[..., None]
    area = torch.linalg.cross(chi, psi, dim=-1)
    area_gradient = torch.zeros_like(area)
    for offset, weight in rank.native.DIRECTIONS:
        area_gradient = area_gradient + weight * (
            area - torch.roll(area, shifts=offset, dims=(0, 1, 2)))
    area_gradient = 2 * area_gradient / rank.R**2
    force_psi = (-(p + c) * psi
                 - torch.linalg.cross(area_gradient, torch.conj(chi), dim=-1))
    force_chi = (-(c - rank.R**2 + p) * chi
                 - torch.linalg.cross(torch.conj(psi), area_gradient, dim=-1)) / rank.B
    return pack(force_psi, force_chi)


def update(previous: torch.Tensor, current: torch.Tensor,
           dt: torch.Tensor) -> torch.Tensor:
    return 2 * current - previous + dt**2 * canonical_force(current)


def phase(value: torch.Tensor) -> torch.Tensor:
    return torch.complex(torch.cos(value), torch.sin(value))


def translate(raw: torch.Tensor, psi_phase: torch.Tensor,
              chi_phase: torch.Tensor) -> torch.Tensor:
    psi, chi = unpack(raw)
    return pack(
        torch.roll(psi, SIGN, AXIS) * phase(psi_phase),
        torch.roll(chi, SIGN, AXIS) * phase(chi_phase))


def dt_from_raw(raw: torch.Tensor) -> torch.Tensor:
    return DT_MIN + (DT_MAX - DT_MIN) * torch.sigmoid(raw)


def raw_from_dt(value: float) -> float:
    x = (value - DT_MIN) / (DT_MAX - DT_MIN)
    x = min(max(x, 1e-8), 1 - 1e-8)
    return float(np.log(x / (1 - x)))


def components(raw: torch.Tensor):
    psi, chi = unpack(raw)
    vacuum = torch.zeros_like(chi)
    vacuum[..., 2] = rank.R
    return psi, chi - vacuum


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    ap, ac = components(actual)
    ep, ec = components(expected)
    return (torch.sum(torch.abs(ap - ep)**2) / torch.sum(torch.abs(ep)**2)
            + torch.sum(torch.abs(ac - ec)**2) / torch.sum(torch.abs(ec)**2))


def shoot(updates: int, iterations: int) -> dict:
    rank.lock_inertia()
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi_np, chi_np = rank.native.carrier_fields(SIDE, amplitude_sq)
    q0 = torch.tensor(rank.pack(psi_np, chi_np), dtype=torch.float64)

    dt_guess = min(0.04, 1.0 / (2 * updates))
    total_phase_guess = rank.R * updates * dt_guess
    shifted = translate(q0, torch.tensor(total_phase_guess), torch.tensor(0.0))
    # Backward secant seed for one cell per N updates.  It is an initial
    # previous layer only and is subsequently solved without restriction.
    qminus = (q0 - (shifted - q0) / updates).detach().requires_grad_(True)
    raw_dt = torch.tensor(raw_from_dt(dt_guess), dtype=torch.float64,
                          requires_grad=True)
    psi_phase = torch.tensor(total_phase_guess, dtype=torch.float64,
                             requires_grad=True)
    chi_phase = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    parameters = [qminus, raw_dt, psi_phase, chi_phase]
    optimizer = torch.optim.LBFGS(
        parameters, lr=0.4, max_iter=iterations, max_eval=iterations * 2,
        tolerance_grad=1e-11, tolerance_change=1e-15, history_size=60,
        line_search_fn="strong_wolfe")
    trace = []

    def evolve():
        dt = dt_from_raw(raw_dt)
        states = [qminus, q0]
        for _ in range(updates):
            states.append(update(states[-2], states[-1], dt))
        return dt, states

    def objective():
        optimizer.zero_grad()
        dt, states = evolve()
        end = translate(q0, psi_phase, chi_phase)
        before_end = translate(qminus, psi_phase, chi_phase)
        loss = relative_error(states[-1], end) + relative_error(
            states[-2], before_end)
        # Prevent the free period from collapsing onto either bound.
        fraction = (dt - DT_MIN) / (DT_MAX - DT_MIN)
        loss = loss + 1e-10 * (1 / fraction + 1 / (1 - fraction))
        loss.backward()
        trace.append(float(torch.sqrt(loss.detach())))
        return loss

    initial = float(torch.sqrt(objective().detach()))
    optimizer.step(objective)
    dt, states = evolve()
    end = translate(q0, psi_phase, chi_phase)
    before_end = translate(qminus, psi_phase, chi_phase)
    endpoint = float(torch.sqrt(relative_error(states[-1], end)).detach())
    layer = float(torch.sqrt(relative_error(
        states[-2], before_end)).detach())
    combined = float(np.sqrt(endpoint**2 + layer**2))

    # Free-run identity check: apply a second interval without any reset.
    second = list(states)
    for _ in range(updates):
        second.append(update(second[-2], second[-1], dt))
    end2 = translate(end, psi_phase, chi_phase)
    second_endpoint = float(torch.sqrt(relative_error(
        second[-1], end2)).detach())

    return {
        "updates_per_cell": updates,
        "initial_relative_boundary_residual": initial,
        "final_relative_boundary_residual": combined,
        "endpoint_relative_residual": endpoint,
        "preceding_layer_relative_residual": layer,
        "unreset_second_cell_endpoint_residual": second_endpoint,
        "solved_dt": float(dt.detach()),
        "solved_translation_time": float(updates * dt.detach()),
        "solved_psi_phase": float(psi_phase.detach()),
        "solved_chi_phase": float(chi_phase.detach()),
        "optimizer_evaluations": len(trace),
        "minimum_objective_trace": min(trace),
        "converged": combined < 1e-8 and second_endpoint < 1e-7,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", nargs="+", type=int, default=(2, 4))
    parser.add_argument("--iterations", type=int, default=400)
    args = parser.parse_args()
    rows = [shoot(value, args.iterations) for value in args.updates]
    passed = any(row["converged"] for row in rows)
    result = {
        "status": ("V4_CANONICAL_RELATIVE_PERIODIC_CELL_PASS" if passed else
                   "V4_CANONICAL_RELATIVE_PERIODIC_CELL_INCONCLUSIVE"),
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "axis": AXIS,
        "sign": SIGN,
        "rows": rows,
        "verdict": ("EXACT_ONE_CELL_RELATIVE_PERIODIC_ORBIT_FOUND" if passed
                    else "NO_ORBIT_CONVERGED_FROM_TESTED_SHOOTING_SEEDS"),
        "claim_boundary": (
            "A converged row would prove a discrete numerical orbit for one "
            "axis at the stated tolerance, not all-axis Dirac transport or a "
            "Lean theorem. Nonconvergence of these seeds is inconclusive, "
            "not a no-go theorem."),
    }
    output = HERE / "canonical_relative_periodic_cell_orbit.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
