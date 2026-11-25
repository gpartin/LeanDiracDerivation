"""Allow the canonical relative-periodic carrier to deform while moving.

This removes the fixed-rest-snapshot restriction from the shooting test.  The
initial current and previous layers, endpoint phases, and timestep are solved
together.  Only total Psi charge and two phase gauges are constrained.  The
result is accepted as the same electron only if its independently measured
eight-site fidelity and containment remain high.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import canonical_relative_periodic_cell_orbit as orbit
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent


def field_metrics(raw: torch.Tensor, reference: torch.Tensor) -> dict:
    psi, chi = orbit.unpack(raw)
    ref_psi, ref_chi = orbit.unpack(reference)
    overlap = torch.sum(torch.conj(ref_psi) * psi)
    aligned = psi * torch.exp(-1j * torch.angle(overlap))
    fidelity = torch.abs(torch.sum(torch.conj(ref_psi) * aligned))**2 / (
        torch.sum(torch.abs(ref_psi)**2) * torch.sum(torch.abs(aligned)**2))
    mask = torch.sum(torch.abs(ref_psi)**2, dim=-1) > 1e-12
    containment = torch.sum(torch.abs(psi[mask])**2) / torch.sum(torch.abs(psi)**2)
    vacuum = torch.zeros_like(chi)
    vacuum[..., 2] = rank.R
    chi_overlap = torch.sum(torch.conj(ref_chi - vacuum) * (chi - vacuum))
    chi_denominator = (
        torch.sum(torch.abs(ref_chi - vacuum)**2)
        * torch.sum(torch.abs(chi - vacuum)**2)).detach()
    chi_fidelity = torch.abs(chi_overlap)**2 / torch.clamp(
        chi_denominator, min=1e-300)
    return {
        "eight_site_psi_fidelity": float(fidelity.detach()),
        "psi_core_containment": float(containment.detach()),
        "chi_track_fidelity": float(chi_fidelity.detach()),
        "psi_charge": float(torch.sum(torch.abs(psi)**2).detach()),
    }


def shoot(updates: int, iterations: int) -> dict:
    rank.lock_inertia()
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi_np, chi_np = rank.native.carrier_fields(orbit.SIDE, amplitude_sq)
    reference = torch.tensor(rank.pack(psi_np, chi_np), dtype=torch.float64)
    ref_psi, ref_chi_delta = orbit.components(reference)
    psi_scale = torch.sum(torch.abs(ref_psi)**2).detach()
    chi_scale = torch.sum(torch.abs(ref_chi_delta)**2).detach()

    dt_guess = 0.45 / updates
    phase_guess = rank.R * updates * dt_guess
    shifted = orbit.translate(reference, torch.tensor(phase_guess),
                              torch.tensor(0.0))
    q0 = reference.detach().clone().requires_grad_(True)
    qminus = (reference - (shifted - reference) / updates
              ).detach().requires_grad_(True)
    raw_dt = torch.tensor(orbit.raw_from_dt(dt_guess), dtype=torch.float64,
                          requires_grad=True)
    psi_phase = torch.tensor(phase_guess, dtype=torch.float64,
                             requires_grad=True)
    chi_phase = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [qminus, q0, raw_dt, psi_phase, chi_phase], lr=0.25,
        max_iter=iterations, max_eval=iterations * 2,
        tolerance_grad=1e-11, tolerance_change=1e-15,
        history_size=80, line_search_fn="strong_wolfe")
    trace = []

    def evolve():
        dt = orbit.dt_from_raw(raw_dt)
        states = [qminus, q0]
        for _ in range(updates):
            states.append(orbit.update(states[-2], states[-1], dt))
        return dt, states

    def normalized_difference(left: torch.Tensor, right: torch.Tensor):
        lp, lc = orbit.components(left)
        rp, rc = orbit.components(right)
        return (torch.sum(torch.abs(lp-rp)**2) / psi_scale
                + torch.sum(torch.abs(lc-rc)**2) / chi_scale)

    def objective():
        optimizer.zero_grad()
        dt, states = evolve()
        end = orbit.translate(q0, psi_phase, chi_phase)
        before_end = orbit.translate(qminus, psi_phase, chi_phase)
        loss = normalized_difference(states[-1], end)
        loss = loss + normalized_difference(states[-2], before_end)
        psi0, chi0 = orbit.unpack(q0)
        charge = torch.sum(torch.abs(psi0)**2)
        loss = loss + 10 * ((charge-psi_scale)/psi_scale)**2
        # Gauge conditions remove the two arbitrary common field phases.
        loss = loss + (torch.imag(torch.sum(torch.conj(ref_psi)*psi0))/psi_scale)**2
        vacuum = torch.zeros_like(chi0)
        vacuum[..., 2] = rank.R
        loss = loss + (torch.imag(torch.sum(
            torch.conj(ref_chi_delta)*(chi0-vacuum)))/chi_scale)**2
        loss.backward()
        trace.append(float(torch.sqrt(loss.detach())))
        return loss

    initial = float(torch.sqrt(objective().detach()))
    optimizer.step(objective)
    dt, states = evolve()
    end = orbit.translate(q0, psi_phase, chi_phase)
    before_end = orbit.translate(qminus, psi_phase, chi_phase)
    boundary = float(torch.sqrt(
        normalized_difference(states[-1], end)
        + normalized_difference(states[-2], before_end)).detach())
    metrics = field_metrics(q0, reference)

    second = list(states)
    for _ in range(updates):
        second.append(orbit.update(second[-2], second[-1], dt))
    end2 = orbit.translate(end, psi_phase, chi_phase)
    second_residual = float(torch.sqrt(
        normalized_difference(second[-1], end2)).detach())
    same_carrier = (metrics["eight_site_psi_fidelity"] > 0.99
                    and metrics["psi_core_containment"] > 0.99
                    and metrics["chi_track_fidelity"] > 0.99)
    return {
        "updates_per_cell": updates,
        "initial_objective": initial,
        "final_boundary_residual": boundary,
        "unreset_second_cell_residual": second_residual,
        "solved_dt": float(dt.detach()),
        "solved_translation_time": float(updates*dt.detach()),
        "solved_psi_phase": float(psi_phase.detach()),
        "solved_chi_phase": float(chi_phase.detach()),
        "optimizer_evaluations": len(trace),
        "minimum_objective_trace": min(trace),
        "carrier_metrics": metrics,
        "same_carrier": same_carrier,
        "converged_same_carrier": boundary < 1e-8 and second_residual < 1e-7
                                  and same_carrier,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    row = shoot(args.updates, args.iterations)
    result = {
        "status": ("V4_CANONICAL_DEFORMED_CELL_ORBIT_PASS"
                   if row["converged_same_carrier"] else
                   "V4_CANONICAL_DEFORMED_CELL_ORBIT_INCONCLUSIVE"),
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "row": row,
        "verdict": ("FREELY_DEFORMED_SAME_CARRIER_ORBIT_FOUND"
                    if row["converged_same_carrier"] else
                    "NO_SAME_CARRIER_ORBIT_CONVERGED"),
        "claim_boundary": (
            "Charge-constrained numerical shooting test in one direction. "
            "Nonconvergence is not a universal no-go. A pass would still "
            "require refinement, all axes/signs, Dirac current, and Lean."),
    }
    output = HERE / "canonical_deformed_relative_periodic_orbit.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
