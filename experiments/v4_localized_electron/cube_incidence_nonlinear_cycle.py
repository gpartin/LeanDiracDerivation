"""Unrestricted nonlinear cycle gate for the unit cube-incidence action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as native_box
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_implicit_update as update
import cube_incidence_nonlinear_action as action
import localized_stability as native


HERE = Path(__file__).resolve().parent
SIDE = 5


def diagnostics(previous, current, expected, rest_psi, carrier):
    current_tensor = torch.tensor(current, dtype=torch.float64)
    previous_tensor = torch.tensor(previous, dtype=torch.float64)
    expected_tensor = torch.tensor(expected, dtype=torch.float64)
    psi, chi = native_box.unpack(current_tensor, SIDE)
    expected_psi, _ = native_box.unpack(expected_tensor, SIDE)
    total = float(np.sum(abs(psi.detach().numpy())**2))
    carrier_power = float(np.sum(abs(psi.detach().numpy()[carrier])**2))
    overlap = np.vdot(expected_psi.detach().numpy()[carrier],
                      psi.detach().numpy()[carrier])
    phase = overlap/abs(overlap) if abs(overlap) else 1+0j
    pattern = float(np.linalg.norm(
        psi.detach().numpy()[carrier]
        -phase*expected_psi.detach().numpy()[carrier])
        / max(np.linalg.norm(rest_psi[carrier]), 1e-30))
    ppsi, pchi = native_box.unpack(previous_tensor, SIDE)
    psi_velocity = (psi-ppsi)/native_time.DT
    chi_velocity = (chi-pchi)/native_time.DT
    kinetic = 0.5*float(np.sum(abs(psi_velocity.detach().numpy())**2))
    kinetic += native_time.B/2*float(np.sum(
        abs(chi_velocity.detach().numpy())**2))
    potential = float(native_box.potential(current_tensor, SIDE).detach()
                      + action.extra_potential(current_tensor).detach())
    return {
        "corotating_relative_deviation": float(
            np.linalg.norm(current-expected)/np.linalg.norm(expected)),
        "carrier_containment": carrier_power/max(total, 1e-30),
        "carrier_pattern_defect": pattern,
        "energy": kinetic+potential,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=float, default=1.0)
    parser.add_argument("--output", default="cube_incidence_nonlinear_cycle.json")
    args = parser.parse_args()
    bare, matrices = action.expected_blocks()
    theta = float(np.arccos(
        1-native_time.DT**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(bare["rest"].size, theta)
    factor = lu_factor(update.rest_following_jacobian(rotation, matrices))
    archived = np.load(HERE/"cube_incidence_implicit_update.npz")
    previous = archived["perturbed_previous"]
    current = archived["perturbed_current"]
    rest_psi, _rest_chi = native.unpack(bare["rest"], SIDE)
    carrier = native.norm2(rest_psi) > 1e-12
    steps = int(np.ceil(args.cycles*2*np.pi/theta))
    expected = bare["rest"].copy()
    rows = []
    converged = True
    total_iterations = 0
    for step in range(steps+1):
        if step == 0 or step == steps or step % max(steps//20, 1) == 0:
            row = diagnostics(previous, current, expected,
                              rest_psi, carrier)
            row["step"] = step
            rows.append(row)
        if step == steps:
            break
        guess = 2*current-previous
        following, history, passed = update.solve_following(
            previous, current, guess, factor, tolerance=1e-8,
            maximum_iterations=16)
        total_iterations += len(history)-1
        if not passed:
            converged = False
            rows.append({"step": step+1,
                         "solver_residual_RMS": history[-1],
                         "solver_failure": True})
            break
        previous, current = current, following
        expected = rotation@expected
    physical = [row for row in rows if "energy" in row]
    energies = np.asarray([row["energy"] for row in physical])
    initial_energy = max(abs(float(energies[0])), 1e-30)
    result = {
        "status": "V4_CUBE_INCIDENCE_NONLINEAR_CYCLE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "requested_cycles": args.cycles,
        "requested_steps": steps,
        "completed_steps": int(physical[-1]["step"]),
        "solver_converged_every_step": converged,
        "average_Newton_iterations": (
            total_iterations/max(int(physical[-1]["step"]), 1)),
        "maximum_corotating_relative_deviation": max(
            row["corotating_relative_deviation"] for row in physical),
        "maximum_carrier_pattern_defect": max(
            row["carrier_pattern_defect"] for row in physical),
        "minimum_carrier_containment": min(
            row["carrier_containment"] for row in physical),
        "maximum_relative_energy_drift": float(
            np.max(abs(energies-energies[0]))/initial_energy),
        "sampled_rows": rows,
        "verdict": (
            "NONLINEAR_CYCLE_PRELIMINARY_PASS"
            if converged and int(physical[-1]["step"]) == steps
            else "NONLINEAR_CYCLE_FAIL"),
        "claim_boundary": (
            "Finite-duration one-percent 001 perturbation run. A preliminary "
            "pass is not the required 100-cycle, movement, current, or V4-006 "
            "pass."),
    }
    (HERE/args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "sampled_rows"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
