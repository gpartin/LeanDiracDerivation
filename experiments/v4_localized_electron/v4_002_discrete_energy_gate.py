"""Fixed-step discrete-energy audit for the V4-002 nonlinear trajectory.

For an autonomous discrete Lagrangian ``Ld(q_n,q_{n+1}; dt)``, the natural
interval energy is ``-partial Ld / partial dt`` with the endpoint fields held
fixed.  The production action contains a centered-clock expression evaluated
through NumPy, so this audit takes a symmetric numerical derivative and checks
that both the derivative and its drift converge under refinement of the
derivative step.  This is an energy diagnostic of the full native state; it is
not the later Dirac energy/readout identification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import live_clifford_bond_action as action


HERE = Path(__file__).resolve().parent


def interval_energy(left: np.ndarray, right: np.ndarray, dt: float,
                    relative_step: float) -> float:
    h = relative_step*dt
    q0 = torch.tensor(left, dtype=torch.float64)
    q1 = torch.tensor(right, dtype=torch.float64)
    plus = action.interval(q0, q1, dt+h)
    minus = action.interval(q0, q1, dt-h)
    return float(-(plus-minus)/(2*h))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--relative-steps", nargs="+", type=float,
                        default=(1e-4, 5e-5, 2.5e-5))
    args = parser.parse_args()

    archive_path = (Path(args.archive) if Path(args.archive).is_absolute()
                    else HERE/args.archive)
    archive = np.load(archive_path)
    states = archive["states"]
    dt = float(archive["dt"])
    rows = []
    for relative_step in args.relative_steps:
        values = [interval_energy(states[n], states[n+1], dt, relative_step)
                  for n in range(len(states)-1)]
        scale = max(max(abs(value) for value in values), 1e-300)
        rows.append({
            "relative_derivative_step": relative_step,
            "interval_energies": values,
            "maximum_relative_drift": max(
                abs(value-values[0]) for value in values)/scale,
        })
    reference = rows[-1]
    refinement_changes = []
    for coarse, fine in zip(rows, rows[1:]):
        coarse_values = np.asarray(coarse["interval_energies"])
        fine_values = np.asarray(fine["interval_energies"])
        refinement_changes.append(float(
            np.max(abs(coarse_values-fine_values))
            / max(np.max(abs(fine_values)), 1e-300)))
    passed = (reference["maximum_relative_drift"] < 1e-6
              and (not refinement_changes
                   or refinement_changes[-1] < 1e-8))
    result = {
        "status": "V4_002_DISCRETE_ENERGY_GATE",
        "action_configuration": {
            "link_source": action.LINK_SOURCE,
            "scale": action.SCALE,
            "spacetime_scale": action.SPACETIME_PLAQUETTE_SCALE,
            "parallel_matched_scale": action.PARALLEL_MATCHED_SCALE,
            "parallel_potential_scale": (
                action.PARALLEL_MATCHED_POTENTIAL_SCALE),
        },
        "trajectory_updates": len(states)-2,
        "dt": dt,
        "rows": rows,
        "successive_refinement_changes": refinement_changes,
        "verdict": ("DISCRETE_ENERGY_BOUNDED_PASS" if passed
                    else "DISCRETE_ENERGY_CHECK_OPEN"),
        "claim_boundary": (
            "Full-state autonomous discrete-Lagrangian interval energy on "
            "the short unrestricted nonlinear V4-002 trajectory. Dirac "
            "energy identification and long moving-carrier endurance are "
            "later gates."),
    }
    output = (Path(args.output) if Path(args.output).is_absolute()
              else HERE/args.output)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
