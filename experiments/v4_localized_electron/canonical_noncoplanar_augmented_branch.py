"""Full augmented Newton solve of the canonical noncoplanar branch.

The reduced four-direction search provides a finite seed.  This calculation
solves the complete constrained relative-equilibrium equations

    grad S_orbit(x) + lambda d = 0,
    d . (x - x0) = epsilon,

so the solution cannot collapse back to the coplanar rest state.  No action
term, target force, or Dirac structure is added.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as audit
import canonical_noncoplanar_branch_probe as branch


HERE = Path(__file__).resolve().parent
TARGET_FRACTIONS = (0.01, 0.02, 0.03)


def main() -> None:
    data = np.load(HERE / "canonical_noncoplanar_solvability_search.npz")
    rest = data["rest"]
    direction = data["direction"]
    direction = direction / np.linalg.norm(direction)
    seed_fractions = data["fractions"]
    seed_states = data["states"]
    theta = float(np.arccos(1 - audit.DT**2 * audit.CHI0**2 / 2))
    phase = np.exp(1j * theta)

    def orbit_action(raw: torch.Tensor) -> torch.Tensor:
        return (audit.interval_action(
            audit.phase_rotate(raw, np.conjugate(phase)), raw)
            + audit.interval_action(raw, audit.phase_rotate(raw, phase)))

    def residual(raw: np.ndarray) -> np.ndarray:
        value = torch.tensor(raw, dtype=torch.float64, requires_grad=True)
        return torch.autograd.grad(
            orbit_action(value), value)[0].detach().numpy()

    rest_norm = np.linalg.norm(rest)
    rows = []
    solutions = []
    previous_state = None
    previous_fraction = None
    for fraction in TARGET_FRACTIONS:
        epsilon = fraction * rest_norm
        if previous_state is None:
            index = int(np.argmin(abs(seed_fractions - fraction)))
            state = seed_states[index].copy()
            state += (epsilon - direction @ (state-rest)) * direction
        else:
            state = previous_state + (
                fraction-previous_fraction) * rest_norm * direction
        multiplier = 0.0
        trace = []
        for iteration in range(10):
            value = residual(state)
            constraint = float(direction @ (state-rest) - epsilon)
            equations = np.r_[value + multiplier*direction, constraint]
            rms = float(np.linalg.norm(value) / np.sqrt(value.size))
            trace.append({
                "iteration": iteration,
                "orbit_residual_rms": rms,
                "amplitude_constraint": constraint,
                "multiplier": multiplier,
                "augmented_norm": float(np.linalg.norm(equations)),
            })
            if rms < 1e-11 and abs(constraint) < 1e-11:
                break
            tensor = torch.tensor(state, dtype=torch.float64,
                                  requires_grad=True)
            hessian = torch.autograd.functional.hessian(
                orbit_action, tensor, vectorize=True).detach().numpy()
            hessian = (hessian + hessian.T) / 2
            augmented = np.block([
                [hessian, direction[:, None]],
                [direction[None, :], np.zeros((1, 1))],
            ])
            step, _, _, _ = np.linalg.lstsq(
                augmented, -equations, rcond=1e-10)
            accepted = False
            old_norm = np.linalg.norm(equations)
            for power in range(12):
                scale = 0.5**power
                trial_state = state + scale*step[:-1]
                trial_multiplier = multiplier + scale*step[-1]
                trial_value = residual(trial_state)
                trial_constraint = float(
                    direction @ (trial_state-rest) - epsilon)
                trial_equations = np.r_[
                    trial_value + trial_multiplier*direction,
                    trial_constraint]
                if np.linalg.norm(trial_equations) < old_norm:
                    state = trial_state
                    multiplier = float(trial_multiplier)
                    accepted = True
                    break
            if not accepted:
                break
        final = residual(state)
        constraint = float(direction @ (state-rest) - epsilon)
        metrics = branch.identity_metrics(rest, state)
        row = {
            "fraction_of_rest_norm": fraction,
            "orbit_residual_rms": float(
                np.linalg.norm(final) / np.sqrt(final.size)),
            "amplitude_constraint": constraint,
            "multiplier": multiplier,
            "unconstrained_residual_rms": float(
                np.linalg.norm(final) / np.sqrt(final.size)),
            "trace": trace,
            **metrics,
        }
        rows.append(row)
        solutions.append(state)
        previous_state = state
        previous_fraction = fraction

    exact = all(row["orbit_residual_rms"] < 1e-10
                and abs(row["amplitude_constraint"]) < 1e-10
                and abs(row["multiplier"]) < 1e-8
                for row in rows)
    identity = all(row["psi_locked_fidelity"] > 0.999
                   and row["maximum_opposite_pair_error"] < 0.01
                   for row in rows)
    verdict = (
        "FINITE_CANONICAL_NONCOPLANAR_REST_BRANCH_SOLVED"
        if exact and identity else
        "FINITE_BRANCH_SOLVED_BUT_LOCKED_IDENTITY_FAILS"
        if exact else
        "AUGMENTED_NONCOPLANAR_BRANCH_NOT_SOLVED")
    result = {
        "status": "V4_CANONICAL_NONCOPLANAR_AUGMENTED_BRANCH",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "verdict": verdict,
        "claim_boundary": (
            "Finite constrained relative equilibria on side three. A pass is "
            "not stability, translation, Dirac current, or box refinement."),
    }
    output = HERE / "canonical_noncoplanar_augmented_branch.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    np.savez_compressed(
        HERE / "canonical_noncoplanar_augmented_branch.npz",
        rest=rest, direction=direction,
        fractions=np.asarray(TARGET_FRACTIONS), states=np.asarray(solutions))
    print(json.dumps({"verdict": verdict, "rows": rows,
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
