"""Search the four canonical pair-preserving transverse directions jointly.

The individual strongest direction has a cubic Lyapunov--Schmidt residual.
This script constructs all four physical transverse directions and minimizes
the complete nullspace solvability residual over their unit sphere.  It tests
whether their nonlinear responses cancel without changing the V40 action.
"""

from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

import canonical_noncoplanar_bifurcation_audit as audit
import canonical_noncoplanar_branch_probe as branch
import localized_stability as native


HERE = Path(__file__).resolve().parent
FRACTIONS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def main() -> None:
    psi, chi = native.carrier_fields(audit.SIDE, audit.AMPLITUDE_SQ)
    current = audit.pack(psi, chi)
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

    base = torch.tensor(current, dtype=torch.float64, requires_grad=True)
    hessian = torch.autograd.functional.hessian(
        orbit_action, base, vectorize=True).detach().numpy()
    hessian = (hessian + hessian.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    scale = max(float(np.max(abs(eigenvalues))), 1e-300)
    null_mask = abs(eigenvalues) < 1e-8 * scale
    null_basis = eigenvectors[:, null_mask]
    range_basis = eigenvectors[:, ~null_mask]
    range_values = eigenvalues[~null_mask]
    occupied = np.sum(abs(psi)**2, axis=-1) > 1e-12

    pair_constraint = np.column_stack([
        audit.opposite_pair_constraint(
            current, null_basis[:, index], occupied)
        for index in range(null_basis.shape[1])
    ])
    pair_kernel = audit.kernel(pair_constraint)
    candidate_basis = null_basis @ pair_kernel
    transverse = np.column_stack([
        audit.carrier_transverse_w_tangent(
            current, candidate_basis[:, index], occupied)
        for index in range(candidate_basis.shape[1])
    ])
    symmetries = audit.global_tangents(current)
    transverse_symmetry = np.column_stack([
        audit.carrier_transverse_w_tangent(
            current, symmetries[:, index], occupied)
        for index in range(symmetries.shape[1])
    ])
    symmetry_basis = audit.orth(transverse_symmetry)
    quotient = transverse - symmetry_basis @ (symmetry_basis.T @ transverse)
    _u, singular, vh = np.linalg.svd(quotient, full_matrices=False)
    rank = int(np.sum(singular > 1e-8 * singular[0]))
    physical = candidate_basis @ vh[:rank].T
    physical, _ = np.linalg.qr(physical)

    def range_correct(direction: np.ndarray, epsilon: float):
        state = current + epsilon * direction
        for _iteration in range(10):
            value = residual(state)
            coordinates = range_basis.T @ value
            old = np.linalg.norm(coordinates)
            if old < 1e-13:
                break
            correction = -range_basis @ (coordinates / range_values)
            for power in range(12):
                trial = state + 0.5**power * correction
                if np.linalg.norm(range_basis.T @ residual(trial)) < old:
                    state = trial
                    break
            else:
                break
        value = residual(state)
        return state, null_basis.T @ value, range_basis.T @ value

    rest_norm = np.linalg.norm(current)
    rng = np.random.default_rng(20261123)

    # A homogeneous cubic in four variables has only 20 monomials.  Rebuild
    # the reduced Lyapunov--Schmidt map once instead of solving the full
    # lattice inside every optimizer evaluation.
    exponents = [value for value in itertools.product(range(4), repeat=rank)
                 if sum(value) == 3]

    def features(coefficients: np.ndarray) -> np.ndarray:
        return np.asarray([
            np.prod(coefficients**np.asarray(exponent))
            for exponent in exponents])

    fit_fraction = 3e-4
    fit_epsilon = fit_fraction * rest_norm
    sample_coefficients = []
    sample_outputs = []
    deterministic = [np.eye(rank)[index] for index in range(rank)]
    deterministic += [-np.eye(rank)[index] for index in range(rank)]
    samples = deterministic + [rng.normal(size=rank) for _ in range(32)]
    for sample in samples:
        coefficients = sample / np.linalg.norm(sample)
        direction = physical @ coefficients
        _state, null_value, _range_value = range_correct(
            direction, fit_epsilon)
        sample_coefficients.append(coefficients)
        sample_outputs.append(null_value / fit_epsilon**3)
    design = np.vstack([features(value) for value in sample_coefficients])
    outputs = np.vstack(sample_outputs)
    cubic_coefficients, _, _, _ = np.linalg.lstsq(
        design, outputs, rcond=1e-11)
    fit_defect = float(np.linalg.norm(design @ cubic_coefficients - outputs)
                       / max(np.linalg.norm(outputs), 1e-300))

    def reduced_objective(raw_coefficients: np.ndarray) -> float:
        norm = np.linalg.norm(raw_coefficients)
        if norm < 1e-14:
            return 1e30
        coefficients = raw_coefficients / norm
        predicted = features(coefficients) @ cubic_coefficients
        return float(np.vdot(predicted, predicted).real)

    seeds = [np.eye(rank)[index] for index in range(rank)]
    seeds += [rng.normal(size=rank) for _ in range(64)]
    reduced_solutions = []
    for seed in seeds:
        optimization = minimize(
            reduced_objective, seed, method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 1e-10, "fatol": 1e-16})
        coefficients = optimization.x / np.linalg.norm(optimization.x)
        reduced_solutions.append((reduced_objective(coefficients),
                                  coefficients, optimization))
    reduced_solutions.sort(key=lambda value: value[0])
    predicted_norm_sq, best_coefficients, best_optimization = \
        reduced_solutions[0]

    rows = []
    solved_states = []
    for fraction in FRACTIONS:
        epsilon = fraction * rest_norm
        direction = physical @ best_coefficients
        state, null_value, range_value = range_correct(direction, epsilon)
        null_norm = np.linalg.norm(null_value)
        metrics = branch.identity_metrics(current, state)
        rows.append({
            "fraction_of_rest_norm": fraction,
            "epsilon": epsilon,
            "physical_direction_coefficients": [float(x)
                                                  for x in best_coefficients],
            "null_solvability_norm": float(null_norm),
            "null_norm_over_epsilon_cubed": float(
                null_norm / max(epsilon**3, 1e-300)),
            "range_norm": float(np.linalg.norm(range_value)),
            "total_residual_rms": float(
                np.linalg.norm(residual(state)) / np.sqrt(state.size)),
            "optimizer_success": bool(best_optimization.success),
            "optimizer_message": str(best_optimization.message),
            **metrics,
        })
        solved_states.append(state)

    ratios = np.asarray([row["null_norm_over_epsilon_cubed"]
                         for row in rows])
    # The smallest-amplitude normalized ratios are noise-amplified.  Require
    # the two largest direct full-system tests to remain at roundoff instead.
    canceled = bool(all(row["total_residual_rms"] < 1e-10
                        for row in rows[-2:]))
    improved = bool(rows[-1]["null_solvability_norm"]
                    < 0.1 * 5.501174680716718e-9)
    verdict = (
        "FOUR_DIRECTION_CUBIC_OBSTRUCTION_CANCELED" if canceled else
        "FOUR_DIRECTION_SEARCH_REDUCES_BUT_DOES_NOT_CANCEL_OBSTRUCTION"
        if improved else
        "FOUR_DIRECTION_SEARCH_DOES_NOT_REMOVE_CUBIC_OBSTRUCTION")
    result = {
        "status": "V4_CANONICAL_NONCOPLANAR_SOLVABILITY_SEARCH",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "physical_pair_preserving_transverse_rank": rank,
        "singular_values": [float(x) for x in singular[:rank]],
        "cubic_monomial_count": len(exponents),
        "cubic_fit_sample_count": len(samples),
        "cubic_fit_relative_defect": fit_defect,
        "best_predicted_cubic_residual_norm": float(
            np.sqrt(predicted_norm_sq)),
        "rows": rows,
        "verdict": verdict,
        "claim_boundary": (
            "Deterministic multistart numerical minimization of the complete "
            "nullspace residual over the four linear physical directions. "
            "Failure is not an analytic no-go for disconnected branches."),
    }
    output = HERE / "canonical_noncoplanar_solvability_search.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    np.savez_compressed(
        HERE / "canonical_noncoplanar_solvability_search.npz",
        rest=current, physical=physical,
        best_coefficients=best_coefficients,
        direction=physical @ best_coefficients,
        fractions=np.asarray(FRACTIONS), states=np.asarray(solved_states))
    print(json.dumps({
        "verdict": verdict,
        "physical_rank": rank,
        "rows": rows,
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
