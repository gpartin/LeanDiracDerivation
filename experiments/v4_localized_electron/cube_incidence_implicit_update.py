"""Implicit discrete variational update for the unit cube-incidence action."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_nonlinear_action as action


HERE = Path(__file__).resolve().parent


def rest_following_jacobian(rotation: np.ndarray, matrices):
    plus = -matrices["mass"]/native_time.DT-matrices["gyro"]/2
    return plus@rotation.T


def residual(previous: np.ndarray, current: np.ndarray,
             following: np.ndarray) -> np.ndarray:
    tensors = (
        torch.tensor(previous, dtype=torch.float64),
        torch.tensor(current, dtype=torch.float64, requires_grad=True),
        torch.tensor(following, dtype=torch.float64, requires_grad=True),
    )
    return action.del_residual(*tensors).detach().numpy()


def solve_following(previous: np.ndarray, current: np.ndarray,
                    guess: np.ndarray, factor, tolerance: float = 1e-10,
                    maximum_iterations: int = 16):
    following = guess.copy()
    history = []
    for iteration in range(maximum_iterations):
        value = residual(previous, current, following)
        norm = float(np.linalg.norm(value)/np.sqrt(value.size))
        history.append(norm)
        if norm < tolerance:
            return following, history, True
        step = lu_solve(factor, -value)
        accepted = False
        for power in range(9):
            scale = 0.5**power
            trial = following+scale*step
            trial_norm = float(np.linalg.norm(
                residual(previous, current, trial))/np.sqrt(value.size))
            if trial_norm < norm:
                following = trial
                accepted = True
                break
        if not accepted:
            return following, history, False
    final = float(np.linalg.norm(
        residual(previous, current, following))/np.sqrt(following.size))
    history.append(final)
    return following, history, final < tolerance


def main() -> None:
    bare, matrices = action.expected_blocks()
    theta = float(np.arccos(
        1-native_time.DT**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(bare["rest"].size, theta)
    factor = lu_factor(rest_following_jacobian(rotation, matrices))
    previous = rotation.T@bare["rest"]
    current = bare["rest"]
    expected = rotation@bare["rest"]
    guess = 2*current-previous
    following, history, converged = solve_following(
        previous, current, guess, factor)

    # Perturb the carrier using the previously identified common 001 defect.
    import cube_geometry_bifurcation_residual as mode
    fit = json.loads((HERE/"cube_geometry_bifurcation_solve.json").read_text())
    values = np.asarray(
        fit["continuous_solution"]["coefficients_in_B_chi0_units"])
    rest, _blocks, old_matrices = mode.assemble(values)
    _multiplier, complex_vector = mode.fastest_mode(rest, old_matrices)
    pivot = int(np.argmax(abs(complex_vector)))
    vector = np.real(np.exp(-1j*np.angle(complex_vector[pivot]))
                     * complex_vector)
    vector *= 0.01*np.linalg.norm(current)/max(np.linalg.norm(vector), 1e-30)
    perturbed_current = current+vector
    perturbed_previous = rotation.T@perturbed_current
    perturbed_guess = 2*perturbed_current-perturbed_previous
    perturbed_following, perturbed_history, perturbed_converged = (
        solve_following(perturbed_previous, perturbed_current,
                        perturbed_guess, factor))
    perturbation_gain = float(np.linalg.norm(
        rotation.T@perturbed_following-current)/np.linalg.norm(vector))

    result = {
        "status": "V4_CUBE_INCIDENCE_IMPLICIT_UPDATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "solver": "fixed-rest-Jacobian Newton with residual-decrease line search",
        "rest_update": {
            "converged": converged,
            "iterations": len(history)-1,
            "residual_history_RMS": history,
            "relative_next_layer_error": float(
                np.linalg.norm(following-expected)/np.linalg.norm(expected)),
        },
        "one_percent_001_update": {
            "converged": perturbed_converged,
            "iterations": len(perturbed_history)-1,
            "residual_history_RMS": perturbed_history,
            "one_update_corotating_perturbation_gain": perturbation_gain,
        },
        "verdict": (
            "IMPLICIT_VARIATIONAL_UPDATE_PASS"
            if converged and perturbed_converged
            and np.linalg.norm(following-expected)/np.linalg.norm(expected)
            < 1e-9 else "IMPLICIT_VARIATIONAL_UPDATE_FAIL"),
        "claim_boundary": (
            "One exact-rest and one perturbed update. It does not establish "
            "100-cycle stability, conservation, movement, current, or V4-006."),
    }
    np.savez_compressed(
        HERE/"cube_incidence_implicit_update.npz",
        rest_previous=previous, rest_current=current,
        rest_following=following, perturbed_previous=perturbed_previous,
        perturbed_current=perturbed_current,
        perturbed_following=perturbed_following)
    (HERE/"cube_incidence_implicit_update.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
