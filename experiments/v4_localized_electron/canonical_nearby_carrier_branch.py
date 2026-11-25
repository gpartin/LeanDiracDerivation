"""Continue a nearby rest orbit along the measured unstable carrier mode.

The solve changes the full Psi/chi state and the common Psi phase step.  It
does not hold a corner target or add a runtime force.  Two fixed symmetry
gauges remove the native global Psi and chi phases; their multipliers must
return to zero for a genuine unconstrained orbit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lstsq

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_box_transfer as box
import canonical_noncoplanar_large_box_audit as large
import canonical_transfer_oscillator_implicit_update as action


HERE = Path(__file__).resolve().parent
SIDE = large.SIDE
FRACTIONS = (1e-4,)


def rotate_psi(raw: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    shaped = raw.reshape(SIDE, SIDE, SIDE, 4, 3)
    c, s = torch.cos(theta), torch.sin(theta)
    result = shaped.clone()
    result[..., 0, :] = c*shaped[..., 0, :]-s*shaped[..., 1, :]
    result[..., 1, :] = s*shaped[..., 0, :]+c*shaped[..., 1, :]
    return result.reshape(-1)


def orbit_residual(raw: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    return action.del_residual(
        rotate_psi(raw, -theta), raw, rotate_psi(raw, theta))


def phase_tangents(rest: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shaped = rest.reshape(SIDE, SIDE, SIDE, 4, 3)
    psi = np.zeros_like(shaped)
    chi = np.zeros_like(shaped)
    psi[..., 0, :] = -shaped[..., 1, :]
    psi[..., 1, :] = shaped[..., 0, :]
    chi[..., 2, :] = -shaped[..., 3, :]
    chi[..., 3, :] = shaped[..., 2, :]
    return psi.reshape(-1), chi.reshape(-1)


def orthogonalize(vector: np.ndarray, columns: list[np.ndarray]) -> np.ndarray:
    result = vector.copy()
    for column in columns:
        result -= column*np.dot(column, result)/np.dot(column, column)
    return result/np.linalg.norm(result)


def augmented_value(state: np.ndarray, theta: float, lambdas: np.ndarray,
                    rest: np.ndarray, direction: np.ndarray, target: float,
                    phases: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    raw = torch.tensor(state, dtype=torch.float64)
    angle = torch.tensor(theta, dtype=torch.float64)
    residual = orbit_residual(raw, angle).detach().numpy()
    residual += lambdas[0]*phases[0]+lambdas[1]*phases[1]
    displacement = state-rest
    return np.r_[residual,
                 np.dot(direction, displacement)-target,
                 np.dot(phases[0], displacement),
                 np.dot(phases[1], displacement)]


def solve_branch(rest: np.ndarray, direction: np.ndarray, theta0: float,
                 fraction: float, phases: tuple[np.ndarray, np.ndarray]) -> dict:
    target = fraction*np.linalg.norm(rest)
    state = rest+target*direction
    theta = theta0
    lambdas = np.zeros(2)
    trace = []
    for iteration in range(5):
        value = augmented_value(
            state, theta, lambdas, rest, direction, target, phases)
        rms = float(np.linalg.norm(value[:rest.size])/np.sqrt(rest.size))
        total = float(np.linalg.norm(value)/np.sqrt(value.size))
        trace.append({"iteration": iteration, "field_rms": rms,
                      "augmented_rms": total})
        if total < 1e-11:
            break
        raw = torch.tensor(state, dtype=torch.float64)
        angle = torch.tensor(theta, dtype=torch.float64)
        jq = torch.autograd.functional.jacobian(
            lambda q: orbit_residual(q, angle), raw,
            vectorize=True).detach().numpy()
        jt = torch.autograd.functional.jacobian(
            lambda t: orbit_residual(raw, t), angle,
            vectorize=True).detach().numpy()
        n = rest.size
        matrix = np.zeros((n+3, n+3))
        matrix[:n, :n] = jq
        matrix[:n, n] = jt
        matrix[:n, n+1] = phases[0]
        matrix[:n, n+2] = phases[1]
        matrix[n, :n] = direction
        matrix[n+1, :n] = phases[0]
        matrix[n+2, :n] = phases[1]
        correction, _residuals, rank, singular = lstsq(
            matrix, -value, cond=1e-12, check_finite=False,
            lapack_driver="gelsy")
        trace[-1]["bordered_rank"] = int(rank)
        trace[-1]["bordered_dimension"] = int(matrix.shape[0])
        accepted = False
        for power in range(12):
            scale = 0.5**power
            trial = augmented_value(
                state+scale*correction[:n], theta+scale*correction[n],
                lambdas+scale*correction[n+1:], rest, direction, target,
                phases)
            if np.linalg.norm(trial) < np.linalg.norm(value):
                state += scale*correction[:n]
                theta += scale*correction[n]
                lambdas += scale*correction[n+1:]
                trace[-1]["accepted_scale"] = scale
                accepted = True
                break
        if not accepted:
            trace[-1]["accepted_scale"] = 0.0
            break
    final = augmented_value(
        state, theta, lambdas, rest, direction, target, phases)
    shaped = state.reshape(SIDE, SIDE, SIDE, 4, 3)
    psi_power = np.sum(shaped[..., :2, :]**2)
    core_psi_power = np.sum(shaped[1:3, 1:3, 1:3, :2, :]**2)
    return {
        "fraction": fraction,
        "theta": theta,
        "relative_state_change": float(
            np.linalg.norm(state-rest)/np.linalg.norm(rest)),
        "field_residual_rms": float(
            np.linalg.norm(final[:rest.size])/np.sqrt(rest.size)),
        "augmented_residual_rms": float(
            np.linalg.norm(final)/np.sqrt(final.size)),
        "phase_multipliers": lambdas.tolist(),
        "Psi_core_containment": float(core_psi_power/psi_power),
        "trace": trace,
        "state": state,
    }


def main() -> None:
    action.STRENGTH = 0.0
    action.EVEN_STRENGTH = 2.0
    action.DENSITY_BALANCE_STRENGTH = 0.5
    archive = np.load(
        HERE / "canonical_transfer_oscillator_spectrum_0_even_2_density_0p5_leading_mode.npz")
    rest = archive["rest"]
    phase_space = archive["phase_space_vector"]
    current_mode = phase_space[:rest.size]
    phases = phase_tangents(rest)
    candidates = [np.real(current_mode), np.imag(current_mode)]
    direction = max(candidates, key=np.linalg.norm)
    direction = orthogonalize(direction, list(phases))
    theta0 = float(np.arccos(1-small.DT**2*small.CHI0**2/2))
    rows = []
    states = {}
    for sign in (-1, 1):
        for fraction in FRACTIONS:
            row = solve_branch(rest, sign*direction, theta0, fraction, phases)
            states[f"sign_{sign}_fraction_{fraction:g}"] = row.pop("state")
            row["sign"] = sign
            rows.append(row)
    passed = [row for row in rows
              if row["field_residual_rms"] < 1e-9
              and max(abs(x) for x in row["phase_multipliers"]) < 1e-9]
    result = {
        "status": "V4_NEARBY_CARRIER_BRANCH",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "research_coefficients": {
            "odd_transfer": 0.0, "even_transfer": 2.0,
            "density_balance": 0.5},
        "rows": rows,
        "genuine_unconstrained_orbit_count": len(passed),
        "verdict": ("NEARBY_NATIVE_REST_BRANCH_FOUND" if passed else
                    "NEARBY_BRANCH_NOT_CLOSED"),
        "claim_boundary": (
            "Full-field local continuation along the measured unstable mode. "
            "A solved rest orbit still requires spectrum, pattern, continuum, "
            "transport, and finite-handoff gates."),
    }
    (HERE / "canonical_nearby_carrier_branch.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    np.savez_compressed(HERE / "canonical_nearby_carrier_branch.npz", **states)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
