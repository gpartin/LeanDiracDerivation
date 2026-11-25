"""Derive the native two-layer graph for the localized four-component lift.

For a configuration embedding Q, an invariant first-order sector of the
native two-layer recurrence has the graph form E=[Q; Q C], where C maps the
current retained coordinates to the previous layer.  If the native update is
M=[[F,G],[I,0]], invariance requires

    (I-QQ^T)(F Q + G Q C) = 0,
    F Q + G Q C = Q C^-1.

The first condition is a linear least-squares problem for C; the second then
tests the complete recurrence.  This derives the time-layer relation from the
native discrete action and never inserts a Dirac Hamiltonian or target shift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize
import torch

import density_mold_rest_spectrum as rest
import edge_snap_action_rank as rank
import localized_embedding_native_closure as local
import relative_kinetic_probe as relative
import transfer_neighbor_reflection_rest_spectrum as reflection


HERE = Path(__file__).resolve().parent


def native_update_blocks() -> tuple[np.ndarray, np.ndarray, dict]:
    layers, theta = rest.orbit_layers()
    coefficient = reflection.COEFFICIENT

    def two_interval_action(previous, current, following):
        return (reflection.interval_action(previous, current, coefficient)
                + reflection.interval_action(current, following, coefficient))

    variables = tuple(value.detach().requires_grad_(True) for value in layers)
    action = two_interval_action(*variables)
    gradient = torch.autograd.grad(
        action, variables[1], create_graph=False)[0].detach().numpy()
    blocks = torch.autograd.functional.hessian(
        two_interval_action, variables, vectorize=True)
    previous = blocks[1][0].detach().numpy()
    current = blocks[1][1].detach().numpy()
    following = blocks[1][2].detach().numpy()
    rotation = relative.raw_rotation_matrix(rest.SIDE, theta, 0.0, 1)
    inverse_rotation = rotation.T
    lhs = following @ rotation
    forward = -np.linalg.solve(lhs, current)
    backward = -np.linalg.solve(lhs, previous @ inverse_rotation)
    return forward, backward, {
        "rest_del_rms": float(np.linalg.norm(gradient) / np.sqrt(gradient.size)),
        "following_block_condition": float(np.linalg.cond(lhs)),
    }


def solve_graph(forward: np.ndarray, backward: np.ndarray,
                q: np.ndarray, frame_rotation: np.ndarray | None = None,
                dt: float | None = None) -> dict:
    retained = q @ q.T
    complement = np.eye(q.shape[0]) - retained
    a = complement @ backward @ q
    target = -complement @ forward @ q
    c, _, _, singular = np.linalg.lstsq(a, target, rcond=1e-12)
    cross = a @ c - target
    cross_relative = float(np.linalg.norm(cross)
                           / max(np.linalg.norm(target), 1e-30))
    condition = float(np.linalg.cond(c))
    if not np.isfinite(condition) or condition > 1e14:
        return {
            "cross_block_relative_residual": cross_relative,
            "previous_layer_map_condition": condition,
            "full_invariance_relative_residual": float("inf"),
            "retained_generator_spectral_radius": None,
            "least_squares_rank": int(np.linalg.matrix_rank(a, tol=1e-11)),
            "least_squares_singular_values": [float(value) for value in singular],
        }
    generator = np.linalg.inv(c)
    generator_values = np.linalg.eigvals(generator)
    top = forward @ q + backward @ q @ c
    phase_embedding = np.vstack((q, q @ c))
    monodromy_image = np.vstack((top, q))
    retained_image = phase_embedding @ generator
    full_relative = float(np.linalg.norm(monodromy_image-retained_image)
                          / max(np.linalg.norm(monodromy_image), 1e-30))
    result = {
        "cross_block_relative_residual": cross_relative,
        "previous_layer_map_condition": condition,
        "full_invariance_relative_residual": full_relative,
        "retained_generator_spectral_radius": float(max(abs(
            generator_values))),
        "retained_generator_eigenvalues": [
            {"real": float(value.real), "imag": float(value.imag),
             "phase": float(np.angle(value)), "modulus": float(abs(value))}
            for value in sorted(generator_values,
                                key=lambda value: (np.angle(value), abs(value)))
        ],
        "least_squares_rank": int(np.linalg.matrix_rank(a, tol=1e-11)),
        "least_squares_singular_values": [float(value) for value in singular],
    }
    if frame_rotation is not None and dt is not None:
        rotated = frame_rotation @ q
        frame_retained = q @ (q.T @ rotated)
        frame_leakage = float(np.linalg.norm(rotated-frame_retained)
                              / max(np.linalg.norm(rotated), 1e-30))
        frame_internal = q.T @ rotated
        lab_internal = frame_internal @ generator
        lab_values = np.linalg.eigvals(lab_internal)
        frequencies = sorted(abs(float(np.angle(value) / dt))
                             for value in lab_values)
        result.update({
            "carrier_frame_relative_leakage": frame_leakage,
            "lab_internal_eigenvalues": [
                {"real": float(value.real), "imag": float(value.imag),
                 "phase": float(np.angle(value)),
                 "modulus": float(abs(value)),
                 "absolute_frequency": float(abs(np.angle(value) / dt))}
                for value in sorted(
                    lab_values, key=lambda value: (np.angle(value), abs(value)))
            ],
            "lab_absolute_frequency_min": float(min(frequencies)),
            "lab_absolute_frequency_max": float(max(frequencies)),
            "lab_absolute_frequency_relative_spread": float(
                (max(frequencies)-min(frequencies)) /
                max(float(np.mean(frequencies)), 1e-30)),
        })
    return result


def objective(parameters: np.ndarray, basis: np.ndarray,
              psi_frame: np.ndarray, chi_frame: np.ndarray,
              forward: np.ndarray, backward: np.ndarray) -> float:
    q = local.real_embedding(
        basis, psi_frame, chi_frame,
        complex(parameters[0], parameters[1]))
    row = solve_graph(forward, backward, q)
    value = row["full_invariance_relative_residual"]
    return value if np.isfinite(value) else 1e6


def classify(name: str, basis: np.ndarray,
             psi_frame: np.ndarray, chi_frame: np.ndarray,
             forward: np.ndarray, backward: np.ndarray,
             frame_rotation: np.ndarray) -> dict:
    bounds = [(-1, 1), (-1, 1)]
    global_fit = differential_evolution(
        lambda value: objective(
            value, basis, psi_frame, chi_frame, forward, backward),
        bounds, seed=40517 + (0 if name == "plus" else 1),
        maxiter=100, popsize=12, tol=1e-9, polish=False)
    local_fit = minimize(
        lambda value: objective(
            value, basis, psi_frame, chi_frame, forward, backward),
        global_fit.x, method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-11, "fatol": 1e-11})
    scale = complex(local_fit.x[0], local_fit.x[1])
    q = local.real_embedding(basis, psi_frame, chi_frame, scale)
    result = solve_graph(
        forward, backward, q, frame_rotation=frame_rotation, dt=rest.DT)
    result.update({
        "flavor": name,
        "native_selected_chi_scale_real": float(scale.real),
        "native_selected_chi_scale_imag": float(scale.imag),
        "optimizer_success": bool(local_fit.success),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=rest.DT)
    parser.add_argument(
        "--output", default="localized_embedding_two_layer_closure.json")
    args = parser.parse_args()
    rest.DT = args.dt
    rank.lock_inertia()
    forward, backward, update = native_update_blocks()
    _, theta = rest.orbit_layers()
    frame_rotation = relative.raw_rotation_matrix(
        rest.SIDE, theta, 0.0, 1)
    _, _, psi_frame, chi_frame = local.carrier_frames()
    rows = [classify(name, basis, psi_frame, chi_frame, forward, backward,
                     frame_rotation)
            for name, basis in local.flavor_bases().items()]
    best = min(row["full_invariance_relative_residual"] for row in rows)
    result = {
        "status": "V4_LOCALIZED_EMBEDDING_TWO_LAYER_CLOSURE_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "native_action": (
            "transfer-softened incumbent plus transfer-activated chi bond "
            "and 1/(2 chi0) native frame reflection"),
        "dt": rest.DT,
        "update_diagnostics": update,
        "rows": rows,
        "best_full_invariance_relative_residual": float(best),
        "verdict": (
            "CORE_ONLY_TWO_LAYER_SECTOR_CLOSED" if best < 1e-8 else
            "CORE_ONLY_TWO_LAYER_SECTOR_NOT_CLOSED"),
        "claim_boundary": (
            "This derives the best same-core previous-layer graph from the "
            "complete implicit native update. It does not permit a halo graph "
            "or change the action, and it does not establish nonlinear motion."),
        "corrective_action": (
            "If the residual remains nonzero, solve the block invariant-graph "
            "problem with a first-shell response while retaining the same "
            "four-component coordinates. Only a residual left after that "
            "extension becomes an action-repair target."),
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
