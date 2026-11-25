"""Solve the previous-layer graph over the four-component spatial image.

For the native recurrence q+ = A q + B q-, write

    q = Q z,       q- = Q X z.

Closure requires both

    (I-QQ^T)(A Q + B Q X) = 0

and, with H = Q^T(AQ+BQX),

    X H = I.

The first equation determines whether velocity can cancel the configuration
leakage.  The second determines whether that velocity graph reproduces itself
after the update.  No Dirac generator is supplied.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_nonlinear_action as action


HERE = Path(__file__).resolve().parent


def main():
    _bare, matrices = action.expected_blocks()
    mass = matrices["mass"]
    gyro = matrices["gyro"]
    stiffness = matrices["stiffness"]
    dt = native_time.DT
    factor = lu_factor(mass/dt+gyro/2)
    advance = -lu_solve(factor, dt*stiffness-2*mass/dt)
    memory = -lu_solve(factor, mass/dt-gyro/2)

    archive = np.load(HERE/"cube_incidence_dirac_envelope_gate.npz")
    q = archive["plus"]
    projector = q@q.T
    normal_advance = advance@q-projector@(advance@q)
    normal_memory = memory@q-projector@(memory@q)

    # Resolve the memory control space relative to its largest singular value.
    # An absolute machine-epsilon cutoff is invalid here: the smallest values
    # are roundoff-level directions that manufacture enormous prior layers.
    u, singular, vh = np.linalg.svd(normal_memory, full_matrices=False)
    cutoff = 1e-4
    rank = int(np.sum(singular > cutoff*singular[0]))
    x = -(vh[:rank].T/singular[:rank])@(u[:, :rank].T@normal_advance)
    following = advance@q+memory@q@x
    exterior = following-projector@following
    h = q.T@following
    identity = np.eye(q.shape[1])
    reproduction = x@h-identity
    x_singular = np.linalg.svd(x, compute_uv=False)
    h_singular = np.linalg.svd(h, compute_uv=False)

    sweep = []
    for relative_cutoff in (1e-2, 1e-4, 1e-6, 1e-8):
        retained = int(np.sum(singular > relative_cutoff*singular[0]))
        trial_x = -(vh[:retained].T/singular[:retained])@(
            u[:, :retained].T@normal_advance)
        trial_following = advance@q+memory@q@trial_x
        trial_exterior = trial_following-projector@trial_following
        trial_h = q.T@trial_following
        sweep.append({
            "relative_singular_cutoff": relative_cutoff,
            "retained_rank": retained,
            "fraction_of_original_leakage_remaining": float(
                np.linalg.norm(normal_advance+normal_memory@trial_x)
                / max(np.linalg.norm(normal_advance), 1e-30)),
            "previous_graph_operator_norm": float(
                np.linalg.norm(trial_x, 2)),
            "relative_graph_reproduction_defect": float(
                np.linalg.norm(trial_x@trial_h-identity)
                / np.linalg.norm(identity)),
            "relative_exterior_residual": float(
                np.linalg.norm(trial_exterior)
                / max(np.linalg.norm(trial_following), 1e-30)),
        })

    result = {
        "status": "V4_CUBE_INCIDENCE_TWO_LAYER_GRAPH",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "spatial_rank": int(q.shape[1]),
        "exterior_cancellation_matrix_rank": int(rank),
        "memory_control_relative_singular_cutoff": cutoff,
        "memory_control_min_retained_singular": float(
            singular[rank-1] if rank else 0.0),
        "uncontrolled_current_layer_relative_leakage": float(
            np.linalg.norm(normal_advance)
            / max(np.linalg.norm(advance@q), 1e-30)),
        "fraction_of_original_leakage_remaining": float(
            np.linalg.norm(normal_advance+normal_memory@x)
            / max(np.linalg.norm(normal_advance), 1e-30)),
        "relative_exterior_residual": float(
            np.linalg.norm(exterior)/max(np.linalg.norm(following), 1e-30)),
        "previous_graph_min_singular": float(x_singular[-1]),
        "previous_graph_condition": float(
            x_singular[0]/max(x_singular[-1], 1e-30)),
        "effective_step_min_singular": float(h_singular[-1]),
        "effective_step_condition": float(
            h_singular[0]/max(h_singular[-1], 1e-30)),
        "relative_graph_reproduction_defect": float(
            np.linalg.norm(reproduction)/np.linalg.norm(identity)),
        "singular_cutoff_sweep": sweep,
        "verdict": (
            "SAME_SPATIAL_IMAGE_TWO_LAYER_GRAPH_PASS"
            if np.linalg.norm(exterior)/max(np.linalg.norm(following), 1e-30)
            < 1e-8
            and np.linalg.norm(reproduction)/np.linalg.norm(identity) < 1e-8
            and x_singular[-1] > 1e-6
            else "SAME_SPATIAL_IMAGE_TWO_LAYER_GRAPH_FAIL"),
        "claim_boundary": (
            "Two-layer graph over the fixed spatial image only. Failure may "
            "require a phase-space tail; pass would still require Clifford "
            "compression, current, nonlinear motion and endurance."),
    }
    np.savez_compressed(
        HERE/"cube_incidence_two_layer_graph.npz", previous_graph=x,
        effective_step=h)
    (HERE/"cube_incidence_two_layer_graph.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
