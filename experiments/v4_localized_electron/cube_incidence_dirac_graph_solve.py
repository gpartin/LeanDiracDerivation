"""Solve the native invariant graph over the four-component spatial image."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve, solve_sylvester

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_nonlinear_action as action
import current_action_localized_envelope_closure as envelope


HERE = Path(__file__).resolve().parent


def main():
    _bare, matrices = action.expected_blocks()
    mass = matrices["mass"]
    gyro = matrices["gyro"]
    stiffness = matrices["stiffness"]
    acceleration = -np.linalg.solve(mass, stiffness)
    archive = np.load(HERE/"cube_incidence_dirac_envelope_gate.npz")
    q = archive["plus"]
    # Both copy images coincide on the complete periodic translation span.
    copy_image_defect = float(np.linalg.norm(
        q@q.T-archive["minus"]@archive["minus"].T))
    complete, _ = np.linalg.qr(q, mode="complete")
    q = complete[:, :q.shape[1]]
    normal = complete[:, q.shape[1]:]
    aqq = q.T@acceleration@q
    aqn = q.T@acceleration@normal
    anq = normal.T@acceleration@q
    ann = normal.T@acceleration@normal
    t = np.zeros((normal.shape[1], q.shape[1]))
    history = [{
        "iteration": -1,
        "damping": 0.0,
        "relative_Riccati_residual": 1.0,
        "graph_operator_norm": 0.0,
    }]
    for iteration in range(8):
        rhs = -anq+t@aqn@t
        candidate = solve_sylvester(ann, -aqq, rhs)
        # Damping protects the graph iteration near shared spectra.
        accepted = False
        for power in range(8):
            trial = t+(candidate-t)*(0.5**power)
            residual = anq+ann@trial-trial@aqq-trial@aqn@trial
            norm = float(np.linalg.norm(residual)/max(
                np.linalg.norm(anq), 1e-30))
            if norm < history[-1]["relative_Riccati_residual"]:
                t = trial
                history.append({
                    "iteration": iteration,
                    "damping": 0.5**power,
                    "relative_Riccati_residual": norm,
                    "graph_operator_norm": float(np.linalg.norm(t, 2)),
                })
                accepted = True
                break
        if not accepted or history[-1]["relative_Riccati_residual"] < 1e-10:
            break
    dressed = envelope.orth(q+normal@t)
    configuration = envelope.closure(acceleration, dressed)
    projection = np.linalg.svd(q.T@dressed, compute_uv=False)

    dt = native_time.DT
    left = mass/dt+gyro/2
    backward = mass/dt-gyro/2
    zero = dt*stiffness-2*mass/dt
    factor = lu_factor(left)
    forward = -lu_solve(factor, zero@dressed)
    previous = -lu_solve(factor, backward@dressed)
    response = np.column_stack((forward, previous))
    projector = dressed@dressed.T
    time_residual = response-projector@response
    time_relative = float(
        np.linalg.norm(time_residual)/max(np.linalg.norm(response), 1e-30))
    singular = np.linalg.svd(time_residual, compute_uv=False)
    result = {
        "status": "V4_CUBE_INCIDENCE_DIRAC_GRAPH_SOLVE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "plus_minus_complete_image_projector_defect": copy_image_defect,
        "Riccati_history": history,
        "configuration_relative_cross_block": configuration[
            "relative_cross_block"],
        "projection_min_singular": float(projection[-1]),
        "projection_condition": float(
            projection[0]/max(projection[-1], 1e-30)),
        "two_layer_relative_cross_block": time_relative,
        "two_layer_residual_rank": int(np.sum(
            singular > 1e-10*max(float(singular[0]), 1.0))),
        "configuration_verdict": (
            "WELL_CONDITIONED_INVARIANT_GRAPH_PASS"
            if configuration["relative_cross_block"] < 1e-8
            and projection[-1] > 1e-3 else
            "INVARIANT_GRAPH_SOLVE_FAIL"),
        "two_layer_verdict": (
            "TWO_LAYER_GRAPH_PASS" if time_relative < 1e-8 else
            "TWO_LAYER_GRAPH_OPEN"),
        "claim_boundary": (
            "Configuration Riccati graph and two-layer test only. It does "
            "not prove Dirac generator/current equality or nonlinear motion."),
    }
    np.savez_compressed(
        HERE/"cube_incidence_dirac_graph_solve.npz", dressed=dressed,
        graph=t)
    (HERE/"cube_incidence_dirac_graph_solve.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
