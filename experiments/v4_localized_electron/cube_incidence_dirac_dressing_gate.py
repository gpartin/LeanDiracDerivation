"""Derive the same-rank native dressing of the Dirac envelope candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, lu_factor, lu_solve

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_nonlinear_action as action
import current_action_localized_envelope_closure as envelope


HERE = Path(__file__).resolve().parent


def channel_fractions(vectors):
    shaped = vectors.reshape(5, 5, 5, 4, 3, -1)
    power = np.sum(shaped**2, axis=(0, 1, 2, 5))
    total = max(float(np.sum(power)), 1e-30)
    return {
        "Psi_fraction": float(np.sum(power[:2])/total),
        "chi_fraction": float(np.sum(power[2:])/total),
        "C3_component_fractions": [float(value/total)
                                    for value in np.sum(power, axis=0)],
    }


def main():
    _bare, matrices = action.expected_blocks()
    mass = (matrices["mass"]+matrices["mass"].T)/2
    stiffness = (matrices["stiffness"]+matrices["stiffness"].T)/2
    gyro = (matrices["gyro"]-matrices["gyro"].T)/2
    acceleration = -np.linalg.solve(mass, stiffness)
    eigenvalues, eigenvectors = eigh(stiffness, mass, check_finite=False)
    archive = np.load(HERE/"cube_incidence_dirac_envelope_gate.npz")
    rows = []
    dressed_archive = {}
    dt = native_time.DT
    left = mass/dt+gyro/2
    backward = mass/dt-gyro/2
    zero = dt*stiffness-2*mass/dt
    factor = lu_factor(left)
    for flavor in ("plus", "minus"):
        q = archive[flavor]
        projector = q@q.T
        euclidean_norm = np.sum(eigenvectors**2, axis=0)
        normal = eigenvectors-projector@eigenvectors
        normal_fraction = np.sum(normal**2, axis=0)/np.maximum(
            euclidean_norm, 1e-30)
        selected = np.argsort(normal_fraction)[:q.shape[1]]
        dressed = envelope.orth(eigenvectors[:, selected])
        dressed_projector = dressed@dressed.T
        projection_singular = np.linalg.svd(
            q.T@dressed, compute_uv=False)
        configuration = envelope.closure(acceleration, dressed)
        tail = (np.eye(q.shape[0])-projector)@dressed

        forward = -lu_solve(factor, zero@dressed)
        previous = -lu_solve(factor, backward@dressed)
        response = np.column_stack((forward, previous))
        two_layer_residual = response-dressed_projector@response
        singular = np.linalg.svd(two_layer_residual, compute_uv=False)
        two_layer_relative = float(
            np.linalg.norm(two_layer_residual)
            / max(np.linalg.norm(response), 1e-30))
        rows.append({
            "flavor": flavor,
            "undressed_rank": int(q.shape[1]),
            "selected_native_mode_count": int(dressed.shape[1]),
            "projection_min_singular": float(projection_singular[-1]),
            "projection_condition": float(
                projection_singular[0]/max(projection_singular[-1], 1e-30)),
            "configuration_relative_cross_block": configuration[
                "relative_cross_block"],
            "tail_relative_norm": float(
                np.linalg.norm(tail)/np.linalg.norm(dressed)),
            "tail_channels": channel_fractions(tail),
            "two_layer_relative_cross_block": two_layer_relative,
            "two_layer_residual_rank": int(np.sum(
                singular > 1e-10*max(float(singular[0]), 1.0))),
            "selected_stiffness_min": float(np.min(eigenvalues[selected])),
            "selected_stiffness_max": float(np.max(eigenvalues[selected])),
        })
        dressed_archive[flavor] = dressed
    configuration_pass = max(row[
        "configuration_relative_cross_block"] for row in rows) < 1e-8
    time_pass = max(row["two_layer_relative_cross_block"]
                    for row in rows) < 1e-8
    result = {
        "status": "V4_CUBE_INCIDENCE_DIRAC_DRESSING_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "configuration_verdict": (
            "SAME_RANK_CONFIGURATION_DRESSING_PASS"
            if configuration_pass else
            "SAME_RANK_CONFIGURATION_DRESSING_FAIL"),
        "two_layer_verdict": (
            "SAME_RANK_TWO_LAYER_DRESSING_PASS"
            if time_pass else "SAME_RANK_TWO_LAYER_DRESSING_FAIL"),
        "claim_boundary": (
            "Action-derived invariant-mode dressing only. A pass would still "
            "require the compressed Dirac generators, current, nonlinear "
            "moving branch and endurance gates."),
    }
    np.savez_compressed(
        HERE/"cube_incidence_dirac_dressing_gate.npz", **dressed_archive)
    (HERE/"cube_incidence_dirac_dressing_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
