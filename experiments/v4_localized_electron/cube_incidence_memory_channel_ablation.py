"""Ablate the native action terms that can orient the previous-layer response.

The current configuration leakage can be corrected by preparation only if the
previous-layer block has a resolved image in the missing normal directions.
This audit removes the reciprocal density connection and cube-incidence
symplectic completion as complete action terms and measures that image.  It
does not fit a Dirac generator or change the admitted action.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_geometry_bifurcation_solve as solve
import cube_incidence_nonlinear_action as action
import cube_incidence_symplectic_completion_gate as cube_gate


HERE = Path(__file__).resolve().parent


def blocks():
    _bare, mass0, gyro0, stiffness0 = solve.base_matrices()
    density = solve.load_blocks("rho_Psi")
    cube = np.load(cube_gate.CACHE)
    natural = native_time.B*native_time.CHI0
    zero = np.zeros_like(mass0)
    return {
        "base_without_density_or_cube_connection": (
            mass0, gyro0, stiffness0),
        "base_plus_density_connection": (
            mass0+natural*density["mass"],
            gyro0+natural*density["gyro"],
            stiffness0+natural*density["stiffness"]),
        "base_plus_cube_incidence_connection": (
            mass0+cube["mass"], gyro0+cube["gyro"],
            stiffness0+cube["stiffness"]),
        "complete_research_action": (
            mass0+natural*density["mass"]+cube["mass"],
            gyro0+natural*density["gyro"]+cube["gyro"],
            stiffness0+natural*density["stiffness"]+cube["stiffness"]),
    }


def measure(mass, gyro, stiffness, q):
    dt = native_time.DT
    factor = lu_factor(mass/dt+gyro/2)
    advance = -lu_solve(factor, dt*stiffness-2*mass/dt)
    memory = -lu_solve(factor, mass/dt-gyro/2)
    projector = q@q.T
    normal_advance = advance@q-projector@(advance@q)
    normal_memory = memory@q-projector@(memory@q)
    u, singular, vh = np.linalg.svd(normal_memory, full_matrices=False)
    # Cross blocks below 1e-10 are roundoff relative to the O(1) update.
    # Treating their relative numerical rank as physical creates 1e11-sized
    # previous layers and a false controllability result.
    threshold = max(1e-10, 1e-4*max(singular[0], 1e-30))
    rank = int(np.sum(singular > threshold))
    if rank:
        graph = -(vh[:rank].T/singular[:rank])@(
            u[:, :rank].T@normal_advance)
    else:
        graph = np.zeros((q.shape[1], q.shape[1]))
    remaining = normal_advance+normal_memory@graph
    return {
        "resolved_memory_control_rank": rank,
        "maximum_memory_cross_singular": float(singular[0]),
        "memory_control_absolute_singular_threshold": threshold,
        "current_layer_relative_leakage": float(
            np.linalg.norm(normal_advance)
            / max(np.linalg.norm(advance@q), 1e-30)),
        "fraction_of_leakage_remaining_after_bounded_control": float(
            np.linalg.norm(remaining)
            / max(np.linalg.norm(normal_advance), 1e-30)),
        "bounded_previous_graph_operator_norm": float(
            np.linalg.norm(graph, 2)),
    }


def main():
    archive = np.load(HERE/"cube_incidence_dirac_envelope_gate.npz")
    q = archive["plus"]
    rows = []
    for name, values in blocks().items():
        row = measure(*values, q)
        row["action_ablation"] = name
        rows.append(row)
    full = next(row for row in rows
                if row["action_ablation"] == "complete_research_action")
    result = {
        "status": "V4_CUBE_INCIDENCE_MEMORY_CHANNEL_ABLATION",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "verdict": (
            "PREPARATION_CHANNEL_RANK_DEFICIENT"
            if full["fraction_of_leakage_remaining_after_bounded_control"]
            > 0.99 else "PREPARATION_CHANNEL_HAS_MATERIAL_CONTROL"),
        "claim_boundary": (
            "Linear two-layer response around the exact rest orbit and the "
            "fixed all-center spatial image. This does not exclude a spatial "
            "phase-space tail or a deformed nonlinear moving branch."),
    }
    (HERE/"cube_incidence_memory_channel_ablation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
