"""Centered-time refinement of the exact scale-six tangent candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig, lu_factor, lu_solve

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_geometry_bifurcation_solve as solve
import cube_incidence_symplectic_completion_gate as gate


HERE = Path(__file__).resolve().parent


def monodromy(mass, gyro, stiffness, dt):
    factor = lu_factor(mass/dt+gyro/2)
    forward = -lu_solve(factor, dt*stiffness-2*mass/dt)
    backward = -lu_solve(factor, mass/dt-gyro/2)
    identity = np.eye(mass.shape[0])
    return np.block([[forward, backward],
                     [identity, np.zeros_like(identity)]])


def main() -> None:
    _bare, mass0, gyro0, stiffness0 = solve.base_matrices()
    connection = solve.load_blocks("rho_Psi")
    natural = native_time.B*native_time.CHI0
    cached = np.load(gate.CACHE)
    rows = []
    for scale in (1, 6):
        matrices = {
            "mass": mass0+natural*connection["mass"]+scale*cached["mass"],
            "gyro": gyro0+natural*connection["gyro"]+scale*cached["gyro"],
            "stiffness": (stiffness0+natural*connection["stiffness"]
                          + scale*cached["stiffness"]),
        }
        for divisor in (1, 2, 4):
            dt = native_time.DT/divisor
            values = eig(monodromy(**matrices, dt=dt),
                         right=False, check_finite=False)
            maximum = float(max(abs(values)))
            rows.append({
                "scale": scale,
                "dt": dt,
                "maximum_multiplier_modulus": maximum,
                "excess_modulus": maximum-1,
                "growth_rate": float(np.log(maximum)/dt),
                "growing_multiplier_count_at_1e_minus_7": int(np.sum(
                    abs(values) > 1+1e-7)),
            })
    result = {
        "status": "V4_CUBE_INCIDENCE_SCALE6_REFINEMENT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "verdict": (
            "UNIT_AND_SCALE6_CENTERED_TIME_REFINEMENT_PASS"
            if all(row["growing_multiplier_count_at_1e_minus_7"] == 0
                   for row in rows)
            else "UNIT_OR_SCALE6_CENTERED_TIME_REFINEMENT_FAIL"),
        "claim_boundary": (
            "Refinement of the fixed rotating-frame tangent pencil. The "
            "underlying action blocks are those derived at dt=1/256; a full "
            "nonlinear re-derived orbit and implicit update remain open."),
    }
    (HERE/"cube_incidence_scale6_refinement.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
