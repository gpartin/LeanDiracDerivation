"""Uncompressed full-state spectrum at the exact six-face coefficient."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig, eigvalsh

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_geometry_bifurcation_solve as solve
import cube_incidence_symplectic_completion_gate as gate


HERE = Path(__file__).resolve().parent


def main() -> None:
    bare, mass0, gyro0, stiffness0 = solve.base_matrices()
    connection = solve.load_blocks("rho_Psi")
    natural = native_time.B*native_time.CHI0
    delta_cache = np.load(gate.CACHE)
    delta = {kind: delta_cache[kind]
             for kind in ("mass", "gyro", "stiffness")}
    matrices = {
        "mass": mass0+natural*connection["mass"]+6*delta["mass"],
        "gyro": gyro0+natural*connection["gyro"]+6*delta["gyro"],
        "stiffness": (stiffness0+natural*connection["stiffness"]
                      + 6*delta["stiffness"]),
    }
    multipliers = eig(solve.reduced_monodromy(**matrices),
                      right=False, check_finite=False)
    result = {
        "status": "V4_CUBE_INCIDENCE_SCALE6_FULL_IDENTITY",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "scale": 6,
        "scale_hypothesis": "number of oriented cube faces",
        "configuration_dimension": int(matrices["mass"].shape[0]),
        "phase_space_dimension": int(2*matrices["mass"].shape[0]),
        "minimum_mass_eigenvalue": float(eigvalsh(
            matrices["mass"], subset_by_index=[0, 0], driver="evr")[0]),
        "maximum_multiplier_modulus": float(max(abs(multipliers))),
        "maximum_growth_rate": float(
            np.log(max(abs(multipliers)))/native_time.DT),
        "growing_multiplier_count_at_1e_minus_7": int(np.sum(
            abs(multipliers) > 1+1e-7)),
        "verdict": (
            "SCALE6_UNCOMPRESSED_LINEAR_SPECTRUM_PASS"
            if not np.any(abs(multipliers) > 1+1e-7)
            else "SCALE6_UNCOMPRESSED_LINEAR_SPECTRUM_FAIL"),
        "claim_boundary": (
            "Uncompressed finite-box linear spectrum at one timestep. It "
            "does not prove coefficient uniqueness, timestep/box convergence, "
            "nonlinear stability, movement, current, or V4-006."),
    }
    (HERE/"cube_incidence_scale6_full_identity.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
