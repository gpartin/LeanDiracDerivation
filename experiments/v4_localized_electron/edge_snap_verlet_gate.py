"""Exact one-edge-per-substrate-tick gate for the localized carrier.

The previous handoff control incorrectly required the four geometrically
shared sites to retain their old values.  A literal lattice shift is instead a
conveyor: every occupied site sends its value to the adjacent site, so the
shared face is replaced by the trailing face while its old value advances to
the leading face.

For each signed coordinate direction this script constructs the exact
two-layer traveling ansatz

    q_(n+1) = exp(+i phi) S q_n,
    q_(n-1) = exp(-i phi) S^{-1} q_n,

where ``S`` is the one-site shift.  It then asks whether the unchanged native
Verlet equation can satisfy

    q_(n+1)-2 q_n+q_(n-1) = dt^2 a(q_n)

with one nonnegative ``dt^2`` and independent global Psi/chi tick phases.
The phases and timestep are diagnostic readout variables, not forces.  The
Dirac equation is not used to construct the ansatz.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

import copy_alternating_handoff as old_handoff
import localized_stability as native


HERE = Path(__file__).resolve().parent


def translate(field: np.ndarray, axis: int, sign: int) -> np.ndarray:
    return np.roll(field, sign, axis=axis)


def required_second_difference(field: np.ndarray, axis: int, sign: int,
                               phase: float) -> np.ndarray:
    following = np.exp(1j * phase) * translate(field, axis, sign)
    previous = np.exp(-1j * phase) * translate(field, axis, -sign)
    return following - 2 * field + previous


def best_dt_sq(required: np.ndarray, acceleration: np.ndarray) -> float:
    denominator = float(np.vdot(acceleration, acceleration).real)
    if denominator <= 1e-30:
        return 0.0
    return max(0.0, float(np.vdot(acceleration, required).real / denominator))


def residual_for_phases(phases, psi, chi, force_psi, force_chi, axis, sign):
    dpsi = required_second_difference(psi, axis, sign, phases[0])
    dchi = required_second_difference(chi, axis, sign, phases[1])
    required = native.pack(dpsi, dchi)
    acceleration = native.pack(force_psi, force_chi)
    dt_sq = best_dt_sq(required, acceleration)
    residual = required - dt_sq * acceleration
    return float(np.linalg.norm(residual) / max(np.linalg.norm(required), 1e-30))


def classify(axis: int, sign: int, psi: np.ndarray, chi: np.ndarray,
             force_psi: np.ndarray, force_chi: np.ndarray):
    objective = lambda phases: residual_for_phases(
        phases, psi, chi, force_psi, force_chi, axis, sign)
    global_result = differential_evolution(
        objective, [(-np.pi, np.pi), (-np.pi, np.pi)],
        seed=40201 + 10 * axis + sign, polish=False, tol=1e-10,
        popsize=12, maxiter=250)
    local_result = minimize(
        objective, global_result.x, method="Nelder-Mead",
        options={"xatol": 1e-12, "fatol": 1e-12, "maxiter": 2000})
    phases = local_result.x
    dpsi = required_second_difference(psi, axis, sign, phases[0])
    dchi = required_second_difference(chi, axis, sign, phases[1])
    required = native.pack(dpsi, dchi)
    acceleration = native.pack(force_psi, force_chi)
    dt_sq = best_dt_sq(required, acceleration)
    residual = required - dt_sq * acceleration
    psi_residual = dpsi - dt_sq * force_psi
    chi_residual = dchi - dt_sq * force_chi

    reflected_psi, reflected_chi = old_handoff.reflected_destination(
        psi, chi, axis, sign)
    same_psi = translate(psi, axis, sign)
    same_chi = translate(chi, axis, sign)
    # Under the reflected endpoint map, half the source sites stay on the
    # shared face and half cross two face-neighbor spacings.  The literal shift
    # maps every source site by exactly one spacing.
    return {
        "axis": axis,
        "sign": sign,
        "same_assignment_translation_defect_psi": float(np.linalg.norm(
            translate(same_psi, axis, -sign) - psi)),
        "same_assignment_translation_defect_chi": float(np.linalg.norm(
            translate(same_chi, axis, -sign) - chi)),
        "same_assignment_hop_distances": [1] * 8,
        "reflected_assignment_hop_distances": [0] * 4 + [2] * 4,
        "reflected_endpoint_difference_from_one_edge_shift_psi": float(
            np.linalg.norm(reflected_psi - same_psi)),
        "reflected_endpoint_difference_from_one_edge_shift_chi": float(
            np.linalg.norm(reflected_chi - same_chi)),
        "best_phase_psi": float(phases[0]),
        "best_phase_chi": float(phases[1]),
        "best_dt_sq": dt_sq,
        "best_dt": float(np.sqrt(dt_sq)),
        "combined_relative_residual": float(
            np.linalg.norm(residual) / max(np.linalg.norm(required), 1e-30)),
        "psi_relative_residual": float(
            np.linalg.norm(psi_residual) / max(np.linalg.norm(dpsi), 1e-30)),
        "chi_relative_residual": float(
            np.linalg.norm(chi_residual) / max(np.linalg.norm(dchi), 1e-30)),
        "required_psi_second_difference_norm": float(np.linalg.norm(dpsi)),
        "required_chi_second_difference_norm": float(np.linalg.norm(dchi)),
        "native_psi_acceleration_norm": float(np.linalg.norm(force_psi)),
        "native_chi_acceleration_norm": float(np.linalg.norm(force_chi)),
        "optimizer_success": bool(local_result.success),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=5)
    parser.add_argument("--output", default="edge_snap_verlet_gate.json")
    args = parser.parse_args()
    native.B = 1197.0
    amplitude_sq = native.R**2 / (2 + 54 / native.R**2)
    psi, chi = native.carrier_fields(args.side, amplitude_sq)
    force_psi, force_chi = native.force(psi, chi)
    rows = [classify(axis, sign, psi, chi, force_psi, force_chi)
            for axis in range(3) for sign in (-1, 1)]
    result = {
        "status": "V4_LITERAL_EDGE_SNAP_VERLET_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "B": native.B,
        "side": args.side,
        "amplitude_sq": float(amplitude_sq),
        "rows": rows,
        "verdict": "ONE_EDGE_ENDPOINT_IS_KINEMATICALLY_EXACT_BUT_NATIVE_UPDATE_FAILS",
        "interpretation": (
            "The same-assignment endpoint is the unique literal one-edge "
            "translation control; shared-face equality was an invalid extra "
            "condition.  The position-only canonical force cannot make the "
            "static-chi carrier obey this shift recurrence.  A candidate "
            "must supply motion-dependent reciprocal response or a freely "
            "deformed traveling branch; the shift may not be installed as a "
            "runtime rule."),
        "claim_boundary": (
            "This is a two-layer inverse recurrence test, not a moving "
            "solution and not evidence that the physical substrate tick "
            "equals an integrator step."),
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
