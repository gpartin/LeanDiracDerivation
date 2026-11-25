"""Targeted exact discrete spectrum for vacuum-activated chi orientation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs

import canonical_noncoplanar_bifurcation_audit as small
import personal_touch_face_psd_cone as cone
import personal_touch_face_shift_invert_scan as face_scan


HERE = Path(__file__).resolve().parent
ACTIVATION_CACHE = HERE/"personal_touch_vacuum_activated_chi_orientation_cache.npz"
SIGMA = 1.00005
STRENGTHS = (0.0, 0.001, 0.01, 0.1, 1.0)
REQUESTED = 4
FACE_COEFFICIENTS = np.asarray((
    0.14705479982595762, 0.14705479982595762,
    0.075, 0.075, 0.10294520017404238))


def targeted_spectrum(mass, gyro, stiffness):
    left = -mass/small.DT-gyro/2
    backward = -mass/small.DT+gyro/2
    zero = 2*mass/small.DT-small.DT*stiffness
    operator_dtype = np.complex128 if np.iscomplexobj(SIGMA) else np.float64
    left = left.astype(operator_dtype)
    backward = backward.astype(operator_dtype)
    zero = zero.astype(operator_dtype)
    left_factor = lu_factor(left)
    qfactor = lu_factor(SIGMA**2*left+SIGMA*zero+backward)
    dimension = len(mass)

    def matvec(raw):
        current, previous = raw[:dimension], raw[dimension:]
        following = -lu_solve(
            left_factor, zero@current+backward@previous)
        return np.r_[following, current]

    def inverse(raw):
        upper, lower = raw[:dimension], raw[dimension:]
        previous = -lu_solve(
            qfactor, left@upper+zero@lower+SIGMA*(left@lower))
        return np.r_[lower+SIGMA*previous, previous]

    operator = LinearOperator(
        (2*dimension, 2*dimension), matvec=matvec, dtype=operator_dtype)
    opinverse = LinearOperator(
        (2*dimension, 2*dimension), matvec=inverse, dtype=operator_dtype)
    converged = True
    try:
        v0 = np.linspace(1.0, 2.0, 2*dimension, dtype=operator_dtype)
        v0 /= np.linalg.norm(v0)
        values = eigs(
            operator, k=REQUESTED, sigma=SIGMA, OPinv=opinverse, which="LM",
            tol=1e-9, maxiter=800, ncv=32, v0=v0,
            return_eigenvectors=False)
    except ArpackNoConvergence as error:
        converged = False
        values = error.eigenvalues
    values = values[np.argsort(np.abs(values))[::-1]]
    return values, converged


def main() -> None:
    data = np.load(cone.CACHE)
    activated = np.load(ACTIVATION_CACHE)["hessian"]
    face_hessians = [data[f"hessian_{name}"] for name in face_scan.NAMES]
    base = data["stiffness"]+sum(
        value*hessian for value, hessian in zip(
            FACE_COEFFICIENTS, face_hessians, strict=True))
    rows = []
    for strength in STRENGTHS:
        values, converged = targeted_spectrum(
            data["mass"], data["gyro"], base+strength*activated)
        moduli = np.abs(values)
        row = {
            "strength": strength,
            "converged_all_requested": converged and len(values) == REQUESTED,
            "converged_count": int(len(values)),
            "leading_modulus": float(moduli[0]) if len(values) else None,
            "growth_rate": (float(np.log(moduli[0])/small.DT)
                            if len(values) else None),
            "count_above_1_plus_1e_minus_7": int(
                np.sum(moduli > 1+1e-7)),
            "values": [
                {"real": float(value.real), "imag": float(value.imag),
                 "modulus": float(abs(value))} for value in values],
        }
        rows.append(row)
        print(strength, row["growth_rate"], row["converged_count"],
              row["converged_all_requested"], flush=True)
    eligible = [row for row in rows if row["converged_all_requested"]]
    best = min(eligible, key=lambda row: row["growth_rate"]) if eligible else None
    result = {
        "status": "V4_VACUUM_CHI_TARGETED_EXACT_SCAN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "shift": SIGMA,
        "rows": rows,
        "best_complete_row": best,
        "verdict": (
            "TARGETED_EXACT_SCAN_CLOSES_RESOLVED_MODES"
            if best is not None
            and best["count_above_1_plus_1e_minus_7"] == 0 else
            "TARGETED_EXACT_SCAN_RETAINS_GROWTH"
            if best is not None else "TARGETED_EXACT_SCAN_INCONCLUSIVE"),
        "claim_boundary": (
            "Four exact discrete multipliers nearest the declared real "
            "shift. This is not a complete contour count or nonlinear gate."),
    }
    (HERE/"personal_touch_vacuum_chi_exact_scan.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"best_complete_row": best,
                      "verdict": result["verdict"]},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
