"""Exact discrete shift-invert scan of the cached face-Hessian cone."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, eigs


HERE = Path(__file__).resolve().parent
CACHE = HERE/"personal_touch_face_psd_cone_cache.npz"
DT = 1/256
SIGMA = 1.0005
NAMES = (
    "diagonal_symmetric", "diagonal_antisymmetric",
    "diagonal_reversed", "diagonal_conjugate", "edge_commutator")


def spectrum(mass: np.ndarray, gyro: np.ndarray, stiffness: np.ndarray,
             hessians: list[np.ndarray], coefficients: np.ndarray) -> np.ndarray:
    n = len(mass)
    left = -mass/DT-gyro/2
    backward = -mass/DT+gyro/2
    candidate = stiffness+sum(value*hessian for value, hessian
                              in zip(coefficients, hessians, strict=True))
    zero = 2*mass/DT-DT*candidate
    left_factor = lu_factor(left)
    qfactor = lu_factor(SIGMA**2*left+SIGMA*zero+backward)

    def matvec(raw):
        current, previous = raw[:n], raw[n:]
        following = -lu_solve(left_factor,
                              zero@current+backward@previous)
        return np.r_[following, current]

    def inverse(raw):
        upper, lower = raw[:n], raw[n:]
        previous = -lu_solve(
            qfactor, left@upper+(zero+SIGMA*left)@lower)
        return np.r_[lower+SIGMA*previous, previous]

    operator = LinearOperator((2*n, 2*n), matvec=matvec, dtype=float)
    opinverse = LinearOperator((2*n, 2*n), matvec=inverse, dtype=float)
    values = eigs(
        operator, k=32, sigma=SIGMA, OPinv=opinverse, which="LM",
        tol=1e-10, maxiter=3000, ncv=96, return_eigenvectors=False)
    return values[np.argsort(np.abs(values))[::-1]]


def main() -> None:
    data = np.load(CACHE)
    mass, gyro, stiffness = data["mass"], data["gyro"], data["stiffness"]
    hessians = [data[f"hessian_{name}"] for name in NAMES]
    fraction = np.asarray((0.7205479982595763, 0.7205479982595763,
                           0.0, 0.0, 0.27945200174042373))
    base = 0.1*fraction
    trials = [("incumbent", np.zeros(5)), ("two_term_cone", base)]
    for index, name in enumerate(NAMES):
        trial = base.copy()
        trial[index] += 0.05
        trials.append((f"cone_plus_{name}", trial))
    trials.append(("cone_plus_all", base+0.05))
    rows = []
    for label, coefficients in trials:
        values = spectrum(mass, gyro, stiffness, hessians, coefficients)
        moduli = np.abs(values)
        rows.append({
            "label": label,
            "coefficients": {name: float(value)
                             for name, value in zip(NAMES, coefficients,
                                                    strict=True)
                             if value > 0},
            "leading_modulus": float(moduli[0]),
            "leading_growth_rate": float(np.log(moduli[0])/DT),
            "count_above_1_plus_1e_minus_7_in_32": int(
                np.sum(moduli > 1+1e-7)),
            "leading_values": [{"real": float(value.real),
                                "imag": float(value.imag),
                                "modulus": float(abs(value))}
                               for value in values[:12]],
        })
        print(label, rows[-1]["leading_growth_rate"], flush=True)
    best = min(rows[1:], key=lambda row: row["leading_growth_rate"])
    result = {
        "status": "V4_FACE_EXACT_SHIFT_INVERT_SCAN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "shift": SIGMA,
        "resolved_outer_mode_count": 32,
        "rows": rows,
        "best_trial": best["label"],
        "best_growth_rate": best["leading_growth_rate"],
        "verdict": (
            "FACE_SCAN_CLOSES_RESOLVED_OUTER_SPECTRUM"
            if best["count_above_1_plus_1e_minus_7_in_32"] == 0 else
            "FACE_SCAN_RETAINS_RESOLVED_INSTABILITY"),
        "claim_boundary": (
            "Exact discrete shift-invert modes nearest the declared real "
            "shift. This is not yet a contour proof of the complete spectrum, "
            "coefficient provenance, nonlinear stability, or movement."),
    }
    (HERE/"personal_touch_face_shift_invert_scan.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"best_trial": result["best_trial"],
                      "best_growth_rate": result["best_growth_rate"],
                      "verdict": result["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
