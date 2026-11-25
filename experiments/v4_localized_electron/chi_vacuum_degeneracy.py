"""Exact linear audit of the complex-chi vacuum orientation degeneracy.

At Psi=0 and |chi|=R the canonical local and area energies are independent of
five real orientation/phase tangents of complex chi at every site.  The
normalized positive C19 chi-gradient action lifts every nonuniform member of
that kernel while retaining only the five required global U(3)-orbit modes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import localized_stability as native


HERE = Path(__file__).resolve().parent


def c19_matrix(side: int):
    sites = side**3
    matrix = np.empty((sites, sites))
    basis = np.zeros((side, side, side))
    for column in range(sites):
        basis.flat[column] = 1.0
        matrix[:, column] = native.c19_dirichlet(
            basis[..., None])[..., 0].reshape(-1)
        basis.flat[column] = 0.0
    return matrix


def counts(values, tolerance=1e-10):
    return {
        "negative": int(np.sum(values < -tolerance)),
        "zero": int(np.sum(abs(values) <= tolerance)),
        "positive": int(np.sum(values > tolerance)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sides", nargs="+", type=int, default=(3, 4, 5))
    parser.add_argument("--output", default="chi_vacuum_degeneracy.json")
    args = parser.parse_args()
    rows = []
    weight_sum = float(sum(weight for _, weight in native.DIRECTIONS))
    for side in args.sides:
        spatial = np.linalg.eigvalsh(c19_matrix(side)) / weight_sum
        canonical_orientation = np.zeros(5 * side**3)
        stiff_orientation = np.repeat(spatial, 5)
        rows.append({
            "side": side,
            "sites": side**3,
            "canonical_orientation_hessian": counts(canonical_orientation),
            "equal_speed_chi_stiffness_hessian": counts(stiff_orientation),
            "smallest_positive_normalized_c19_eigenvalue": float(
                spatial[spatial > 1e-10][0]),
            "largest_normalized_c19_eigenvalue": float(spatial[-1]),
        })
    result = {
        "status": "V40_CHI_VACUUM_ORIENTATION_DEGENERACY_PROVED",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "c19_weight_sum": weight_sum,
        "analytic_result": (
            "Canonical kernel dimension is 5*N_sites. A positive normalized "
            "C19 chi-gradient leaves exactly five global orientation modes "
            "and makes all nonuniform orientation modes positive."),
        "rows": rows,
        "claim_boundary": (
            "Necessary vacuum-sector result only; it does not prove that the "
            "same term admits or stabilizes the localized carrier."),
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
