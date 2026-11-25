"""Test the proper four-component small-tail carrier under the stable action.

Unlike the rejected compact common-phase launch, this gate builds the
translation-equivariant overlapping-cube lift of each geometry-derived
four-component copy.  It asks whether the complete all-center carrier family
is invariant under the native configuration tangent of the unit
cube-incidence action.  No Dirac operator is inserted in the action.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize

import cube_incidence_nonlinear_action as action
import current_action_localized_envelope_closure as envelope


HERE = Path(__file__).resolve().parent


def classify(acceleration, basis, psi_frame, chi_frame, sites, offsets,
             seed, optimize=True):
    def build(values):
        scale = complex(values[0], values[1])
        return envelope.orth(envelope.raw_columns(
            basis, psi_frame, chi_frame, sites, offsets, scale))

    def objective(values):
        if (not np.all(np.isfinite(values))
                or np.max(np.abs(values)) > 0.1):
            return 1e6
        try:
            return envelope.closure(acceleration, build(values))[
                "relative_cross_block"]
        except np.linalg.LinAlgError:
            return 1e6

    if optimize:
        fit = minimize(
            objective, np.asarray((seed.real, seed.imag)), method="Nelder-Mead",
            options={"maxiter": 120, "xatol": 1e-8, "fatol": 1e-8})
        values = fit.x
        optimizer_success = bool(fit.success)
        evaluations = int(fit.nfev)
    else:
        values = np.asarray((seed.real, seed.imag))
        optimizer_success = None
        evaluations = 1
    q = build(values)
    row = envelope.closure(acceleration, q)
    row.update({
        "chi_scale_real": float(values[0]),
        "chi_scale_imag": float(values[1]),
        "optimizer_success": optimizer_success,
        "optimizer_evaluations": evaluations,
    })
    return row, q


def main():
    bare, matrices = action.expected_blocks()
    # This is the rotating-frame configuration response at zero tangent
    # velocity.  The independent two-layer/gyro closure remains a later gate.
    acceleration = -solve(
        matrices["mass"], matrices["stiffness"], assume_a="sym")
    psi_frame, chi_frame, sites = envelope.carrier_frames(bare["rest"])
    sets = {
        "nearest_27_centers": list(
            itertools.product((-1, 0, 1), repeat=3)),
        "all_125_centers": list(
            itertools.product((-2, -1, 0, 1, 2), repeat=3)),
    }
    rows = []
    saved = {}
    for flavor, basis in envelope.flavor_bases().items():
        seed = 1/np.sqrt(1197)+0j
        cases = []
        for name, offsets in sets.items():
            row, q = classify(
                acceleration, basis, psi_frame, chi_frame, sites, offsets,
                seed, optimize=(name == "nearest_27_centers"))
            row["envelope"] = name
            cases.append(row)
            seed = complex(row["chi_scale_real"], row["chi_scale_imag"])
            if name == "all_125_centers":
                saved[flavor] = q
        rows.append({"flavor": flavor, "cases": cases})
    worst_all = max(next(case for case in row["cases"]
                         if case["envelope"] == "all_125_centers")[
                             "relative_cross_block"] for row in rows)
    np.savez_compressed(
        HERE/"cube_incidence_dirac_envelope_gate.npz", **saved)
    result = {
        "status": "V4_CUBE_INCIDENCE_DIRAC_ENVELOPE_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "construction": (
            "translation-equivariant overlapping-cube lift of each native "
            "four-component flavor copy, with a freely optimized complex "
            "chi/Psi tangent ratio"),
        "rows": rows,
        "worst_all_center_cross_block": float(worst_all),
        "verdict": (
            "FOUR_COMPONENT_CONFIGURATION_ENVELOPE_PASS"
            if worst_all < 1e-8 else
            "FOUR_COMPONENT_CONFIGURATION_ENVELOPE_FAIL"),
        "claim_boundary": (
            "Configuration-tangent closure only. A pass still requires the "
            "two committed layers, native current pullback, nonlinear moving "
            "branch and endurance gates before V4-006."),
    }
    (HERE/"cube_incidence_dirac_envelope_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
