"""Exact discrete test of the activated chi/Psi bond cone candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import personal_touch_face_psd_cone as cone
import personal_touch_face_shift_invert_scan as face_scan
import personal_touch_vacuum_chi_exact_scan as exact


HERE = Path(__file__).resolve().parent
FACE_COEFFICIENTS = np.asarray((
    0.14705479982595762, 0.14705479982595762,
    0.075, 0.075, 0.10294520017404238))
ACTIVATED_NAMES = (
    "vacuum_vacuum_chi",
    "carrier_vacuum_chi",
    "carrier_carrier_chi",
    "carrier_carrier_Psi_tight_frame",
)
ACTIVATED_COEFFICIENTS = np.asarray((
    0.000753123914826186,
    0.0967079843896992,
    0.0,
    6.94406813381766,
))
SCALES = (2.0, 2.5, 3.0, 4.0)
REFINED_SHIFT = 1.00000005
REFINED_REQUESTED = 2


def main() -> None:
    exact.SIGMA = REFINED_SHIFT
    exact.REQUESTED = REFINED_REQUESTED
    data = np.load(cone.CACHE)
    activated = np.load(
        HERE/"personal_touch_activated_bond_basis_v2_cache.npz")
    face_hessians = [data[f"hessian_{name}"] for name in face_scan.NAMES]
    activated_hessians = [activated[f"hessian_{name}"]
                          for name in ACTIVATED_NAMES]
    base = data["stiffness"]+sum(
        value*hessian for value, hessian in zip(
            FACE_COEFFICIENTS, face_hessians, strict=True))
    rows = []
    for scale in SCALES:
        stiffness = base+sum(
            scale*value*hessian for value, hessian in zip(
                ACTIVATED_COEFFICIENTS, activated_hessians, strict=True))
        values, converged = exact.targeted_spectrum(
            data["mass"], data["gyro"], stiffness)
        moduli = np.abs(values)
        row = {
            "scale": scale,
            "converged_all_requested": converged and len(values) == exact.REQUESTED,
            "converged_count": int(len(values)),
            "leading_modulus": float(moduli[0]) if len(values) else None,
            "growth_rate": (float(np.log(moduli[0])/exact.small.DT)
                            if len(values) else None),
            "count_above_1_plus_1e_minus_7": int(
                np.sum(moduli > 1+1e-7)),
            "values": [
                {"real": float(value.real), "imag": float(value.imag),
                 "modulus": float(abs(value))} for value in values],
        }
        rows.append(row)
        print(scale, row["growth_rate"], row["converged_count"],
              row["converged_all_requested"], flush=True)
    eligible = [row for row in rows if row["converged_all_requested"]]
    best = min(eligible, key=lambda row: row["growth_rate"]) if eligible else None
    result = {
        "status": "V4_ACTIVATED_BOND_EXACT_DISCRETE_NEAR_UNIT_SCAN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "face_coefficients": dict(zip(
            face_scan.NAMES, FACE_COEFFICIENTS.tolist(), strict=True)),
        "activated_coefficients": dict(zip(
            ACTIVATED_NAMES, ACTIVATED_COEFFICIENTS.tolist(), strict=True)),
        "rows": rows,
        "best_complete_row": best,
        "verdict": (
            "ACTIVATED_BOND_CANDIDATE_CLOSES_TARGETED_DISCRETE_MODES"
            if best is not None
            and best["count_above_1_plus_1e_minus_7"] == 0 else
            "ACTIVATED_BOND_CANDIDATE_RETAINS_TARGETED_GROWTH"
            if best is not None else "ACTIVATED_BOND_EXACT_SCAN_INCONCLUSIVE"),
        "claim_boundary": (
            "Four multipliers nearest the formerly leading real shift. A "
            "targeted pass still requires complete contour count, coefficient "
            "provenance, refinements, conservation, nonlinear and movement gates."),
    }
    (HERE/"personal_touch_activated_bond_exact_near_unit_scan.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"best_complete_row": best,
                      "verdict": result["verdict"]},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
