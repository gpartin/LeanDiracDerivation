"""Coefficient-free unit-weight geometric action spectrum screen.

Every independently normalized positive face identity and activated bond-class
identity receives unit weight.  No stability, Dirac, or movement observable
selects a coefficient.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import personal_touch_activated_bond_exact_scan as activated_names
import personal_touch_face_psd_cone as cone
import personal_touch_face_shift_invert_scan as face_scan
import personal_touch_vacuum_chi_exact_scan as exact


HERE = Path(__file__).resolve().parent
SHIFTS = (
    1.00000005+0j,
    1.000001+1e-6j,
    1.000001+5e-6j,
    1.000001+1e-5j,
    1.000005+5e-5j,
    1.00001+1e-4j,
)


def main() -> None:
    data = np.load(cone.CACHE)
    activated = np.load(
        HERE/"personal_touch_activated_bond_basis_v2_cache.npz")
    stiffness = data["stiffness"].copy()
    for name in face_scan.NAMES:
        stiffness += data[f"hessian_{name}"]
    for name in activated_names.ACTIVATED_NAMES:
        stiffness += activated[f"hessian_{name}"]
    exact.REQUESTED = 2
    rows, collected = [], []
    for shift in SHIFTS:
        exact.SIGMA = shift
        values, converged = exact.targeted_spectrum(
            data["mass"], data["gyro"], stiffness)
        collected.extend(values.tolist())
        row = {
            "shift": {"real": float(shift.real), "imag": float(shift.imag)},
            "converged": converged and len(values) == exact.REQUESTED,
            "values": [
                {"real": float(value.real), "imag": float(value.imag),
                 "modulus": float(abs(value))} for value in values],
        }
        rows.append(row)
        print(shift, row["converged"], row["values"], flush=True)
    unique = []
    for value in collected:
        if not any(abs(value-existing) < 1e-8 for existing in unique):
            unique.append(value)
    moduli = np.abs(unique)
    result = {
        "status": "V4_UNIT_GEOMETRIC_ACTION_MULTISHIFT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "coefficient_rule": (
            "unit weight for every independently normalized positive face "
            "and activated bond identity; no fitted coefficient"),
        "face_coefficients": {name: 1.0 for name in face_scan.NAMES},
        "activated_coefficients": {
            name: 1.0 for name in activated_names.ACTIVATED_NAMES},
        "rows": rows,
        "maximum_resolved_modulus": float(np.max(moduli)),
        "count_resolved_above_1_plus_1e_minus_7": int(
            np.sum(moduli > 1+1e-7)),
        "verdict": (
            "UNIT_GEOMETRIC_ACTION_RESOLVED_BAND_PASS"
            if all(row["converged"] for row in rows)
            and not np.any(moduli > 1+1e-7)
            else "UNIT_GEOMETRIC_ACTION_RETAINS_GROWTH"
            if np.any(moduli > 1+1e-7) else
            "UNIT_GEOMETRIC_ACTION_INCONCLUSIVE"),
        "claim_boundary": (
            "Exact modes nearest six complex shifts across the prior unstable "
            "band. Still requires contour count, refinements, conservation, "
            "nonlinear stability, motion, and downstream Dirac regressions."),
    }
    (HERE/"personal_touch_unit_geometric_action.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({
        "maximum_resolved_modulus": result["maximum_resolved_modulus"],
        "count_resolved_above_1_plus_1e_minus_7": result[
            "count_resolved_above_1_plus_1e_minus_7"],
        "verdict": result["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
