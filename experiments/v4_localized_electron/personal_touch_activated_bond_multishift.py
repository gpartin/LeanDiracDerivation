"""Multi-shift exact discrete audit of the scale-three activated bond cone."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import personal_touch_activated_bond_exact_scan as candidate
import personal_touch_face_psd_cone as cone
import personal_touch_face_shift_invert_scan as face_scan
import personal_touch_vacuum_chi_exact_scan as exact


HERE = Path(__file__).resolve().parent
SCALE = 3.0
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
    face_hessians = [data[f"hessian_{name}"] for name in face_scan.NAMES]
    activated_hessians = [activated[f"hessian_{name}"]
                          for name in candidate.ACTIVATED_NAMES]
    stiffness = data["stiffness"]+sum(
        value*hessian for value, hessian in zip(
            candidate.FACE_COEFFICIENTS, face_hessians, strict=True))
    stiffness += sum(
        SCALE*value*hessian for value, hessian in zip(
            candidate.ACTIVATED_COEFFICIENTS,
            activated_hessians, strict=True))
    exact.REQUESTED = 2
    rows, collected = [], []
    for shift in SHIFTS:
        exact.SIGMA = shift
        values, converged = exact.targeted_spectrum(
            data["mass"], data["gyro"], stiffness)
        collected.extend(values.tolist())
        row = {
            "shift": {"real": float(shift.real),
                      "imag": float(shift.imag)},
            "converged": converged and len(values) == exact.REQUESTED,
            "count": int(len(values)),
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
        "status": "V4_ACTIVATED_BOND_SCALE3_MULTISHIFT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "scale": SCALE,
        "rows": rows,
        "unique_resolved_values": [
            {"real": float(value.real), "imag": float(value.imag),
             "modulus": float(abs(value))} for value in unique],
        "maximum_resolved_modulus": float(np.max(moduli)),
        "count_resolved_above_1_plus_1e_minus_7": int(
            np.sum(moduli > 1+1e-7)),
        "verdict": (
            "MULTISHIFT_RESOLVED_BAND_PASS"
            if all(row["converged"] for row in rows)
            and not np.any(moduli > 1+1e-7)
            else "MULTISHIFT_RESOLVED_BAND_RETAINS_GROWTH"
            if np.any(moduli > 1+1e-7) else
            "MULTISHIFT_RESOLVED_BAND_INCONCLUSIVE"),
        "claim_boundary": (
            "Exact modes nearest seven real shifts across the prior unstable "
            "band. This is stronger than one targeted shift but is not yet a "
            "closed complex-contour eigenvalue count."),
    }
    (HERE/"personal_touch_activated_bond_multishift.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({
        "maximum_resolved_modulus": result["maximum_resolved_modulus"],
        "count_resolved_above_1_plus_1e_minus_7": result[
            "count_resolved_above_1_plus_1e_minus_7"],
        "verdict": result["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
