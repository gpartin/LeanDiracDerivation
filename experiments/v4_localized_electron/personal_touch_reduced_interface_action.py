"""Full multishift screen of the greedy one-term geometric reduction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import personal_touch_face_psd_cone as cone
import personal_touch_unit_geometric_action as unit
import personal_touch_unit_geometric_canonical as canonical
import personal_touch_vacuum_chi_exact_scan as exact


HERE = Path(__file__).resolve().parent
TERMS = ("carrier_carrier_chi", "carrier_carrier_Psi_tight_frame")


def main() -> None:
    data = np.load(canonical.CACHE)
    activated = np.load(
        HERE/"personal_touch_activated_bond_basis_v2_cache.npz")
    stiffness = data["stiffness"]+sum(
        activated[f"hessian_{term}"] for term in TERMS)
    exact.REQUESTED = 2
    rows = []
    values_all: list[complex] = []
    for shift in canonical.SHIFTS:
        exact.SIGMA = shift
        values, converged = exact.targeted_spectrum(
            data["mass"], data["gyro"], stiffness)
        values_all.extend(values.tolist())
        row = {
            "shift": {"real": float(shift.real),
                      "imag": float(shift.imag)},
            "converged": bool(converged and len(values) == exact.REQUESTED),
            "values": [
                {"real": float(value.real), "imag": float(value.imag),
                 "modulus": float(abs(value))} for value in values],
        }
        rows.append(row)
        print(shift, row["converged"], row["values"], flush=True)
    maximum = max(abs(value) for value in values_all)
    result = {
        "status": "V4_REDUCED_INTERFACE_ACTION_MULTISHIFT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "retained_unit_terms": [f"bond:{term}" for term in TERMS],
        "rows": rows,
        "maximum_resolved_modulus": float(maximum),
        "count_resolved_above_1_plus_1e_minus_7": int(sum(
            abs(value) > 1+1e-7 for value in values_all)),
        "verdict": (
            "REDUCED_TIGHT_FRAME_RESOLVED_BAND_PASS"
            if all(row["converged"] for row in rows)
            and maximum <= 1+1e-7
            else "REDUCED_TIGHT_FRAME_RETAINS_GROWTH"),
        "claim_boundary": (
            "Six-shift screen of the previously unstable band for the greedy "
            "one-term reduction. Complete contour, nonlinear, refinement, "
            "movement, and Dirac gates remain open."),
    }
    (HERE/"personal_touch_reduced_interface_action.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
