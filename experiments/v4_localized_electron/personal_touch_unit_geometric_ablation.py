"""Leave-one-out screen for the unit geometric research action.

This is a reduction diagnostic, not a complete contour proof.  It removes
one independently normalized positive identity at a time and probes the
previously unstable multiplier band with exact shift-invert solves.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import personal_touch_activated_bond_exact_scan as activated_names
import personal_touch_unit_geometric_canonical as canonical
import personal_touch_face_psd_cone as cone
import personal_touch_face_shift_invert_scan as face_scan
import personal_touch_vacuum_chi_exact_scan as exact


HERE = Path(__file__).resolve().parent
SHIFTS = canonical.SHIFTS
THRESHOLD = 1.0+1e-7


def main() -> None:
    data = np.load(canonical.CACHE)
    faces = np.load(cone.CACHE)
    activated = np.load(
        HERE/"personal_touch_activated_bond_basis_v2_cache.npz")
    terms = {
        **{f"face:{name}": faces[f"hessian_{name}"]
           for name in face_scan.NAMES},
        **{f"bond:{name}": activated[f"hessian_{name}"]
           for name in activated_names.ACTIVATED_NAMES},
    }
    full = data["stiffness"]+sum(terms.values())
    exact.REQUESTED = 2
    rows = []
    for omitted, hessian in terms.items():
        values_all: list[complex] = []
        shift_rows = []
        for shift in SHIFTS:
            exact.SIGMA = shift
            values, converged = exact.targeted_spectrum(
                data["mass"], data["gyro"], full-hessian)
            values_all.extend(values.tolist())
            shift_rows.append({
                "shift": {"real": float(shift.real),
                          "imag": float(shift.imag)},
                "converged": bool(converged and len(values) == exact.REQUESTED),
                "values": [
                    {"real": float(value.real),
                     "imag": float(value.imag),
                     "modulus": float(abs(value))} for value in values],
            })
        maximum = max((abs(value) for value in values_all), default=float("nan"))
        row = {
            "omitted": omitted,
            "maximum_resolved_modulus": float(maximum),
            "reopens_resolved_growth": bool(maximum > THRESHOLD),
            "all_shifts_converged": all(
                item["converged"] for item in shift_rows),
            "shifts": shift_rows,
        }
        rows.append(row)
        print(omitted, row["maximum_resolved_modulus"],
              row["reopens_resolved_growth"], flush=True)
    retained = list(terms)
    removed: list[str] = []
    greedy_steps = []
    changed = True
    while changed:
        changed = False
        for omitted in list(retained):
            stiffness = data["stiffness"]+sum(
                terms[name] for name in retained if name != omitted)
            maxima = []
            converged_all = True
            for shift in SHIFTS:
                exact.SIGMA = shift
                values, converged = exact.targeted_spectrum(
                    data["mass"], data["gyro"], stiffness)
                converged_all = converged_all and bool(
                    converged and len(values) == exact.REQUESTED)
                maxima.extend(abs(value) for value in values)
            maximum = max(maxima, default=float("nan"))
            accepted = bool(converged_all and maximum <= THRESHOLD)
            greedy_steps.append({
                "omitted": omitted,
                "retained_before": list(retained),
                "maximum_resolved_modulus": float(maximum),
                "all_shifts_converged": converged_all,
                "removal_accepted": accepted,
            })
            print("greedy", omitted, maximum, accepted, flush=True)
            if accepted:
                retained.remove(omitted)
                removed.append(omitted)
                changed = True
                break
    result = {
        "status": "V4_UNIT_GEOMETRIC_LEAVE_ONE_OUT_SCREEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "threshold": THRESHOLD,
        "shifts": [
            {"real": float(shift.real), "imag": float(shift.imag)}
            for shift in SHIFTS],
        "rows": rows,
        "terms_required_by_this_screen": [
            row["omitted"] for row in rows
            if row["reopens_resolved_growth"]],
        "terms_not_resolved_as_required": [
            row["omitted"] for row in rows
            if not row["reopens_resolved_growth"]],
        "greedy_reduction": {
            "removed": removed,
            "retained": retained,
            "steps": greedy_steps
        },
        "claim_boundary": (
            "Two-shift leave-one-out screen of the previously unstable band. "
            "A reopened multiplier establishes necessity for this local "
            "screen; a clean omission is inconclusive until the reduced "
            "combination passes the complete multishift and nonlinear gates."),
    }
    (HERE/"personal_touch_unit_geometric_ablation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
