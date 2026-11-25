"""Resolve the native common-pair gyroscopic stability minimum.

This diagnostic locates a possible stability interval; it does not select or
admit a coefficient.  The scale is reported relative to B*chi0 so that any
critical value can subsequently be compared with a native bifurcation or
discrete-clock identity rather than fitted to Dirac motion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import canonical_noncoplanar_bifurcation_audit as native_time
import complete_pair_wave_action_gate as spectrum
import current_action_two_layer_spectrum as current
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent


def main() -> None:
    bare = np.load(canonical.CACHE)
    additions = np.load(HERE/"personal_touch_activated_bond_basis_v2_cache.npz")
    cached = np.load(HERE/"density_oriented_pair_connection_matrices.npz")
    blocks = {kind: cached[f"{kind}_common_pair_connection"]
              for kind in ("mass", "gyro", "stiffness")}
    base_stiffness = (
        bare["stiffness"]
        + additions["hessian_carrier_carrier_chi"]
        + additions["hessian_carrier_carrier_Psi_tight_frame"])
    natural = native_time.B*native_time.CHI0
    rows = []
    for factor in np.linspace(0.85, 1.15, 13):
        coefficient = float(factor*natural)
        mass = bare["mass"]+coefficient*blocks["mass"]
        gyro = bare["gyro"]+coefficient*blocks["gyro"]
        stiffness = base_stiffness+coefficient*blocks["stiffness"]
        dressed = current.dressed_configuration(stiffness, mass, bare["rest"])
        row = spectrum.spectrum_row(
            mass, gyro, stiffness, dressed, native_time.DT)
        row.update({"factor": float(factor), "coefficient": coefficient})
        rows.append(row)
    best = min(rows, key=lambda row: row["maximum_growth_rate"])
    result = {
        "status": "V4_COMMON_CONNECTION_BIFURCATION_SCAN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "natural_scale": natural,
        "rows": rows,
        "best": best,
        "verdict": (
            "COARSE_STABLE_INTERVAL_FOUND"
            if best["growing_multiplier_count_at_1e_minus_7"] == 0
            else "COMMON_CONNECTION_MINIMUM_RETAINS_GROWTH"),
        "claim_boundary": (
            "A scan diagnoses the gyroscopic bifurcation. Its minimum is not "
            "coefficient provenance and no action is installed."),
    }
    (HERE/"common_connection_bifurcation_scan.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
