"""Test the existing positive chi interface terms on the new limiting mode.

After the common pair connection stabilizes the in-plane Psi family, the
limiting mode is an out-of-plane chi interface deformation.  The unit
density-activated chi bond squares were derived earlier and are re-tested
here at their geometric unit coefficient; no new target term is invented.
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
    activated = np.load(HERE/"personal_touch_activated_bond_basis_v2_cache.npz")
    connection = np.load(HERE/"density_oriented_pair_connection_matrices.npz")
    common = {kind: connection[f"{kind}_common_pair_connection"]
              for kind in ("mass", "gyro", "stiffness")}
    natural = native_time.B*native_time.CHI0
    combinations = (
        ("no_interface", 0.0, 0.0),
        ("carrier_vacuum_unit", 1.0, 0.0),
        ("vacuum_vacuum_unit", 0.0, 1.0),
        ("both_interface_unit", 1.0, 1.0),
    )
    rows = []
    for common_factor in (0.95, 1.0):
        coefficient = common_factor*natural
        mass = bare["mass"]+coefficient*common["mass"]
        gyro = bare["gyro"]+coefficient*common["gyro"]
        for label, carrier_vacuum, vacuum_vacuum in combinations:
            stiffness = (
                bare["stiffness"]
                + activated["hessian_carrier_carrier_chi"]
                + activated["hessian_carrier_carrier_Psi_tight_frame"]
                + carrier_vacuum*activated["hessian_carrier_vacuum_chi"]
                + vacuum_vacuum*activated["hessian_vacuum_vacuum_chi"]
                + coefficient*common["stiffness"])
            dressed = current.dressed_configuration(
                stiffness, mass, bare["rest"])
            row = spectrum.spectrum_row(
                mass, gyro, stiffness, dressed, native_time.DT)
            row.update({
                "label": label,
                "common_factor": common_factor,
                "common_coefficient": coefficient,
                "carrier_vacuum_coefficient": carrier_vacuum,
                "vacuum_vacuum_coefficient": vacuum_vacuum,
            })
            rows.append(row)
    best = min(rows, key=lambda row: row["maximum_growth_rate"])
    result = {
        "status": "V4_CONNECTION_CHI_INTERFACE_COMPLETION_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "best": best,
        "verdict": (
            "CHI_INTERFACE_COMPLETION_FULL_SPECTRUM_PASS"
            if best["growing_multiplier_count_at_1e_minus_7"] == 0
            else "CHI_INTERFACE_COMPLETION_RETAINS_GROWTH"),
        "claim_boundary": (
            "This reuses prior positive unit bond terms. A spectral pass "
            "would still require action minimality, refinements, nonlinear "
            "motion and current. No action is installed here."),
    }
    (HERE/"connection_chi_interface_completion_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
